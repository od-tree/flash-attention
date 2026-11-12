"""Paged head-TP decode on two peer GPUs.

Short rows use a normal page: every KV head, ``block_size`` tokens. Long rows
pack this rank's KV heads into the same physical page, ``d`` logical blocks
each. A short row is never padded out to the longest request.

The five pure-long shapes are the ones in ``operator_refactor.md``. The mixed
shapes are a short batch plus a few 128k or 256k requests. Those do not fit
if every row is padded to the max length; paged storage does.

``d = 4`` communication is not run. Two GPUs can only time ``d = 2``.
"""

import ctypes
import importlib.util
import math
import os
import time
from pathlib import Path

import torch

BLOCK = 16
DEGREE = 2
KV = 2
Q = 16
DIM = 256
Q_PER = Q // DEGREE
SCALE = DIM ** -0.5
OWNER = 0
PARTNER = 1
DTYPE = torch.bfloat16
WARMUP = 3
ITERS = 8

_ROOT = Path(__file__).resolve().parents[1]


def _page_tests():
    for name in ("test_head_tp_page_d2", "test_head_tp_page_d4"):
        path = Path(__file__).resolve().parent / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in sorted(dir(module)):
            if attr.startswith("test_"):
                getattr(module, attr)()
                print(f"page {name}.{attr} ok", flush=True)


def _enable_peers():
    if torch.cuda.device_count() < 2:
        raise SystemExit("two CUDA devices are required")
    if not torch.cuda.can_device_access_peer(0, 1) or not torch.cuda.can_device_access_peer(1, 0):
        raise SystemExit("cuda peer access is not available")
    lib = None
    for soname in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            lib = ctypes.CDLL(soname)
            break
        except OSError:
            continue
    if lib is None:
        raise SystemExit("libcudart was not found")
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    lib.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
    for src, dst in ((0, 1), (1, 0)):
        err = lib.cudaSetDevice(src)
        if err != 0:
            raise SystemExit(f"cudaSetDevice({src}) returned {err}")
        err = lib.cudaDeviceEnablePeerAccess(dst, 0)
        if err not in (0, 704):
            raise SystemExit(f"peer access {src}->{dst} returned {err}")


def _ops():
    import vllm_flash_attn._vllm_fa2_C  # noqa: F401

    return torch.ops._vllm_fa2_C


def _blocks(seq):
    return (seq + BLOCK - 1) // BLOCK


def _packed_pages(seq):
    span = DEGREE * BLOCK
    if seq % span != 0:
        raise ValueError(f"long seq {seq} is not a multiple of {span}")
    return seq // span


def _standard_cache(seqs, device, seed, fill="randn"):
    counts = [_blocks(seq) for seq in seqs]
    total = sum(counts)
    if fill == "empty":
        k = torch.empty(total, BLOCK, KV, DIM, dtype=DTYPE, device=device)
        v = torch.empty(total, BLOCK, KV, DIM, dtype=DTYPE, device=device)
        generator = None
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        k = torch.randn(total, BLOCK, KV, DIM, dtype=DTYPE, device=device, generator=generator)
        v = torch.randn(total, BLOCK, KV, DIM, dtype=DTYPE, device=device, generator=generator)
    table = torch.zeros(len(seqs), max(counts), dtype=torch.int32, device=device)
    cursor = 0
    spans = []
    for row, count in enumerate(counts):
        table[row, :count] = torch.arange(cursor, cursor + count, device=device, dtype=torch.int32)
        spans.append((cursor, count))
        cursor += count
    if generator is None:
        q = torch.empty(len(seqs), Q, DIM, dtype=DTYPE, device=device)
    else:
        q = torch.randn(len(seqs), Q, DIM, dtype=DTYPE, device=device, generator=generator)
    seqused = torch.tensor(seqs, dtype=torch.int32, device=device)
    return k, v, table, spans, q, seqused


def _logical_head(cache, table_row, seq, head):
    count = _blocks(seq)
    pages = cache[table_row[:count].long()]
    return pages[:, :, head, :].reshape(-1, DIM)[:seq]


def _pack_head(tokens):
    span = DEGREE * BLOCK
    pages, slots, width, dim = tokens.numel() // (span * DIM), DEGREE, BLOCK, DIM
    del pages
    view = tokens.view(-1, slots, width, dim)
    return view.permute(0, 2, 1, 3).contiguous()


def _cat(parts):
    offsets = []
    cursor = 0
    for part in parts:
        offsets.append(cursor)
        cursor += part.shape[0]
    return torch.cat(parts, dim=0), offsets


def _table(offsets, counts, device):
    width = max(counts)
    table = torch.zeros(len(counts), width, dtype=torch.int32, device=device)
    for row, (offset, count) in enumerate(zip(offsets, counts)):
        table[row, :count] = torch.arange(offset, offset + count, device=device, dtype=torch.int32)
    return table


def _copy_pack(dst_k, dst_v, cursor, tokens_k, tokens_v):
    pages_k = _pack_head(tokens_k)
    pages_v = _pack_head(tokens_v)
    count = pages_k.shape[0]
    dst_k[cursor : cursor + count] = pages_k
    dst_v[cursor : cursor + count] = pages_v
    return count


def _split_cache(k, v, table, seqs, n_long, head, device):
    """Pack long rows of one KV head straight into `device`."""
    counts = [_packed_pages(seqs[row]) for row in range(n_long)]
    cache_k = torch.empty(sum(counts), BLOCK, KV, DIM, dtype=DTYPE, device=device)
    cache_v = torch.empty_like(cache_k)
    cursor = 0
    for row, count in enumerate(counts):
        pages_k = _pack_head(_logical_head(k, table[row], seqs[row], head))
        pages_v = _pack_head(_logical_head(v, table[row], seqs[row], head))
        cache_k[cursor : cursor + count] = pages_k
        cache_v[cursor : cursor + count] = pages_v
        cursor += count
    offsets = []
    cursor = 0
    for count in counts:
        offsets.append(cursor)
        cursor += count
    return cache_k, cache_v, _table(offsets, counts, device)


def _append_local(long_k, long_v, long_counts, short_k, short_v, short_spans):
    """Keep the packed long rows, then the partner's own full-head short pages."""
    short_counts = [count for _, count in short_spans]
    counts = list(long_counts) + short_counts
    cache_k = torch.empty(sum(counts), BLOCK, KV, DIM, dtype=DTYPE, device=long_k.device)
    cache_v = torch.empty_like(cache_k)
    cache_k[: long_k.shape[0]] = long_k
    cache_v[: long_v.shape[0]] = long_v
    cursor = long_k.shape[0]
    for start, count in short_spans:
        cache_k[cursor : cursor + count] = short_k[start : start + count]
        cache_v[cursor : cursor + count] = short_v[start : start + count]
        cursor += count
    offsets = []
    cursor = 0
    for count in counts:
        offsets.append(cursor)
        cursor += count
    return cache_k, cache_v, _table(offsets, counts, long_k.device)


def _random_pages(seqs, n_long, device):
    counts = [
        _packed_pages(seq) if row < n_long else _blocks(seq)
        for row, seq in enumerate(seqs)
    ]
    cache_k = torch.empty(sum(counts), BLOCK, KV, DIM, dtype=DTYPE, device=device)
    cache_v = torch.empty_like(cache_k)
    offsets = []
    cursor = 0
    for count in counts:
        offsets.append(cursor)
        cursor += count
    return cache_k, cache_v, _table(offsets, counts, device)


def _owner_cache(k, v, table, spans, seqs, n_long):
    """Long rows are packed head 0. Short rows keep every KV head."""
    counts = [
        _packed_pages(seq) if row < n_long else _blocks(seq)
        for row, seq in enumerate(seqs)
    ]
    cache_k = torch.empty(sum(counts), BLOCK, KV, DIM, dtype=DTYPE, device=k.device)
    cache_v = torch.empty_like(cache_k)
    cursor = 0
    for row, (seq, count) in enumerate(zip(seqs, counts)):
        if row < n_long:
            _copy_pack(
                cache_k, cache_v, cursor,
                _logical_head(k, table[row], seq, 0),
                _logical_head(v, table[row], seq, 0),
            )
        else:
            start, width = spans[row]
            cache_k[cursor : cursor + count] = k[start : start + width]
            cache_v[cursor : cursor + count] = v[start : start + width]
        cursor += count
    offsets = []
    cursor = 0
    for count in counts:
        offsets.append(cursor)
        cursor += count
    return cache_k, cache_v, _table(offsets, counts, k.device)


def _current(k, v, table, seqs):
    n = len(seqs)
    k_token = torch.empty(n, KV, DIM, dtype=DTYPE, device=k.device)
    v_token = torch.empty(n, KV, DIM, dtype=DTYPE, device=k.device)
    for row, seq in enumerate(seqs):
        pos = seq - 1
        block = int(table[row, pos // BLOCK])
        token = pos % BLOCK
        k_token[row] = k[block, token]
        v_token[row] = v[block, token]
    return k_token, v_token


def _cu(batch, device):
    return (
        torch.arange(batch + 1, dtype=torch.int32, device=device),
        torch.zeros(batch + 1, dtype=torch.int32, device=device),
    )


def _pack_tile(row, head, split):
    # Bits 0-15 row, 16-23 KV head, 24-31 split. Same layout as
    # vllm.v1.head_tp.pack_head_tp_tile.
    return (row & 0xFFFF) | ((head & 0xFF) << 16) | ((split & 0xFF) << 24)


def _tile_plan(seqs, n_long, kv_begin, num_sms, device):
    """Long rows share about num_sms CTAs, capped at 16 splits. Short rows stay at 1.

    The split count matches vllm.v1.head_tp.build_head_tp_tile_plan.
    """
    if os.environ.get("HEAD_TP_TILES", "1") == "0":
        return None, None, 0
    n = len(seqs)
    long_rows = list(range(n_long))
    kv_per = 1
    row_splits = [1] * n
    if long_rows and n_long * kv_per < num_sms:
        total = sum(seqs[i] for i in long_rows) or 1
        for i in long_rows:
            raw = (num_sms * seqs[i]) // (total * kv_per)
            row_splits[i] = min(16, max(1, raw))
    tiles = []
    for row in long_rows:
        for split in range(row_splits[row]):
            tiles.append(_pack_tile(row, kv_begin, split))
    for row in range(n_long, n):
        for head in range(KV):
            tiles.append(_pack_tile(row, head, 0))
    if not tiles:
        tiles.append(_pack_tile(0, kv_begin, 0))
        row_splits[0] = 1
    return (
        torch.tensor(tiles, dtype=torch.int32, device=device),
        torch.tensor(row_splits, dtype=torch.int8, device=device),
        max(row_splits),
    )


def _varlen(ops, q, k, v, block_table, seqused, cu_q, cu_k, max_k, out, row_kind, pack, kv_begin, src, xchg,
            num_splits=None, tiles=None, row_splits=None, max_tile_splits=0):
    if num_splits is None:
        num_splits = int(os.environ.get("HEAD_TP_SPLITS", "1"))
    # Sixteen exchange tensors, then max_q / max_kv / payload_tokens. The ints
    # have C++ defaults, but tiles are positional after them.
    tail = ([None] * 16 + [0, 0, 0]) if xchg is None else list(xchg)
    ops.varlen_fwd(
        q, k, v, out, cu_q, cu_k, seqused, None, block_table, None,
        1, max_k, 0.0, SCALE, True, False, -1, -1, 0.0, False, num_splits, None,
        row_kind, pack, kv_begin, src, *tail, tiles, row_splits, max_tile_splits,
    )


def _pointers(buffers, device):
    return torch.tensor([buf.data_ptr() for buf in buffers], dtype=torch.int64, device=device)


def _plan(n, n_long, device):
    row_kind = torch.zeros(n, dtype=torch.int8, device=device)
    slot = torch.full((n, DEGREE), -1, dtype=torch.int32, device=device)
    q_range = torch.zeros(n, DEGREE, 2, dtype=torch.int32, device=device)
    kv_range = torch.zeros(n, DEGREE, 2, dtype=torch.int32, device=device)
    slot[:, OWNER] = OWNER
    if n_long:
        row_kind[:n_long] = 1
        slot[:n_long, PARTNER] = PARTNER
        q_range[:n_long, 0, 1] = Q_PER
        q_range[:n_long, 1, 0] = Q_PER
        q_range[:n_long, 1, 1] = Q
        kv_range[:n_long, 0, 1] = 1
        kv_range[:n_long, 1, 0] = 1
        kv_range[:n_long, 1, 1] = KV
    return row_kind, slot, kv_range, q_range


class _Case:
    def __init__(self, ops, seqs, n_long, partner_seqs=None, fill="randn"):
        self.ops = ops
        self.seqs = list(seqs)
        self.n = len(self.seqs)
        self.n_long = n_long
        self.partner_seqs = list(partner_seqs or [])
        self.partner_n = n_long + len(self.partner_seqs)
        self.s0 = torch.cuda.Stream(device=OWNER)
        self.s1 = torch.cuda.Stream(device=PARTNER)
        self.base_splits = int(os.environ.get("HEAD_TP_SPLITS", "1"))
        self.owner_bytes = 0
        k, v, table, spans, q, seqused = _standard_cache(self.seqs, OWNER, 11, fill)
        self.max_k = max(self.seqs)
        self.paged_bytes = k.nbytes + v.nbytes
        self.cu_owner = _cu(self.n, OWNER)
        self.base_batch = (q, k, v, table, seqused)
        self.spans = spans
        self.base_out = torch.empty_like(q)
        self.partner_base = None
        self.partner_spans = None
        self.cu_partner_base = None
        self.partner_max_k = 0
        if self.partner_seqs:
            pk, pv, ptable, pspans, pq, pseq = _standard_cache(self.partner_seqs, PARTNER, 29)
            self.paged_bytes += pk.nbytes + pv.nbytes
            self.partner_base = (pq, pk, pv, ptable, pseq)
            self.partner_spans = pspans
            self.partner_base_out = torch.empty_like(pq)
            self.cu_partner_base = _cu(len(self.partner_seqs), PARTNER)
            self.partner_max_k = max(self.partner_seqs)
        self.owner = None
        self.partner = None
        self.owner_out = torch.empty_like(q)
        self.partner_out = torch.empty(self.partner_n, Q, DIM, dtype=DTYPE, device=PARTNER)
        self.owner_plan = _plan(self.n, n_long, OWNER)
        self.partner_plan = _plan(self.partner_n, n_long, PARTNER)
        self.cu_partner = _cu(self.partner_n, PARTNER)
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.owner_tiles = _tile_plan(self.seqs, n_long, 0, num_sms, OWNER)
        partner_seqs = self.seqs[:n_long] + self.partner_seqs
        self.partner_tiles = _tile_plan(partner_seqs, n_long, 1, num_sms, PARTNER)

    def pack(self):
        """Build the split caches, then drop the full-head cache."""
        if self.owner is not None:
            return
        q, k, v, table, seqused = self.base_batch
        k_token, v_token = _current(k, v, table, self.seqs)
        owner_k, owner_v, owner_table = _owner_cache(k, v, table, self.spans, self.seqs, self.n_long)
        partner_k, partner_v, partner_table = _split_cache(
            k, v, table, self.seqs, self.n_long, head=1, device=PARTNER
        )
        partner_seq = seqused[: self.n_long].to(PARTNER)
        partner_q = torch.zeros(self.partner_n, Q, DIM, dtype=DTYPE, device=PARTNER)
        if self.partner_base is not None:
            pq, pk, pv, _, pseq = self.partner_base
            partner_k, partner_v, partner_table = _append_local(
                partner_k, partner_v,
                [_packed_pages(seq) for seq in self.seqs[: self.n_long]],
                pk, pv, self.partner_spans,
            )
            partner_q[self.n_long :] = pq
            partner_seq = torch.cat((partner_seq, pseq))
        self.owner = (q, owner_k, owner_v, owner_table, seqused, k_token, v_token)
        self.partner = (partner_q, partner_k, partner_v, partner_table, partner_seq)
        self._exchange(k_token, v_token)
        self.owner_bytes = owner_k.nbytes + owner_v.nbytes
        self.base_batch = (q, None, None, None, seqused)
        self.spans = None
        del k, v, table
        torch.cuda.empty_cache()

    def pack_random(self):
        """Drop the full-head cache, then allocate packed pages without a second copy."""
        q, k, v, table, seqused = self.base_batch
        self.base_batch = (q, None, None, None, seqused)
        self.spans = None
        partner_base = self.partner_base
        self.partner_base = None
        self.partner_spans = None
        del k, v, table, partner_base
        torch.cuda.empty_cache()
        owner_k, owner_v, owner_table = _random_pages(self.seqs, self.n_long, OWNER)
        partner_seqs = self.seqs[: self.n_long]
        partner_k, partner_v, partner_table = _random_pages(partner_seqs, self.n_long, PARTNER)
        partner_seq = seqused[: self.n_long].to(PARTNER)
        partner_q = torch.empty(self.partner_n, Q, DIM, dtype=DTYPE, device=PARTNER)
        k_token = torch.empty(self.n, KV, DIM, dtype=DTYPE, device=OWNER)
        v_token = torch.empty(self.n, KV, DIM, dtype=DTYPE, device=OWNER)
        self.owner_bytes = owner_k.nbytes + owner_v.nbytes
        self.owner = (q, owner_k, owner_v, owner_table, seqused, k_token, v_token)
        self.partner = (partner_q, partner_k, partner_v, partner_table, partner_seq)
        self._exchange(k_token, v_token)

    def _exchange(self, k_token, v_token):
        tokens = max(self.n, self.partner_n)
        self.arrived = [
            torch.zeros(DEGREE, tokens, dtype=torch.int32, device=OWNER),
            torch.zeros(DEGREE, tokens, dtype=torch.int32, device=PARTNER),
        ]
        self.return_count = [
            torch.zeros(tokens, dtype=torch.int32, device=OWNER),
            torch.zeros(tokens, dtype=torch.int32, device=PARTNER),
        ]
        self.payload_q = [
            torch.zeros(DEGREE, tokens, Q_PER, DIM, dtype=DTYPE, device=OWNER),
            torch.zeros(DEGREE, tokens, Q_PER, DIM, dtype=DTYPE, device=PARTNER),
        ]
        self.payload_k = [
            torch.zeros(DEGREE, tokens, 1, DIM, dtype=DTYPE, device=OWNER),
            torch.zeros(DEGREE, tokens, 1, DIM, dtype=DTYPE, device=PARTNER),
        ]
        self.payload_v = [
            torch.zeros(DEGREE, tokens, 1, DIM, dtype=DTYPE, device=OWNER),
            torch.zeros(DEGREE, tokens, 1, DIM, dtype=DTYPE, device=PARTNER),
        ]
        self.payload_out = [
            torch.zeros(DEGREE, tokens, Q_PER, DIM, dtype=DTYPE, device=OWNER),
            torch.zeros(DEGREE, tokens, Q_PER, DIM, dtype=DTYPE, device=PARTNER),
        ]
        self.k_token = k_token
        self.v_token = v_token
        tokens = max(self.n, self.partner_n)
        self.k_token_partner = torch.zeros(self.partner_n, KV, DIM, dtype=DTYPE, device=PARTNER)
        self.v_token_partner = torch.zeros(self.partner_n, KV, DIM, dtype=DTYPE, device=PARTNER)
        if self.n_long:
            self.k_token_partner[: self.n_long] = k_token[: self.n_long]
            self.v_token_partner[: self.n_long] = v_token[: self.n_long]
        self.payload_tokens = tokens

    def _xchg(self, src):
        device = PARTNER if src == PARTNER else OWNER
        plan = self.partner_plan if src == PARTNER else self.owner_plan
        _, slot, kv_range, q_range = plan
        payload_tokens = self.payload_tokens
        return (
            slot,
            kv_range,
            q_range,
            self.k_token_partner if src == PARTNER else self.k_token,
            self.v_token_partner if src == PARTNER else self.v_token,
            _pointers(self.payload_q, device),
            _pointers(self.payload_k, device),
            _pointers(self.payload_v, device),
            _pointers(self.payload_out, device),
            _pointers(self.arrived, device),
            self.arrived[src],
            self.payload_out[src],
            self.payload_q[src],
            self.payload_k[src],
            self.payload_v[src],
            self.return_count[src],
            Q_PER,
            1,
            payload_tokens,
        )

    def base(self, starts=None, ends=None):
        q, k, v, table, seqused = self.base_batch
        with torch.cuda.stream(self.s0):
            if starts is not None:
                starts[0].record(self.s0)
            _varlen(
                self.ops, q, k, v, table, seqused, *self.cu_owner, self.max_k,
                self.base_out, None, 0, 0, -1, None, num_splits=self.base_splits,
            )
            if ends is not None:
                ends[0].record(self.s0)
        if self.partner_base is None:
            return
        q, k, v, table, seqused = self.partner_base
        with torch.cuda.stream(self.s1):
            if starts is not None:
                starts[1].record(self.s1)
            _varlen(
                self.ops, q, k, v, table, seqused, *self.cu_partner_base, self.partner_max_k,
                self.partner_base_out, None, 0, 0, -1, None, num_splits=self.base_splits,
            )
            if ends is not None:
                ends[1].record(self.s1)

    def _split_timed(self, starts, ends):
        self._launch(self.s0, self.owner, self.owner_out, self.owner_plan, 0, -1, None, starts, ends, 0)
        self._launch(self.s1, self.partner, self.partner_out, self.partner_plan, 1, -1, None, starts, ends, 1)

    def _std_timed(self, starts, ends):
        self._launch(self.s0, self.owner, self.owner_out, self.owner_plan, 0, OWNER, self._xchg(OWNER), starts, ends, 0)
        self._launch(self.s1, self.partner, self.partner_out, self.partner_plan, 1, PARTNER, self._xchg(PARTNER), starts, ends, 1)

    def _launch(self, stream, batch, out, plan, kv_begin, src, xchg, starts, ends, index):
        row_kind = plan[0]
        q, k, v, table, seqused = batch[:5]
        cu = self.cu_partner if index == 1 else self.cu_owner
        with torch.cuda.stream(stream):
            if starts is not None:
                starts[index].record(stream)
            tiles, row_splits, max_splits = self.owner_tiles if index == 0 else self.partner_tiles
            _varlen(
                self.ops, q, k, v, table, seqused, *cu, self.max_k, out,
                row_kind, DEGREE, kv_begin, src, xchg,
                tiles=tiles, row_splits=row_splits, max_tile_splits=max_splits,
            )
            if ends is not None:
                ends[index].record(stream)

    def clear_flags(self):
        with torch.cuda.stream(self.s0):
            self.arrived[0].zero_()
            self.return_count[0].zero_()
        with torch.cuda.stream(self.s1):
            self.arrived[1].zero_()
            self.return_count[1].zero_()
        self.s0.synchronize()
        self.s1.synchronize()

    def cooperative(self):
        self.clear_flags()
        self._std_timed(None, None)
        self.s0.synchronize()
        self.s1.synchronize()


def _bf16_ulps(got, ref):
    gi = got.view(torch.int16).to(torch.int32) & 0xFFFF
    ri = ref.view(torch.int16).to(torch.int32) & 0xFFFF

    def ordered(bits):
        return torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits)

    return (ordered(gi) - ordered(ri)).abs()


def _check(case):
    case.base()
    case.s0.synchronize()
    ref = case.base_out
    case.pack()
    case.cooperative()
    n = case.n_long
    long_ulps = _bf16_ulps(case.owner_out[:n], ref[:n])
    if int(long_ulps.max()) > 1:
        raise AssertionError(f"spliced long rows differ by {int(long_ulps.max())} bf16 ulps")
    if case.n > n and not torch.equal(case.owner_out[n:], ref[n:]):
        local_ulps = int(_bf16_ulps(case.owner_out[n:], ref[n:]).max())
        raise AssertionError(f"a local row changed ({local_ulps} bf16 ulps)")
    partner = case.partner_out[:n]
    if not torch.equal(partner[:, :Q_PER], torch.zeros_like(partner[:, :Q_PER])):
        raise AssertionError("partner wrote the owner's Q heads")
    remote_ulps = _bf16_ulps(partner[:, Q_PER:], ref[:n, Q_PER:].to(PARTNER))
    if int(remote_ulps.max()) > 1:
        raise AssertionError(f"partner Q heads differ by {int(remote_ulps.max())} bf16 ulps")


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _time(launch, streams, before=None):
    for _ in range(WARMUP):
        if before is not None:
            before()
        launch(None, None)
    for stream in streams:
        stream.synchronize()
    columns = [[] for _ in streams]
    walls = []
    for _ in range(ITERS):
        if before is not None:
            before()
        for stream in streams:
            stream.synchronize()
        starts = []
        ends = []
        for stream in streams:
            with torch.cuda.device(stream.device):
                starts.append(torch.cuda.Event(True))
                ends.append(torch.cuda.Event(True))
        t0 = time.perf_counter()
        launch(starts, ends)
        for stream in streams:
            stream.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        for index, stream in enumerate(streams):
            del stream
            columns[index].append(starts[index].elapsed_time(ends[index]))
    return [_median(column) for column in columns], _median(walls)


def _dense_bytes(seqs):
    return len(seqs) * max(seqs) * KV * DIM * 2 * DTYPE.itemsize


def _report(name, base_ev, base_wall, split_ev, split_wall, std_ev, std_wall, paged, dense):
    half_split = split_ev[1] / (base_ev[0] / 2) if len(split_ev) > 1 else 0.0
    half_std = std_ev[1] / (base_ev[0] / 2) if len(std_ev) > 1 else 0.0
    split0 = split_ev[0]
    split1 = split_ev[1] if len(split_ev) > 1 else 0.0
    std0 = std_ev[0]
    std1 = std_ev[1] if len(std_ev) > 1 else 0.0
    base1 = f" base1 {base_ev[1]:8.3f}" if len(base_ev) > 1 else ""
    print(
        f"{name:<28} {base_ev[0]:8.3f} {split0:8.3f} {split1:8.3f} {std0:8.3f} {std1:8.3f} "
        f"{base_wall:8.3f} {split_wall:8.3f} {std_wall:8.3f} "
        f"{split_wall / base_wall:7.3f} {std_wall / base_wall:7.3f} "
        f"{half_split:7.3f} {half_std:7.3f} {paged / 2**30:7.2f} {dense / 2**30:7.2f}{base1}",
        flush=True,
    )


def _perf(case):
    base_streams = [case.s0] if case.partner_base is None else [case.s0, case.s1]
    base_ev, base_wall = _time(
        lambda starts, ends: case.base(starts, ends),
        base_streams,
    )
    case.pack()
    split_ev, split_wall = _time(
        lambda starts, ends: case._split_timed(starts, ends),
        [case.s0, case.s1],
    )
    std_ev, std_wall = _time(
        lambda starts, ends: case._std_timed(starts, ends),
        [case.s0, case.s1],
        before=case.clear_flags,
    )
    return base_ev, base_wall, split_ev, split_wall, std_ev, std_wall, case.paged_bytes, _dense_bytes(case.seqs)


def _run_named(ops, name, seqs, n_long, check, partner_seqs=None):
    try:
        case = _Case(ops, seqs, n_long, partner_seqs)
        if check:
            _check(case)
            print(f"check {name} ok", flush=True)
        else:
            stats = _perf(case)
            _report(name, *stats)
        del case
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{name} out of memory", flush=True)


def _splits_heuristic(batch_nheads_mblocks, num_sms, num_n_blocks, max_splits=128):
    """Same rule as flash_api.cpp num_splits_heuristic."""
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    def ceildiv(a, b):
        return (a + b - 1) // b
    def eligible(num_splits):
        return num_splits == 1 or ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1)
    efficiency = []
    max_eff = 0.0
    for num_splits in range(1, max_splits + 1):
        if not eligible(num_splits):
            efficiency.append(0.0)
            continue
        waves = (batch_nheads_mblocks * num_splits) / num_sms
        eff = waves / math.ceil(waves)
        max_eff = max(max_eff, eff)
        efficiency.append(eff)
    for num_splits in range(1, max_splits + 1):
        if not eligible(num_splits):
            continue
        if efficiency[num_splits - 1] >= 0.85 * max_eff:
            return num_splits
    return 1


def _predict_splits(batch, seqlen, num_sms):
    num_n_blocks = (seqlen + 64 - 1) // 64
    # After the decode swap, seqlen_q is the group size 8 and one m-block remains.
    return _splits_heuristic(batch * KV, num_sms * 2, num_n_blocks)


def _sweep_case(ops, group, name, seqs, n_long, base_splits, num_sms):
    print(f"run {group} {name}", flush=True)
    try:
        case = _Case(ops, seqs, n_long, fill="empty")
        case.base_splits = base_splits
        base_ev, base_wall = _time(lambda starts, ends: case.base(starts, ends), [case.s0])
        full_bytes = case.paged_bytes
        case.pack_random()
        torch.cuda.synchronize()
        std_ev, std_wall = _time(
            lambda starts, ends: case._std_timed(starts, ends),
            [case.s0, case.s1],
            before=case.clear_flags,
        )
        splits = 1 if base_splits == 1 else _predict_splits(case.n, case.max_k, num_sms)
        base_cta = splits * case.n * KV
        std_cta = n_long + (case.n - n_long) * KV
        base_gb = full_bytes / (base_ev[0] * 1e-3) / 1e9
        std_gb = case.owner_bytes / (std_ev[0] * 1e-3) / 1e9
        print(
            f"{group:<3} {name:<22} {base_ev[0]:8.3f} {std_ev[0]:8.3f} {std_ev[1]:8.3f} "
            f"{std_wall / base_wall:7.3f} {base_gb:8.1f} {std_gb:8.1f} {base_cta:6d} {std_cta:6d} "
            f"{base_gb / base_cta:7.2f} {std_gb / std_cta:7.2f} {splits:4d} {full_bytes / 2**30:7.2f}",
            flush=True,
        )
        del case
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{group:<3} {name:<22} out of memory", flush=True)
    except Exception as exc:
        print(f"{group:<3} {name:<22} failed: {type(exc).__name__}: {exc}", flush=True)
        raise


def _arm(case, mode):
    """Zero flags, then optionally publish them before the launch.

    ``send`` makes the partner's wait succeed immediately. ``both`` also makes
    the owner's splice succeed immediately. The timed region starts afterwards.
    """
    case.clear_flags()
    if mode == "std":
        return
    with torch.cuda.stream(case.s1):
        case.arrived[1].fill_(1)
    if mode == "both":
        with torch.cuda.stream(case.s0):
            case.arrived[0].fill_(1)
    case.s0.synchronize()
    case.s1.synchronize()


def _comm_case(ops, n, seq):
    name = f"n{n}_s{seq}"
    print(f"run {name}", flush=True)
    try:
        case = _Case(ops, [seq] * n, n, fill="empty")
        base_ev, base_wall = _time(lambda starts, ends: case.base(starts, ends), [case.s0])
        full_bytes = case.paged_bytes
        case.pack_random()
        torch.cuda.synchronize()
        timed = {}
        for mode in ("std", "send", "both"):
            timed[mode] = _time(
                lambda starts, ends: case._std_timed(starts, ends),
                [case.s0, case.s1],
                before=lambda mode=mode: _arm(case, mode),
            )
        owner_bytes = case.owner_bytes
        def gb(ev):
            return owner_bytes / (ev[0] * 1e-3) / 1e9
        base_gb = full_bytes / (base_ev[0] * 1e-3) / 1e9
        std_ev, std_wall = timed["std"]
        send_ev, send_wall = timed["send"]
        both_ev, both_wall = timed["both"]
        print(
            f"{name:<14} {base_ev[0]:8.3f} {std_ev[0]:8.3f} {std_ev[1]:8.3f} {std_wall / base_wall:7.3f} "
            f"{send_ev[0]:8.3f} {send_ev[1]:8.3f} {send_wall / base_wall:7.3f} "
            f"{both_ev[0]:8.3f} {both_ev[1]:8.3f} {both_wall / base_wall:7.3f} "
            f"{base_gb:8.1f} {gb(both_ev):8.1f} {std_ev[0] - both_ev[0]:8.3f}",
            flush=True,
        )
        del case
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{name:<14} out of memory", flush=True)
    except Exception as exc:
        print(f"{name:<14} failed: {type(exc).__name__}: {exc}", flush=True)
        raise


def _comm():
    _enable_peers()
    ops = _ops()
    print(
        "comm  send=partner flag already set, both=send and return flags already set. "
        "gap is std0-both0. bothGB is owner bytes over the no-wait kernel.",
        flush=True,
    )
    print(
        f"{'case':<14} {'base0':>8} {'std0':>8} {'std1':>8} {'w_std':>7} "
        f"{'send0':>8} {'send1':>8} {'w_send':>7} {'both0':>8} {'both1':>8} {'w_both':>7} "
        f"{'baseGB':>8} {'bothGB':>8} {'gap':>8}",
        flush=True,
    )
    for n, seq in (
        (4, 262144),
        (8, 262144),
        (16, 262144),
        (32, 32768),
        (48, 32768),
        (64, 32768),
        (80, 32768),
        (108, 32768),
        (128, 32768),
        (256, 32768),
    ):
        _comm_case(ops, n, seq)
    print("comm done", flush=True)


def _sweep():
    _enable_peers()
    ops = _ops()
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(
        f"sweep sms {num_sms}  GB/s uses 1e9. base GB/s is both KV heads. "
        "std GB/s is the owner pages and includes the flag wait.",
        flush=True,
    )
    print(
        f"{'grp':<3} {'case':<22} {'base0':>8} {'std0':>8} {'std1':>8} "
        f"{'w_std':>7} {'baseGB':>8} {'stdGB':>8} {'cta0':>6} {'ctaS':>6} "
        f"{'GBcta0':>7} {'GBctaS':>7} {'spl':>4} {'GiB':>7}",
        flush=True,
    )
    ab = [(n, 32768) for n in (1, 2, 4, 8, 16, 32, 48, 64, 80, 108, 128, 160, 256)]
    for n, seq in ab:
        _sweep_case(ops, "ab", f"n{n}_s{seq}", [seq] * n, n, 1, num_sms)
    length = [(8, seq) for seq in (8192, 32768, 131072, 262144)]
    length += [(128, seq) for seq in (8192, 32768, 131072)]
    for n, seq in length:
        _sweep_case(ops, "c", f"n{n}_s{seq}", [seq] * n, n, 1, num_sms)
    for n_long in (4, 16, 32):
        _sweep_case(ops, "e", f"long{n_long}_s262144", [262144] * n_long, n_long, 1, num_sms)
        mixed = [262144] * n_long + [512] * 512
        _sweep_case(ops, "e", f"long{n_long}_s262144_sh", mixed, n_long, 1, num_sms)
    for n, seq in ((4, 262144), (16, 262144), (32, 262144), (8, 32768), (128, 32768), (256, 32768)):
        _sweep_case(ops, "f", f"n{n}_s{seq}", [seq] * n, n, 0, num_sms)
    print("sweep done", flush=True)


def main():
    if os.environ.get("HEAD_TP_COMM"):
        _comm()
        return
    if os.environ.get("HEAD_TP_SWEEP"):
        _sweep()
        return
    _page_tests()
    _enable_peers()
    ops = _ops()
    print("check: mixed lengths, local row kept, long rows spliced", flush=True)
    _run_named(ops, "check_mix", [64, 96, 128, 17], 3, check=True)
    print(
        f"{'case':<28} {'base0':>8} {'split0':>8} {'split1':>8} {'std0':>8} {'std1':>8} "
        f"{'base_w':>8} {'split_w':>8} {'std_w':>8} {'w_spl':>7} {'w_std':>7} "
        f"{'half_s':>7} {'half_c':>7} {'paged':>7} {'dense':>7}",
        flush=True,
    )
    pure = [
        ("20x32768", [32768] * 20),
        ("128x8192", [8192] * 128),
        ("128x32768", [32768] * 128),
        ("128x65536", [65536] * 128),
        ("256x32768", [32768] * 256),
    ]
    # Long rows first so the partner batch is a prefix of the owner batch.
    mixed = [
        ("mix_128x256_2x128k_1x256k", [131072, 131072, 262144] + [256] * 128, 3),
        ("mix_64x1024_4x128k", [131072] * 4 + [1024] * 64, 4),
        ("mix_32x512_2x256k", [262144, 262144] + [512] * 32, 2),
        ("mix_512x512_16x256k", [262144] * 16 + [512] * 512, 16),
    ]
    # Partner keeps its own short batch. Long rows stay owned by the first card.
    paired = [
        ("mix_1024x1024_4x256k", [262144] * 4 + [1024] * 1024, 4, [1024] * 1024),
    ]
    only = os.environ.get("HEAD_TP_ONLY")
    for name, seqs in pure:
        if only is None or only == name:
            _run_named(ops, name, seqs, len(seqs), check=False)
    for name, seqs, n_long in mixed:
        if only is None or only == name:
            _run_named(ops, name, seqs, n_long, check=False)
    for name, seqs, n_long, partner_seqs in paired:
        if only is None or only == name:
            _run_named(ops, name, seqs, n_long, check=False, partner_seqs=partner_seqs)
    print("perf done", flush=True)


if __name__ == "__main__":
    main()
