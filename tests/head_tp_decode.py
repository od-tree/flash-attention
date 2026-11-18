"""Two-rank paged GQA decode fixture for FA2 benchmarks."""
import os
from itertools import accumulate

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


def _move(src, device, *, dtype=None):
    """Cross-GPU copy. PyTorch ``.to()`` fails with error 205 on this 5090."""
    from cuda_ipc_p2p import copy_tensor

    return copy_tensor(src, device, dtype=dtype)


def _put(dst, src):
    from cuda_ipc_p2p import assign_tensor

    assign_tensor(dst, src)
_MEMCPY = False


def _enable_peers():
    """Enable cross-GPU access for the head-TP exchange buffers.

    Default is raw ``EnablePeerAccess`` + foreign ``data_ptr``. On sm_120 /
    5090, or when ``P2P_BACKEND=memcpy``, payloads stay local and the host
    copies them with ``cudaMemcpyPeer``.
    """
    global _MEMCPY
    if torch.cuda.device_count() < 2:
        raise SystemExit("two CUDA devices are required")
    import cuda_ipc_p2p as ipc

    backend = ipc.enable_peers([0, 1])
    _MEMCPY = backend == "memcpy"
    if _MEMCPY:
        os.environ["P2P_BACKEND"] = "memcpy"
    else:
        os.environ.pop("HEAD_TP_PHASE", None)


def _ops():
    import vllm_flash_attn._vllm_fa2_C  # noqa: F401

    return torch.ops._vllm_fa2_C


def _blocks(seq):
    return (seq + BLOCK - 1) // BLOCK


def _packed_pages(seq):
    span = DEGREE * BLOCK
    return (seq + span - 1) // span


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
    padding = (-tokens.shape[0]) % span
    if padding:
        tokens = torch.nn.functional.pad(tokens, (0, 0, 0, padding))
    view = tokens.view(-1, DEGREE, BLOCK, DIM)
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
        _put(cache_k[cursor : cursor + count], pages_k)
        _put(cache_v[cursor : cursor + count], pages_v)
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


# FA2 split-KV kBlockN for head_dim 256. Same rule as flash_api set_params_splitkv.
KV_TILE = 64
# int8 row_splits, and FA2 combine is compiled up to 128 partials.
SPLIT_LIMIT = 127


def _fill_wave_splits(seqs, num_sms, short_work=0):
    """One local KV head. Match vllm.v1.head_tp._fill_wave_splits."""
    total = sum(seqs)
    total_work = total + max(short_work, 0)
    budget = num_sms if total_work == total else (num_sms * total) // total_work
    budget = max(budget, len(seqs))
    budget = min(budget, num_sms)
    limits = []
    splits = []
    for seq in seqs:
        n_blocks = max(1, (seq + KV_TILE - 1) // KV_TILE)
        limit = min(SPLIT_LIMIT, n_blocks)
        limits.append(limit)
        raw = (budget * seq) // total
        splits.append(min(limit, max(1, raw)))

    def tail_cmp(left, right):
        return seqs[left] * splits[right] - seqs[right] * splits[left]

    while True:
        spare = budget - sum(splits)
        if spare <= 0:
            break
        worst = 0
        for index in range(1, len(splits)):
            if tail_cmp(index, worst) > 0:
                worst = index
        group = []
        blocked = False
        for index in range(len(splits)):
            if tail_cmp(index, worst) != 0:
                continue
            if splits[index] >= limits[index]:
                blocked = True
                break
            group.append(index)
        if blocked or not group or len(group) > spare:
            break
        for index in group:
            splits[index] += 1
    return splits


def _tile_plan(seqs, n_long, kv_begin, num_sms, device):
    """Long rows share one wave of CTAs. One KV head uses more splits.

    The split count matches vllm.v1.head_tp.build_head_tp_tile_plan.
    """
    if os.environ.get("HEAD_TP_TILES", "1") == "0":
        return None, None, 0
    n = len(seqs)
    long_rows = list(range(n_long))
    kv_per = 1
    row_splits = [1] * n
    if long_rows and n_long * kv_per < num_sms:
        seqs_long = [max(seqs[i], 1) for i in long_rows]
        short_work = sum(max(seqs[i], 1) * KV for i in range(n_long, n))
        for i, count in zip(long_rows, _fill_wave_splits(seqs_long, num_sms, short_work)):
            row_splits[i] = count
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
        # 0 asks the FA2 heuristic. The tile path ignores this once a plan is set.
        num_splits = int(os.environ.get("HEAD_TP_SPLITS", "0"))
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


def _copy_rank_plane(buffers, src, n_rows):
    """Copy the leading ``n_rows`` of rank ``src``'s plane onto the other GPU.

    Long rows are that prefix, and they are the only tokens the peer reads.
    Short rows stay on their own card. The prefix is contiguous, so each
    buffer is one copy.
    """
    if n_rows <= 0:
        return
    from cuda_ipc_p2p import memcpy_peer

    dst = PARTNER if src == OWNER else OWNER
    for buf in buffers:
        memcpy_peer(buf[dst][src][:n_rows], buf[src][src][:n_rows])


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


class HeadTPDecode:
    def __init__(self, ops, seqs, n_long, partner_seqs=None, fill="randn", *, inputs=None, tile_plan=_tile_plan):
        self.ops = ops
        self.seqs = list(seqs)
        self.n = len(self.seqs)
        self.n_long = n_long
        self.partner_seqs = list(partner_seqs or [])
        self.partner_n = n_long + len(self.partner_seqs)
        self.s0 = torch.cuda.Stream(device=OWNER)
        self.s1 = torch.cuda.Stream(device=PARTNER)
        # Single-card baseline uses the FA2 heuristic unless HEAD_TP_SPLITS overrides.
        self.base_splits = int(os.environ.get("HEAD_TP_SPLITS", "0"))
        self.owner_bytes = 0
        def rank_cache(rank, lengths, seed):
            if inputs is None:
                return _standard_cache(lengths, rank, seed, fill)
            q, k, v, table, used = (value.to(rank) for value in inputs[rank])
            counts = [_blocks(length) for length in lengths]
            spans = list(zip(accumulate(counts, initial=0), counts))
            return k, v, table, spans, q, used

        k, v, table, spans, q, seqused = rank_cache(OWNER, self.seqs, 11)
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
            pk, pv, ptable, pspans, pq, pseq = rank_cache(PARTNER, self.partner_seqs, 29)
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
        self.owner_tiles = tile_plan(self.seqs, n_long, 0, num_sms, OWNER)
        partner_seqs = self.seqs[:n_long] + self.partner_seqs
        self.partner_tiles = tile_plan(partner_seqs, n_long, 1, torch.cuda.get_device_properties(1).multi_processor_count, PARTNER)

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
        partner_seq = _move(seqused[: self.n_long], PARTNER)
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
        partner_seq = _move(seqused[: self.n_long], PARTNER)
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
            _put(self.k_token_partner[: self.n_long], k_token[: self.n_long])
            _put(self.v_token_partner[: self.n_long], v_token[: self.n_long])
        self.payload_tokens = tokens

    def _xchg(self, src):
        # The pointer tables are invariant. Building them inside a capture is a
        # pageable host copy, which CUDA graph rejects.
        cached = getattr(self, "_xchg_cache", None)
        if cached is None:
            cached = {}
            self._xchg_cache = cached
        if src not in cached:
            cached[src] = self._xchg_build(src)
        return cached[src]

    def _xchg_build(self, src):
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

    def compute_step(self, mark):
        self._split_timed(None, None)
        mark("attention")
        return self.owner_out, self.partner_out[self.n_long:]

    def prepare_exchange(self):
        # Upload both pointer tables before either rank starts waiting on its peer.
        self._xchg(OWNER)
        self._xchg(PARTNER)

    def decode_step(self, mark):
        # Replayed graphs cannot rely on a host-side reset between steps.
        # Each device must see its cleared flags before any peer can signal.
        streams = (self.s0, self.s1)
        ready = []
        for rank, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                self.arrived[rank].zero_()
                self.return_count[rank].zero_()
                ready.append(stream.record_event())
        streams[0].wait_event(ready[1])
        streams[1].wait_event(ready[0])
        mark("reset")
        original_phase = os.environ.get("HEAD_TP_PHASE")
        try:
            self._std_timed(None, None, mark)
        finally:
            if original_phase is None:
                os.environ.pop("HEAD_TP_PHASE", None)
            else:
                os.environ["HEAD_TP_PHASE"] = original_phase
        return self.owner_out, self.partner_out[self.n_long:]

    def _split_timed(self, starts, ends):
        self._launch(self.s0, self.owner, self.owner_out, self.owner_plan, 0, -1, None, starts, ends, 0)
        self._launch(self.s1, self.partner, self.partner_out, self.partner_plan, 1, -1, None, starts, ends, 1)

    def _std_timed(self, starts, ends, mark=lambda name: None):
        if not _MEMCPY:
            self._launch(self.s0, self.owner, self.owner_out, self.owner_plan, 0, OWNER, self._xchg(OWNER), starts, ends, 0)
            self._launch(self.s1, self.partner, self.partner_out, self.partner_plan, 1, PARTNER, self._xchg(PARTNER), starts, ends, 1)
            mark("fused_exchange_attention")
            return
        # Send writes the owner's local plane. The partner is not the row
        # owner, so it publishes nothing there. Compute waits on the copied
        # plane and the partner writes its Q heads. Only that return is copied,
        # then splice puts those heads into the owner output. Each copy is the
        # long-row prefix, not the short rows that stay local.
        planes = (self.payload_q, self.payload_k, self.payload_v, self.arrived)
        from cuda_ipc_p2p import graph_streams_active

        graphing = graph_streams_active()
        n_rows = self.n_long
        os.environ["HEAD_TP_PHASE"] = "send"
        self._launch(self.s0, self.owner, self.owner_out, self.owner_plan, 0, OWNER, self._xchg(OWNER), starts, None, 0)
        self._launch(self.s1, self.partner, self.partner_out, self.partner_plan, 1, PARTNER, self._xchg(PARTNER), starts, None, 1)
        if not graphing:
            self.s0.synchronize()
            self.s1.synchronize()
        mark("send")
        _copy_rank_plane(planes, OWNER, n_rows)
        mark("qkv_transfer")
        os.environ["HEAD_TP_PHASE"] = "compute"
        self._launch(self.s0, self.owner, self.owner_out, self.owner_plan, 0, OWNER, self._xchg(OWNER), None, None, 0)
        self._launch(self.s1, self.partner, self.partner_out, self.partner_plan, 1, PARTNER, self._xchg(PARTNER), None, None, 1)
        if not graphing:
            self.s0.synchronize()
            self.s1.synchronize()
        mark("attention")
        _copy_rank_plane((self.payload_out, self.arrived), PARTNER, n_rows)
        mark("output_transfer")
        os.environ["HEAD_TP_PHASE"] = "splice"
        self._launch(self.s0, self.owner, self.owner_out, self.owner_plan, 0, OWNER, self._xchg(OWNER), None, ends, 0)
        self._launch(self.s1, self.partner, self.partner_out, self.partner_plan, 1, PARTNER, self._xchg(PARTNER), None, ends, 1)
        os.environ.pop("HEAD_TP_PHASE", None)
        mark("splice")

    def _launch(self, stream, batch, out, plan, kv_begin, src, xchg, starts, ends, index):
        row_kind = plan[0]
        q, k, v, table, seqused = batch[:5]
        cu = self.cu_partner if index == 1 else self.cu_owner
        max_k = max(self.max_k, self.partner_max_k) if index == 1 else self.max_k
        with torch.cuda.stream(stream):
            if starts is not None:
                starts[index].record(stream)
            tiles, row_splits, max_splits = self.owner_tiles if index == 0 else self.partner_tiles
            _varlen(
                self.ops, q, k, v, table, seqused, *cu, max_k, out,
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
        self.prepare_exchange()
        self.clear_flags()
        self._std_timed(None, None)
        self.s0.synchronize()
        self.s1.synchronize()
