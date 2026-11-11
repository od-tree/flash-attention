"""Two-GPU wall time for one head-TP decode step.

Rank 0 owns the long rows. Rank 1 holds the other half of the short rows.
Baseline keeps each long row on rank 0 and runs all 16 Q heads there.
The cooperative step runs Q [0, 8) on rank 0 and Q [8, 16) on rank 1.
Send, the partner wait, the return, and the splice run inside fwd_kvcache.
A long row uses one shared Q and KV cache on both ranks. Splice writes the
partner shard into the owner's attention output. The check compares that
merged output with one unplanned forward of the same cache. The 1% and 5% mixes are marked as the NanoCP-shaped loads. No
50 ms or 3 x T0 threshold is applied.

Requires two CUDA devices with peer access. Skips otherwise.
"""

import ctypes
import importlib.util
import sys
import time
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_tp_eval_shapes import eval_cases  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_Q = 16
_KV = 2
_DIM = 256
_SHARD = _Q // 2
_DP = 2
_SCALE = _DIM ** -0.5
_OWNER = 0
_PARTNER = 1


def _load_exchange():
    path = _ROOT / "flash_attn" / "head_tp_exchange.py"
    spec = importlib.util.spec_from_file_location("head_tp_exchange", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ops():
    import vllm_flash_attn._vllm_fa2_C  # noqa: F401

    return torch.ops._vllm_fa2_C


def _require_two_gpus():
    if torch.cuda.device_count() < 2:
        raise unittest.SkipTest("two CUDA devices are required")
    try:
        ops = _ops()
    except Exception as exc:
        raise unittest.SkipTest(str(exc)) from exc
    exchange = _load_exchange()
    try:
        exchange._ops()
    except RuntimeError as exc:
        raise unittest.SkipTest(str(exc)) from exc
    if not torch.cuda.can_device_access_peer(0, 1):
        raise unittest.SkipTest("cuda:0 cannot access cuda:1")
    lib = None
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        raise unittest.SkipTest("libcudart was not found")
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    lib.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
    for src, dst in ((0, 1), (1, 0)):
        err = lib.cudaSetDevice(src)
        if err != 0:
            raise unittest.SkipTest(f"cudaSetDevice({src}) returned {err}")
        err = lib.cudaDeviceEnablePeerAccess(dst, 0)
        if err not in (0, 704):
            raise unittest.SkipTest(f"peer access {src}->{dst} returned {err}")
    return ops, exchange


def _selected_cases():
    cases = []
    for case in eval_cases():
        if case.num_tokens == 1 or case.name.endswith("_local"):
            continue
        cases.append(case)
    return cases


def _split(case):
    seqs = case.seq_lens()
    longs = list(range(case.long_count))
    shorts = list(range(case.long_count, case.num_tokens))
    return longs, shorts[0::2], shorts[1::2], seqs


def _cache(ids, seqs, device, seed):
    n = len(ids)
    if n == 0:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    max_seq = max(seqs[i] for i in ids)
    q = torch.randn(n, 1, _Q, _DIM, dtype=torch.float16, device=device, generator=generator)
    k = torch.randn(n, max_seq, _KV, _DIM, dtype=torch.float16, device=device, generator=generator)
    v = torch.randn(n, max_seq, _KV, _DIM, dtype=torch.float16, device=device, generator=generator)
    seqlens = torch.tensor([seqs[i] for i in ids], dtype=torch.int32, device=device)
    return q, k, v, seqlens


def _plan(n, n_long, q_begin, q_end, device):
    row_kind = torch.zeros(n, dtype=torch.int8, device=device)
    q_range = torch.zeros(n, _DP, 2, dtype=torch.int32, device=device)
    if n_long:
        row_kind[:n_long] = 1
        q_range[:n_long, 0, 0] = q_begin
        q_range[:n_long, 0, 1] = q_end
    return row_kind, q_range


def _fwd(ops, batch, out, row_kind, q_range, dp_size, xchg=None):
    q, k, v, seqlens = batch
    tail = () if xchg is None else xchg
    ops.fwd_kvcache(
        q, k, v,
        None, None, seqlens,
        None, None, None, None, None, None,
        out, _SCALE, False, -1, -1, 0.0, False, 0,
        row_kind, q_range, dp_size,
        *tail,
    )


def _pointers(buffers, device):
    return torch.tensor([buf.data_ptr() for buf in buffers], dtype=torch.int64, device=device)


def _repeats(case):
    work = case.num_tokens * max(case.seq_lens())
    # Long rows dominate the step, so fewer samples still cover the noise.
    if work >= 20 * 8192:
        return 5, 20
    return 10, 30


def _median(samples):
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def _sync_all():
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def _sync_wall(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync_all()
    samples = []
    for _ in range(iters):
        _sync_all()
        start = time.perf_counter()
        fn()
        _sync_all()
        samples.append((time.perf_counter() - start) * 1e3)
    return _median(samples)


class _Step:
    def __init__(self, ops, exchange, case):
        self.ops = ops
        self.exchange = exchange
        longs, owner_shorts, partner_shorts, seqs = _split(case)
        self.longs = longs
        self.n_long = len(longs)
        self.global_n = case.num_tokens
        self.owner_ids = longs + owner_shorts
        self.partner_base_ids = partner_shorts
        self.partner_coop_ids = longs + partner_shorts
        self.seqs = seqs
        self.owner = torch.device("cuda:0")
        self.partner = torch.device("cuda:1")
        self.s0 = torch.cuda.Stream(device=self.owner)
        self.s1 = torch.cuda.Stream(device=self.partner)
        self.owner_batch = _cache(self.owner_ids, seqs, self.owner, 0)
        self.partner_base = _cache(self.partner_base_ids, seqs, self.partner, 1)
        self.partner_coop = _cache(self.partner_coop_ids, seqs, self.partner, 1)
        self.owner_out = None if self.owner_batch is None else torch.empty_like(self.owner_batch[0])
        self.partner_base_out = None if self.partner_base is None else torch.empty_like(self.partner_base[0])
        self.partner_coop_out = None if self.partner_coop is None else torch.empty_like(self.partner_coop[0])
        self.owner_plan = None
        self.partner_plan = None
        if self.owner_batch is not None:
            self.owner_plan = _plan(len(self.owner_ids), self.n_long, 0, _SHARD, self.owner)
        if self.partner_coop is not None:
            self.partner_plan = _plan(len(self.partner_coop_ids), self.n_long, _SHARD, _Q, self.partner)
        self._share_long_rows()
        self._exchange_buffers()

    def _share_long_rows(self):
        # Partner attention has to see the same long-row Q and KV as the owner.
        # Short rows stay on the rank that owns them.
        if not self.n_long or self.partner_coop is None:
            return
        n = self.n_long
        for src, dst in zip(self.owner_batch[:3], self.partner_coop[:3]):
            dst[:n].copy_(src[:n])

    def _exchange_buffers(self):
        g = self.global_n
        device = self.owner
        self.send_q = torch.randn(g, _Q, _DIM, dtype=torch.float16, device=device)
        self.send_k = torch.randn(g, _KV, _DIM, dtype=torch.float16, device=device)
        self.send_v = torch.randn(g, _KV, _DIM, dtype=torch.float16, device=device)
        self.send_row = torch.zeros(g, dtype=torch.int8, device=device)
        self.slot_rank = torch.full((g, _DP), -1, dtype=torch.int32, device=device)
        self.slot_rank[:, 0] = _OWNER
        self.send_q_range = torch.zeros(g, _DP, 2, dtype=torch.int32, device=device)
        self.send_kv_range = torch.zeros(g, _DP, 2, dtype=torch.int32, device=device)
        if self.n_long:
            ids = torch.tensor(self.longs, dtype=torch.long, device=device)
            self.send_row[ids] = 1
            self.slot_rank[ids, 1] = _PARTNER
            self.send_q_range[ids, 0, 1] = _SHARD
            self.send_q_range[ids, 1, 0] = _SHARD
            self.send_q_range[ids, 1, 1] = _Q
            self.send_kv_range[ids, 0, 1] = 1
            self.send_kv_range[ids, 1, 0] = 1
            self.send_kv_range[ids, 1, 1] = _KV
            q, k, v, seqlens = self.owner_batch
            n = self.n_long
            pos = (seqlens[:n] - 1).to(torch.long)
            self.send_q[:n] = q[:n, 0]
            index = pos.view(n, 1, 1, 1).expand(n, 1, _KV, _DIM)
            self.send_k[:n] = k[:n].gather(1, index).squeeze(1)
            self.send_v[:n] = v[:n].gather(1, index).squeeze(1)
        p = self.partner
        self.payload_q = [
            torch.zeros(_DP, g, _SHARD, _DIM, dtype=torch.float16, device=self.owner),
            torch.zeros(_DP, g, _SHARD, _DIM, dtype=torch.float16, device=p),
        ]
        self.payload_k = [
            torch.zeros(_DP, g, 1, _DIM, dtype=torch.float16, device=self.owner),
            torch.zeros(_DP, g, 1, _DIM, dtype=torch.float16, device=p),
        ]
        self.payload_v = [
            torch.zeros(_DP, g, 1, _DIM, dtype=torch.float16, device=self.owner),
            torch.zeros(_DP, g, 1, _DIM, dtype=torch.float16, device=p),
        ]
        self.payload_out = [
            torch.zeros(_DP, g, _SHARD, _DIM, dtype=torch.float16, device=self.owner),
            torch.zeros(_DP, g, _SHARD, _DIM, dtype=torch.float16, device=p),
        ]
        self.arrived = [
            torch.zeros(_DP, g, dtype=torch.int32, device=self.owner),
            torch.zeros(_DP, g, dtype=torch.int32, device=p),
        ]
        self.ptr_q = _pointers(self.payload_q, device)
        self.ptr_k = _pointers(self.payload_k, device)
        self.ptr_v = _pointers(self.payload_v, device)
        self.ptr_arrived_owner = _pointers(self.arrived, device)
        self.ptr_out_owner = _pointers(self.payload_out, device)
        self.ptr_q_partner = _pointers(self.payload_q, p)
        self.ptr_k_partner = _pointers(self.payload_k, p)
        self.ptr_v_partner = _pointers(self.payload_v, p)
        self.ptr_out = _pointers(self.payload_out, p)
        self.ptr_arrived_partner = _pointers(self.arrived, p)
        self.slot_partner = self.slot_rank.to(p)
        self.kv_partner = self.send_kv_range.to(p)
        self.xchg_q_partner = self.send_q_range.to(p)
        self.return_count = [
            torch.zeros(g, dtype=torch.int32, device=device),
            torch.zeros(g, dtype=torch.int32, device=p),
        ]
        self.inbound = torch.tensor([self.n_long, 0], dtype=torch.int32, device=p)
        self.expect = torch.tensor([self.n_long], dtype=torch.int32, device=device)

    def _xchg(self, src):
        partner = src == _PARTNER
        device_slot = self.slot_partner if partner else self.slot_rank
        device_kv = self.kv_partner if partner else self.send_kv_range
        device_q = self.xchg_q_partner if partner else self.send_q_range
        ptr_q = self.ptr_q_partner if partner else self.ptr_q
        ptr_k = self.ptr_k_partner if partner else self.ptr_k
        ptr_v = self.ptr_v_partner if partner else self.ptr_v
        ptr_out = self.ptr_out if partner else self.ptr_out_owner
        ptr_arrived = self.ptr_arrived_partner if partner else self.ptr_arrived_owner
        arrived = self.arrived[src]
        payload = self.payload_out[src]
        count = self.return_count[src]
        return (
            device_slot, device_kv, device_q, ptr_q, ptr_k, ptr_v, ptr_out, ptr_arrived,
            arrived, payload, count, src, _SHARD, 1, self.global_n,
        )

    def _attn(self, stream, ops_batch, out, plan, xchg=None):
        if ops_batch is None:
            return
        row_kind, q_range, dp = (None, None, 0) if plan is None else (*plan, _DP)
        with torch.cuda.stream(stream):
            _fwd(self.ops, ops_batch, out, row_kind, q_range, dp, xchg)

    def baseline(self):
        self._attn(self.s0, self.owner_batch, self.owner_out, None)
        self._attn(self.s1, self.partner_base, self.partner_base_out, None)
        self.s0.synchronize()
        self.s1.synchronize()

    def cooperative(self):
        # Flags and the return counter have to be clear before either forward.
        # Both forwards are then queued without a host wait between them: the
        # owner tile sends while local tiles run, and the partner tile waits
        # inside its own kernel.
        with torch.cuda.stream(self.s0):
            self.arrived[0].zero_()
            self.return_count[0].zero_()
        with torch.cuda.stream(self.s1):
            self.arrived[1].zero_()
            self.return_count[1].zero_()
        self.s0.synchronize()
        self.s1.synchronize()
        with torch.cuda.stream(self.s0):
            if self.owner_out is not None:
                self.owner_out.zero_()
            if self.owner_batch is not None:
                _fwd(
                    self.ops, self.owner_batch, self.owner_out, *self.owner_plan, _DP,
                    self._xchg(_OWNER),
                )
        with torch.cuda.stream(self.s1):
            if self.partner_coop_out is not None:
                self.partner_coop_out.zero_()
            if self.partner_coop is not None:
                _fwd(
                    self.ops, self.partner_coop, self.partner_coop_out, *self.partner_plan, _DP,
                    self._xchg(_PARTNER),
                )
        self.s0.synchronize()
        self.s1.synchronize()

    def check(self):
        if not self.n_long:
            return
        n = self.n_long
        ref = torch.empty_like(self.owner_out)
        _fwd(self.ops, self.owner_batch, ref, None, None, 0)
        torch.cuda.synchronize(self.owner)
        ref_long = ref[:n, 0]
        owner_long = self.owner_out[:n, 0]
        if not torch.equal(owner_long, ref_long):
            raise AssertionError("merged owner output differs from the full forward")
        if self.owner_out.size(0) > n:
            if not torch.equal(self.owner_out[n:], ref[n:]):
                raise AssertionError("an owner local row changed")
        partner_long = self.partner_coop_out[:n, 0]
        partner_shard = partner_long[:, _SHARD:].to(ref.device)
        if not torch.equal(partner_shard, ref_long[:, _SHARD:]):
            raise AssertionError("partner shard differs from the full forward")
        partner_other = partner_long[:, :_SHARD]
        if not torch.equal(partner_other, torch.zeros_like(partner_other)):
            raise AssertionError("partner forward wrote the owner Q heads")
        sent_q = self.send_q[:n, _SHARD:]
        got_q = self.payload_q[_PARTNER][_OWNER, :n].to(sent_q.device)
        if not torch.equal(got_q, sent_q):
            raise AssertionError("partner did not receive the current Q shard")
        sent_k = self.send_k[:n, 1:]
        got_k = self.payload_k[_PARTNER][_OWNER, :n].to(sent_k.device)
        if not torch.equal(got_k, sent_k):
            raise AssertionError("partner did not receive the current K")
        sent_v = self.send_v[:n, 1:]
        got_v = self.payload_v[_PARTNER][_OWNER, :n].to(sent_v.device)
        if not torch.equal(got_v, sent_v):
            raise AssertionError("partner did not receive the current V")


def _nanocp(name):
    return name.startswith("mix_1pct") or name.startswith("mix_5pct")


def test_two_gpu_decode_step():
    ops, exchange = _require_two_gpus()
    print(
        f"{'case':<22} {'tokens':>6} {'long':>4} {'baseline_ms':>12} "
        f"{'head_tp_ms':>12} {'ratio':>8} nanocp"
    )
    failures = []
    ran = 0
    for case in _selected_cases():
        try:
            step = _Step(ops, exchange, case)
            warmup, iters = _repeats(case)
            baseline_ms = _sync_wall(step.baseline, warmup, iters)
            head_tp_ms = _sync_wall(step.cooperative, warmup, iters)
            step.check()
            mismatch = ""
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            failures.append(f"{case.name}: out of memory")
            print(f"{case.name:<22} out of memory")
            continue
        except AssertionError as exc:
            mismatch = f" check_failed: {exc}"
            failures.append(f"{case.name}: {exc}")
        ratio = head_tp_ms / baseline_ms if baseline_ms else 0.0
        mark = "yes" if _nanocp(case.name) else ""
        print(
            f"{case.name:<22} {case.num_tokens:6d} {case.long_count:4d} "
            f"{baseline_ms:12.3f} {head_tp_ms:12.3f} {ratio:8.3f} {mark}{mismatch}",
            flush=True,
        )
        ran += 1
        torch.cuda.empty_cache()
    if failures:
        raise AssertionError("\n".join(failures))
    if ran == 0:
        raise AssertionError("no dual-GPU loads were timed")


if __name__ == "__main__":
    try:
        test_two_gpu_decode_step()
    except unittest.SkipTest as exc:
        print(f"skip: {exc}")
    else:
        print("ok")
