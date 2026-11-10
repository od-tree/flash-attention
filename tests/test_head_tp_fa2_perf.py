"""Time eval-plan decode attention against the unplanned FA2 forward.

One case is one ``fwd_kvcache`` call, not the 200-request trace. Heads are
Qwen3.5 Full Attention: 16 Q heads, 2 KV heads, dim 256, fp16, d = 2.
Sequence lengths are 1024, 1279, 8192, and 32768. A cooperative case marks
the long rows as owner shards and keeps Q heads [0, 8).

The 1% and 5% mixes are the attention loads that line up with NanoCP's
uneven request mixes. This file times those loads with and without the
head-TP plan. It does not launch a separate NanoCP kernel, and it does not
treat 50 ms or 3 x T0 as a pass line.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_tp_eval_shapes import NUM_Q, eval_cases  # noqa: E402

_SCALE = 256 ** -0.5
_SHARD = NUM_Q // 2


def _ops():
    import vllm_flash_attn._vllm_fa2_C  # noqa: F401

    return torch.ops._vllm_fa2_C


def _require_cuda_ops():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")
    try:
        return _ops()
    except Exception as exc:
        raise unittest.SkipTest(str(exc)) from exc


def _repeats(case):
    work = case.num_tokens * max(case.seq_lens())
    if work >= 20 * 32768:
        return 1, 2
    if work >= 20 * 8192:
        return 1, 3
    return 2, 5


def _plan(case):
    row_kind = torch.zeros(case.num_tokens, dtype=torch.int8, device="cuda")
    if case.coop:
        row_kind[: case.long_count] = 1
    q_range = torch.zeros(case.num_tokens, case.degree, 2, dtype=torch.int32, device="cuda")
    q_range[:, 0, 1] = _SHARD
    return row_kind, q_range


def _allocate(case):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    seq = case.seq_lens()
    max_seq = max(seq)
    q = torch.randn(
        case.num_tokens, 1, case.num_q, case.head_dim,
        dtype=torch.float16, device="cuda", generator=generator,
    )
    k = torch.randn(
        case.num_tokens, max_seq, case.num_kv, case.head_dim,
        dtype=torch.float16, device="cuda", generator=generator,
    )
    v = torch.randn(
        case.num_tokens, max_seq, case.num_kv, case.head_dim,
        dtype=torch.float16, device="cuda", generator=generator,
    )
    seqlens = torch.tensor(seq, dtype=torch.int32, device="cuda")
    return q, k, v, seqlens


def _launch(ops, q, k, v, seqlens, out, row_kind, q_range, dp_size):
    ops.fwd_kvcache(
        q, k, v,
        None, None, seqlens,
        None, None, None, None, None, None,
        out, _SCALE, False, -1, -1, 0.0, False, 0,
        row_kind, q_range, dp_size,
    )


def _median_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _check(case, original, planned):
    original = original.squeeze(1)
    planned = planned.squeeze(1)
    if case.coop and case.long_count:
        long_p = planned[: case.long_count]
        long_o = original[: case.long_count]
        if not torch.equal(long_p[:, :_SHARD], long_o[:, :_SHARD]):
            raise AssertionError(f"{case.name}: owner shard differs from the full forward")
        outside = long_p[:, _SHARD:]
        if not torch.equal(outside, torch.zeros_like(outside)):
            raise AssertionError(f"{case.name}: heads outside the owner shard were written")
        if case.long_count < case.num_tokens:
            if not torch.equal(planned[case.long_count :], original[case.long_count :]):
                raise AssertionError(f"{case.name}: a local row changed")
        return
    if not torch.equal(planned, original):
        raise AssertionError(f"{case.name}: local plan changed the full forward")


def _nanocp_load(name):
    return name.startswith("mix_1pct") or name.startswith("mix_5pct")


def _run_case(ops, case):
    warmup, iters = _repeats(case)
    q, k, v, seqlens = _allocate(case)
    out = torch.empty_like(q)
    row_kind, q_range = _plan(case)
    original_ms = _median_ms(
        lambda: _launch(ops, q, k, v, seqlens, out, None, None, 0),
        warmup, iters,
    )
    original = out.detach().clone()
    head_tp_ms = _median_ms(
        lambda: _launch(ops, q, k, v, seqlens, out, row_kind, q_range, case.degree),
        warmup, iters,
    )
    _check(case, original, out)
    return original_ms, head_tp_ms


def test_eval_plan_attention_perf():
    ops = _require_cuda_ops()
    rows = []
    failures = []
    print(
        f"{'case':<24} {'tokens':>6} {'long':>4} {'max_seq':>8} "
        f"{'original_ms':>12} {'head_tp_ms':>12} {'ratio':>8} nanocp"
    )
    for case in eval_cases():
        try:
            original_ms, head_tp_ms = _run_case(ops, case)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            failures.append(f"{case.name}: out of memory")
            print(f"{case.name:<24} out of memory")
            continue
        ratio = head_tp_ms / original_ms if original_ms else 0.0
        mark = "yes" if _nanocp_load(case.name) else ""
        print(
            f"{case.name:<24} {case.num_tokens:6d} {case.long_count:4d} "
            f"{max(case.seq_lens()):8d} {original_ms:12.3f} {head_tp_ms:12.3f} "
            f"{ratio:8.3f} {mark}"
        )
        rows.append((case.name, original_ms, head_tp_ms))
        torch.cuda.empty_cache()
    if failures:
        raise AssertionError("\n".join(failures))
    if not rows:
        raise AssertionError("no attention loads were timed")


if __name__ == "__main__":
    try:
        test_eval_plan_attention_perf()
    except unittest.SkipTest as exc:
        print(f"skip: {exc}")
    else:
        print("ok")
