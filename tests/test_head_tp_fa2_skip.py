"""FA2 head-TP tile skip through varlen_fwd and fwd_kvcache.

Q heads and KV heads are equal, so the decode GQA transpose does not run.
That transpose would make the kernel head index a KV head. The predicate
indexes Q heads.

A launch with no plan is the reference. With a plan, the local row matches
it, the owner row keeps shard 0 and leaves the other heads at zero, and the
empty row stays zero. fwd_kvcache with num_splits=2 takes the split-KV
kernel. Its combine step skips the same tiles, so a pre-zeroed output stays
zero. The test skips when CUDA or the extension is absent.
"""

import unittest

import torch

_BATCH = 3
_HEADS = 4
_DIM = 256
_SEQ_K = 128
_SHARD_END = 2
_DP = 2


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


def _inputs():
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    shape = (_BATCH * _SEQ_K, _HEADS, _DIM)
    q = torch.randn(_BATCH, _HEADS, _DIM, dtype=torch.float16, device="cuda", generator=generator)
    k = torch.randn(shape, dtype=torch.float16, device="cuda", generator=generator)
    v = torch.randn(shape, dtype=torch.float16, device="cuda", generator=generator)
    return q, k, v


def _plan():
    row_kind = torch.tensor([0, 1, 2], dtype=torch.int8, device="cuda")
    q_range = torch.zeros(_BATCH, _DP, 2, dtype=torch.int32, device="cuda")
    q_range[:, 0, 1] = _SHARD_END
    return row_kind, q_range


def _varlen(ops, q, k, v, row_kind, q_range, dp_size):
    cu_q = torch.arange(_BATCH + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.arange(_BATCH + 1, dtype=torch.int32, device="cuda") * _SEQ_K
    out = torch.zeros_like(q)
    returned = ops.varlen_fwd(
        q, k, v, out, cu_q, cu_k,
        None, None, None, None,
        1, _SEQ_K,
        0.0, _DIM ** -0.5, True, False, -1, -1, 0.0, False,
        0, None,
        row_kind, q_range, dp_size,
    )
    return returned[0]


def _kvcache(ops, q, k, v, row_kind, q_range, dp_size):
    q_b = q.view(_BATCH, 1, _HEADS, _DIM)
    k_b = k.view(_BATCH, _SEQ_K, _HEADS, _DIM)
    v_b = v.view(_BATCH, _SEQ_K, _HEADS, _DIM)
    out = torch.zeros_like(q_b)
    returned = ops.fwd_kvcache(
        q_b, k_b, v_b,
        None, None, None, None, None, None, None, None, None,
        out, _DIM ** -0.5, False, -1, -1, 0.0, False, 2,
        row_kind, q_range, dp_size,
    )
    return returned[0].squeeze(1)


def _check(full, planned):
    if not torch.equal(planned[0], full[0]):
        raise AssertionError("local row changed under a plan that keeps every head")
    if not torch.equal(planned[1, :_SHARD_END], full[1, :_SHARD_END]):
        raise AssertionError("owner shard does not match the full forward")
    outside = planned[1, _SHARD_END:]
    if not torch.equal(outside, torch.zeros_like(outside)):
        raise AssertionError("owner heads outside the shard were written")
    if not torch.equal(planned[2], torch.zeros_like(planned[2])):
        raise AssertionError("empty row was written")


def test_fa2_tile_skip_matches_the_unplanned_forward():
    ops = _require_cuda_ops()
    q, k, v = _inputs()
    row_kind, q_range = _plan()
    full = _varlen(ops, q, k, v, None, None, 0)
    planned = _varlen(ops, q, k, v, row_kind, q_range, _DP)
    _check(full, planned)
    full_split = _kvcache(ops, q, k, v, None, None, 0)
    planned_split = _kvcache(ops, q, k, v, row_kind, q_range, _DP)
    _check(full_split, planned_split)


if __name__ == "__main__":
    try:
        test_fa2_tile_skip_matches_the_unplanned_forward()
    except unittest.SkipTest as exc:
        print(f"skip: {exc}")
    else:
        print("ok")
