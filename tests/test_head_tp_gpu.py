"""GPU exchange correctness. Skips when CUDA or the extension is absent.

One device holds every rank's buffers. The wait kernel runs on that device:
an inbound sum of 0 returns immediately, and a send on the same stream is
visible to the following wait. Expecting more flags than were published is not
run; that path traps.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_head = _load("head_tp", _ROOT / "flash_attn" / "head_tp.py")
_exchange = _load("head_tp_exchange", _ROOT / "flash_attn" / "head_tp_exchange.py")
_ref = _load("test_head_tp", _TESTS / "test_head_tp.py")


def _require_cuda_ops():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")
    try:
        _exchange._ops()
    except RuntimeError as exc:
        raise unittest.SkipTest(str(exc)) from exc


def _pointers(buffers):
    return torch.tensor(
        [buf.data_ptr() for buf in buffers],
        dtype=torch.int64,
        device=buffers[0].device,
    )


def _gpu_exchange(world):
    """Run send, both waits, return, and splice on one CUDA device."""
    dp = len(world.ranks)
    tokens, num_q, dim = world.ranks[0].q.shape
    num_kv = world.ranks[0].k.shape[1]
    degree = dp
    q_per = num_q // degree
    kv_per = num_kv // degree
    device = torch.device("cuda")
    dtype = world.ranks[0].q.dtype

    def alloc(per):
        return [
            torch.zeros(dp, tokens, per, dim, dtype=dtype, device=device)
            for _ in range(dp)
        ]

    payload_q = alloc(q_per)
    payload_k = alloc(kv_per)
    payload_v = alloc(kv_per)
    payload_out = alloc(q_per)
    arrived = [
        torch.zeros(dp, tokens, dtype=torch.int32, device=device) for _ in range(dp)
    ]
    ptr_q = _pointers(payload_q)
    ptr_k = _pointers(payload_k)
    ptr_v = _pointers(payload_v)
    ptr_out = _pointers(payload_out)
    ptr_arrived = _pointers(arrived)
    slot_rank = world.ranks[0].slot_rank.to(device)
    q_range = world.ranks[0].q_range.to(device)
    kv_range = world.ranks[0].kv_range.to(device)
    for rank in world.ranks:
        _exchange.send(
            rank.q.to(device),
            rank.k.to(device),
            rank.v.to(device),
            rank.row_kind.to(device),
            slot_rank,
            q_range,
            kv_range,
            ptr_q,
            ptr_k,
            ptr_v,
            ptr_arrived,
            rank.rank,
            q_per,
            kv_per,
        )
    for rank in world.ranks:
        _exchange.wait(arrived[rank.rank], inbound=rank.inbound.to(device))
    partial = torch.zeros(dp, tokens, q_per, dim, dtype=dtype, device=device)
    q_len = torch.zeros(dp, tokens, dtype=torch.int32, device=device)
    for src in range(dp):
        for token in range(tokens):
            if int(world.arrived[1, src, token]) != 1:
                continue
            shard = world.payload_out[src, 1, token]
            width = shard.shape[0]
            partial[src, token, :width] = shard.to(device)
            q_len[src, token] = width
    _exchange.publish_return(partial, q_len, ptr_out, ptr_arrived, 1, q_per)
    owner = world.ranks[0]
    expect = torch.tensor([_head.remote_shard_count(owner)], dtype=torch.int32, device=device)
    _exchange.wait(arrived[0], expect=expect)
    out = torch.zeros(tokens, num_q, dim, dtype=dtype, device=device)
    _exchange.splice(
        out,
        payload_out[0],
        owner.row_kind.to(device),
        slot_rank,
        q_range,
        q_per,
    )
    torch.cuda.synchronize()
    return payload_q, payload_k, payload_v, payload_out, arrived, out


def test_gpu_exchange_matches_cpu_reference():
    _require_cuda_ops()
    torch.manual_seed(0)
    tokens, seq = 3, 8
    degree = 2
    q = torch.randn(tokens, 16, 256)
    k = torch.randn(tokens, 2, 256)
    v = torch.randn(tokens, 2, 256)
    history_k = torch.randn(tokens, seq, 2, 256)
    history_v = torch.randn(tokens, seq, 2, 256)
    seq_len = torch.full((tokens,), seq, dtype=torch.int32)
    row_kind = torch.tensor(
        [_head.ROW_LOCAL, _head.ROW_OWNER, _head.ROW_EMPTY], dtype=torch.int8
    )
    slot_rank, q_range, kv_range = _ref._ranges(16, 2, degree, tokens, [-1, 0, -1])
    world, _scale = _ref._world(
        q, k, v, history_k, history_v, seq_len, row_kind, slot_rank, q_range, kv_range, degree
    )
    _head.send_current(world)
    sent_q = world.payload_q.clone()
    sent_k = world.payload_k.clone()
    sent_v = world.payload_v.clone()
    sent_flags = world.arrived.clone()
    _head.phase_partner(world)
    _head.phase_splice(world)
    payload_q, payload_k, payload_v, payload_out, arrived, out = _gpu_exchange(world)
    for dst in range(degree):
        torch.testing.assert_close(payload_q[dst].cpu(), sent_q[dst])
        torch.testing.assert_close(payload_k[dst].cpu(), sent_k[dst])
        torch.testing.assert_close(payload_v[dst].cpu(), sent_v[dst])
        torch.testing.assert_close(arrived[dst].cpu(), world.arrived[dst])
        torch.testing.assert_close(payload_out[dst].cpu(), world.payload_out[dst])
    assert int(sent_flags[:, :, 0].sum()) == 0
    assert int(sent_flags[:, :, 2].sum()) == 0
    assert int(sent_flags[1, 0, 1]) == 1
    assert int(sent_flags[0, 1, 1]) == 0
    qb, qe = int(q_range[1, 1, 0]), int(q_range[1, 1, 1])
    torch.testing.assert_close(out[1, qb:qe].cpu(), world.ranks[0].out[1, qb:qe])


if __name__ == "__main__":
    try:
        test_gpu_exchange_matches_cpu_reference()
    except unittest.SkipTest as exc:
        print(f"skip: {exc}")
    else:
        print("ok")
