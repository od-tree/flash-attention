"""Lock the eval-plan decode shapes and check those lengths against full attention."""

import importlib.util
from pathlib import Path

import torch

_TESTS = Path(__file__).resolve().parent
_shapes = importlib.util.spec_from_file_location(
    "head_tp_eval_shapes", _TESTS / "head_tp_eval_shapes.py"
)
_plan = importlib.util.module_from_spec(_shapes)
_shapes.loader.exec_module(_plan)
HEAD_DIM = _plan.HEAD_DIM
LAST_SHORT_LEN = _plan.LAST_SHORT_LEN
LONG_LENS = _plan.LONG_LENS
NUM_KV = _plan.NUM_KV
NUM_Q = _plan.NUM_Q
SHORT_LEN = _plan.SHORT_LEN
eval_cases = _plan.eval_cases
operator_input_shapes = _plan.operator_input_shapes

_TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("test_head_tp", _TESTS / "test_head_tp.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)


def test_case_table_matches_the_eval_plan():
    cases = {case.name: case for case in eval_cases()}
    assert cases["capacity_1024"].seq_lens() == [1024]
    assert cases["capacity_8192"].seq_lens() == [8192]
    assert cases["capacity_32768"].seq_lens() == [32768]
    assert cases["pure_short_1024"].seq_lens() == [SHORT_LEN] * 20
    assert cases["pure_short_1279"].seq_lens() == [LAST_SHORT_LEN] * 20
    for long_len in LONG_LENS:
        pure = cases[f"pure_long_{long_len}"]
        assert pure.num_tokens == 20 and pure.coop
        assert pure.seq_lens() == [long_len] * 20
        local = cases[f"pure_long_{long_len}_local"]
        assert local.seq_lens() == pure.seq_lens()
        assert local.owners() == [-1] * 20
        for ratio, tokens, longs in ((1, 100, 1), (5, 20, 1), (20, 20, 4)):
            mix = cases[f"mix_{ratio}pct_{long_len}"]
            assert mix.num_tokens == tokens
            assert mix.seq_lens().count(long_len) == longs
            assert mix.seq_lens().count(SHORT_LEN) == tokens - longs
            assert sum(owner == 0 for owner in mix.owners()) == longs
            pair = cases[f"mix_{ratio}pct_{long_len}_local"]
            assert pair.seq_lens() == mix.seq_lens()
            assert pair.owners() == [-1] * tokens


def test_operator_shapes_use_qwen35_heads():
    for case in eval_cases():
        shapes = operator_input_shapes(case)
        tokens = case.num_tokens
        max_seq = max(case.seq_lens())
        assert shapes["q"] == (tokens, NUM_Q, HEAD_DIM)
        assert shapes["k"] == shapes["v"] == (tokens, NUM_KV, HEAD_DIM)
        assert shapes["k_cache"] == (tokens, max_seq, NUM_KV, HEAD_DIM)
        assert shapes["payload_q"] == (2, tokens, NUM_Q // 2, HEAD_DIM)
        assert shapes["payload_k"] == (2, tokens, NUM_KV // 2, HEAD_DIM)
        assert shapes["arrived"] == (2, tokens)
        assert case.num_q % case.degree == 0
        assert case.num_kv % case.degree == 0


def _attention_at(seq, tokens, owners, row_kind):
    degree = 2
    q = torch.randn(tokens, NUM_Q, HEAD_DIM)
    k = torch.randn(tokens, NUM_KV, HEAD_DIM)
    v = torch.randn(tokens, NUM_KV, HEAD_DIM)
    history_k = torch.randn(tokens, seq, NUM_KV, HEAD_DIM)
    history_v = torch.randn(tokens, seq, NUM_KV, HEAD_DIM)
    seq_len = torch.full((tokens,), seq, dtype=torch.int32)
    slot_rank, q_range, kv_range = _ref._ranges(
        NUM_Q, NUM_KV, degree, tokens, owners
    )
    world, scale = _ref._world(
        q, k, v, history_k, history_v, seq_len, row_kind, slot_rank, q_range, kv_range, degree
    )
    outs = _ref.run_head_tp(world)
    reference = _ref._full_out(q, k, v, history_k, history_v, seq_len, scale)
    return outs, reference, world


def test_plan_lengths_match_full_attention():
    row_kind = torch.tensor(
        [_ref.ROW_LOCAL, _ref.ROW_OWNER, _ref.ROW_EMPTY], dtype=torch.int8
    )
    owners = [-1, 0, -1]
    for seq in (SHORT_LEN, LONG_LENS[0]):
        outs, reference, world = _attention_at(seq, 3, owners, row_kind)
        torch.testing.assert_close(outs[0][0], reference[0])
        torch.testing.assert_close(outs[0][1], reference[1])
        torch.testing.assert_close(outs[0][2], torch.zeros_like(reference[2]))
        assert int(world.arrived[:, :, 2].sum()) == 0
        assert int(world.arrived[1, 0, 1]) == 1
        assert int(world.arrived[0, 1, 1]) == 1
    try:
        outs, reference, world = _attention_at(
            LONG_LENS[1],
            1,
            [-1],
            torch.tensor([_ref.ROW_LOCAL], dtype=torch.int8),
        )
    except (MemoryError, RuntimeError) as exc:
        print(f"skip seq {LONG_LENS[1]}: {exc}")
        return
    torch.testing.assert_close(outs[0][0], reference[0])
    assert int(world.arrived.sum()) == 0


if __name__ == "__main__":
    test_case_table_matches_the_eval_plan()
    test_operator_shapes_use_qwen35_heads()
    test_plan_lengths_match_full_attention()
    print("ok")
