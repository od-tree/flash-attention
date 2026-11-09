"""Standalone head-TP checks. No FlashAttention build and no vLLM."""

import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "head_tp",
    Path(__file__).resolve().parents[1] / "flash_attn" / "head_tp.py",
)
_head_tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_head_tp)

ROW_EMPTY = _head_tp.ROW_EMPTY
ROW_LOCAL = _head_tp.ROW_LOCAL
ROW_OWNER = _head_tp.ROW_OWNER
Rank = _head_tp.Rank
World = _head_tp.World
attend = _head_tp.attend
head_tp_mask_mod = _head_tp.head_tp_mask_mod
publish_return = _head_tp.publish_return
remote_shard_count = _head_tp.remote_shard_count
run_head_tp = _head_tp.run_head_tp
send_current = _head_tp.send_current
tile_active = _head_tp.tile_active
wait_arrived = _head_tp.wait_arrived


def _full_out(q, k_now, v_now, history_k, history_v, seq_len, scale):
    out = torch.zeros_like(q)
    for token in range(q.shape[0]):
        sl = int(seq_len[token])
        k = history_k[token].clone()
        v = history_v[token].clone()
        k[sl - 1] = k_now[token]
        v[sl - 1] = v_now[token]
        out[token] = attend(q[token], k[:sl], v[:sl], scale)
    return out


def _ranges(num_q, num_kv, degree, tokens, owners):
    """owners[token] is the owner rank, or -1 for a local/empty row."""
    dp = degree if owners is None else max(degree, max(owners) + 1)
    q_per = num_q // degree
    kv_per = num_kv // degree
    q_range = torch.full((tokens, dp, 2), -1, dtype=torch.int32)
    kv_range = torch.full((tokens, dp, 2), -1, dtype=torch.int32)
    slot_rank = torch.full((tokens, dp), -1, dtype=torch.int32)
    for token, owner in enumerate(owners):
        if owner < 0:
            continue
        slot_rank[token, 0] = owner
        for shard in range(degree):
            q_range[token, shard, 0] = shard * q_per
            q_range[token, shard, 1] = (shard + 1) * q_per
            kv_range[token, shard, 0] = shard * kv_per
            kv_range[token, shard, 1] = (shard + 1) * kv_per
            if shard > 0:
                slot_rank[token, shard] = (owner + shard) % dp
    return slot_rank, q_range, kv_range


def _world(q, k, v, history_k, history_v, seq_len, row_kind, slot_rank, q_range, kv_range, degree):
    tokens, num_q, dim = q.shape
    num_kv = k.shape[1]
    dp = slot_rank.shape[1]
    q_per = num_q // degree
    kv_per = num_kv // degree
    scale = dim ** -0.5
    ranks = []
    peer_k = {}
    peer_v = {}
    inbound = torch.zeros((dp, dp), dtype=torch.int32)
    for rank in range(dp):
        own_kind = torch.full((tokens,), ROW_EMPTY, dtype=torch.int8)
        for token in range(tokens):
            if int(row_kind[token]) == ROW_LOCAL:
                own_kind[token] = ROW_LOCAL
            elif int(row_kind[token]) == ROW_OWNER and int(slot_rank[token, 0]) == rank:
                own_kind[token] = ROW_OWNER
        k_cache = history_k.clone()
        v_cache = history_v.clone()
        ranks.append(
            Rank(
                rank=rank,
                q=q.clone(),
                k=k.clone(),
                v=v.clone(),
                k_cache=k_cache,
                v_cache=v_cache,
                row_kind=own_kind,
                slot_rank=slot_rank,
                q_range=q_range,
                kv_range=kv_range,
                seq_len=seq_len.clone(),
                inbound=torch.zeros(dp, dtype=torch.int32),
            )
        )
    for token in range(tokens):
        if int(row_kind[token]) != ROW_OWNER:
            continue
        owner = int(slot_rank[token, 0])
        for slot in range(1, dp):
            peer = int(slot_rank[token, slot])
            if peer < 0:
                continue
            inbound[peer, owner] += 1
            kb = int(kv_range[token, slot, 0])
            ke = int(kv_range[token, slot, 1])
            peer_k[(peer, owner, token)] = history_k[token, :, kb:ke].clone()
            peer_v[(peer, owner, token)] = history_v[token, :, kb:ke].clone()
    for rank in ranks:
        rank.inbound = inbound[rank.rank].clone()
    world = World(
        ranks=ranks,
        payload_q=torch.zeros(dp, dp, tokens, q_per, dim),
        payload_k=torch.zeros(dp, dp, tokens, kv_per, dim),
        payload_v=torch.zeros(dp, dp, tokens, kv_per, dim),
        payload_out=torch.zeros(dp, dp, tokens, q_per, dim),
        arrived=torch.zeros(dp, dp, tokens, dtype=torch.int32),
        scale=scale,
        peer_k=peer_k,
        peer_v=peer_v,
    )
    return world, scale


def test_tile_predicate_matches_fa4_full_mask():
    row_kind = torch.tensor([ROW_LOCAL, ROW_OWNER, ROW_EMPTY], dtype=torch.int8)
    q_range = torch.zeros(3, 2, 2, dtype=torch.int32)
    q_range[1, 0] = torch.tensor([0, 4])
    mask_mod = head_tp_mask_mod(row_kind, q_range, dp_size=2)
    for batch, head, active in (
        (0, 0, True),
        (0, 7, True),
        (1, 3, True),
        (1, 4, False),
        (2, 0, False),
    ):
        assert tile_active(row_kind, q_range, 2, batch, head) is active
        # Every KV index is masked off, so FA4 classifies the block as fully masked.
        assert mask_mod(batch, head, 0, 0, None) is active
        assert mask_mod(batch, head, 0, 5, None) is active


def test_send_and_return_use_different_flags():
    torch.manual_seed(0)
    tokens, num_q, num_kv, dim, seq = 3, 8, 2, 16, 5
    degree = 2
    q = torch.randn(tokens, num_q, dim)
    k = torch.randn(tokens, num_kv, dim)
    v = torch.randn(tokens, num_kv, dim)
    history_k = torch.randn(tokens, seq, num_kv, dim)
    history_v = torch.randn(tokens, seq, num_kv, dim)
    seq_len = torch.tensor([seq, seq, seq], dtype=torch.int32)
    row_kind = torch.tensor([ROW_LOCAL, ROW_OWNER, ROW_EMPTY], dtype=torch.int8)
    slot_rank, q_range, kv_range = _ranges(num_q, num_kv, degree, tokens, owners=[-1, 0, -1])
    world, _scale = _world(
        q, k, v, history_k, history_v, seq_len, row_kind, slot_rank, q_range, kv_range, degree
    )
    send_current(world)
    token, peer = 1, 1
    qb, qe = (int(q_range[token, peer, 0]), int(q_range[token, peer, 1]))
    kb, ke = (int(kv_range[token, peer, 0]), int(kv_range[token, peer, 1]))
    torch.testing.assert_close(world.payload_q[peer, 0, token, : qe - qb], q[token, qb:qe])
    torch.testing.assert_close(world.payload_k[peer, 0, token, : ke - kb], k[token, kb:ke])
    torch.testing.assert_close(world.payload_v[peer, 0, token, : ke - kb], v[token, kb:ke])
    assert int(world.arrived[peer, 0, token]) == 1
    assert int(world.arrived[0, peer, token]) == 0
    assert int(world.arrived[:, :, 0].sum()) == 0
    assert int(world.arrived[:, :, 2].sum()) == 0
    wait_arrived(world.arrived[0], 0)
    wait_arrived(world.arrived[peer], int(world.ranks[peer].inbound.sum()))
    try:
        wait_arrived(world.arrived[peer], 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a short arrival count must fail")
    partial = torch.ones(qe - qb, dim)
    publish_return(world, 0, peer, token, partial)
    assert int(world.arrived[peer, 0, token]) == 1
    assert int(world.arrived[0, peer, token]) == 1
    torch.testing.assert_close(world.payload_out[0, peer, token, : qe - qb], partial)
    wait_arrived(world.arrived[0], remote_shard_count(world.ranks[0]))


def test_local_and_d2_match_full_attention():
    torch.manual_seed(0)
    tokens, num_q, num_kv, dim, seq = 3, 8, 2, 16, 5
    degree = 2
    q = torch.randn(tokens, num_q, dim)
    k = torch.randn(tokens, num_kv, dim)
    v = torch.randn(tokens, num_kv, dim)
    history_k = torch.randn(tokens, seq, num_kv, dim)
    history_v = torch.randn(tokens, seq, num_kv, dim)
    seq_len = torch.tensor([seq, seq, seq], dtype=torch.int32)
    row_kind = torch.tensor([ROW_LOCAL, ROW_OWNER, ROW_EMPTY], dtype=torch.int8)
    slot_rank, q_range, kv_range = _ranges(num_q, num_kv, degree, tokens, owners=[-1, 0, -1])
    world, scale = _world(
        q, k, v, history_k, history_v, seq_len, row_kind, slot_rank, q_range, kv_range, degree
    )
    arrived_before = world.arrived.clone()
    outs = run_head_tp(world)
    reference = _full_out(q, k, v, history_k, history_v, seq_len, scale)
    torch.testing.assert_close(outs[0][0], reference[0])
    torch.testing.assert_close(outs[0][1], reference[1])
    torch.testing.assert_close(outs[0][2], torch.zeros_like(reference[2]))
    # The empty row does not gain an arrival. The cooperative row does, on the
    # partner and then back on the owner.
    assert int(world.arrived[:, :, 2].sum()) == int(arrived_before[:, :, 2].sum())
    assert int(world.arrived[1, 0, 1]) == 1
    assert int(world.arrived[0, 1, 1]) == 1
    assert int(world.ranks[1].inbound.sum()) == 1
    assert int(world.ranks[0].inbound.sum()) == 0


def test_d4_one_row_matches_full_attention():
    torch.manual_seed(1)
    tokens, num_q, num_kv, dim, seq = 1, 8, 4, 16, 4
    degree = 4
    q = torch.randn(tokens, num_q, dim)
    k = torch.randn(tokens, num_kv, dim)
    v = torch.randn(tokens, num_kv, dim)
    history_k = torch.randn(tokens, seq, num_kv, dim)
    history_v = torch.randn(tokens, seq, num_kv, dim)
    seq_len = torch.tensor([seq], dtype=torch.int32)
    row_kind = torch.tensor([ROW_OWNER], dtype=torch.int8)
    slot_rank, q_range, kv_range = _ranges(num_q, num_kv, degree, tokens, owners=[0])
    world, scale = _world(
        q, k, v, history_k, history_v, seq_len, row_kind, slot_rank, q_range, kv_range, degree
    )
    outs = run_head_tp(world)
    reference = _full_out(q, k, v, history_k, history_v, seq_len, scale)
    torch.testing.assert_close(outs[0][0], reference[0])
    for peer in range(1, degree):
        assert int(world.ranks[peer].inbound.sum()) == 1
        assert int(world.arrived[peer, 0, 0]) == 1
        assert int(world.arrived[0, peer, 0]) == 1


if __name__ == "__main__":
    test_tile_predicate_matches_fa4_full_mask()
    test_send_and_return_use_different_flags()
    test_local_and_d2_match_full_attention()
    test_d4_one_row_matches_full_attention()
    print("ok")
