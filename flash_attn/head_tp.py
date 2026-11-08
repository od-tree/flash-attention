# Head-TP attention: tile predicate plus the exchange that sits beside the
# FlashAttention mainloop.
#
# The predicate matches csrc/flash_attn/src/head_tp_skip.h. On the FA2 kernel
# it returns before the mainloop, the same way a fully masked block is
# skipped. On Hopper the warp-specialized pipeline must not return from inside
# the consumer loop; head_tp_mask_mod makes those tiles fully masked so the
# FA4 block-sparsity prepass omits them before the mainloop starts.
#
# Payload exchange, arrival flags, and the splice are not part of that prune.
# run_head_tp performs them in the fixed order from the head-TP spec. History
# KV is already caught up; this function only appends the current token.

from dataclasses import dataclass, field

import torch

ROW_LOCAL = 0
ROW_OWNER = 1
ROW_EMPTY = 2


def tile_active(row_kind, q_range, dp_size, batch, head):
    """Return whether this (batch, head) tile runs the attention mainloop."""
    kind = int(row_kind[batch])
    if kind == ROW_EMPTY:
        return False
    if kind == ROW_LOCAL:
        return True
    if kind != ROW_OWNER or dp_size <= 0:
        return False
    q_begin = int(q_range[batch, 0, 0])
    q_end = int(q_range[batch, 0, 1])
    return q_begin <= head < q_end


def head_tp_mask_mod(row_kind, q_range, dp_size):
    """FA4 mask_mod. Inactive tiles are false for every KV index.

    compute_block_sparsity then records neither a full nor a partial block,
    which is the fully-masked prune.
    """

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, *aux):
        del q_idx, kv_idx, seqlen_info, aux
        return tile_active(row_kind, q_range, dp_size, int(batch_idx), int(head_idx))

    return mask_mod


def attend(q, k, v, scale):
    """GQA attention. q is [Hq, D], k and v are [S, Hkv, D]."""
    hq = q.shape[0]
    hkv = k.shape[1]
    if hq % hkv != 0:
        raise ValueError(f"Hq {hq} is not divisible by Hkv {hkv}")
    group = hq // hkv
    k_g = k.repeat_interleave(group, dim=1)
    v_g = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("hd,shd->hs", q.float(), k_g.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,shd->hd", probs, v_g.float()).to(dtype=q.dtype)


@dataclass
class Rank:
    rank: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    row_kind: torch.Tensor
    slot_rank: torch.Tensor
    q_range: torch.Tensor
    kv_range: torch.Tensor
    seq_len: torch.Tensor
    inbound: torch.Tensor
    out: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.out = torch.zeros_like(self.q)


@dataclass
class World:
    """Shared payload buffers. Index is [dst, src, token, ...]."""

    ranks: list
    payload_q: torch.Tensor
    payload_k: torch.Tensor
    payload_v: torch.Tensor
    payload_out: torch.Tensor
    arrived: torch.Tensor
    scale: float
    # Partner-side history, already caught up, keyed by (dst, src, token).
    peer_k: dict
    peer_v: dict


def _write_kv(cache_k, cache_v, token, seq_len, k_now, v_now, kv_begin, kv_end):
    pos = seq_len - 1
    cache_k[token, pos, kv_begin:kv_end] = k_now[kv_begin:kv_end]
    cache_v[token, pos, kv_begin:kv_end] = v_now[kv_begin:kv_end]


def phase_local_and_send(world: World):
    """Local rows, owner shard, then payload stores. No waiting."""
    dp = len(world.ranks)
    for rank in world.ranks:
        hkv = rank.k.shape[1]
        for token in range(rank.q.shape[0]):
            kind = int(rank.row_kind[token])
            if kind == ROW_EMPTY:
                continue
            seq_len = int(rank.seq_len[token])
            if kind == ROW_LOCAL:
                _write_kv(rank.k_cache, rank.v_cache, token, seq_len, rank.k[token], rank.v[token], 0, hkv)
                rank.out[token] = attend(
                    rank.q[token],
                    rank.k_cache[token, :seq_len],
                    rank.v_cache[token, :seq_len],
                    world.scale,
                )
                continue
            if kind != ROW_OWNER:
                raise ValueError(f"bad row_kind {kind}")
            q_begin = int(rank.q_range[token, 0, 0])
            q_end = int(rank.q_range[token, 0, 1])
            kv_begin = int(rank.kv_range[token, 0, 0])
            kv_end = int(rank.kv_range[token, 0, 1])
            _write_kv(
                rank.k_cache, rank.v_cache, token, seq_len, rank.k[token], rank.v[token], kv_begin, kv_end
            )
            rank.out[token, q_begin:q_end] = attend(
                rank.q[token, q_begin:q_end],
                rank.k_cache[token, :seq_len, kv_begin:kv_end],
                rank.v_cache[token, :seq_len, kv_begin:kv_end],
                world.scale,
            )
            for slot in range(1, dp):
                peer = int(rank.slot_rank[token, slot])
                if peer < 0:
                    continue
                qb = int(rank.q_range[token, slot, 0])
                qe = int(rank.q_range[token, slot, 1])
                kb = int(rank.kv_range[token, slot, 0])
                ke = int(rank.kv_range[token, slot, 1])
                world.payload_q[peer, rank.rank, token, : qe - qb] = rank.q[token, qb:qe]
                world.payload_k[peer, rank.rank, token, : ke - kb] = rank.k[token, kb:ke]
                world.payload_v[peer, rank.rank, token, : ke - kb] = rank.v[token, kb:ke]
                world.arrived[peer, rank.rank, token] = 1


def phase_partner(world: World):
    """Device-side wait, then attention on received shards.

    Ranks whose inbound sum is 0 do not touch arrived.
    """
    dp = len(world.ranks)
    for rank in world.ranks:
        if int(rank.inbound.sum()) == 0:
            continue
        got = int((world.arrived[rank.rank] == 1).sum())
        if got != int(rank.inbound.sum()):
            raise RuntimeError(f"rank {rank.rank} expected {int(rank.inbound.sum())} shards, saw {got}")
        for src in range(dp):
            if src == rank.rank:
                continue
            owner = world.ranks[src]
            for token in range(owner.q.shape[0]):
                if int(world.arrived[rank.rank, src, token]) != 1:
                    continue
                slot = None
                for candidate in range(1, dp):
                    if int(owner.slot_rank[token, candidate]) == rank.rank:
                        slot = candidate
                        break
                if slot is None:
                    raise RuntimeError("arrived shard has no slot_rank")
                qb = int(owner.q_range[token, slot, 0])
                qe = int(owner.q_range[token, slot, 1])
                kb = int(owner.kv_range[token, slot, 0])
                ke = int(owner.kv_range[token, slot, 1])
                seq_len = int(owner.seq_len[token])
                peer_k = world.peer_k[(rank.rank, src, token)]
                peer_v = world.peer_v[(rank.rank, src, token)]
                peer_k[seq_len - 1] = world.payload_k[rank.rank, src, token, : ke - kb]
                peer_v[seq_len - 1] = world.payload_v[rank.rank, src, token, : ke - kb]
                partial = attend(
                    world.payload_q[rank.rank, src, token, : qe - qb],
                    peer_k[:seq_len],
                    peer_v[:seq_len],
                    world.scale,
                )
                world.payload_out[src, rank.rank, token, : qe - qb] = partial
                world.arrived[src, rank.rank, token] = 1


def phase_splice(world: World):
    dp = len(world.ranks)
    for rank in world.ranks:
        for token in range(rank.q.shape[0]):
            if int(rank.row_kind[token]) != ROW_OWNER:
                continue
            for slot in range(1, dp):
                peer = int(rank.slot_rank[token, slot])
                if peer < 0:
                    continue
                if int(world.arrived[rank.rank, peer, token]) != 1:
                    raise RuntimeError(f"rank {rank.rank} missing return from {peer} token {token}")
                qb = int(rank.q_range[token, slot, 0])
                qe = int(rank.q_range[token, slot, 1])
                rank.out[token, qb:qe] = world.payload_out[rank.rank, peer, token, : qe - qb]


def run_head_tp(world: World):
    """One step. Sends from every rank finish before any rank waits."""
    phase_local_and_send(world)
    phase_partner(world)
    phase_splice(world)
    return [rank.out for rank in world.ranks]
