"""d=4 page layout and splice indexes. 8 KV heads, 64 Q heads, 2 KV and 16 Q per rank.

Two cards are enough for the byte layout and the gather indexes. The four-card
exchange is not run here.
"""

import importlib.util
from pathlib import Path

import torch

BLOCK = 16
DEGREE = 4
KV_HEADS = 8
Q_HEADS = 64
DIM = 256
DTYPE = torch.bfloat16


def _layout():
    path = Path(__file__).resolve().parents[1] / "flash_attn" / "head_tp_page.py"
    spec = importlib.util.spec_from_file_location("head_tp_page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logical(seq_len: int) -> torch.Tensor:
    kv_per = KV_HEADS // DEGREE
    pos = torch.arange(seq_len, dtype=torch.float32)
    logical = pos.view(1, seq_len, 1).expand(kv_per, seq_len, DIM).to(DTYPE).contiguous()
    heads = torch.arange(kv_per, dtype=torch.float32).to(DTYPE)
    logical[:, :, 1] = heads.view(kv_per, 1)
    return logical


def test_each_local_head_is_a_contiguous_run():
    layout = _layout()
    seq_len = DEGREE * BLOCK
    logical = _logical(seq_len)
    pages = layout.pack_long_pages(logical, DEGREE, BLOCK)
    kv_per = KV_HEADS // DEGREE
    assert pages.dtype == DTYPE
    assert pages.shape == (1, KV_HEADS, BLOCK, DIM)
    for head in range(kv_per):
        restored = layout.logical_tokens(pages, head=head, degree=DEGREE)
        assert torch.equal(restored, logical[head])


def test_physical_pages_are_one_over_d_of_a_full_head_allocation():
    layout = _layout()
    seq_len = 3 * DEGREE * BLOCK
    pages = layout.pack_long_pages(_logical(seq_len), DEGREE, BLOCK)
    full_head_pages = (seq_len + BLOCK - 1) // BLOCK
    assert pages.shape[0] * DEGREE == full_head_pages


def test_short_page_keeps_all_eight_kv_heads():
    layout = _layout()
    heads = torch.arange(KV_HEADS, dtype=torch.float32).to(DTYPE).view(KV_HEADS, 1, 1).expand(KV_HEADS, BLOCK, DIM).contiguous()
    page = layout.short_page(heads)
    assert page.shape == (KV_HEADS, BLOCK, DIM)
    for head in range(KV_HEADS):
        assert torch.equal(page[head], heads[head])


def test_paged_address_reads_both_local_heads():
    layout = _layout()
    seq_len = 2 * DEGREE * BLOCK
    logical = _logical(seq_len)
    pages = layout.pack_long_pages(logical, DEGREE, BLOCK)
    view = pages.transpose(1, 2)
    block_table = torch.arange(pages.shape[0], dtype=torch.int32)
    kv_per = KV_HEADS // DEGREE
    for head in range(kv_per):
        for pos in (0, BLOCK, DEGREE * BLOCK, seq_len - 1):
            got = _read_long(view, block_table, pos, local_head=head)
            assert torch.equal(got, logical[head, pos])


def test_first_three_idle_heads_send_the_three_remote_slots():
    layout = _layout()
    kv_per = KV_HEADS // DEGREE
    sent = [layout.owner_send_slot(head, kv_begin=0, kv_per=kv_per, degree=DEGREE) for head in range(KV_HEADS)]
    assert sent == [-1, -1, 1, 2, 3, -1, -1, -1]
    # Shard 1 owns KV heads 2 and 3. Those heads are active on that rank.
    partner = [layout.owner_send_slot(head, kv_begin=2, kv_per=kv_per, degree=DEGREE) for head in range(KV_HEADS)]
    assert partner[2] == -1 and partner[3] == -1


def test_returned_q_heads_cover_sixteen_per_rank():
    layout = _layout()
    group = Q_HEADS // KV_HEADS
    shard0 = [layout.original_q_head(kv, group, row) for kv in range(2) for row in range(group)]
    assert shard0 == list(range(16))


def test_splice_indexes_cover_every_remote_slot():
    q_per = Q_HEADS // DEGREE
    kv_per = KV_HEADS // DEGREE
    assert q_per == 16
    assert kv_per == 2
    remote = [(slot * q_per, (slot + 1) * q_per) for slot in range(1, DEGREE)]
    assert remote == [(16, 32), (32, 48), (48, 64)]
    covered = [index for start, end in remote for index in range(start, end)]
    assert covered == list(range(q_per, Q_HEADS))
    assert len(covered) == (DEGREE - 1) * q_per


def _read_long(view: torch.Tensor, block_table: torch.Tensor, pos: int, local_head: int) -> torch.Tensor:
    page_tokens = DEGREE * BLOCK
    physical = int(block_table[pos // page_tokens])
    intra = pos % page_tokens
    slot = local_head * DEGREE + intra // BLOCK
    token = intra % BLOCK
    return view[physical, token, slot]
