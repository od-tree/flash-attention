"""d=2 page layout. Qwen3.5: 16 Q heads, 2 KV heads, one local KV head per rank."""

import importlib.util
from pathlib import Path

import torch

BLOCK = 16
DEGREE = 2
KV_HEADS = 2
Q_HEADS = 16
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


def test_long_page_stores_one_head_as_contiguous_tokens():
    layout = _layout()
    seq_len = DEGREE * BLOCK
    logical = _logical(seq_len)
    pages = layout.pack_long_pages(logical, DEGREE, BLOCK)
    assert pages.dtype == DTYPE
    assert pages.shape == (1, KV_HEADS, BLOCK, DIM)
    restored = layout.logical_tokens(pages, head=0, degree=DEGREE)
    assert torch.equal(restored, logical[0])


def test_physical_pages_are_one_over_d_of_a_full_head_allocation():
    layout = _layout()
    seq_len = 4 * DEGREE * BLOCK
    logical = _logical(seq_len)
    pages = layout.pack_long_pages(logical, DEGREE, BLOCK)
    full_head_pages = (seq_len + BLOCK - 1) // BLOCK
    assert pages.shape[0] * DEGREE == full_head_pages


def test_short_page_keeps_both_kv_heads():
    layout = _layout()
    heads = torch.arange(KV_HEADS, dtype=torch.float32).to(DTYPE).view(KV_HEADS, 1, 1).expand(KV_HEADS, BLOCK, DIM).contiguous()
    page = layout.short_page(heads)
    assert page.shape == (KV_HEADS, BLOCK, DIM)
    assert torch.equal(page[0], heads[0])
    assert torch.equal(page[1], heads[1])


def test_paged_address_reads_the_packed_head():
    layout = _layout()
    seq_len = 2 * DEGREE * BLOCK
    logical = _logical(seq_len)
    pages = layout.pack_long_pages(logical, DEGREE, BLOCK)
    view = pages.transpose(1, 2)
    block_table = torch.arange(pages.shape[0], dtype=torch.int32)
    for pos in (0, BLOCK - 1, BLOCK, seq_len - 1):
        got = _read_long(view, block_table, pos, local_head=0)
        assert torch.equal(got, logical[0, pos])


def test_owner_gathers_the_one_remote_q_range():
    q_per = Q_HEADS // DEGREE
    remote = [(slot * q_per, (slot + 1) * q_per) for slot in range(1, DEGREE)]
    assert remote == [(q_per, Q_HEADS)]


def test_idle_kv_head_sends_the_only_remote_slot():
    layout = _layout()
    kv_per = KV_HEADS // DEGREE
    assert layout.owner_send_slot(0, kv_begin=0, kv_per=kv_per, degree=DEGREE) == -1
    assert layout.owner_send_slot(1, kv_begin=0, kv_per=kv_per, degree=DEGREE) == 1
    # The partner owns KV head 1, so it does not use the owner's send map.
    assert layout.owner_send_slot(1, kv_begin=1, kv_per=kv_per, degree=DEGREE) == -1


def test_returned_q_head_follows_the_gqa_group():
    layout = _layout()
    group = Q_HEADS // KV_HEADS
    assert [layout.original_q_head(0, group, row) for row in range(group)] == list(range(8))
    assert [layout.original_q_head(1, group, row) for row in range(group)] == list(range(8, 16))


def _read_long(view: torch.Tensor, block_table: torch.Tensor, pos: int, local_head: int) -> torch.Tensor:
    page_tokens = DEGREE * BLOCK
    physical = int(block_table[pos // page_tokens])
    intra = pos % page_tokens
    slot = local_head * DEGREE + intra // BLOCK
    token = intra % BLOCK
    return view[physical, token, slot]
