# Head-TP KV page layout shared by the tests and the paged kernel.
#
# A physical page is [num_kv_heads, block_size, dim] in head-major order, the
# same bytes vLLM keeps before transpose(1, 2). Short rows use every head for
# block_size tokens. Long rows keep only this rank's kv_per heads: local head
# h occupies slots [h * d, (h + 1) * d), and those slots are d * block_size
# contiguous tokens of that head.

import torch


def pack_long_pages(tokens_per_head: torch.Tensor, degree: int, block_size: int) -> torch.Tensor:
    """Pack per-head tokens into physical pages.

    Args:
        tokens_per_head: [kv_per, seq_len, dim]. seq_len is a multiple of
            degree * block_size.
        degree: head-TP width d.
        block_size: physical block size before packing.

    Returns:
        [num_pages, kv_per * degree, block_size, dim]
    """
    if tokens_per_head.ndim != 3:
        raise ValueError("tokens_per_head must be [kv_per, seq_len, dim]")
    kv_per, seq_len, dim = tokens_per_head.shape
    span = degree * block_size
    if degree < 2 or seq_len % span != 0:
        raise ValueError(f"seq_len {seq_len} is not a multiple of d * block_size ({span})")
    num_pages = seq_len // span
    num_slots = kv_per * degree
    pages = torch.empty(
        (num_pages, num_slots, block_size, dim),
        dtype=tokens_per_head.dtype,
        device=tokens_per_head.device,
    )
    for head in range(kv_per):
        for page in range(num_pages):
            chunk = tokens_per_head[head, page * span : (page + 1) * span]
            for slot_in_head in range(degree):
                slot = head * degree + slot_in_head
                start = slot_in_head * block_size
                pages[page, slot] = chunk[start : start + block_size]
    return pages


def logical_tokens(pages: torch.Tensor, head: int, degree: int) -> torch.Tensor:
    """Read one local head back out of packed pages, in token order."""
    span_slots = pages[:, head * degree : (head + 1) * degree]
    num_pages, _, block_size, dim = span_slots.shape
    return span_slots.reshape(num_pages, degree * block_size, dim).reshape(-1, dim)


def owner_send_slot(kv_head: int, kv_begin: int, kv_per: int, degree: int) -> int:
    """Remote slot sent by this KV head, or -1.

    Inactive KV heads in increasing order send slots 1 .. degree-1. An active
    head, or an idle head past that, sends nothing. The device function
    head_tp_owner_send_slot applies the same rule and then skips an empty rank.
    """
    local = kv_head - kv_begin
    if 0 <= local < kv_per:
        return -1
    inactive = 0
    for head in range(kv_head):
        other = head - kv_begin
        if other < 0 or other >= kv_per:
            inactive += 1
    slot = inactive + 1
    if slot >= degree:
        return -1
    return slot


def original_q_head(kv_head: int, group: int, row: int) -> int:
    """Q head stored for KV head `kv_head` and group row `row` after the GQA swap."""
    return kv_head * group + row


def short_page(tokens_per_head: torch.Tensor) -> torch.Tensor:
    """One short page: [num_kv_heads, block_size, dim] with every head present."""
    if tokens_per_head.ndim != 3:
        raise ValueError("tokens_per_head must be [num_kv_heads, block_size, dim]")
    return tokens_per_head
