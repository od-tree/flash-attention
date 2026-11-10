"""Launch the head-TP exchange kernels.

The kernels live in csrc/flash_attn/src/head_tp_exchange.cu and are registered on
whichever FlashAttention extension is loaded. Peer buffers are caller-owned.
arrived is zeroed by the caller before each replay. Attention stays outside
these launches: send, wait, partner attention, return, wait, splice.
"""


def _ops():
    import torch

    for name in ("_vllm_fa2_C", "_vllm_fa3_C"):
        module = getattr(torch.ops, name, None)
        if module is not None and hasattr(module, "head_tp_send"):
            return module
    raise RuntimeError("head_tp exchange kernels are not loaded")


def send(
    q,
    k,
    v,
    row_kind,
    slot_rank,
    q_range,
    kv_range,
    payload_q_ptrs,
    payload_k_ptrs,
    payload_v_ptrs,
    arrived_ptrs,
    src_rank,
    max_q_per,
    max_kv_per,
):
    _ops().head_tp_send(
        q, k, v, row_kind, slot_rank, q_range, kv_range, payload_q_ptrs, payload_k_ptrs,
        payload_v_ptrs, arrived_ptrs, src_rank, max_q_per, max_kv_per,
    )


def wait(arrived, inbound=None, expect=None):
    """Spin on device until `inbound` sums to enough 1s, or `expect` is met.

    Pass inbound for the request wait and expect for the return wait. A zero
    sum returns without spinning.
    """
    _ops().head_tp_wait(arrived, inbound, expect)


def publish_return(
    out, row_kind, slot_rank, q_range, payload_out_ptrs, arrived_ptrs, src_rank, max_q_per
):
    """Copy this rank's attention heads back to the owner.

    ``out`` is the attention output, indexed like the Q that ``send`` reads.
    The kernel copies ``q_range`` for the slot owned by ``src_rank``.
    """
    _ops().head_tp_return(
        attention_output_rows(out),
        row_kind,
        slot_rank,
        q_range,
        payload_out_ptrs,
        arrived_ptrs,
        src_rank,
        max_q_per,
    )


def splice(out, payload_out, row_kind, slot_rank, q_range, max_q_per):
    _ops().head_tp_splice(out, payload_out, row_kind, slot_rank, q_range, max_q_per)


def attention_output_rows(out):
    """View an attention output as [tokens, heads, dim].

    FA2 ``fwd_kvcache`` and FA3/FA4 decode both use one query position, stored
    as [tokens, 1, heads, dim]. FA4 varlen and the CPU reference store
    [tokens, heads, dim]. The storage is the attention output either way.
    """
    if out.dim() == 3:
        rows = out
    elif out.dim() == 4 and out.size(1) == 1:
        rows = out.view(out.size(0), out.size(2), out.size(3))
    else:
        raise ValueError(
            "attention output must be [tokens, heads, dim] or [tokens, 1, heads, dim]"
        )
    if not rows.is_contiguous():
        raise ValueError("attention output rows must be contiguous")
    return rows


def splice_attention_output(out, payload_out, row_kind, slot_rank, q_range, max_q_per):
    """Copy returned shards into the attention output, in place.

    ``out`` is the tensor the attention kernel already wrote. Partner heads
    that the kernel left untouched are filled from ``payload_out``.
    """
    splice(
        attention_output_rows(out),
        payload_out,
        row_kind,
        slot_rank,
        q_range,
        max_q_per,
    )
