"""Launch the head-TP exchange kernels.

The kernels live in csrc/flash_attn/src/head_tp_exchange.cu and are registered on
whichever FlashAttention extension is loaded. Peer buffers are caller-owned.
arrived is zeroed by the caller before each replay. Attention stays outside
these launches: send, wait, partner attention, return, wait, splice.
"""


def _loaded_ops():
    import torch

    for name in ("_vllm_fa2_C", "_vllm_fa3_C"):
        module = getattr(torch.ops, name, None)
        if module is not None and hasattr(module, "head_tp_send"):
            return module
    return None


def _ops():
    module = _loaded_ops()
    if module is not None:
        return module
    for name in ("vllm_flash_attn._vllm_fa2_C", "vllm_flash_attn._vllm_fa3_C"):
        try:
            __import__(name)
        except ImportError:
            continue
        module = _loaded_ops()
        if module is not None:
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
    out,
    row_kind,
    slot_rank,
    q_range,
    payload_out_ptrs,
    arrived_ptrs,
    src_rank,
    max_q_per,
    payload_tokens,
):
    """Copy this rank's attention heads back to the owner.

    ``out`` is the attention output, indexed like the Q that ``send`` reads.
    The kernel copies ``q_range`` for the slot owned by ``src_rank``.
    ``payload_tokens`` is the token count of the peer buffer
    ``[dp, payload_tokens, max_q_per, dim]``. ``out`` may be a prefix of that
    axis; the copy still lands at ``[src_rank, token]``.
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
        payload_tokens,
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
