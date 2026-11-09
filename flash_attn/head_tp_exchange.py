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
    partial, q_len, payload_out_ptrs, arrived_ptrs, src_rank, max_q_per
):
    _ops().head_tp_return(partial, q_len, payload_out_ptrs, arrived_ptrs, src_rank, max_q_per)


def splice(out, payload_out, row_kind, slot_rank, q_range, max_q_per):
    _ops().head_tp_splice(out, payload_out, row_kind, slot_rank, q_range, max_q_per)
