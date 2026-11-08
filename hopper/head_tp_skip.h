/******************************************************************************
 * Head-TP tile predicate. Same rule as csrc/flash_attn/src/head_tp_skip.h.
 * row_kind is indexed by batch: 0 local, 1 owner shard, 2 empty padding.
 * q_range is [batch, dp_size, 2]; shard 0 is the head interval this rank computes.
 ******************************************************************************/

#pragma once

#include <cstdint>

template <typename Params>
__device__ __forceinline__ bool head_tp_tile_active(const Params &params, int batch, int head) {
    const int8_t *row_kind = params.head_tp_row_kind;
    if (row_kind == nullptr) {
        return true;
    }
    const int8_t kind = row_kind[batch];
    if (kind == 2) {
        return false;
    }
    if (kind == 0) {
        return true;
    }
    if (kind != 1 || params.head_tp_q_range == nullptr || params.head_tp_dp_size <= 0) {
        return false;
    }
    const int base = (batch * params.head_tp_dp_size) * 2;
    const int q_begin = params.head_tp_q_range[base];
    const int q_end = params.head_tp_q_range[base + 1];
    return head >= q_begin && head < q_end;
}
