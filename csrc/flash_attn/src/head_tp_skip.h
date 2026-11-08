/******************************************************************************
 * Head-TP tile predicate.
 *
 * Same shape as the fully-masked-block prune in FA4: a tile that has nothing
 * to compute returns before the attention mainloop. Hopper's warp-specialized
 * pipeline must not branch out of the consumer loop; those tiles are omitted
 * by the FA4 block-sparsity prepass (see flash_attn/head_tp.py).
 ******************************************************************************/

#pragma once

#include <cstdint>

// row_kind: 0 local (all heads), 1 owner shard, 2 empty padding row.
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
    // Shard 0 of q_range[batch, dp_size, 2].
    const int base = (batch * params.head_tp_dp_size) * 2;
    const int q_begin = params.head_tp_q_range[base];
    const int q_end = params.head_tp_q_range[base + 1];
    return head >= q_begin && head < q_end;
}
