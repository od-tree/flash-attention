/******************************************************************************
 * Per-row KV page size for head-TP.
 *
 * Short rows keep page_block_size and one physical slot per KV head.
 * Long rows (row_kind == 1) store only this rank's kv_per heads. Local head h
 * occupies physical slots [h * pack, (h + 1) * pack). Those slots are one
 * contiguous run of pack * page_block_size tokens, so the paged reader uses
 * that larger page and a slot base of h * pack.
 ******************************************************************************/

#pragma once

#include <cstdint>

template <typename Params>
__device__ __forceinline__ bool head_tp_long_page(const Params &params, int batch) {
    return params.head_tp_row_kind != nullptr && params.head_tp_pack > 1 &&
           params.head_tp_row_kind[batch] == 1;
}

// A combine row owns partial `split` when the host tile table said so.
// Without a table every partial of params.num_splits is live, as before.
template <typename Params>
__device__ __forceinline__ bool head_tp_partial_used(const Params &params, int idx, int split) {
    if (params.head_tp_row_splits == nullptr) {
        return true;
    }
    const int span = params.b * params.h * params.seqlen_q;
    if (idx < 0 || idx >= span || params.seqlen_q <= 0) {
        return false;
    }
    const int batch = idx / (params.h * params.seqlen_q);
    const int rem = idx - batch * params.h * params.seqlen_q;
    const int head = rem / params.seqlen_q;
    if (params.head_tp_tiles != nullptr && head_tp_long_page(params, batch)) {
        const int local = head - params.head_tp_kv_begin;
        if (local < 0 || local >= params.head_tp_kv_per) {
            return false;
        }
    }
    return split < static_cast<int>(params.head_tp_row_splits[batch]);
}

template <typename Params>
__device__ __forceinline__ int head_tp_page_tokens(const Params &params, int batch) {
    if (!head_tp_long_page(params, batch)) {
        return params.page_block_size;
    }
    return params.page_block_size * params.head_tp_pack;
}

template <typename Params>
__device__ __forceinline__ int head_tp_kv_slot(const Params &params, int batch, int kv_head) {
    if (!head_tp_long_page(params, batch)) {
        return kv_head;
    }
    return kv_head * params.head_tp_pack;
}

// Element offset of one head vector. `local_head` is the KV head for a short
// page and the local head (0 .. kv_per-1) for a long page. A long page's slot
// is local_head * pack plus the logical block inside that page.
template <typename index_t>
__device__ __forceinline__ index_t head_tp_paged_offset(
    int pos, int local_head, int page_block_size, int pack, const int *block_table,
    index_t page_stride, index_t row_stride, index_t head_stride) {
    const int width = pack > 1 ? pack : 1;
    const int page_tokens = page_block_size * width;
    const int physical = block_table[pos / page_tokens];
    const int intra = pos % page_tokens;
    const int slot = local_head * width + intra / page_block_size;
    const int token = intra % page_block_size;
    return static_cast<index_t>(physical) * page_stride + static_cast<index_t>(token) * row_stride +
           static_cast<index_t>(slot) * head_stride;
}
