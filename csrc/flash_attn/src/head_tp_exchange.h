/******************************************************************************
 * Head-TP exchange inside the paged FA2 forward.
 *
 * Send CTAs are the first linear block ids. Each one copies the current token
 * and publishes the flag, then returns. It does not read KV and it does not
 * wait. The active tile waits only on the partner, runs the mainloop, and the
 * owner splices after its own store (or after the split-KV combine). Idle KV
 * heads do not send and do not spin.
 *
 * The current token's partner K/V is not in this rank's packed page. It is read
 * from head_tp_k_token / head_tp_v_token, one row per request. Only split 0 of
 * each local head writes that token into the page. The other splits wait on
 * head_tp_kv_ready, so a later split cannot read the page mid-copy.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <cstdio>

#include "head_tp_page.h"

namespace FLASH_NAMESPACE {

constexpr uint64_t kHeadTpSpinLimit = 100000000ull;

__device__ __forceinline__ void head_tp_store_release(int *ptr) {
    unsigned bits = 1u;
    uint64_t address = reinterpret_cast<uint64_t>(ptr);
    asm volatile("st.global.release.sys.u32 [%0], %1;" ::"l"(address), "r"(bits) : "memory");
}

__device__ __forceinline__ int head_tp_load_acquire(const int *ptr) {
    unsigned bits;
    uint64_t address = reinterpret_cast<uint64_t>(ptr);
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(bits) : "l"(address) : "memory");
    return static_cast<int>(bits);
}

__device__ __forceinline__ void *head_tp_peer(const int64_t *table, int peer) {
    return reinterpret_cast<void *>(static_cast<uintptr_t>(table[peer]));
}

__device__ __forceinline__ void head_tp_copy(char *dst, const char *src, int nbytes, int tid, int nthreads) {
    const int nvec = nbytes / static_cast<int>(sizeof(uint4));
    const uint4 *src4 = reinterpret_cast<const uint4 *>(src);
    uint4 *dst4 = reinterpret_cast<uint4 *>(dst);
    for (int i = tid; i < nvec; i += nthreads) {
        dst4[i] = src4[i];
    }
    for (int i = nvec * static_cast<int>(sizeof(uint4)) + tid; i < nbytes; i += nthreads) {
        dst[i] = src[i];
    }
}

template <typename Params>
__device__ __forceinline__ bool head_tp_exchange_on(const Params &params) {
    return params.head_tp_src_rank >= 0 && params.head_tp_slot_rank != nullptr &&
           params.head_tp_row_kind != nullptr;
}

template <typename Params>
__device__ __forceinline__ bool head_tp_is_owner(const Params &params, int token) {
    const int dp = params.head_tp_dp_size;
    return params.head_tp_slot_rank[token * dp] == params.head_tp_src_rank;
}

template <typename Params>
__device__ int head_tp_seqlen(const Params &params, int token) {
    if (params.seqused_k != nullptr) {
        return params.seqused_k[token];
    }
    if (params.cu_seqlens_k == nullptr) {
        return params.seqlen_k;
    }
    if (params.is_seqlens_k_cumulative) {
        return params.cu_seqlens_k[token + 1] - params.cu_seqlens_k[token];
    }
    return params.cu_seqlens_k[token];
}

// Inactive KV heads, in increasing order, send slots 1 .. d-1. Matches
// flash_attn/head_tp_page.py:owner_send_slot, then drops a slot whose rank is empty.
template <typename Params>
__device__ int head_tp_owner_send_slot(const Params &params, int token, int kv_head) {
    if (!head_tp_exchange_on(params) || params.head_tp_row_kind[token] != 1) {
        return -1;
    }
    if (!head_tp_is_owner(params, token)) {
        return -1;
    }
    const int begin = params.head_tp_kv_begin;
    const int kv_per = params.head_tp_kv_per;
    const int local = kv_head - begin;
    if (local >= 0 && local < kv_per) {
        return -1;
    }
    int inactive = 0;
    for (int head = 0; head < kv_head; ++head) {
        const int other = head - begin;
        if (other < 0 || other >= kv_per) {
            ++inactive;
        }
    }
    const int slot = inactive + 1;
    const int dp = params.head_tp_dp_size;
    if (slot >= dp) {
        return -1;
    }
    const int dst = params.head_tp_slot_rank[token * dp + slot];
    if (dst < 0 || dst == params.head_tp_src_rank) {
        return -1;
    }
    return slot;
}

template <typename Params>
__device__ int64_t head_tp_vec_offset(const Params &params, int token, int q_head, bool output) {
    const int group = params.head_tp_q_per_kv > 1 ? params.head_tp_q_per_kv : 1;
    const int64_t batch_stride = output ? params.o_batch_stride : params.q_batch_stride;
    const int64_t row_stride = output ? params.o_row_stride : params.q_row_stride;
    const int64_t head_stride = output ? params.o_head_stride : params.q_head_stride;
    if (params.seqlenq_ngroups_swapped) {
        const int kv_head = q_head / group;
        const int row = q_head - kv_head * group;
        return static_cast<int64_t>(token) * batch_stride + static_cast<int64_t>(row) * row_stride +
               static_cast<int64_t>(kv_head) * head_stride;
    }
    return (static_cast<int64_t>(token) * params.head_tp_num_q_heads + q_head) * params.d;
}

// Later splits spin until each local head's split 0 has published.
__device__ __forceinline__ void head_tp_wait_count(int *ptr, int expected) {
    if (expected <= 0) {
        return;
    }
    if (threadIdx.x == 0) {
        for (uint64_t spin = 0; spin < kHeadTpSpinLimit; ++spin) {
            const int got = head_tp_load_acquire(ptr);
            if (got >= expected) {
                break;
            }
            if (spin + 1 == kHeadTpSpinLimit) {
                printf("head_tp kv ready timeout got=%d expected=%d\n", got, expected);
                asm volatile("trap;");
            }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
    }
    __syncthreads();
}

template <typename Params>
__device__ void head_tp_wait_flag(const Params &params, int *flag, int token) {
    if (threadIdx.x == 0) {
        for (uint64_t spin = 0; spin < kHeadTpSpinLimit; ++spin) {
            if (head_tp_load_acquire(flag) == 1) {
                break;
            }
            if (spin + 1 == kHeadTpSpinLimit) {
                printf("head_tp wait timeout token=%d\n", token);
                asm volatile("trap;");
            }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
    }
    __syncthreads();
}

template <typename Element, typename Params>
__device__ void head_tp_owner_send(const Params &params, int token, int slot, int tid, int nthreads) {
    const int dp = params.head_tp_dp_size;
    const int dst = params.head_tp_slot_rank[token * dp + slot];
    const int *q_pair = params.head_tp_xchg_q_range + (static_cast<int64_t>(token) * dp + slot) * 2;
    const int *kv_pair = params.head_tp_kv_range + (static_cast<int64_t>(token) * dp + slot) * 2;
    const int q_len = q_pair[1] - q_pair[0];
    const int kv_len = kv_pair[1] - kv_pair[0];
    if (q_len <= 0 || kv_len <= 0) {
        return;
    }
    const int es = static_cast<int>(sizeof(Element));
    const int dim = params.d;
    char *payload_q = static_cast<char *>(head_tp_peer(params.head_tp_payload_q_ptrs, dst));
    char *payload_k = static_cast<char *>(head_tp_peer(params.head_tp_payload_k_ptrs, dst));
    char *payload_v = static_cast<char *>(head_tp_peer(params.head_tp_payload_v_ptrs, dst));
    int *arrived = static_cast<int *>(head_tp_peer(params.head_tp_arrived_ptrs, dst));
    const char *q = static_cast<const char *>(params.q_ptr);
    const char *k = static_cast<const char *>(params.head_tp_k_token);
    const char *v = static_cast<const char *>(params.head_tp_v_token);
    for (int local = 0; local < q_len; ++local) {
        const int q_head = q_pair[0] + local;
        const int64_t src = head_tp_vec_offset(params, token, q_head, false) * es;
        const int64_t dst_off =
            ((static_cast<int64_t>(params.head_tp_src_rank) * params.head_tp_payload_tokens + token) *
                 params.head_tp_max_q_per +
             local) *
            dim * es;
        head_tp_copy(payload_q + dst_off, q + src, dim * es, tid, nthreads);
    }
    for (int local = 0; local < kv_len; ++local) {
        const int kv_head = kv_pair[0] + local;
        const int64_t src_k = (static_cast<int64_t>(token) * params.head_tp_k_token_batch_stride +
                               static_cast<int64_t>(kv_head) * params.head_tp_k_token_head_stride) *
                              es;
        const int64_t src_v = (static_cast<int64_t>(token) * params.head_tp_v_token_batch_stride +
                               static_cast<int64_t>(kv_head) * params.head_tp_v_token_head_stride) *
                              es;
        const int64_t dst_off =
            ((static_cast<int64_t>(params.head_tp_src_rank) * params.head_tp_payload_tokens + token) *
                 params.head_tp_max_kv_per +
             local) *
            dim * es;
        head_tp_copy(payload_k + dst_off, k + src_k, dim * es, tid, nthreads);
        head_tp_copy(payload_v + dst_off, v + src_v, dim * es, tid, nthreads);
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        head_tp_store_release(arrived + params.head_tp_src_rank * params.head_tp_payload_tokens + token);
    }
    __syncthreads();
}

// One CTA per (token, remote slot), launched ahead of every compute tile.
// Short rows and non-owners return without touching KV.
template <typename Element, typename Params>
__device__ void head_tp_priority_send(const Params &params, int linear, int tid, int nthreads) {
    if (!head_tp_exchange_on(params)) {
        return;
    }
    const int span = params.head_tp_dp_size - 1;
    if (span <= 0) {
        return;
    }
    const int token = linear / span;
    const int slot = 1 + linear % span;
    if (token < 0 || token >= params.b || params.head_tp_row_kind[token] != 1) {
        return;
    }
    if (!head_tp_is_owner(params, token)) {
        return;
    }
    const int dst = params.head_tp_slot_rank[token * params.head_tp_dp_size + slot];
    if (dst < 0 || dst == params.head_tp_src_rank) {
        return;
    }
    head_tp_owner_send<Element>(params, token, slot, tid, nthreads);
}

template <typename Element, typename Params>
__device__ void head_tp_partner_prepare(const Params &params, int token, int kv_head, int m_block, int k_block_m,
                                       int seqlen_q, int n_split_idx, int tid, int nthreads) {
    if (!head_tp_exchange_on(params) || params.head_tp_row_kind[token] != 1 || params.block_table == nullptr) {
        return;
    }
    if (head_tp_is_owner(params, token)) {
        return;
    }
    const int dp = params.head_tp_dp_size;
    int slot = -1;
    for (int candidate = 1; candidate < dp; ++candidate) {
        if (params.head_tp_slot_rank[token * dp + candidate] == params.head_tp_src_rank) {
            slot = candidate;
        }
    }
    if (slot < 0) {
        return;
    }
    const int owner = params.head_tp_slot_rank[token * dp];
    if (owner < 0) {
        return;
    }
    head_tp_wait_flag(params, params.head_tp_arrived_local + owner * params.head_tp_payload_tokens + token, token);
    const int *kv_pair = params.head_tp_kv_range + (static_cast<int64_t>(token) * dp + slot) * 2;
    const int local = kv_head - kv_pair[0];
    if (local < 0 || kv_head >= kv_pair[1]) {
        return;
    }
    const int seqlen = head_tp_seqlen(params, token);
    if (seqlen <= 0) {
        return;
    }
    // Every split used to copy this token. Those copies race the split that
    // has already started reading the page. One CTA per local head writes it.
    const bool publish = n_split_idx == 0 && m_block == 0;
    if (params.head_tp_kv_ready != nullptr && !publish) {
        head_tp_wait_count(params.head_tp_kv_ready + token, params.head_tp_kv_per);
        return;
    }
    const int es = static_cast<int>(sizeof(Element));
    const int dim = params.d;
    const int pack = head_tp_long_page(params, token) ? params.head_tp_pack : 1;
    const int *block_table = params.block_table + token * params.block_table_batch_stride;
    const int64_t page_off = head_tp_paged_offset<int64_t>(
        seqlen - 1, local, params.page_block_size, pack, block_table, params.k_batch_stride, params.k_row_stride,
        params.k_head_stride);
    const int64_t page_off_v = head_tp_paged_offset<int64_t>(
        seqlen - 1, local, params.page_block_size, pack, block_table, params.v_batch_stride, params.v_row_stride,
        params.v_head_stride);
    const char *local_k = static_cast<const char *>(params.head_tp_payload_k_local);
    const char *local_v = static_cast<const char *>(params.head_tp_payload_v_local);
    const int64_t src_off =
        ((static_cast<int64_t>(owner) * params.head_tp_payload_tokens + token) * params.head_tp_max_kv_per + local) *
        dim * es;
    head_tp_copy(static_cast<char *>(params.k_ptr) + page_off * es, local_k + src_off, dim * es, tid, nthreads);
    head_tp_copy(static_cast<char *>(params.v_ptr) + page_off_v * es, local_v + src_off, dim * es, tid, nthreads);
    const int *q_pair = params.head_tp_xchg_q_range + (static_cast<int64_t>(token) * dp + slot) * 2;
    const int q_begin = q_pair[0];
    const int q_end = q_pair[1];
    const char *local_q = static_cast<const char *>(params.head_tp_payload_q_local);
    const int group = params.head_tp_q_per_kv > 1 ? params.head_tp_q_per_kv : 1;
    if (params.seqlenq_ngroups_swapped) {
        const int row_begin = m_block * k_block_m;
        int row_end = row_begin + k_block_m;
        if (row_end > seqlen_q) {
            row_end = seqlen_q;
        }
        for (int row = row_begin; row < row_end; ++row) {
            const int q_head = kv_head * group + row;
            if (q_head < q_begin || q_head >= q_end) {
                continue;
            }
            const int q_local = q_head - q_begin;
            const int64_t q_src =
                ((static_cast<int64_t>(owner) * params.head_tp_payload_tokens + token) * params.head_tp_max_q_per +
                 q_local) *
                dim * es;
            const int64_t q_dst = head_tp_vec_offset(params, token, q_head, false) * es;
            head_tp_copy(static_cast<char *>(params.q_ptr) + q_dst, local_q + q_src, dim * es, tid, nthreads);
        }
    }
    __threadfence();
    __syncthreads();
    if (params.head_tp_kv_ready != nullptr && tid == 0) {
        atomicAdd(params.head_tp_kv_ready + token, 1);
    }
    __syncthreads();
}

template <typename Element, typename Params>
__device__ void head_tp_partner_return(const Params &params, int token, int bidh, int m_block, int k_block_m,
                                       int seqlen_q, int tid, int nthreads) {
    if (!head_tp_exchange_on(params) || params.head_tp_row_kind[token] != 1 || head_tp_is_owner(params, token)) {
        return;
    }
    const int dp = params.head_tp_dp_size;
    int slot = -1;
    for (int candidate = 1; candidate < dp; ++candidate) {
        if (params.head_tp_slot_rank[token * dp + candidate] == params.head_tp_src_rank) {
            slot = candidate;
        }
    }
    if (slot < 0 || params.head_tp_return_count == nullptr) {
        return;
    }
    const int owner = params.head_tp_slot_rank[token * dp];
    const int *q_pair = params.head_tp_xchg_q_range + (static_cast<int64_t>(token) * dp + slot) * 2;
    const int q_begin = q_pair[0];
    const int q_end = q_pair[1];
    const int q_len = q_end - q_begin;
    if (q_len <= 0) {
        return;
    }
    const int es = static_cast<int>(sizeof(Element));
    const int dim = params.d;
    const char *out = static_cast<const char *>(params.o_ptr);
    char *payload = static_cast<char *>(head_tp_peer(params.head_tp_payload_out_ptrs, owner));
    int *arrived = static_cast<int *>(head_tp_peer(params.head_tp_arrived_ptrs, owner));
    const int group = params.head_tp_q_per_kv > 1 ? params.head_tp_q_per_kv : 1;
    int copied = 0;
    if (params.seqlenq_ngroups_swapped) {
        const int row_begin = m_block * k_block_m;
        int row_end = row_begin + k_block_m;
        if (row_end > seqlen_q) {
            row_end = seqlen_q;
        }
        for (int row = row_begin; row < row_end; ++row) {
            const int q_head = bidh * group + row;
            if (q_head < q_begin || q_head >= q_end) {
                continue;
            }
            const int local = q_head - q_begin;
            const int64_t src = head_tp_vec_offset(params, token, q_head, true) * es;
            const int64_t dst =
                ((static_cast<int64_t>(params.head_tp_src_rank) * params.head_tp_payload_tokens + token) *
                     params.head_tp_max_q_per +
                 local) *
                dim * es;
            head_tp_copy(payload + dst, out + src, dim * es, tid, nthreads);
            ++copied;
        }
    } else if (m_block == 0 && bidh >= q_begin && bidh < q_end) {
        const int local = bidh - q_begin;
        const int64_t src = head_tp_vec_offset(params, token, bidh, true) * es;
        const int64_t dst =
            ((static_cast<int64_t>(params.head_tp_src_rank) * params.head_tp_payload_tokens + token) *
                 params.head_tp_max_q_per +
             local) *
            dim * es;
        head_tp_copy(payload + dst, out + src, dim * es, tid, nthreads);
        copied = 1;
    }
    if (copied == 0) {
        return;
    }
    // Publish only after every head this tile owns is stored. One atomic for
    // the whole tile, so an earlier head cannot release the flag while a later
    // head in the same tile is still being copied.
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        const int done = atomicAdd(params.head_tp_return_count + token, copied);
        if (done + copied == q_len) {
            head_tp_store_release(arrived + params.head_tp_src_rank * params.head_tp_payload_tokens + token);
        }
    }
    __syncthreads();
}

template <typename Element, typename Params>
__device__ void head_tp_owner_splice(const Params &params, int token, int slot, int tid, int nthreads) {
    const int dp = params.head_tp_dp_size;
    const int peer = params.head_tp_slot_rank[token * dp + slot];
    if (peer < 0) {
        return;
    }
    head_tp_wait_flag(params, params.head_tp_arrived_local + peer * params.head_tp_payload_tokens + token, token);
    const int *q_pair = params.head_tp_xchg_q_range + (static_cast<int64_t>(token) * dp + slot) * 2;
    const int q_len = q_pair[1] - q_pair[0];
    if (q_len <= 0) {
        return;
    }
    const int es = static_cast<int>(sizeof(Element));
    const int dim = params.d;
    char *out = static_cast<char *>(params.o_ptr);
    const char *payload = static_cast<const char *>(params.head_tp_payload_out_local);
    for (int local = 0; local < q_len; ++local) {
        const int q_head = q_pair[0] + local;
        const int64_t src =
            ((static_cast<int64_t>(peer) * params.head_tp_payload_tokens + token) * params.head_tp_max_q_per + local) *
            dim * es;
        const int64_t dst = head_tp_vec_offset(params, token, q_head, true) * es;
        head_tp_copy(out + dst, payload + src, dim * es, tid, nthreads);
    }
}

// Split-KV publishes the return only after combine has written the final out.
// Each combine block copies the Q rows it owns; the last block releases the flag.
// The owner's block that owns row 0 of the active KV head splices after that.
template <typename Element, typename Params>
__device__ void head_tp_after_combine(const Params &params, int bidx, int kBlockM, int tid, int nthreads) {
    if (!head_tp_exchange_on(params) || !params.seqlenq_ngroups_swapped) {
        return;
    }
    const int span = params.b * params.h * params.seqlen_q;
    const int group = params.head_tp_q_per_kv > 1 ? params.head_tp_q_per_kv : 1;
    const int dp = params.head_tp_dp_size;
    const int es = static_cast<int>(sizeof(Element));
    const int dim = params.d;
    int copied = 0;
    int publish_token = -1;
    int publish_q_len = 0;
    int splice_token = -1;
    for (int m = 0; m < kBlockM; ++m) {
        const int idx = bidx * kBlockM + m;
        if (idx >= span) {
            break;
        }
        const int batch_idx = idx / (params.h * params.seqlen_q);
        const int rem = idx - batch_idx * params.h * params.seqlen_q;
        const int head_idx = rem / params.seqlen_q;
        const int row = rem - head_idx * params.seqlen_q;
        if (params.head_tp_row_kind[batch_idx] != 1 || head_idx != params.head_tp_kv_begin) {
            continue;
        }
        if (head_tp_is_owner(params, batch_idx)) {
            if (row == 0) {
                splice_token = batch_idx;
            }
            continue;
        }
        int slot = -1;
        for (int candidate = 1; candidate < dp; ++candidate) {
            if (params.head_tp_slot_rank[batch_idx * dp + candidate] == params.head_tp_src_rank) {
                slot = candidate;
            }
        }
        if (slot < 0) {
            continue;
        }
        const int *q_pair = params.head_tp_xchg_q_range + (static_cast<int64_t>(batch_idx) * dp + slot) * 2;
        const int q_head = head_idx * group + row;
        if (q_head < q_pair[0] || q_head >= q_pair[1]) {
            continue;
        }
        const int local = q_head - q_pair[0];
        const int64_t src = head_tp_vec_offset(params, batch_idx, q_head, true) * es;
        const int64_t dst =
            ((static_cast<int64_t>(params.head_tp_src_rank) * params.head_tp_payload_tokens + batch_idx) *
                 params.head_tp_max_q_per +
             local) *
            dim * es;
        const int owner = params.head_tp_slot_rank[batch_idx * dp];
        char *payload = static_cast<char *>(head_tp_peer(params.head_tp_payload_out_ptrs, owner));
        const char *out = static_cast<const char *>(params.o_ptr);
        head_tp_copy(payload + dst, out + src, dim * es, tid, nthreads);
        copied += 1;
        publish_token = batch_idx;
        publish_q_len = q_pair[1] - q_pair[0];
    }
    if (copied > 0) {
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            const int owner = params.head_tp_slot_rank[publish_token * dp];
            int *arrived = static_cast<int *>(head_tp_peer(params.head_tp_arrived_ptrs, owner));
            const int done = atomicAdd(params.head_tp_return_count + publish_token, copied);
            if (done + copied == publish_q_len) {
                head_tp_store_release(
                    arrived + params.head_tp_src_rank * params.head_tp_payload_tokens + publish_token);
            }
        }
        __syncthreads();
    }
    if (splice_token >= 0) {
        for (int slot = 1; slot < dp; ++slot) {
            head_tp_owner_splice<Element>(params, splice_token, slot, tid, nthreads);
        }
    }
}

}  // namespace FLASH_NAMESPACE
