/******************************************************************************
 * Head-TP exchange inside the FA2 forward.
 *
 * The inactive head tile sends the current token, then waits and splices the
 * partner shard. Local tiles never wait. A partner tile waits for that send
 * before its mainloop. Return runs after the final store: the non-split tile
 * or the combine kernel, not a split partial.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <cstdio>

#include "head_tp_skip.h"

namespace FLASH_NAMESPACE {

constexpr uint64_t kHeadTpSpinLimit = 100000000;

__device__ __forceinline__ void head_tp_store_release(int* ptr) {
    unsigned bits = 1u;
    uint64_t address = reinterpret_cast<uint64_t>(ptr);
    asm volatile("st.global.release.sys.u32 [%0], %1;" ::"l"(address), "r"(bits) : "memory");
}

__device__ __forceinline__ int head_tp_load_acquire(const int* ptr) {
    unsigned bits;
    uint64_t address = reinterpret_cast<uint64_t>(ptr);
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(bits) : "l"(address) : "memory");
    return static_cast<int>(bits);
}

__device__ __forceinline__ void* head_tp_peer(const int64_t* table, int peer) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(table[peer]));
}

__device__ __forceinline__ void head_tp_copy(char* dst, const char* src, int nbytes, int tid, int nthreads) {
    int nvec = nbytes / static_cast<int>(sizeof(uint4));
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (int i = tid; i < nvec; i += nthreads) {
        dst4[i] = src4[i];
    }
}

__device__ __forceinline__ void head_tp_publish(int* flag) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        head_tp_store_release(flag);
    }
}

template <typename Params>
__device__ __forceinline__ bool head_tp_exchange_on(const Params& params) {
    return params.head_tp_src_rank >= 0 && params.head_tp_slot_rank != nullptr &&
           params.head_tp_row_kind != nullptr;
}

template <typename Params>
__device__ __forceinline__ bool head_tp_is_owner(const Params& params, int token) {
    return params.head_tp_slot_rank[token * params.head_tp_dp_size] == params.head_tp_src_rank;
}

template <typename Params>
__device__ int head_tp_seqlen(const Params& params, int token) {
    if (params.cu_seqlens_k == nullptr) {
        return params.seqlen_k;
    }
    if (params.is_seqlens_k_cumulative) {
        return params.cu_seqlens_k[token + 1] - params.cu_seqlens_k[token];
    }
    return params.cu_seqlens_k[token];
}

template <typename Params>
__device__ bool head_tp_owner_block(const Params& params, int token, int head, int m_block, int split) {
    if (!head_tp_exchange_on(params) || m_block != 0 || split != 0) {
        return false;
    }
    if (token < 0 || token >= params.b || params.head_tp_row_kind[token] != 1) {
        return false;
    }
    if (!head_tp_is_owner(params, token) || head_tp_tile_active(params, token, head)) {
        return false;
    }
    for (int prev = 0; prev < head; ++prev) {
        if (!head_tp_tile_active(params, token, prev)) {
            return false;
        }
    }
    return true;
}

template <typename Element, typename Params>
__device__ void head_tp_owner_send_wait_splice(const Params& params, int token, int tid, int nthreads) {
    int dp = params.head_tp_dp_size;
    int es = static_cast<int>(sizeof(Element));
    int dim = params.d;
    int q_heads = params.head_tp_num_q_heads;
    const char* q = static_cast<const char*>(params.q_ptr);
    const char* k = static_cast<const char*>(params.k_ptr);
    const char* v = static_cast<const char*>(params.v_ptr);
    int seqlen = head_tp_seqlen(params, token);
    int pos = seqlen - 1;
    for (int slot = 1; slot < dp; ++slot) {
        int dst = params.head_tp_slot_rank[token * dp + slot];
        if (dst < 0 || dst >= dp || dst == params.head_tp_src_rank) {
            continue;
        }
        const int* q_pair = params.head_tp_xchg_q_range + (token * dp + slot) * 2;
        const int* kv_pair = params.head_tp_kv_range + (token * dp + slot) * 2;
        int q_len = q_pair[1] - q_pair[0];
        int kv_len = kv_pair[1] - kv_pair[0];
        if (q_len <= 0 || kv_len <= 0 || pos < 0) {
            continue;
        }
        char* payload_q = static_cast<char*>(head_tp_peer(params.head_tp_payload_q_ptrs, dst));
        char* payload_k = static_cast<char*>(head_tp_peer(params.head_tp_payload_k_ptrs, dst));
        char* payload_v = static_cast<char*>(head_tp_peer(params.head_tp_payload_v_ptrs, dst));
        int* arrived = static_cast<int*>(head_tp_peer(params.head_tp_arrived_ptrs, dst));
        int64_t q_src = (static_cast<int64_t>(token) * q_heads + q_pair[0]) * dim * es;
        int64_t q_dst = (static_cast<int64_t>(params.head_tp_src_rank) * params.head_tp_payload_tokens + token) *
                        params.head_tp_max_q_per * dim * es;
        int64_t kv_src = (static_cast<int64_t>(token) * params.k_batch_stride +
                          static_cast<int64_t>(pos) * params.k_row_stride +
                          static_cast<int64_t>(kv_pair[0]) * params.k_head_stride) * es;
        int64_t kv_src_v = (static_cast<int64_t>(token) * params.v_batch_stride +
                            static_cast<int64_t>(pos) * params.v_row_stride +
                            static_cast<int64_t>(kv_pair[0]) * params.v_head_stride) * es;
        int64_t kv_dst = (static_cast<int64_t>(params.head_tp_src_rank) * params.head_tp_payload_tokens + token) *
                         params.head_tp_max_kv_per * dim * es;
        head_tp_copy(payload_q + q_dst, q + q_src, q_len * dim * es, tid, nthreads);
        head_tp_copy(payload_k + kv_dst, k + kv_src, kv_len * dim * es, tid, nthreads);
        head_tp_copy(payload_v + kv_dst, v + kv_src_v, kv_len * dim * es, tid, nthreads);
        head_tp_publish(arrived + params.head_tp_src_rank * params.head_tp_payload_tokens + token);
    }
    // One return flag per remote slot. d = 2 publishes a single cell.
    if (tid == 0) {
        for (uint64_t spin = 0; spin < kHeadTpSpinLimit; ++spin) {
            int got = 0;
            int need = 0;
            for (int slot = 1; slot < dp; ++slot) {
                int peer = params.head_tp_slot_rank[token * dp + slot];
                if (peer < 0 || peer >= dp || peer == params.head_tp_src_rank) {
                    continue;
                }
                ++need;
                int cell = peer * params.head_tp_payload_tokens + token;
                if (head_tp_load_acquire(params.head_tp_arrived_local + cell) == 1) {
                    ++got;
                }
            }
            if (need == 0 || got >= need) {
                break;
            }
            if (spin + 1 == kHeadTpSpinLimit) {
                printf("head_tp owner splice timeout token=%d\n", token);
                asm volatile("trap;");
            }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
    }
    __syncthreads();
    char* out = static_cast<char*>(params.o_ptr);
    const char* payload = static_cast<const char*>(params.head_tp_payload_out_local);
    for (int slot = 1; slot < dp; ++slot) {
        int peer = params.head_tp_slot_rank[token * dp + slot];
        if (peer < 0 || peer >= dp) {
            continue;
        }
        const int* q_pair = params.head_tp_xchg_q_range + (token * dp + slot) * 2;
        int q_len = q_pair[1] - q_pair[0];
        if (q_len <= 0) {
            continue;
        }
        int64_t src = (static_cast<int64_t>(peer) * params.head_tp_payload_tokens + token) *
                      params.head_tp_max_q_per * dim * es;
        int64_t dst = (static_cast<int64_t>(token) * q_heads + q_pair[0]) * dim * es;
        head_tp_copy(out + dst, payload + src, q_len * dim * es, tid, nthreads);
    }
}

template <typename Params>
__device__ void head_tp_partner_wait(const Params& params, int token) {
    if (!head_tp_exchange_on(params) || params.head_tp_row_kind[token] != 1) {
        return;
    }
    if (head_tp_is_owner(params, token)) {
        return;
    }
    int dp = params.head_tp_dp_size;
    int owner = params.head_tp_slot_rank[token * dp];
    bool mine = false;
    for (int slot = 1; slot < dp; ++slot) {
        if (params.head_tp_slot_rank[token * dp + slot] == params.head_tp_src_rank) {
            mine = true;
        }
    }
    if (!mine || owner < 0) {
        return;
    }
    int cell = owner * params.head_tp_payload_tokens + token;
    if (threadIdx.x == 0) {
        for (uint64_t spin = 0; spin < kHeadTpSpinLimit; ++spin) {
            if (head_tp_load_acquire(params.head_tp_arrived_local + cell) == 1) {
                break;
            }
            if (spin + 1 == kHeadTpSpinLimit) {
                printf("head_tp partner wait timeout token=%d\n", token);
                asm volatile("trap;");
            }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
    }
    __syncthreads();
}

// Copy one finished Q head from the caller's output storage into the owner's payload.
template <typename Element, typename Params>
__device__ void head_tp_partner_return_head(const Params& params, int token, int q_head, int tid, int nthreads) {
    if (!head_tp_exchange_on(params) || token < 0 || token >= params.b) {
        return;
    }
    if (params.head_tp_row_kind[token] != 1 || head_tp_is_owner(params, token)) {
        return;
    }
    int dp = params.head_tp_dp_size;
    int owner = -1;
    int q_begin = 0;
    int q_end = 0;
    for (int slot = 1; slot < dp; ++slot) {
        if (params.head_tp_slot_rank[token * dp + slot] != params.head_tp_src_rank) {
            continue;
        }
        const int* q_pair = params.head_tp_xchg_q_range + (token * dp + slot) * 2;
        owner = params.head_tp_slot_rank[token * dp];
        q_begin = q_pair[0];
        q_end = q_pair[1];
    }
    if (owner < 0 || q_head < q_begin || q_head >= q_end || params.head_tp_return_count == nullptr) {
        return;
    }
    int es = static_cast<int>(sizeof(Element));
    int dim = params.d;
    int local = q_head - q_begin;
    int q_len = q_end - q_begin;
    const char* out = static_cast<const char*>(params.o_ptr);
    char* payload = static_cast<char*>(head_tp_peer(params.head_tp_payload_out_ptrs, owner));
    int* arrived = static_cast<int*>(head_tp_peer(params.head_tp_arrived_ptrs, owner));
    int64_t src = (static_cast<int64_t>(token) * params.head_tp_num_q_heads + q_head) * dim * es;
    int64_t dst = ((static_cast<int64_t>(params.head_tp_src_rank) * params.head_tp_payload_tokens + token) *
                       params.head_tp_max_q_per * dim +
                   static_cast<int64_t>(local) * dim) *
                  es;
    head_tp_copy(payload + dst, out + src, dim * es, tid, nthreads);
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        int done = atomicAdd(params.head_tp_return_count + token, 1);
        if (done + 1 == q_len) {
            head_tp_store_release(arrived + params.head_tp_src_rank * params.head_tp_payload_tokens + token);
        }
    }
    __syncthreads();
}

template <typename Element, typename Params>
__device__ void head_tp_partner_return_span(const Params& params, int token, int head, int tid, int nthreads) {
    int group = params.head_tp_q_per_kv > 1 ? params.head_tp_q_per_kv : 1;
    int begin = head * group;
    for (int q_head = begin; q_head < begin + group; ++q_head) {
        head_tp_partner_return_head<Element>(params, token, q_head, tid, nthreads);
    }
}

}  // namespace FLASH_NAMESPACE
