#include "head_tp_exchange.h"

#include <cstdio>

namespace {

constexpr uint64_t kSpinLimit = 100000000;
constexpr int kThreads = 128;

__device__ __forceinline__ void store_release_system(int* ptr, int value) {
    unsigned bits = static_cast<unsigned>(value);
    uint64_t address = reinterpret_cast<uint64_t>(ptr);
    asm volatile("st.global.release.sys.u32 [%0], %1;" ::"l"(address), "r"(bits) : "memory");
}

__device__ __forceinline__ int load_acquire_system(const int* ptr) {
    unsigned bits;
    uint64_t address = reinterpret_cast<uint64_t>(ptr);
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(bits) : "l"(address) : "memory");
    return static_cast<int>(bits);
}

__device__ __forceinline__ void* peer_ptr(const int64_t* table, int peer) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(table[peer]));
}

__device__ void copy_bytes(char* dst, const char* src, int nbytes, int tid, int nthreads) {
    int nvec = nbytes / static_cast<int>(sizeof(uint4));
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (int i = tid; i < nvec; i += nthreads) {
        dst4[i] = src4[i];
    }
}

__device__ void publish(int* flag) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        store_release_system(flag, 1);
    }
}

__global__ void head_tp_send_kernel(HeadTpSendArgs args) {
    int token = static_cast<int>(blockIdx.x);
    int slot = static_cast<int>(blockIdx.y);
    if (token >= args.num_tokens || slot <= 0 || slot >= args.dp_size) {
        return;
    }
    if (args.row_kind[token] != 1) {
        return;
    }
    int dst = args.slot_rank[token * args.dp_size + slot];
    if (dst < 0 || dst >= args.dp_size || dst == args.src_rank) {
        return;
    }
    const int* q_pair = args.q_range + (token * args.dp_size + slot) * 2;
    const int* kv_pair = args.kv_range + (token * args.dp_size + slot) * 2;
    int q_len = q_pair[1] - q_pair[0];
    int kv_len = kv_pair[1] - kv_pair[0];
    if (q_pair[0] < 0 || kv_pair[0] < 0 || q_len <= 0 || kv_len <= 0 || q_len > args.max_q_per ||
        kv_len > args.max_kv_per) {
        return;
    }

    int es = args.element_size;
    int q_row = args.head_dim * es;
    const char* q = static_cast<const char*>(args.q);
    const char* k = static_cast<const char*>(args.k);
    const char* v = static_cast<const char*>(args.v);
    char* payload_q = static_cast<char*>(peer_ptr(args.payload_q_ptrs, dst));
    char* payload_k = static_cast<char*>(peer_ptr(args.payload_k_ptrs, dst));
    char* payload_v = static_cast<char*>(peer_ptr(args.payload_v_ptrs, dst));
    int* arrived = static_cast<int*>(peer_ptr(args.arrived_ptrs, dst));

    int q_src = ((token * args.num_q_heads + q_pair[0]) * args.head_dim) * es;
    int kv_src_k = ((token * args.num_kv_heads + kv_pair[0]) * args.head_dim) * es;
    int kv_src_v = kv_src_k;
    int q_dst = ((args.src_rank * args.num_tokens + token) * args.max_q_per * args.head_dim) * es;
    int kv_dst = ((args.src_rank * args.num_tokens + token) * args.max_kv_per * args.head_dim) * es;
    copy_bytes(payload_q + q_dst, q + q_src, q_len * q_row, threadIdx.x, blockDim.x);
    copy_bytes(payload_k + kv_dst, k + kv_src_k, kv_len * q_row, threadIdx.x, blockDim.x);
    copy_bytes(payload_v + kv_dst, v + kv_src_v, kv_len * q_row, threadIdx.x, blockDim.x);
    publish(arrived + args.src_rank * args.num_tokens + token);
}

__global__ void head_tp_wait_kernel(HeadTpWaitArgs args) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int target = 0;
    if (args.inbound != nullptr) {
        for (int i = 0; i < args.dp_size; ++i) {
            target += args.inbound[i];
        }
    } else if (args.expect != nullptr) {
        target = args.expect[0];
    }
    if (target <= 0) {
        return;
    }
    int cells = args.dp_size * args.num_tokens;
    for (uint64_t spin = 0; spin < kSpinLimit; ++spin) {
        int got = 0;
        for (int i = 0; i < cells; ++i) {
            if (load_acquire_system(args.arrived + i) == 1) {
                ++got;
            }
        }
        if (got >= target) {
            return;
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
    printf("head_tp_wait timeout target=%d\n", target);
    asm volatile("trap;");
}

__global__ void head_tp_return_kernel(HeadTpReturnArgs args) {
    int token = static_cast<int>(blockIdx.x);
    int slot = static_cast<int>(blockIdx.y);
    if (token >= args.num_tokens || slot <= 0 || slot >= args.dp_size) {
        return;
    }
    if (args.row_kind[token] != 1) {
        return;
    }
    int peer = args.slot_rank[token * args.dp_size + slot];
    if (peer != args.src_rank) {
        return;
    }
    int owner = args.slot_rank[token * args.dp_size];
    if (owner < 0 || owner >= args.dp_size || owner == args.src_rank) {
        return;
    }
    const int* q_pair = args.q_range + (token * args.dp_size + slot) * 2;
    int q_len = q_pair[1] - q_pair[0];
    if (q_pair[0] < 0 || q_len <= 0 || q_len > args.max_q_per ||
        q_pair[0] + q_len > args.num_q_heads) {
        return;
    }
    int es = args.element_size;
    int row = args.head_dim * es;
    const char* out = static_cast<const char*>(args.out);
    char* payload_out = static_cast<char*>(peer_ptr(args.payload_out_ptrs, owner));
    int* arrived = static_cast<int*>(peer_ptr(args.arrived_ptrs, owner));
    int src_off = ((token * args.num_q_heads + q_pair[0]) * args.head_dim) * es;
    int dst_off =
        ((args.src_rank * args.payload_tokens + token) * args.max_q_per * args.head_dim) * es;
    copy_bytes(payload_out + dst_off, out + src_off, q_len * row, threadIdx.x, blockDim.x);
    publish(arrived + args.src_rank * args.payload_tokens + token);
}

__global__ void head_tp_splice_kernel(HeadTpSpliceArgs args) {
    int token = static_cast<int>(blockIdx.x);
    int slot = static_cast<int>(blockIdx.y);
    if (token >= args.num_tokens || slot <= 0 || slot >= args.dp_size) {
        return;
    }
    if (args.row_kind[token] != 1) {
        return;
    }
    int peer = args.slot_rank[token * args.dp_size + slot];
    if (peer < 0 || peer >= args.dp_size) {
        return;
    }
    const int* q_pair = args.q_range + (token * args.dp_size + slot) * 2;
    int q_len = q_pair[1] - q_pair[0];
    if (q_pair[0] < 0 || q_len <= 0 || q_len > args.max_q_per ||
        q_pair[0] + q_len > args.num_q_heads) {
        return;
    }
    int es = args.element_size;
    int row = args.head_dim * es;
    const char* payload = static_cast<const char*>(args.payload_out);
    char* out = static_cast<char*>(args.out);
    int src_off = ((peer * args.num_tokens + token) * args.max_q_per * args.head_dim) * es;
    int dst_off = ((token * args.num_q_heads + q_pair[0]) * args.head_dim) * es;
    copy_bytes(out + dst_off, payload + src_off, q_len * row, threadIdx.x, blockDim.x);
}

bool aligned_row(int head_dim, int element_size) {
    return element_size > 0 && head_dim > 0 && (head_dim * element_size) % static_cast<int>(sizeof(uint4)) == 0;
}

}  // namespace

cudaError_t head_tp_send(const HeadTpSendArgs& args, cudaStream_t stream) {
    if (!aligned_row(args.head_dim, args.element_size) || args.dp_size <= 0) {
        return cudaErrorInvalidValue;
    }
    if (args.num_tokens <= 0) {
        return cudaSuccess;
    }
    dim3 grid(args.num_tokens, args.dp_size);
    head_tp_send_kernel<<<grid, kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

cudaError_t head_tp_wait(const HeadTpWaitArgs& args, cudaStream_t stream) {
    if (args.dp_size <= 0 || args.num_tokens < 0 || args.arrived == nullptr) {
        return cudaErrorInvalidValue;
    }
    head_tp_wait_kernel<<<1, 1, 0, stream>>>(args);
    return cudaGetLastError();
}

cudaError_t head_tp_return(const HeadTpReturnArgs& args, cudaStream_t stream) {
    if (!aligned_row(args.head_dim, args.element_size) || args.dp_size <= 0) {
        return cudaErrorInvalidValue;
    }
    if (args.num_tokens <= 0) {
        return cudaSuccess;
    }
    dim3 grid(args.num_tokens, args.dp_size);
    head_tp_return_kernel<<<grid, kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

cudaError_t head_tp_splice(const HeadTpSpliceArgs& args, cudaStream_t stream) {
    if (!aligned_row(args.head_dim, args.element_size) || args.dp_size <= 0) {
        return cudaErrorInvalidValue;
    }
    if (args.num_tokens <= 0) {
        return cudaSuccess;
    }
    dim3 grid(args.num_tokens, args.dp_size);
    head_tp_splice_kernel<<<grid, kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}
