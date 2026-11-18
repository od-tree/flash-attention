// One block per token. Copies partner Q heads published by the compute phase.
// It does not read KV and it does not recompute attention.

#include "namespace_config.h"
#include "flash.h"
#include "cuda_check.h"
#include "head_tp_exchange.h"

#include <cutlass/numeric_types.h>
#include <cuda_runtime.h>

namespace FLASH_NAMESPACE {

template <typename Element>
__global__ void head_tp_splice_only_kernel(Flash_fwd_params params) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= params.b || !head_tp_exchange_on(params) || params.head_tp_row_kind == nullptr) {
        return;
    }
    if (params.head_tp_row_kind[token] != 1 || !head_tp_is_owner(params, token)) {
        return;
    }
    const int dp = params.head_tp_dp_size;
    for (int slot = 1; slot < dp; ++slot) {
        head_tp_owner_splice<Element>(params, token, slot, static_cast<int>(threadIdx.x),
                                      static_cast<int>(blockDim.x));
    }
}

void head_tp_launch_splice(const Flash_fwd_params &params, cudaStream_t stream, bool bf16) {
    if (params.b <= 0) {
        return;
    }
    if (bf16) {
        head_tp_splice_only_kernel<cutlass::bfloat16_t><<<params.b, 128, 0, stream>>>(params);
    } else {
        head_tp_splice_only_kernel<cutlass::half_t><<<params.b, 128, 0, stream>>>(params);
    }
    FLASHATTENTION_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace FLASH_NAMESPACE
