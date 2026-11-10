#include "head_tp_exchange.h"
#include "registration.h"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>
#include <optional>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

#define HEAD_TP_CHECK(cond, ...) STD_TORCH_CHECK(cond, __VA_ARGS__)

namespace {

#define STABLE_TORCH_LIBRARY_FRAGMENT_EXPAND(NAME, MODULE) \
    STABLE_TORCH_LIBRARY_FRAGMENT(NAME, MODULE)

cudaStream_t current_stream(const Tensor& tensor) {
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
    return static_cast<cudaStream_t>(stream_ptr);
}

void check_launch(cudaError_t error, const char* what) {
    HEAD_TP_CHECK(error == cudaSuccess, what, " failed: ", cudaGetErrorString(error));
}

int element_size_of(ScalarType dtype) {
    if (dtype == ScalarType::Half || dtype == ScalarType::BFloat16) {
        return 2;
    }
    if (dtype == ScalarType::Float) {
        return 4;
    }
    return 0;
}

void check_cuda_contiguous(const Tensor& tensor, const char* name) {
    HEAD_TP_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    HEAD_TP_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_ptrs(const Tensor& ptrs, int dp_size, const char* name) {
    check_cuda_contiguous(ptrs, name);
    HEAD_TP_CHECK(ptrs.scalar_type() == ScalarType::Long, name, " must be int64");
    HEAD_TP_CHECK(ptrs.numel() == dp_size, name, " must have one pointer per rank");
}

void head_tp_send_op(const Tensor& q,
                     const Tensor& k,
                     const Tensor& v,
                     const Tensor& row_kind,
                     const Tensor& slot_rank,
                     const Tensor& q_range,
                     const Tensor& kv_range,
                     const Tensor& payload_q_ptrs,
                     const Tensor& payload_k_ptrs,
                     const Tensor& payload_v_ptrs,
                     const Tensor& arrived_ptrs,
                     int64_t src_rank,
                     int64_t max_q_per,
                     int64_t max_kv_per) {
    check_cuda_contiguous(q, "q");
    check_cuda_contiguous(k, "k");
    check_cuda_contiguous(v, "v");
    check_cuda_contiguous(row_kind, "row_kind");
    check_cuda_contiguous(slot_rank, "slot_rank");
    check_cuda_contiguous(q_range, "q_range");
    check_cuda_contiguous(kv_range, "kv_range");
    HEAD_TP_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                  "q, k, and v must have the same dtype");
    int element_size = element_size_of(q.scalar_type());
    HEAD_TP_CHECK(element_size != 0, "q must be fp16, bf16, or fp32");
    HEAD_TP_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, and v must be [tokens, heads, dim]");
    int num_tokens = static_cast<int>(q.size(0));
    int num_q_heads = static_cast<int>(q.size(1));
    int head_dim = static_cast<int>(q.size(2));
    int num_kv_heads = static_cast<int>(k.size(1));
    HEAD_TP_CHECK(k.size(0) == num_tokens && v.size(0) == num_tokens, "k and v token count must match q");
    HEAD_TP_CHECK(k.size(2) == head_dim && v.size(2) == head_dim, "k and v head dim must match q");
    HEAD_TP_CHECK(row_kind.scalar_type() == ScalarType::Char, "row_kind must be int8");
    HEAD_TP_CHECK(row_kind.numel() == num_tokens, "row_kind must have one entry per token");
    HEAD_TP_CHECK(slot_rank.scalar_type() == ScalarType::Int, "slot_rank must be int32");
    HEAD_TP_CHECK(slot_rank.dim() == 2 && slot_rank.size(0) == num_tokens, "slot_rank must be [tokens, dp]");
    int dp_size = static_cast<int>(slot_rank.size(1));
    HEAD_TP_CHECK(q_range.sizes().equals({num_tokens, dp_size, 2}), "q_range must be [tokens, dp, 2]");
    HEAD_TP_CHECK(kv_range.sizes().equals({num_tokens, dp_size, 2}), "kv_range must be [tokens, dp, 2]");
    HEAD_TP_CHECK(q_range.scalar_type() == ScalarType::Int && kv_range.scalar_type() == ScalarType::Int,
                  "ranges must be int32");
    check_ptrs(payload_q_ptrs, dp_size, "payload_q_ptrs");
    check_ptrs(payload_k_ptrs, dp_size, "payload_k_ptrs");
    check_ptrs(payload_v_ptrs, dp_size, "payload_v_ptrs");
    check_ptrs(arrived_ptrs, dp_size, "arrived_ptrs");
    HEAD_TP_CHECK(src_rank >= 0 && src_rank < dp_size, "src_rank is out of range");
    HEAD_TP_CHECK(max_q_per > 0 && max_kv_per > 0, "payload head capacity must be positive");
    HEAD_TP_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0, "q must be 16-byte aligned");
    HEAD_TP_CHECK(reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0, "k must be 16-byte aligned");
    HEAD_TP_CHECK(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "v must be 16-byte aligned");

    HeadTpSendArgs args{};
    args.src_rank = static_cast<int>(src_rank);
    args.dp_size = dp_size;
    args.num_tokens = num_tokens;
    args.head_dim = head_dim;
    args.num_q_heads = num_q_heads;
    args.num_kv_heads = num_kv_heads;
    args.max_q_per = static_cast<int>(max_q_per);
    args.max_kv_per = static_cast<int>(max_kv_per);
    args.element_size = element_size;
    args.row_kind = static_cast<const int8_t*>(row_kind.data_ptr());
    args.slot_rank = static_cast<const int*>(slot_rank.data_ptr());
    args.q_range = static_cast<const int*>(q_range.data_ptr());
    args.kv_range = static_cast<const int*>(kv_range.data_ptr());
    args.q = q.data_ptr();
    args.k = k.data_ptr();
    args.v = v.data_ptr();
    args.payload_q_ptrs = static_cast<const int64_t*>(payload_q_ptrs.data_ptr());
    args.payload_k_ptrs = static_cast<const int64_t*>(payload_k_ptrs.data_ptr());
    args.payload_v_ptrs = static_cast<const int64_t*>(payload_v_ptrs.data_ptr());
    args.arrived_ptrs = static_cast<const int64_t*>(arrived_ptrs.data_ptr());
    check_launch(head_tp_send(args, current_stream(q)), "head_tp_send");
}

void head_tp_wait_op(const Tensor& arrived, std::optional<Tensor> inbound, std::optional<Tensor> expect) {
    check_cuda_contiguous(arrived, "arrived");
    HEAD_TP_CHECK(arrived.scalar_type() == ScalarType::Int, "arrived must be int32");
    HEAD_TP_CHECK(arrived.dim() == 2, "arrived must be [dp, tokens]");
    HEAD_TP_CHECK(!(inbound.has_value() && expect.has_value()), "pass inbound or expect, not both");
    int dp_size = static_cast<int>(arrived.size(0));
    int num_tokens = static_cast<int>(arrived.size(1));
    const int* inbound_ptr = nullptr;
    const int* expect_ptr = nullptr;
    if (inbound.has_value()) {
        check_cuda_contiguous(*inbound, "inbound");
        HEAD_TP_CHECK(inbound->scalar_type() == ScalarType::Int, "inbound must be int32");
        HEAD_TP_CHECK(inbound->numel() == dp_size, "inbound must have one count per rank");
        inbound_ptr = static_cast<const int*>(inbound->data_ptr());
    }
    if (expect.has_value()) {
        check_cuda_contiguous(*expect, "expect");
        HEAD_TP_CHECK(expect->scalar_type() == ScalarType::Int, "expect must be int32");
        HEAD_TP_CHECK(expect->numel() == 1, "expect must be a single int32");
        expect_ptr = static_cast<const int*>(expect->data_ptr());
    }
    HeadTpWaitArgs args{};
    args.arrived = static_cast<const int*>(arrived.data_ptr());
    args.inbound = inbound_ptr;
    args.expect = expect_ptr;
    args.dp_size = dp_size;
    args.num_tokens = num_tokens;
    check_launch(head_tp_wait(args, current_stream(arrived)), "head_tp_wait");
}

void head_tp_return_op(const Tensor& out,
                       const Tensor& row_kind,
                       const Tensor& slot_rank,
                       const Tensor& q_range,
                       const Tensor& payload_out_ptrs,
                       const Tensor& arrived_ptrs,
                       int64_t src_rank,
                       int64_t max_q_per,
                       int64_t payload_tokens) {
    check_cuda_contiguous(out, "out");
    check_cuda_contiguous(row_kind, "row_kind");
    check_cuda_contiguous(slot_rank, "slot_rank");
    check_cuda_contiguous(q_range, "q_range");
    int element_size = element_size_of(out.scalar_type());
    HEAD_TP_CHECK(element_size != 0, "out must be fp16, bf16, or fp32");
    HEAD_TP_CHECK(out.dim() == 3, "out must be [tokens, heads, dim]");
    int num_tokens = static_cast<int>(out.size(0));
    int num_q_heads = static_cast<int>(out.size(1));
    int head_dim = static_cast<int>(out.size(2));
    HEAD_TP_CHECK(slot_rank.dim() == 2 && slot_rank.size(0) == num_tokens, "slot_rank must be [tokens, dp]");
    int dp_size = static_cast<int>(slot_rank.size(1));
    HEAD_TP_CHECK(row_kind.scalar_type() == ScalarType::Char && row_kind.numel() == num_tokens,
                  "row_kind must be int8 [tokens]");
    HEAD_TP_CHECK(slot_rank.scalar_type() == ScalarType::Int && q_range.scalar_type() == ScalarType::Int,
                  "slot_rank and q_range must be int32");
    HEAD_TP_CHECK(q_range.sizes().equals({num_tokens, dp_size, 2}), "q_range must be [tokens, dp, 2]");
    check_ptrs(payload_out_ptrs, dp_size, "payload_out_ptrs");
    check_ptrs(arrived_ptrs, dp_size, "arrived_ptrs");
    HEAD_TP_CHECK(src_rank >= 0 && src_rank < dp_size, "src_rank is out of range");
    HEAD_TP_CHECK(payload_tokens >= num_tokens, "payload_tokens is shorter than out");
    HEAD_TP_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "out must be 16-byte aligned");

    HeadTpReturnArgs args{};
    args.src_rank = static_cast<int>(src_rank);
    args.dp_size = dp_size;
    args.num_tokens = num_tokens;
    args.payload_tokens = static_cast<int>(payload_tokens);
    args.head_dim = head_dim;
    args.num_q_heads = num_q_heads;
    args.max_q_per = static_cast<int>(max_q_per);
    args.element_size = element_size;
    args.out = out.data_ptr();
    args.row_kind = static_cast<const int8_t*>(row_kind.data_ptr());
    args.slot_rank = static_cast<const int*>(slot_rank.data_ptr());
    args.q_range = static_cast<const int*>(q_range.data_ptr());
    args.payload_out_ptrs = static_cast<const int64_t*>(payload_out_ptrs.data_ptr());
    args.arrived_ptrs = static_cast<const int64_t*>(arrived_ptrs.data_ptr());
    check_launch(head_tp_return(args, current_stream(out)), "head_tp_return");
}

void head_tp_splice_op(Tensor out,
                       const Tensor& payload_out,
                       const Tensor& row_kind,
                       const Tensor& slot_rank,
                       const Tensor& q_range,
                       int64_t max_q_per) {
    check_cuda_contiguous(out, "out");
    check_cuda_contiguous(payload_out, "payload_out");
    check_cuda_contiguous(row_kind, "row_kind");
    check_cuda_contiguous(slot_rank, "slot_rank");
    check_cuda_contiguous(q_range, "q_range");
    int element_size = element_size_of(out.scalar_type());
    HEAD_TP_CHECK(element_size != 0 && out.scalar_type() == payload_out.scalar_type(),
                  "out and payload_out must be fp16, bf16, or fp32");
    HEAD_TP_CHECK(out.dim() == 3, "out must be [tokens, heads, dim]");
    int num_tokens = static_cast<int>(out.size(0));
    int num_q_heads = static_cast<int>(out.size(1));
    int head_dim = static_cast<int>(out.size(2));
    HEAD_TP_CHECK(slot_rank.dim() == 2 && slot_rank.size(0) == num_tokens, "slot_rank must be [tokens, dp]");
    int dp_size = static_cast<int>(slot_rank.size(1));
    HEAD_TP_CHECK(payload_out.dim() == 4, "payload_out must be [dp, tokens, max_q_per, dim]");
    HEAD_TP_CHECK(payload_out.size(0) == dp_size && payload_out.size(1) == num_tokens, "payload_out shape mismatch");
    HEAD_TP_CHECK(payload_out.size(2) == max_q_per && payload_out.size(3) == head_dim, "payload_out shape mismatch");
    HEAD_TP_CHECK(row_kind.scalar_type() == ScalarType::Char && row_kind.numel() == num_tokens,
                  "row_kind must be int8 [tokens]");
    HEAD_TP_CHECK(slot_rank.scalar_type() == ScalarType::Int && q_range.scalar_type() == ScalarType::Int,
                  "slot_rank and q_range must be int32");
    HEAD_TP_CHECK(q_range.sizes().equals({num_tokens, dp_size, 2}), "q_range must be [tokens, dp, 2]");
    HEAD_TP_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "out must be 16-byte aligned");
    HEAD_TP_CHECK(reinterpret_cast<uintptr_t>(payload_out.data_ptr()) % 16 == 0,
                  "payload_out must be 16-byte aligned");

    HeadTpSpliceArgs args{};
    args.dp_size = dp_size;
    args.num_tokens = num_tokens;
    args.head_dim = head_dim;
    args.num_q_heads = num_q_heads;
    args.max_q_per = static_cast<int>(max_q_per);
    args.element_size = element_size;
    args.out = out.data_ptr();
    args.payload_out = payload_out.data_ptr();
    args.row_kind = static_cast<const int8_t*>(row_kind.data_ptr());
    args.slot_rank = static_cast<const int*>(slot_rank.data_ptr());
    args.q_range = static_cast<const int*>(q_range.data_ptr());
    check_launch(head_tp_splice(args, current_stream(out)), "head_tp_splice");
}

STABLE_TORCH_LIBRARY_FRAGMENT_EXPAND(TORCH_EXTENSION_NAME, head_tp_ops) {
    head_tp_ops.def(
        "head_tp_send(Tensor q, Tensor k, Tensor v, Tensor row_kind, Tensor slot_rank, "
        "Tensor q_range, Tensor kv_range, Tensor payload_q_ptrs, Tensor payload_k_ptrs, "
        "Tensor payload_v_ptrs, Tensor arrived_ptrs, int src_rank, int max_q_per, int max_kv_per) -> ()");
    head_tp_ops.def("head_tp_wait(Tensor arrived, Tensor? inbound, Tensor? expect) -> ()");
    head_tp_ops.def(
        "head_tp_return(Tensor out, Tensor row_kind, Tensor slot_rank, Tensor q_range, "
        "Tensor payload_out_ptrs, Tensor arrived_ptrs, int src_rank, int max_q_per, "
        "int payload_tokens) -> ()");
    head_tp_ops.def(
        "head_tp_splice(Tensor! out, Tensor payload_out, Tensor row_kind, Tensor slot_rank, "
        "Tensor q_range, int max_q_per) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, head_tp_ops) {
    head_tp_ops.impl("head_tp_send", TORCH_BOX(&head_tp_send_op));
    head_tp_ops.impl("head_tp_wait", TORCH_BOX(&head_tp_wait_op));
    head_tp_ops.impl("head_tp_return", TORCH_BOX(&head_tp_return_op));
    head_tp_ops.impl("head_tp_splice", TORCH_BOX(&head_tp_splice_op));
}

}  // namespace
