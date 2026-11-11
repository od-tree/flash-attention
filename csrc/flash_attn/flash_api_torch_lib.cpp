#include "namespace_config.h"

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>

#include "registration.h"

#include <optional>
#include <vector>

using torch::stable::Tensor;

/**
 *  Externs for the flash_attn ops to be exposed as a (stable-ABI) pytorch library.
 */

namespace FLASH_NAMESPACE {

////////////////////////////// From flash_api.cpp //////////////////////////////

std::vector<Tensor>
mha_varlen_fwd(Tensor q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               std::optional<Tensor> out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const Tensor &cu_seqlens_q,  // b+1
               const Tensor &cu_seqlens_k,  // b+1
               std::optional<Tensor> seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               const std::optional<Tensor> &leftpad_k_, // batch_size
               std::optional<Tensor> block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<Tensor> alibi_slopes_, // num_heads or b x num_heads
               int64_t max_seqlen_q,
               const int64_t max_seqlen_k,
               const double p_dropout,
               const double softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int64_t window_size_left,
               int64_t window_size_right,
               const double softcap,
               const bool return_softmax,
               int64_t num_splits,
               // Retained only for backwards-compat arg positioning; must be None.
               std::optional<Tensor> unused_generator_compat,
               std::optional<Tensor> head_tp_row_kind,
               int64_t head_tp_pack,
               int64_t head_tp_kv_begin,
               int64_t head_tp_src_rank,
               std::optional<Tensor> head_tp_slot_rank,
               std::optional<Tensor> head_tp_kv_range,
               std::optional<Tensor> head_tp_xchg_q_range,
               std::optional<Tensor> head_tp_k_token,
               std::optional<Tensor> head_tp_v_token,
               std::optional<Tensor> head_tp_payload_q_ptrs,
               std::optional<Tensor> head_tp_payload_k_ptrs,
               std::optional<Tensor> head_tp_payload_v_ptrs,
               std::optional<Tensor> head_tp_payload_out_ptrs,
               std::optional<Tensor> head_tp_arrived_ptrs,
               std::optional<Tensor> head_tp_arrived_local,
               std::optional<Tensor> head_tp_payload_out,
               std::optional<Tensor> head_tp_payload_q,
               std::optional<Tensor> head_tp_payload_k,
               std::optional<Tensor> head_tp_payload_v,
               std::optional<Tensor> head_tp_return_count,
               int64_t head_tp_max_q_per,
               int64_t head_tp_max_kv_per,
               int64_t head_tp_payload_tokens);

std::vector<Tensor>
mha_fwd_kvcache(Tensor q,                 // batch_size x seqlen_q x num_heads x head_size
                Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const std::optional<Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                const std::optional<Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                const std::optional<Tensor> &seqlens_k_, // batch_size
                const std::optional<Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                const std::optional<Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                const std::optional<Tensor> &cache_batch_idx_, // indices to index into the KV cache
                const std::optional<Tensor> &leftpad_k_, // batch_size
                std::optional<Tensor> block_table_, // batch_size x max_num_blocks_per_seq
                std::optional<Tensor> alibi_slopes_, // num_heads or batch_size x num_heads
                std::optional<Tensor> out_,             // batch_size x seqlen_q x num_heads x head_size
                const double softmax_scale,
                bool is_causal,
                int64_t window_size_left,
                int64_t window_size_right,
                const double softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int64_t num_splits);

/////////////////////////// From flash_api_sparse.cpp //////////////////////////

std::vector<Tensor>
mha_fwd_sparse(Tensor q,         // batch_size x seqlen_q x num_heads x head_size
               const Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
               const Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
               const Tensor &block_count,
               const Tensor &block_offset,
               const Tensor &column_count,
               const Tensor &column_index,
               std::optional<Tensor> out_,             // batch_size x seqlen_q x num_heads x head_size
               std::optional<Tensor> alibi_slopes_, // num_heads or batch_size x num_heads
               const double p_dropout,
               const double softmax_scale,
               bool is_causal,
               const double softcap,
               const bool return_softmax,
               // Retained only for backwards-compat arg positioning; must be None.
               std::optional<Tensor> unused_generator_compat);

std::vector<Tensor>
mha_varlen_fwd_sparse(Tensor q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                      const Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                      const Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                      const Tensor &block_count,
                      const Tensor &block_offset,
                      const Tensor &column_count,
                      const Tensor &column_index,
                      std::optional<Tensor> out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                      const Tensor &cu_seqlens_q,  // b+1
                      const Tensor &cu_seqlens_k,  // b+1
                      std::optional<Tensor> seqused_k, // b. If given, only this many elements of each batch element's keys are used.
                      std::optional<Tensor> alibi_slopes_, // num_heads or b x num_heads
                      int64_t max_seqlen_q,
                      const int64_t max_seqlen_k,
                      const double p_dropout,
                      const double softmax_scale,
                      const bool zero_tensors,
                      bool is_causal,
                      const double softcap,
                      const bool return_softmax,
                      // Retained only for backwards-compat arg positioning; must be None.
                      std::optional<Tensor> unused_generator_compat);

} // namespace FLASH_NAMESPACE

/**
 *  Torch Library Registration (stable ABI)
 */
STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("varlen_fwd(Tensor! q, Tensor k, Tensor v, Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, "
            "int num_splits, Tensor? gen, Tensor? head_tp_row_kind=None, int head_tp_pack=0, "
            "int head_tp_kv_begin=0, int head_tp_src_rank=-1, Tensor? head_tp_slot_rank=None, "
            "Tensor? head_tp_kv_range=None, Tensor? head_tp_xchg_q_range=None, "
            "Tensor? head_tp_k_token=None, Tensor? head_tp_v_token=None, "
            "Tensor? head_tp_payload_q_ptrs=None, Tensor? head_tp_payload_k_ptrs=None, "
            "Tensor? head_tp_payload_v_ptrs=None, Tensor? head_tp_payload_out_ptrs=None, "
            "Tensor? head_tp_arrived_ptrs=None, Tensor? head_tp_arrived_local=None, "
            "Tensor? head_tp_payload_out=None, Tensor? head_tp_payload_q=None, "
            "Tensor? head_tp_payload_k=None, Tensor? head_tp_payload_v=None, "
            "Tensor? head_tp_return_count=None, "
            "int head_tp_max_q_per=0, int head_tp_max_kv_per=0, int head_tp_payload_tokens=0) -> Tensor[]");

    ops.def("fwd_kvcache(Tensor! q, Tensor kcache, Tensor vcache, Tensor? k, Tensor? v, Tensor? seqlens_k, "
            "Tensor? rotary_cos, Tensor? rotary_sin, Tensor? cache_batch_idx, Tensor? leftpad_k, Tensor? block_table, "
            "Tensor? alibi_slopes, Tensor!? out, float softmax_scale, bool is_causal, int window_size_left, "
            "int window_size_right, float softcap, bool is_rotary_interleaved, int num_splits) -> Tensor[]");

#ifndef FA2_MINIMAL_HDIM256_BF16
    ops.def("fwd_sparse(Tensor! q, Tensor k, Tensor v, "
            "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
            "Tensor!? out, Tensor? alibi_slopes, "
            "float p_dropout, float softmax_scale, bool is_causal, "
            "float softcap, bool return_softmax, Tensor? gen)"
            "-> Tensor[]");

    ops.def("varlen_fwd_sparse(Tensor! q, Tensor k, Tensor v, "
            "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
            "Tensor!? out, Tensor cu_seqlens_q, "
            "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? alibi_slopes, "
            "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
            "bool is_causal, float softcap, bool return_softmax, Tensor? gen) -> Tensor[]");
#endif
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("varlen_fwd", TORCH_BOX(&FLASH_NAMESPACE::mha_varlen_fwd));
    ops.impl("fwd_kvcache", TORCH_BOX(&FLASH_NAMESPACE::mha_fwd_kvcache));
#ifndef FA2_MINIMAL_HDIM256_BF16
    ops.impl("fwd_sparse", TORCH_BOX(&FLASH_NAMESPACE::mha_fwd_sparse));
    ops.impl("varlen_fwd_sparse", TORCH_BOX(&FLASH_NAMESPACE::mha_varlen_fwd_sparse));
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
