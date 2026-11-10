/******************************************************************************
 * Head-TP current-token exchange. This sits beside the attention mainloop.
 *
 * Launch order on each rank, after that rank's local and owner-shard attention:
 *   1. head_tp_send    write other shards' Q and current K/V into the peer
 *                      payload, then set arrived[peer][src, token] = 1
 *   2. head_tp_wait    inbound sum 0 returns immediately; otherwise spin on
 *                      this rank's arrived until that many cells are 1
 *   3. attention on the received Q (not this file)
 *   4. head_tp_return  read the partner attention output at q_range, write that
 *                      shard into the owner's payload_out, and set
 *                      arrived[owner][peer, token] = 1
 *   5. head_tp_wait    again, with expect = number of return shards
 *   6. head_tp_splice  copy those shards into out by q_range
 *
 * History KV is already on the partner. arrived is zeroed by the caller before
 * replay. Peer access is enabled by the caller. Payload pointers are the base
 * of a contiguous [dp, tokens, max_per, head_dim] buffer on that rank, indexed
 * by source rank. Request and return flags are different cells.
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

struct HeadTpSendArgs {
    int src_rank;
    int dp_size;
    int num_tokens;
    int head_dim;
    int num_q_heads;
    int num_kv_heads;
    int max_q_per;
    int max_kv_per;
    int element_size;
    const int8_t* row_kind;
    const int* slot_rank;
    const int* q_range;
    const int* kv_range;
    const void* q;
    const void* k;
    const void* v;
    const int64_t* payload_q_ptrs;
    const int64_t* payload_k_ptrs;
    const int64_t* payload_v_ptrs;
    const int64_t* arrived_ptrs;
};

struct HeadTpWaitArgs {
    const int* arrived;
    const int* inbound;
    const int* expect;
    int dp_size;
    int num_tokens;
};

struct HeadTpReturnArgs {
    int src_rank;
    int dp_size;
    int num_tokens;
    int head_dim;
    int num_q_heads;
    int max_q_per;
    int element_size;
    const void* out;
    const int8_t* row_kind;
    const int* slot_rank;
    const int* q_range;
    const int64_t* payload_out_ptrs;
    const int64_t* arrived_ptrs;
};

struct HeadTpSpliceArgs {
    int dp_size;
    int num_tokens;
    int head_dim;
    int num_q_heads;
    int max_q_per;
    int element_size;
    void* out;
    const void* payload_out;
    const int8_t* row_kind;
    const int* slot_rank;
    const int* q_range;
};

cudaError_t head_tp_send(const HeadTpSendArgs& args, cudaStream_t stream);
cudaError_t head_tp_wait(const HeadTpWaitArgs& args, cudaStream_t stream);
cudaError_t head_tp_return(const HeadTpReturnArgs& args, cudaStream_t stream);
cudaError_t head_tp_splice(const HeadTpSpliceArgs& args, cudaStream_t stream);
