// SPDX-License-Identifier: Apache-2.0
// Minimal test bridge, using the same adapters as the production registration.
#include <array>
#include <torch/extension.h>
#include "attention/sparse_kv_request_plan/sparse_kv_request_plan_torch_adpt.h"
#include "attention/sparse_kv_plan/sparse_kv_lru_torch_adpt.h"
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
  ops.def("npu_sparse_kv_request_plan(Tensor req_ids, Tensor topk_indices, Tensor stable_prefix_lens, "
          "Tensor visible_seq_lens, Tensor token_to_req, Tensor block_table, Tensor active_rows, "
          "Tensor(a!) last_req_ids, Tensor(b!) slot_to_token, Tensor(c!) priority, Tensor(d!) free_mask, "
          "Tensor(e!) free_stack, Tensor(f!) free_count, Tensor(g!) clock, Tensor(h!) current_slots, "
          "Tensor(i!) miss_count, Tensor(j!) miss_tokens, Tensor(k!) miss_slots, Tensor(l!) token_to_slot, "
          "Tensor(m!) status, Tensor(n!) resident_end, "
          "int max_queries, int max_token, int block_size, int host_num_blocks) -> ()");
  ops.impl("npu_sparse_kv_request_plan", torch::kPrivateUse1, &vllm_ascend::npu_sparse_kv_request_plan);
  ops.def("npu_sparse_kv_transfer(Tensor miss_count, Tensor miss_tokens, Tensor miss_slots, "
          "Tensor token_to_req, Tensor block_table, Tensor host_cache_bases, Tensor active_rows, "
          "Tensor(a!) resident_k, Tensor(b!) resident_v, int topk, int capacity, int block_size, "
          "int host_num_blocks, int token_size_bytes_k, int token_size_bytes_v) -> ()");
  ops.impl("npu_sparse_kv_transfer", torch::kPrivateUse1, &vllm_ascend::npu_sparse_kv_transfer);
}
