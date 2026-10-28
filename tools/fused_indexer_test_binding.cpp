// SPDX-License-Identifier: Apache-2.0
#include "attention/quant_lightning_indexer_sparse_kv/quant_lightning_indexer_sparse_kv_torch_adpt.h"
#include "attention/sparse_kv_plan/sparse_kv_lru_torch_adpt.h"
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def(
        "npu_quant_lightning_indexer_sparse_kv("
        "Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, "
        "Tensor key_dequant_scale, Tensor actual_seq_lengths_query, "
        "Tensor actual_seq_lengths_key, Tensor indexer_block_table, Tensor req_ids, "
        "Tensor stable_prefix_lens, Tensor visible_seq_lens, Tensor token_to_req, "
        "Tensor kv_block_table, Tensor active_rows, Tensor(a!) last_req_ids, "
        "Tensor(b!) slot_to_token, Tensor(c!) priority, Tensor(d!) free_mask, Tensor(e!) free_stack, "
        "Tensor(f!) free_count, Tensor(g!) clock, Tensor(h!) current_slots, Tensor(i!) miss_count, "
        "Tensor(j!) miss_tokens, Tensor(k!) miss_slots, Tensor(l!) token_to_slot, Tensor(m!) status, "
        "Tensor(n!) resident_end, int capacity, int max_token, int block_size, "
        "int host_num_blocks, int max_queries) -> Tensor");
    ops.impl("npu_quant_lightning_indexer_sparse_kv", torch::kPrivateUse1,
             &vllm_ascend::npu_quant_lightning_indexer_sparse_kv);

    ops.def(
        "npu_sparse_kv_transfer("
        "Tensor miss_count, Tensor miss_tokens, Tensor miss_slots, "
        "Tensor token_to_req, Tensor block_table, Tensor host_cache_bases, "
        "Tensor active_rows, Tensor(a!) resident_k, Tensor(b!) resident_v, "
        "int topk, int capacity, int block_size, int host_num_blocks, "
        "int token_size_bytes_k, int token_size_bytes_v) -> ()");
    ops.impl("npu_sparse_kv_transfer", torch::kPrivateUse1,
             &vllm_ascend::npu_sparse_kv_transfer);

}
