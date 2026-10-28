# SPDX-License-Identifier: Apache-2.0
"""Persistent request state for the fused 950 QLI operator.

TopK and request planning execute in one mixed kernel. Transfer consumes its
request-major misses on the same stream. Capture allocates all state up front.
This operator harness does not adapt the legacy serving row-major hot pool.
"""

import torch

_PROBE_STAT_COLUMNS = 3
_MAX_QUERY_ROWS = 16
_TOPK = 2048
_BLOCK_SIZE = 128
_MAX_CAPACITY = 1 << 24
_INT32_MAX = (1 << 31) - 1


class SparseKVFusedIndexer:
    def __init__(
        self, *, max_requests, max_rows, max_queries, topk, capacity, max_token, block_size, host_num_blocks, device
    ):
        if not (
            1 <= max_queries <= _MAX_QUERY_ROWS
            and topk == _TOPK
            and max_queries * topk <= capacity <= _MAX_CAPACITY
            and 0 < max_requests <= _INT32_MAX
            and 0 < max_rows <= _INT32_MAX
            and 0 < max_token <= _INT32_MAX
            and block_size == _BLOCK_SIZE
            and 0 < host_num_blocks <= _INT32_MAX
            and capacity % block_size == 0
        ):
            raise ValueError("Invalid shared planner shape; capacity must cover max_queries*topk and whole pages")
        self.max_requests, self.max_rows = max_requests, max_rows
        self.max_queries, self.topk, self.capacity = max_queries, topk, capacity
        self.max_token, self.block_size, self.host_num_blocks = max_token, block_size, host_num_blocks
        self.union_capacity = max_queries * topk
        i32 = {"dtype": torch.int32, "device": device}
        i64 = {"dtype": torch.int64, "device": device}
        self.last_req_ids = torch.full((max_requests,), -1, **i64)
        self.slot_to_token = torch.full((max_requests, capacity), -1, **i32)
        self.priority = torch.zeros((max_requests, capacity), **i64)
        self.free_mask = torch.ones((max_requests, capacity), **i32)
        self.free_stack = torch.arange(capacity - 1, -1, -1, **i32).repeat(max_requests, 1)
        self.free_count = torch.full((max_requests,), capacity, **i32)
        self.clock = torch.zeros(max_requests, **i64)
        self.current_slots = torch.full((max_rows, topk), -1, **i32)
        self.miss_count = torch.zeros(max_requests, **i32)
        self.miss_tokens = torch.full((max_requests, self.union_capacity), -1, **i32)
        self.miss_slots = torch.full_like(self.miss_tokens, -1)
        self.token_to_slot = torch.full((max_requests, max_token), -1, **i32)
        self.resident_end = torch.zeros(max_requests, **i32)
        self.status = torch.zeros(max_requests, **i32)
        self.active_rows = torch.zeros(1, **i32)
        self.probe_thresholds = torch.full((max_rows,), float("inf"), dtype=torch.float32, device=device)
        self.probe_stats = torch.zeros((max_rows, _PROBE_STAT_COLUMNS), **i32)
        self.probe_mode = 0
        # Transfer rows are requests, so its existing row*C addressing is shared.
        self.transfer_requests = torch.arange(max_requests, **i32)
        self.transfer_active = torch.full((1,), max_requests, **i32)
        pages = capacity // block_size
        self.request_page_table = torch.arange(max_requests * pages, **i32).view(max_requests, pages)
        self.attention_page_table = torch.empty((max_rows, pages), **i32)
        self.safe_requests = torch.zeros(max_rows, **i32)

    def select_and_plan(
        self,
        *,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        indexer_block_table,
        req_ids,
        stable_prefix_lens,
        visible_seq_lens,
        token_to_req,
        block_table,
        update_page_table=True,
    ):
        if query.shape[0] != self.max_rows:
            raise ValueError("Query rows must match the captured fused state")
        indices = torch.ops._C_ascend.npu_quant_lightning_indexer_sparse_kv(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            indexer_block_table,
            req_ids,
            stable_prefix_lens,
            visible_seq_lens,
            token_to_req,
            block_table,
            self.active_rows,
            self.last_req_ids,
            self.slot_to_token,
            self.priority,
            self.free_mask,
            self.free_stack,
            self.free_count,
            self.clock,
            self.current_slots,
            self.miss_count,
            self.miss_tokens,
            self.miss_slots,
            self.token_to_slot,
            self.status,
            self.resident_end,
            self.probe_thresholds,
            self.probe_stats,
            self.capacity,
            self.max_token,
            self.block_size,
            self.host_num_blocks,
            self.max_queries,
            self.probe_mode,
        )
        # Padding has -1 indices; clamp its page-table owner to an allocated row.
        if update_page_table:
            torch.clamp(token_to_req, 0, self.max_requests - 1, out=self.safe_requests)
            torch.index_select(self.request_page_table, 0, self.safe_requests, out=self.attention_page_table)
        return indices

    def transfer(
        self, *, block_table, host_cache_bases, resident_k, resident_v, token_size_bytes_k, token_size_bytes_v
    ):
        expected_k = self.max_requests * self.capacity * token_size_bytes_k
        expected_v = self.max_requests * self.capacity * token_size_bytes_v
        if resident_k.nbytes != expected_k or resident_v.nbytes != expected_v:
            raise ValueError("Resident KV must have request-major [max_requests, capacity] storage")
        if not resident_k.is_contiguous() or not resident_v.is_contiguous():
            raise ValueError("Resident KV must be contiguous")
        torch.ops._C_ascend.npu_sparse_kv_transfer(
            self.miss_count,
            self.miss_tokens,
            self.miss_slots,
            self.transfer_requests,
            block_table,
            host_cache_bases,
            self.transfer_active,
            resident_k,
            resident_v,
            self.union_capacity,
            self.capacity,
            self.block_size,
            self.host_num_blocks,
            token_size_bytes_k,
            token_size_bytes_v,
        )

    def check_status(self):
        """Outside capture/timing only; callers must reject failed plans."""
        status = self.status.cpu().tolist()
        if any(status):
            raise RuntimeError(f"Request planner status: {status} (1=rows, 2=state, 4=capacity)")
