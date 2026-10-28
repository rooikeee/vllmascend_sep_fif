# SPDX-License-Identifier: Apache-2.0
"""Experimental 950 request-shared planner; serving selection remains explicit.

Use one owner per layer. Call plan, then transfer, then attention on the same
stream. Do not reuse this state before the preceding transfer has completed.
"""

import torch


class SparseKVRequestPlan:
    def __init__(self, *, max_requests, max_rows, max_queries, topk, capacity,
                 max_token, block_size, host_num_blocks, device):
        if not (1 <= max_queries <= 16 and 1 <= topk <= 2048 and capacity >= max_queries * topk
                and max_requests > 0 and max_rows > 0 and max_token > 0 and block_size > 0
                and host_num_blocks > 0 and capacity % block_size == 0):
            raise ValueError("Invalid shared planner shape; capacity must cover max_queries*topk and whole pages")
        self.max_requests, self.max_rows = max_requests, max_rows
        self.max_queries, self.topk, self.capacity = max_queries, topk, capacity
        self.max_token, self.block_size, self.host_num_blocks = max_token, block_size, host_num_blocks
        self.union_capacity = max_queries * topk
        self.hash_capacity = max(32, 1 << (2 * self.union_capacity - 1).bit_length())
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
        self.workspace = torch.empty((max_requests, 2 * self.hash_capacity + capacity), **i32)
        self.status = torch.zeros(max_requests, **i32)
        self.active_rows = torch.zeros(1, **i32)
        # Transfer rows are requests, so its existing row*C addressing is shared.
        self.transfer_requests = torch.arange(max_requests, **i32)
        self.transfer_active = torch.full((1,), max_requests, **i32)
        pages = capacity // block_size
        self.request_page_table = torch.arange(max_requests * pages, **i32).view(max_requests, pages)
        self.attention_page_table = torch.empty((max_rows, pages), **i32)
        self.safe_requests = torch.zeros(max_rows, **i32)

    def plan(self, *, req_ids, topk_indices, stable_prefix_lens, visible_seq_lens,
             token_to_req, block_table):
        if topk_indices.shape != (self.max_rows, self.topk):
            raise ValueError("Pass fixed-capacity TopK rows; refresh active_rows on every replay")
        torch.ops._C_ascend.npu_sparse_kv_request_plan(
            req_ids, topk_indices, stable_prefix_lens, visible_seq_lens, token_to_req,
            block_table, self.active_rows, self.last_req_ids, self.slot_to_token, self.priority,
            self.free_mask, self.free_stack, self.free_count, self.clock, self.current_slots,
            self.miss_count, self.miss_tokens, self.miss_slots, self.workspace, self.status,
            self.max_queries, self.max_token, self.block_size, self.host_num_blocks)
        # Padding has -1 indices; clamp its page-table owner to an allocated row.
        torch.clamp(token_to_req, 0, self.max_requests - 1, out=self.safe_requests)
        torch.index_select(self.request_page_table, 0, self.safe_requests, out=self.attention_page_table)
        return self.current_slots

    def transfer(self, *, block_table, host_cache_bases, resident_k, resident_v,
                 token_size_bytes_k, token_size_bytes_v):
        expected_k = self.max_requests * self.capacity * token_size_bytes_k
        expected_v = self.max_requests * self.capacity * token_size_bytes_v
        if resident_k.nbytes != expected_k or resident_v.nbytes != expected_v:
            raise ValueError("Resident KV must have request-major [max_requests, capacity] storage")
        if not resident_k.is_contiguous() or not resident_v.is_contiguous():
            raise ValueError("Resident KV must be contiguous")
        torch.ops._C_ascend.npu_sparse_kv_transfer(
            self.miss_count, self.miss_tokens, self.miss_slots, self.transfer_requests,
            block_table, host_cache_bases, self.transfer_active, resident_k, resident_v,
            self.union_capacity, self.capacity, self.block_size, self.host_num_blocks,
            token_size_bytes_k, token_size_bytes_v)

    def check_status(self):
        """Outside capture/timing only; callers must reject failed plans."""
        status = self.status.cpu().tolist()
        if any(status):
            raise RuntimeError(f"Request planner status: {status} (1=rows, 2=state, 4=capacity)")
