#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""950 fused QLI correctness, real KV transfer and Graph replay acceptance.

No standalone indexer or planner is invoked. Input keys encode exact 2048-token
selection groups, so the expected TopK set is known without another device op.
"""

import argparse
import ctypes
import importlib.util
import json
import mmap
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)
TOPK = 2048
CONTEXT = 32768
REQUESTS = 2
ROWS = 8
MAX_QUERIES = 4
CAPACITY = MAX_QUERIES * TOPK
BLOCK_SIZE = 128
KV_WIDTH = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50, help="Timed full-hit fused calls after correctness")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    name = torch.npu.get_device_name(args.device)
    if "950" not in name:
        raise RuntimeError(f"Fused QLI execution requires Ascend 950; this machine reports {name}")
    torch.ops.load_library(str(args.library.resolve()))
    spec = importlib.util.spec_from_file_location(
        "fused_indexer", ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_fused_indexer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("npu", args.device)
    state = module.SparseKVFusedIndexer(
        max_requests=REQUESTS,
        max_rows=ROWS,
        max_queries=MAX_QUERIES,
        topk=TOPK,
        capacity=CAPACITY,
        max_token=CONTEXT,
        block_size=BLOCK_SIZE,
        host_num_blocks=REQUESTS * CONTEXT // BLOCK_SIZE,
        device=device,
    )
    req = torch.tensor([101, 202], dtype=torch.int64, device=device)
    owners = torch.tensor([0] * MAX_QUERIES + [1] * MAX_QUERIES, dtype=torch.int32, device=device)
    stable = torch.full((ROWS,), CONTEXT, dtype=torch.int32, device=device)
    visible = stable.clone()
    table = torch.arange(REQUESTS * CONTEXT // BLOCK_SIZE, dtype=torch.int32, device=device).view(REQUESTS, -1)
    indexer_table = table.clone()
    query = torch.zeros((ROWS, 32, 128)).to(torch.float8_e4m3fn).to(device)
    key_cpu = torch.zeros((REQUESTS * CONTEXT, 128), dtype=torch.float32)
    token_ids = torch.arange(REQUESTS * CONTEXT)
    key_cpu[token_ids, (token_ids % CONTEXT) // TOPK] = 1
    key = key_cpu.view(-1, BLOCK_SIZE, 1, 128).to(torch.float8_e4m3fn).to(device)
    weights = torch.zeros((ROWS, 32), dtype=torch.bfloat16, device=device)
    weights[:, 0] = 1
    query_scale = torch.ones((ROWS, 32), device=device)
    key_scale = torch.ones((key.shape[0], BLOCK_SIZE), device=device)
    actual_q = torch.tensor([4, 8], dtype=torch.int32, device=device)
    actual_k = torch.full((REQUESTS,), CONTEXT, dtype=torch.int32, device=device)
    inputs = dict(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_q,
        actual_seq_lengths_key=actual_k,
        indexer_block_table=indexer_table,
        req_ids=req,
        stable_prefix_lens=stable,
        visible_seq_lens=visible,
        token_to_req=owners,
        block_table=table,
    )

    def set_groups(groups):
        q = torch.zeros((ROWS, 32, 128), dtype=torch.float32)
        for row, group in enumerate(groups):
            q[row, 0, group] = 1
        query.copy_(q.to(torch.float8_e4m3fn))

    # Reject malformed metadata and aliased mutable buffers before any launch.
    def expect_validation_error(call):
        try:
            call()
        except RuntimeError:
            return
        raise AssertionError("Malformed fused inputs were accepted")

    invalid = dict(inputs, visible_seq_lens=visible[:-1])
    expect_validation_error(lambda: state.select_and_plan(**invalid))
    invalid = dict(inputs, req_ids=req.to(torch.int32))
    expect_validation_error(lambda: state.select_and_plan(**invalid))
    original_mask = state.free_mask
    state.free_mask = state.slot_to_token
    expect_validation_error(lambda: state.select_and_plan(**inputs))
    state.free_mask = original_mask
    state.max_queries = MAX_QUERIES + 1
    expect_validation_error(lambda: state.select_and_plan(**inputs))
    state.max_queries = MAX_QUERIES

    runtime = ctypes.CDLL("libascendcl.so")
    register = runtime.aclrtHostRegister
    register.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    register.restype = ctypes.c_int
    unregister = runtime.aclrtHostUnregister
    unregister.argtypes = [ctypes.c_void_p]
    unregister.restype = ctypes.c_int
    allocations = [mmap.mmap(-1, REQUESTS * CONTEXT * KV_WIDTH * 2) for _ in range(2)]
    host = [torch.frombuffer(a, dtype=torch.bfloat16).view(REQUESTS * CONTEXT, KV_WIDTH) for a in allocations]
    for kind, tensor in enumerate(host):
        tensor.copy_(torch.arange(tensor.numel()).remainder(997).view_as(tensor) + kind)
    registered = []
    graph = None
    records = []
    try:
        aliases = []
        for tensor in host:
            alias = ctypes.c_void_p()
            code = register(tensor.data_ptr(), tensor.nbytes, 0, ctypes.byref(alias))
            if code:
                raise RuntimeError(f"aclrtHostRegister failed: {code}")
            registered.append(tensor.data_ptr())
            aliases.append(alias.value)
        bases = torch.tensor(aliases, dtype=torch.int64, device=device)
        resident = [torch.empty((REQUESTS, CAPACITY, KV_WIDTH), dtype=torch.bfloat16, device=device) for _ in range(2)]

        def run():
            indices = state.select_and_plan(**inputs)
            state.transfer(
                block_table=table,
                host_cache_bases=bases,
                resident_k=resident[0],
                resident_v=resident[1],
                token_size_bytes_k=KV_WIDTH * 2,
                token_size_bytes_v=KV_WIDTH * 2,
            )
            return indices

        state.active_rows.fill_(ROWS)
        set_groups([0] * ROWS)
        run()
        torch.npu.synchronize()
        state.check_status()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            indices = run()
        req.add_(10)
        pointers = {key: value.data_ptr() for key, value in vars(state).items() if isinstance(value, torch.Tensor)}
        for step in range(12):
            old_tokens, old_ids = state.slot_to_token.cpu(), state.last_req_ids.cpu()
            groups = [0] * ROWS if step < 2 else [((step - 2) * 4 + row % 4) % 12 for row in range(ROWS)]
            if step == 6:
                stable.fill_(1024)
                for tensor in host:
                    for request in range(REQUESTS):
                        tensor[request * CONTEXT + 1024 : (request + 1) * CONTEXT].add_(2)
            else:
                stable.fill_(CONTEXT)
            if step == 7:
                req[0].add_(1)
                owners.copy_(torch.tensor([1, 0] * 4, dtype=torch.int32, device=device))
            active = 3 if step == 8 else 0 if step == 9 else ROWS
            state.active_rows.fill_(active)
            # Exercise mixed/all empty QLI work: every AIV still enters SyncAll.
            actual_k.copy_(
                torch.tensor(
                    [0, CONTEXT] if step == 10 else [0, 0] if step == 11 else [CONTEXT, CONTEXT],
                    dtype=torch.int32,
                    device=device,
                )
            )
            if step == 5:
                table[0, 0] = -1  # planner filtering must not alter indexer scores
            else:
                table.copy_(indexer_table)
            set_groups(groups)
            graph.replay()
            torch.npu.synchronize()
            state.check_status()
            output = indices.cpu().view(ROWS, TOPK)
            row_owners, request_ids, prefixes = owners.cpu(), req.cpu(), stable.cpu()
            slots, tokens, counts = state.current_slots.cpu(), state.slot_to_token.cpu(), state.miss_count.cpu()
            reverse, free, free_count = state.token_to_slot.cpu(), state.free_mask.cpu(), state.free_count.cpu()
            page_table = table.cpu()
            assert (reverse >= -1).all(), "Pending CAS claim survived fused completion"
            selected_rows = []
            for row in range(ROWS):
                empty = (step == 10 and row < MAX_QUERIES) or step == 11
                expected = (
                    torch.full((TOPK,), -1, dtype=torch.int32)
                    if empty
                    else torch.arange(groups[row] * TOPK, (groups[row] + 1) * TOPK, dtype=torch.int32)
                )
                torch.testing.assert_close(output[row].sort().values, expected, rtol=0, atol=0)
                request = int(row_owners[row])
                selected = [
                    int(t)
                    for t in output[row]
                    if t >= 0 and 0 <= int(page_table[request, int(t) // BLOCK_SIZE]) < key.shape[0]
                ]
                selected_rows.append(selected)
            for request in range(REQUESTS):
                rows = [r for r in range(active) if int(row_owners[r]) == request]
                selected = {t for row in rows for t in selected_rows[row]}
                prefix = min((int(prefixes[row]) for row in rows), default=CONTEXT)
                cached = {int(t) for t in old_tokens[request] if 0 <= t < prefix}
                expected_misses = selected - cached if old_ids[request] == request_ids[request] else selected
                actual = state.miss_tokens[request, : int(counts[request])].cpu().tolist()
                assert len(actual) == len(set(actual)) and set(actual) == expected_misses, (step, request)
                occupied = (tokens[request] >= 0).nonzero().flatten()
                mapped = (reverse[request] >= 0).nonzero().flatten()
                assert torch.equal(reverse[request, tokens[request, occupied].long()].long(), occupied)
                assert torch.equal(tokens[request, reverse[request, mapped].long()].long(), mapped)
                assert torch.equal(free[request].bool(), tokens[request] == -1)
                stack = state.free_stack[request, : int(free_count[request])].cpu().tolist()
                assert len(stack) == len(set(stack)) == int(free[request].sum())
                assert set(stack) == set((tokens[request] == -1).nonzero().flatten().tolist())
            resident_cpu = [tensor.cpu() for tensor in resident]
            for row in range(active):
                request = int(row_owners[row])
                valid = torch.tensor([int(t) in selected_rows[row] for t in output[row]])
                assert (slots[row, ~valid] == -1).all()
                ids, selected_slots = output[row, valid].long(), slots[row, valid].long()
                assert (selected_slots >= 0).all()
                assert torch.equal(tokens[request, selected_slots].long(), ids)
                physical = page_table[request, ids // BLOCK_SIZE].long() * BLOCK_SIZE + ids % BLOCK_SIZE
                for kind in range(2):
                    torch.testing.assert_close(
                        resident_cpu[kind][request, selected_slots], host[kind][physical], rtol=0, atol=0
                    )
            assert (slots[active:] == -1).all()
            assert pointers == {
                key: value.data_ptr() for key, value in vars(state).items() if isinstance(value, torch.Tensor)
            }
            record = {"step": step, "active_rows": active, "misses": counts.tolist(), "exact_topk_and_kv": True}
            records.append(record)
            print(json.dumps(record), flush=True)

        # Dynamic error handling is checked on the fused op, without consuming a failed plan.
        state.active_rows.fill_(ROWS + 1)
        state.select_and_plan(**inputs)
        torch.npu.synchronize()
        assert (state.status.cpu() != 0).all() and (state.current_slots.cpu() == -1).all()
        assert (state.miss_count.cpu() == 0).all()
        state.active_rows.fill_(ROWS)
        actual_k.fill_(CONTEXT)
        owners.copy_(torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32, device=device))
        owners[0] = -1
        state.select_and_plan(**inputs)
        torch.npu.synchronize()
        assert int(state.status[0].cpu()) != 0 and (state.current_slots[0].cpu() == -1).all()
        owners.copy_(torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32, device=device))
        set_groups([0] * ROWS)
        stable.fill_(CONTEXT)
        for _ in range(3):
            state.select_and_plan(**inputs)
        torch.npu.synchronize()
        state.check_status()
        assert not state.miss_count.cpu().any()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            state.select_and_plan(**inputs)
        end.record()
        torch.npu.synchronize()
        state.check_status()
        result = {
            "device": name,
            "passed": True,
            "steps": records,
            "fused_with_page_table_us": start.elapsed_time(end) * 1000 / args.iterations,
            "timing_case": "B2 Q4 K2048 C8192 L32768 full-hit; includes page-table update",
            "iterations": args.iterations,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
    finally:
        if graph is not None:
            graph.reset()
        torch.npu.synchronize()
        for address in registered:
            code = unregister(address)
            if code:
                raise RuntimeError(f"aclrtHostUnregister failed: {code}")
        registered.clear()
        host.clear()
        tensor = None
        for allocation in allocations:
            allocation.close()


if __name__ == "__main__":
    main()
