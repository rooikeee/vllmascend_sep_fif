#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""950-only planner + request-major transfer capture/replay correctness gate."""

import argparse
import ctypes
import importlib.util
import json
import mmap
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, required=True, help="Rebuilt Torch extension containing both operators")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    name = torch.npu.get_device_name(args.device)
    if "950" not in name:
        raise RuntimeError(f"This test requires 950PR/950, got {name}")
    torch.ops.load_library(str(args.library.resolve()))
    spec = importlib.util.spec_from_file_location("request_plan",
        ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_request_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("npu", args.device)
    plan = module.SparseKVRequestPlan(max_requests=2, max_rows=8, max_queries=4, topk=2048,
        capacity=12288, max_token=32768, block_size=128, host_num_blocks=512, device=device)
    req = torch.tensor([101, 202], dtype=torch.int64, device=device)
    owners = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32, device=device)
    stable = torch.full((8,), 32768, dtype=torch.int32, device=device)
    visible = stable.clone()
    table = torch.arange(512, dtype=torch.int32, device=device).view(2, 256)
    topk = torch.zeros((8, 2048), dtype=torch.int32, device=device)
    # Real host memory, registered for device access; retained until graph destruction.
    runtime = ctypes.CDLL("libascendcl.so")
    register = runtime.aclrtHostRegister
    register.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    register.restype = ctypes.c_int
    unregister = runtime.aclrtHostUnregister
    unregister.argtypes = [ctypes.c_void_p]
    unregister.restype = ctypes.c_int
    allocations = [mmap.mmap(-1, 65536 * 16 * 2) for _ in range(2)]
    host = [torch.frombuffer(allocation, dtype=torch.bfloat16).view(65536, 16) for allocation in allocations]
    for tensor in host:
        tensor.copy_(torch.arange(65536 * 16, dtype=torch.int32).remainder(997).view(65536, 16))
    host[1].add_(1)
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
            registered.append(tensor)
            aliases.append(alias.value)
        bases = torch.tensor(aliases, dtype=torch.int64, device=device)
        resident = [torch.empty((2, 12288, 16), dtype=torch.bfloat16, device=device) for _ in range(2)]

        def run():
            plan.plan(req_ids=req, topk_indices=topk, stable_prefix_lens=stable,
                visible_seq_lens=visible, token_to_req=owners, block_table=table)
            plan.transfer(block_table=table, host_cache_bases=bases, resident_k=resident[0],
                resident_v=resident[1], token_size_bytes_k=32, token_size_bytes_v=32)

        plan.active_rows.fill_(8)
        topk.copy_(torch.arange(2048, dtype=torch.int32, device=device).repeat(8, 1))
        run()
        torch.npu.synchronize()
        plan.check_status()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        # Identity change forces first-fill on first replay, without reallocating state.
        req.add_(10)
        pointers = {key: value.data_ptr() for key, value in vars(plan).items() if isinstance(value, torch.Tensor)}
        for step in range(10):
            old_tokens = plan.slot_to_token.cpu()
            old_ids = plan.last_req_ids.cpu()
            inputs = torch.stack([torch.arange(2048, dtype=torch.int32) +
                ((row % 4) * 512 if step not in (0, 1) else 0) + (max(step - 2, 0) * 4096 % 24000)
                for row in range(8)])
            if step == 6:
                stable.fill_(1024)
            else:
                stable.fill_(32768)
            if step == 7:
                req[0].add_(1)
                owners.copy_(torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int32, device=device))
            plan.active_rows.fill_(3 if step == 8 else 0 if step == 9 else 8)
            topk.copy_(inputs)
            graph.replay()
            torch.npu.synchronize()
            plan.check_status()
            active = plan.active_rows.item()
            row_owners, request_ids, prefixes = owners.cpu(), req.cpu(), stable.cpu()
            slots, tokens = plan.current_slots.cpu(), plan.slot_to_token.cpu()
            counts = plan.miss_count.cpu()
            for request in range(2):
                rows = [row for row in range(active) if row_owners[row] == request]
                selected = {int(token) for row in rows for token in inputs[row]}
                prefix = min((int(prefixes[row]) for row in rows), default=32768)
                cached = set(int(x) for x in old_tokens[request] if 0 <= x < prefix)
                expected = selected - cached if old_ids[request] == request_ids[request] else selected
                assert int(counts[request]) == len(expected), (step, request, counts, len(expected))
                actual = plan.miss_tokens[request, :int(counts[request])].cpu().tolist()
                assert len(actual) == len(set(actual)) and set(actual) == expected
            for row in range(active):
                request = int(row_owners[row])
                assert torch.equal(tokens[request, slots[row].long()], inputs[row])
                physical = inputs[row].long() + request * 32768
                for kind in range(2):
                    torch.testing.assert_close(resident[kind][request].cpu()[slots[row].long()], host[kind][physical],
                                               rtol=0, atol=0)
            assert (slots[active:] == -1).all()
            assert pointers == {key: value.data_ptr() for key, value in vars(plan).items() if isinstance(value, torch.Tensor)}
            record = {"step": step, "active_rows": active, "misses": counts.tolist(), "replay": True,
                      "host_kv_exact": True}
            records.append(record)
            print(json.dumps(record), flush=True)
        (args.output / "result.json").write_text(json.dumps({"device": name, "passed": True, "steps": records}, indent=2))
    finally:
        if graph is not None:
            graph.reset()
        torch.npu.synchronize()
        for tensor in registered:
            code = unregister(tensor.data_ptr())
            if code:
                raise RuntimeError(f"aclrtHostUnregister failed: {code}")
        registered.clear()
        host.clear()
        tensor = None
        for allocation in allocations:
            allocation.close()


if __name__ == "__main__":
    main()
