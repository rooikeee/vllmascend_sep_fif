#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare off / threshold-only / read-only block lookup in the same fused QLI.

Only the fused kernel runs: no transfer, prefetch or separate planner. All state
restores occur outside device event timing. Optional profiles record per-mode kernel timelines.
"""

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)
# tools/bisect shadows the standard-library module required by statistics.
import statistics  # noqa: E402

TOPK = 2048
BLOCK = 128
MODES = ("off", "threshold", "lookup")
STATE_NAMES = (
    "last_req_ids",
    "slot_to_token",
    "priority",
    "free_mask",
    "free_stack",
    "free_count",
    "clock",
    "current_slots",
    "miss_count",
    "miss_tokens",
    "miss_slots",
    "token_to_slot",
    "status",
    "resident_end",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--queries", type=int, default=4, help="Actual query rows per request; MTP3 means 4")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--capacity", type=int, default=0, help="0 selects queries*2048")
    parser.add_argument("--thresholds", default="inf,1,0.5,-1", help="Comma-separated FP32 score thresholds")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trace", action="store_true", help="Capture separate per-mode NPU traces after timing")
    args = parser.parse_args()
    if not (1 <= args.batch <= 64 and 1 <= args.queries <= 16 and args.context >= TOPK + args.queries):
        parser.error("batch must be 1..64, queries 1..16, context >= 2048+queries")
    args.capacity = args.capacity or args.queries * TOPK
    if args.capacity < args.queries * TOPK or args.capacity % BLOCK or args.capacity > 1 << 24:
        parser.error("capacity must cover queries*2048 and be aligned to 128")
    if args.iterations < 1 or args.rounds < 1:
        parser.error("iterations and rounds must be positive")
    try:
        args.thresholds = [float(value) for value in args.thresholds.split(",")]
    except ValueError:
        parser.error("thresholds must be floating-point numbers")
    if any(math.isnan(value) for value in args.thresholds):
        parser.error("NaN is supported as no candidates but excluded from performance sweeps")
    return args


def load_state_class():
    spec = importlib.util.spec_from_file_location(
        "fused_state", ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_fused_indexer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SparseKVFusedIndexer


def make_fixture(torch, args):
    rows = args.batch * args.queries
    pages = (args.context + BLOCK - 1) // BLOCK
    physical = args.batch * pages
    device = torch.device("npu", args.device)
    state = load_state_class()(
        max_requests=args.batch,
        max_rows=rows,
        max_queries=args.queries,
        topk=TOPK,
        capacity=args.capacity,
        max_token=args.context,
        block_size=BLOCK,
        host_num_blocks=physical,
        device=device,
    )
    state.active_rows.fill_(rows)
    # Every row selects one complete 2048-token group with score 1, all others 0.
    # Groups end before the shortest causal length, including non-page-aligned tails.
    groups = min(args.queries, (args.context - args.queries + 1) // TOPK)
    keys = torch.zeros((pages * BLOCK, 128))
    for group in range(groups):
        keys[group * TOPK : (group + 1) * TOPK, group] = 1
    key = keys.view(pages, BLOCK, 1, 128).repeat(args.batch, 1, 1, 1).to(torch.float8_e4m3fn).to(device)
    query = torch.zeros((rows, 32, 128))
    for row in range(rows):
        query[row, 0, row % args.queries % groups] = 1
    weights = torch.zeros((rows, 32), dtype=torch.bfloat16, device=device)
    weights[:, 0] = 1
    inputs = dict(
        query=query.to(torch.float8_e4m3fn).to(device),
        key=key,
        weights=weights,
        query_dequant_scale=torch.ones((rows, 32), device=device),
        key_dequant_scale=torch.ones((physical, BLOCK), device=device),
        actual_seq_lengths_query=torch.arange(args.queries, rows + 1, args.queries, dtype=torch.int32, device=device),
        actual_seq_lengths_key=torch.full((args.batch,), args.context, dtype=torch.int32, device=device),
        indexer_block_table=torch.arange(physical, dtype=torch.int32, device=device).view(args.batch, pages),
        req_ids=torch.arange(10, 10 + args.batch, dtype=torch.int64, device=device),
        stable_prefix_lens=torch.full((rows,), args.context, dtype=torch.int32, device=device),
        visible_seq_lens=torch.arange(
            args.context - args.queries + 1, args.context + 1, dtype=torch.int32, device=device
        ).repeat(args.batch),
        token_to_req=torch.arange(args.batch, dtype=torch.int32, device=device).repeat_interleave(args.queries),
        block_table=torch.arange(physical, dtype=torch.int32, device=device).view(args.batch, pages),
        update_page_table=False,
    )
    return state, inputs, groups


def snapshot(state):
    return {name: getattr(state, name).clone() for name in STATE_NAMES}


def restore(state, saved):
    for name, value in saved.items():
        getattr(state, name).copy_(value)
    state.probe_stats.fill_(-99)


def check_result(torch, state, inputs, before, indices, groups, args, threshold, mode):
    state.check_status()
    output = indices.cpu().view(-1, TOPK)
    tokens = state.slot_to_token.cpu()
    reverse = state.token_to_slot.cpu()
    slots = state.current_slots.cpu()
    stats = state.probe_stats.cpu()
    owners = inputs["token_to_req"].cpu().tolist()
    prefixes = inputs["stable_prefix_lens"].cpu().tolist()
    visible = inputs["visible_seq_lens"].cpu().tolist()
    active = int(state.active_rows.cpu()[0])
    ids, old_ids = inputs["req_ids"].cpu(), before["last_req_ids"].cpu()
    actual_k = inputs["actual_seq_lengths_key"].cpu().tolist()
    old_reverse = before["token_to_slot"].cpu()
    selected = [set() for _ in range(args.batch)]
    stats_expected = []
    for row in range(output.shape[0]):
        group = row % args.queries % groups
        empty = actual_k[row // args.queries] == 0
        expected = (
            torch.full((TOPK,), -1, dtype=torch.int32)
            if empty
            else torch.arange(group * TOPK, (group + 1) * TOPK, dtype=torch.int32)
        )
        assert torch.equal(output[row].sort().values, expected), f"TopK changed at row {row}"
        request = owners[row]
        if row >= active or empty:
            assert (slots[row] == -1).all()
            stats_expected.append([0, 0])
            continue
        assert (slots[row] >= 0).all()
        assert torch.equal(tokens[request, slots[row].long()], output[row])
        selected[request].update(output[row].tolist())
        limit = min(args.context - args.queries + row % args.queries + 1, visible[row], args.context)
        score = torch.zeros(limit)
        score[group * TOPK : min(limit, (group + 1) * TOPK)] = 1
        candidate_ids = (score > threshold).nonzero().flatten()
        stable = min(prefixes[r] for r in range(active) if owners[r] == request)
        hits = (old_reverse[request, candidate_ids] >= 0) & (old_reverse[request, candidate_ids] < args.capacity)
        hits &= candidate_ids < max(stable, 0)
        if old_ids[request] != ids[request]:
            hits.zero_()
        misses = int((~hits).sum()) if mode == 2 else 0
        stats_expected.append([candidate_ids.numel(), misses])
    if mode:
        assert torch.equal(stats[:, 1:], torch.tensor(stats_expected, dtype=torch.int32)), (stats, stats_expected)
        for row in range(output.shape[0]):
            if row < active and actual_k[row // args.queries] > 0:
                assert stats[row, 0] > 0
            else:
                assert stats[row, 0] == 0
    else:
        assert (stats == -99).all(), "Disabled mode wrote probe statistics"
    counts = state.miss_count.cpu().tolist()
    for request in range(args.batch):
        req_rows = [r for r in range(active) if owners[r] == request]
        stable = min((prefixes[r] for r in req_rows), default=0)
        cached = {int(t) for t in before["slot_to_token"][request].cpu() if 0 <= t < stable}
        expected_misses = selected[request] - cached if old_ids[request] == ids[request] else selected[request]
        actual = state.miss_tokens[request, : counts[request]].cpu().tolist()
        assert len(actual) == len(set(actual)) and set(actual) == expected_misses
        occupied = (tokens[request] >= 0).nonzero().flatten()
        assert torch.equal(reverse[request, tokens[request, occupied].long()].long(), occupied)
        mapped = (reverse[request] >= 0).nonzero().flatten()
        assert torch.equal(tokens[request, reverse[request, mapped].long()].long(), mapped)
        assert (reverse[request] >= -1).all()
        free_count = int(state.free_count[request].cpu())
        stack = state.free_stack[request, :free_count].cpu().tolist()
        assert len(stack) == len(set(stack))
        assert set(stack) == set((tokens[request] == -1).nonzero().flatten().tolist())
    return stats.tolist()


def measure(torch, call, state, saved, iterations):
    # Per-replay state restores and counter initialization are excluded from events.
    events = []
    for _ in range(iterations):
        restore(state, saved)
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        events.append((start, end))
    torch.npu.synchronize()
    state.check_status()
    return [start.elapsed_time(end) * 1000 for start, end in events]


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    name = torch.npu.get_device_name(args.device)
    if "950" not in name:
        raise RuntimeError(f"This experiment requires 950 execution, got {name}; no 910 timing is valid")
    torch.ops.load_library(str(args.library.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    state, inputs, groups = make_fixture(torch, args)
    cold = snapshot(state)
    state.select_and_plan(**inputs)
    torch.npu.synchronize()
    state.check_status()
    warm = snapshot(state)
    records, correctness = [], []
    pointers = {key: value.data_ptr() for key, value in vars(state).items() if isinstance(value, torch.Tensor)}
    # Both modes use the exact same initial state. The extra probe never supplies
    # or replaces the final TopK plan; verify that plan independently every time.
    initial_ids = inputs["req_ids"].clone()
    for cache_name, saved in (
        ("cold", cold),
        ("warm", warm),
        ("rollback", warm),
        ("padding", warm),
        ("replacement", warm),
        ("empty", warm),
        ("mixed-empty", warm),
    ):
        inputs["req_ids"].copy_(initial_ids)
        if cache_name == "replacement":
            inputs["req_ids"].add_(100)
        inputs["actual_seq_lengths_key"].fill_(args.context)
        if cache_name == "empty":
            inputs["actual_seq_lengths_key"].zero_()
        elif cache_name == "mixed-empty":
            inputs["actual_seq_lengths_key"][0] = 0
        inputs["stable_prefix_lens"].fill_(TOPK // 2 if cache_name == "rollback" else args.context)
        state.active_rows.fill_(
            max(0, args.batch * args.queries - 1) if cache_name == "padding" else args.batch * args.queries
        )
        for threshold in args.thresholds:
            state.probe_thresholds.fill_(threshold)
            baseline = None
            for mode in range(len(MODES)):
                restore(state, saved)
                state.probe_mode = mode
                indices = state.select_and_plan(**inputs)
                torch.npu.synchronize()
                stats = check_result(torch, state, inputs, saved, indices, groups, args, threshold, mode)
                if baseline is None:
                    baseline = indices.cpu()
                else:
                    assert torch.equal(baseline, indices.cpu()), "Probe changed final TopK ordering"
                correctness.append(dict(cache=cache_name, threshold=str(threshold), mode=MODES[mode], stats=stats))
    inputs["stable_prefix_lens"].fill_(args.context)
    inputs["req_ids"].copy_(initial_ids)
    inputs["actual_seq_lengths_key"].fill_(args.context)
    state.active_rows.fill_(args.batch * args.queries)
    for cache_name, saved in (("cold", cold), ("warm", warm)):
        for threshold in args.thresholds:
            state.probe_thresholds.fill_(threshold)
            graphs, graph_indices = [], []
            try:
                for mode in range(len(MODES)):
                    state.probe_mode = mode
                    restore(state, saved)
                    state.select_and_plan(**inputs)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    restore(state, saved)
                    with torch.npu.graph(graph):
                        indices = state.select_and_plan(**inputs)
                    graphs.append(graph)
                    graph_indices.append(indices)
                    restore(state, saved)
                    graph.replay()
                    torch.npu.synchronize()
                    check_result(torch, state, inputs, saved, indices, groups, args, threshold, mode)
                    if mode == 2:
                        # Threshold storage is a tensor input, so replay must read
                        # new contents without recapturing or changing addresses.
                        state.probe_thresholds.fill_(0.25)
                        restore(state, saved)
                        graph.replay()
                        torch.npu.synchronize()
                        check_result(torch, state, inputs, saved, indices, groups, args, 0.25, mode)
                        state.probe_thresholds.fill_(threshold)
                timings = {mode: [] for mode in range(len(MODES))}
                for round_id in range(args.rounds):
                    # Rotate order to avoid systematically rewarding the first mode.
                    for index in range(len(MODES)):
                        mode = (round_id + index) % len(MODES)
                        timings[mode].extend(measure(torch, graphs[mode].replay, state, saved, args.iterations))
                        check_result(torch, state, inputs, saved, graph_indices[mode], groups, args, threshold, mode)
                for mode in range(len(MODES)):
                    record = dict(
                        cache=cache_name,
                        threshold=str(threshold),
                        mode=MODES[mode],
                        median_us=statistics.median(timings[mode]),
                        samples_us=timings[mode],
                    )
                    record["overhead_vs_off_pct"] = (record["median_us"] / statistics.median(timings[0]) - 1) * 100
                    records.append(record)
                    print(json.dumps({key: value for key, value in record.items() if key != "samples_us"}), flush=True)
                if args.trace:
                    for mode in range(len(MODES)):
                        restore(state, saved)
                        torch.npu.synchronize()
                        with torch_npu.profiler.profile(
                            activities=[
                                torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU,
                            ]
                        ) as prof:
                            graphs[mode].replay()
                            torch.npu.synchronize()
                        prof.export_chrome_trace(str(args.output / f"{cache_name}-{threshold}-{MODES[mode]}.json"))
            finally:
                torch.npu.synchronize()
                for graph in graphs:
                    graph.reset()
    assert pointers == {key: value.data_ptr() for key, value in vars(state).items() if isinstance(value, torch.Tensor)}
    report = dict(
        passed=True,
        device=name,
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        batch=args.batch,
        queries=args.queries,
        context=args.context,
        capacity=args.capacity,
        timing="NPUGraph single fused kernel; restores excluded; no transfer or page-table update",
        qualification="Kernel latency is not isolated AIV time; use 950 instruction profiling for AIV/Cube overlap.",
        correctness=correctness,
        results=records,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
