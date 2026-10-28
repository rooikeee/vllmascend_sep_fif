# SPDX-License-Identifier: Apache-2.0
"""Verify the request-major fused Python ABI without importing the vLLM plugin."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fused_dispatch_reuses_request_state_and_page_tables(monkeypatch):
    module = load_module(
        "fused_state", "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_fused_indexer.py"
    )
    state = module.SparseKVFusedIndexer(
        max_requests=2,
        max_rows=8,
        max_queries=4,
        topk=2048,
        capacity=8192,
        max_token=32768,
        block_size=128,
        host_num_blocks=512,
        device="cpu",
    )
    seen = []

    def fused(*args):
        seen.append(args)
        return torch.empty((8, 1, 2048), dtype=torch.int32)

    monkeypatch.setattr(module.torch.ops, "_C_ascend", SimpleNamespace(npu_quant_lightning_indexer_sparse_kv=fused))
    inputs = dict(
        query=torch.empty(8, 32, 128),
        key=torch.empty(512, 128, 1, 128),
        weights=torch.empty(8, 32),
        query_dequant_scale=torch.ones(8, 32),
        key_dequant_scale=torch.ones(512, 128),
        actual_seq_lengths_query=torch.tensor([4, 8], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([32768, 32768], dtype=torch.int32),
        indexer_block_table=torch.arange(512, dtype=torch.int32).view(2, 256),
        req_ids=torch.tensor([11, 22]),
        stable_prefix_lens=torch.full((8,), 32768, dtype=torch.int32),
        visible_seq_lens=torch.full((8,), 32768, dtype=torch.int32),
        token_to_req=torch.tensor([1, 0, 1, 0, 1, 0, -1, -1], dtype=torch.int32),
        block_table=torch.arange(512, dtype=torch.int32).view(2, 256),
    )
    pointers = {name: tensor.data_ptr() for name, tensor in vars(state).items() if isinstance(tensor, torch.Tensor)}
    for _ in range(2):
        assert state.select_and_plan(**inputs).shape == (8, 1, 2048)
    assert len(seen) == 2
    args = seen[0]
    assert len(args) == 36  # 30 tensors and six attributes; one fused dispatch
    assert args[15] is state.slot_to_token and args[16] is state.priority
    assert args[18] is state.free_stack and args[19] is state.free_count
    assert args[25] is state.token_to_slot and args[26] is state.status
    assert args[28] is state.probe_thresholds and args[29] is state.probe_stats
    assert args[30:] == (8192, 32768, 128, 512, 4, 0)
    assert state.slot_to_token.shape == (2, 8192)
    assert state.miss_tokens.shape == (2, 8192)
    assert torch.equal(state.attention_page_table[0], state.request_page_table[1])
    assert torch.equal(state.attention_page_table[-1], state.request_page_table[0])
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in vars(state).items() if isinstance(tensor, torch.Tensor)
    }
    state.probe_mode = 2
    state.probe_thresholds.fill_(0.5)
    state.attention_page_table.fill_(-7)
    state.select_and_plan(**inputs, update_page_table=False)
    assert seen[-1][-1] == 2
    assert (state.attention_page_table == -7).all()
    state.status[0] = 2
    with pytest.raises(RuntimeError, match="status"):
        state.check_status()


@pytest.mark.parametrize("changes", [{"topk": 1024}, {"capacity": 4096}, {"max_queries": 17}, {"block_size": 64}])
def test_fused_state_rejects_invalid_pool_contract(changes):
    module = load_module(
        "fused_state", "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_fused_indexer.py"
    )
    options = dict(
        max_requests=2,
        max_rows=8,
        max_queries=4,
        topk=2048,
        capacity=8192,
        max_token=32768,
        block_size=128,
        host_num_blocks=512,
        device="cpu",
    )
    options.update(changes)
    with pytest.raises(ValueError):
        module.SparseKVFusedIndexer(**options)
