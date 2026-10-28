from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "csrc/attention/quant_lightning_indexer_sparse_kv/op_kernel/quant_lightning_indexer_sparse_kv.cpp"
VECTOR = (
    ROOT
    / "csrc/attention/quant_lightning_indexer_sparse_kv/vendor/quant_lightning_indexer/op_kernel/arch35"
    / "quant_lightning_indexer_service_vector_arch35.h"
)
ARCH = VECTOR.with_name("quant_lightning_indexer_kernel_arch35.h")


def test_every_aiv_finishes_topk_before_request_planning():
    entry, vector = ENTRY.read_text(), VECTOR.read_text()
    assert entry.index("qli.Process();") < entry.index("qli.PlanRequests(context, shape);")
    plan = vector[vector.index("QLIVector<QLIT>::PlanRequests") : vector.index("QLIVector<QLIT>::InitParams")]
    assert plan.index("SyncAll();") < plan.index("asc_vf_call<RunRequestPlan>")
    assert plan.index("SyncAll();") < plan.index("return;")  # idle cores must participate too
    assert "GetBlockNum() * 2" in plan
    assert "ProcessSparseKvPlanRow" not in vector
    assert "SparseKvPlanRuntimeRowVf" not in vector


def test_fused_abi_has_request_state_and_no_row_hash_workspace():
    header = (ENTRY.parent / "quant_lightning_indexer_sparse_kv_tiling_data.h").read_text()
    assert "maxQueries" in header
    for name in ("hashCapacity", "workspaceRowElements", "planLocalMemoryBytes"):
        assert name not in header
    adapter = (ENTRY.parent.parent / "quant_lightning_indexer_sparse_kv_torch_adpt.h").read_text()
    assert adapter.count("EXEC_NPU_CMD(") == 1
    assert "aclnnQuantLightningIndexerSparseKv" in adapter
    for name in ("tokenToSlot", "freeStack", "freeCount", "priority", "residentEnd", "status"):
        assert name in adapter


def test_empty_rows_keep_single_writer_before_shared_phase():
    arch = ARCH.read_text()
    assert "if ((tmpBlockIdx & 1U) != 0U)" in arch
    assert "row = tmpBlockIdx; row < totalRows; row += aivCoreNum" in arch
    assert "vectorService.PlanRequests(context, shape);" in arch
