from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "tools/run_quant_lightning_indexer_sparse_kv_tests.py"


def test_old_three_route_benchmark_is_not_an_acceptance_entry(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--suite", "benchmark", "--dry-run", "--output", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "torch_npu" not in result.stderr
    assert not (tmp_path / "results.csv").exists()


def test_help_only_offers_current_fused_suites() -> None:
    result = subprocess.run([sys.executable, str(RUNNER), "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "{static,device}" in result.stdout
    assert "benchmark-static" not in result.stdout


@pytest.mark.parametrize("options", [("--batches", "33"), ("--rounds", "0"), ("--mtps", "0")])
def test_benchmark_rejects_bad_matrix_before_loading_npu(tmp_path: Path, options: tuple[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--suite", "benchmark", "--dry-run", "--output", str(tmp_path), *options],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "torch_npu" not in result.stderr
    assert not (tmp_path / "results.csv").exists()


@pytest.mark.parametrize("failure_stage", ["preflight", "measure"])
def test_benchmark_records_failure_without_a_successful_timing_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str
) -> None:
    package = "tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv"
    benchmark = importlib.import_module(f"{package}.benchmark")
    preflight_module = ModuleType("tools.run_quant_lightning_indexer_sparse_kv_tests")
    device_module = ModuleType(f"{package}.benchmark_device")

    def fail(*_args):
        raise RuntimeError("injected target failure")

    preflight_module.preflight = fail if failure_stage == "preflight" else lambda *_: {}
    device_module.measure_case = fail
    monkeypatch.setitem(sys.modules, preflight_module.__name__, preflight_module)
    monkeypatch.setitem(sys.modules, device_module.__name__, device_module)
    args = argparse.Namespace(
        output=tmp_path,
        trace=False,
        warmup=1,
        iterations=1,
        rounds=2,
        seed=2026,
        batches="1",
        seq_lens="140000",
        mtps="1",
        capacities="2048",
        cache_states="warm",
        execution="eager",
        dry_run=False,
        device=0,
        cann_root=tmp_path,
    )

    result = benchmark.run_benchmark(args)

    assert result == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed" and status["completed_cases"] == 0
    if failure_stage == "measure":
        rows = json.loads((tmp_path / "results.json").read_text())["results"]
        assert len(rows) == 1 and rows[0]["status"] == "failed"
        assert "event_speedup" not in rows[0] and "fused_ms" not in rows[0]
    else:
        assert not (tmp_path / "results.json").exists()


def test_retired_benchmark_command_preserves_existing_results(tmp_path: Path) -> None:
    previous = '{"results": []}\n'
    (tmp_path / "results.json").write_text(previous)

    result = subprocess.run(
        [sys.executable, str(RUNNER), "--suite", "benchmark", "--dry-run", "--output", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert (tmp_path / "results.json").read_text() == previous
