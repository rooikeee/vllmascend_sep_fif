# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def runner():
    spec = importlib.util.spec_from_file_location(
        "fused_runner", ROOT / "tools/run_quant_lightning_indexer_sparse_kv_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_runner_propagates_failure_and_only_targets_fused(monkeypatch):
    module = runner()
    calls = []
    monkeypatch.setattr(module.sys, "argv", ["runner", "--suite", "static"])
    monkeypatch.setattr(
        module.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), SimpleNamespace(returncode=1))[1]
    )
    assert module.main() == 1
    assert all("fused" in p or "quant_lightning_indexer" in p for p in module.STATIC_TESTS)
    assert "tests/ut/kv_offload/test_quant_lightning_indexer_request_state.py" in calls[0]


def test_device_runner_dispatches_fused_gate(monkeypatch, tmp_path):
    module = runner()
    calls = []
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "runner",
            "--suite",
            "device",
            "--device",
            "2",
            "--library",
            str(tmp_path / "ops.so"),
            "--output",
            str(tmp_path / "result"),
        ],
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), SimpleNamespace(returncode=0))[1]
    )
    assert module.main() == 0
    assert calls[0][1].endswith("check_fused_indexer_npu.py")
    assert calls[0][calls[0].index("--device") + 1] == "2"


def test_device_runner_requires_rebuilt_library_and_output(monkeypatch):
    module = runner()
    monkeypatch.setattr(module.sys, "argv", ["runner", "--suite", "device"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
