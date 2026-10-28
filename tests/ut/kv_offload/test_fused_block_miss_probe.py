# SPDX-License-Identifier: Apache-2.0
"""Execute the actual block probe with CPU lanes and sanitizer checks."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_block_probe_read_only_algorithm(tmp_path):
    compiler = shutil.which("g++")
    assert compiler, "A C++20 compiler is required"
    binary = tmp_path / "block-probe"
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-g",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I",
            str(ROOT / "csrc/attention/quant_lightning_indexer_sparse_kv/op_kernel"),
            str(Path(__file__).with_name("block_miss_probe_host_test.cpp")),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_probe_is_after_cube_release_and_before_topk():
    root = ROOT / "csrc/attention/quant_lightning_indexer_sparse_kv"
    vector = (
        root / "vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_service_vector_arch35.h"
    ).read_text()
    score = vector[vector.index("QLIVector<QLIT>::ProcessVec1(") : vector.index("QLIVector<QLIT>::ProcessTopK(")]
    assert score.index("asc_vf_call<RunBlockMissProbe>") > score.rindex("CrossCoreSetFlag<")
    assert score.index("asc_vf_call<RunBlockMissProbe>") > score.index("DataCopyPad(scoreGm[")
    probe = (root / "op_kernel/block_miss_probe.h").read_text()
    assert "asc_atomic" not in probe
    assert "const __gm__ volatile int32_t* reverse" in probe
    assert "c.priority" not in probe and "c.freeStack" not in probe
    entry = (root / "op_kernel/quant_lightning_indexer_sparse_kv.cpp").read_text()
    assert entry.index("qli.InitBlockMissProbe(") < entry.index("qli.Process();") < entry.index("qli.PlanRequests(")


def test_probe_benchmark_timing_excludes_state_restore(monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("probe_bench", ROOT / "tools/bench_fused_block_miss_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    order = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            order.append("event")

        def elapsed_time(self, end):
            return 0.25

    npu = SimpleNamespace(Event=Event, synchronize=lambda: order.append("sync"))
    state = SimpleNamespace(check_status=lambda: order.append("status"))
    monkeypatch.setattr(module, "restore", lambda *_: order.append("restore"))
    timings = module.measure(SimpleNamespace(npu=npu), lambda: order.append("fused"), state, {}, 2)
    assert timings == [250.0, 250.0]
    assert order == ["restore", "event", "fused", "event"] * 2 + ["sync", "status"]


def test_probe_benchmark_bad_arguments_fail_before_npu_import(tmp_path):
    result = subprocess.run(
        [
            shutil.which("python3"),
            str(ROOT / "tools/bench_fused_block_miss_probe.py"),
            "--library",
            "ops.so",
            "--output",
            str(tmp_path / "result"),
            "--queries",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0 and "queries" in result.stderr
    assert "torch_npu" not in result.stderr
    assert not (tmp_path / "result").exists()
