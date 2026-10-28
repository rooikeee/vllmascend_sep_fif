# SPDX-License-Identifier: Apache-2.0
"""Run the production SIMT algorithm with CPU atomics and a block barrier."""

import shutil
import subprocess
from pathlib import Path


def test_fused_request_state_algorithm(tmp_path):
    root = Path(__file__).resolve().parents[3]
    compiler = shutil.which("g++")
    assert compiler, "A C++20 compiler is required"
    output = tmp_path / "request-plan-test"
    compiled = subprocess.run(
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
            str(root / "csrc/attention/quant_lightning_indexer_sparse_kv/op_kernel"),
            str(Path(__file__).with_name("request_plan_host_test.cpp")),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(output)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
