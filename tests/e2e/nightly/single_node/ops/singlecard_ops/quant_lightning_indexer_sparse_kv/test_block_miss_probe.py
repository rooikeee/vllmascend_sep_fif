# SPDX-License-Identifier: Apache-2.0
"""950-only block-probe checks for short tails and both QLI S1 tilings."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("batch,queries,context", [(1, 1, 2177), (2, 4, 32768), (17, 4, 32768)])
def test_block_probe_preserves_final_topk_and_request_plan(tmp_path, batch, queries, context):
    root = next(
        path for path in Path(__file__).resolve().parents if (path / "tools/bench_fused_block_miss_probe.py").is_file()
    )
    library = root / "build/fused-indexer-test-binding/fused_indexer_test_ops.so"
    assert library.is_file(), "Build the current fused Torch bridge before 950 execution"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/bench_fused_block_miss_probe.py"),
            "--library",
            str(library),
            "--output",
            str(tmp_path / "probe"),
            "--batch",
            str(batch),
            "--queries",
            str(queries),
            "--context",
            str(context),
            "--iterations",
            "2",
            "--rounds",
            "1",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
