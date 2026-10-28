# SPDX-License-Identifier: Apache-2.0
"""Current fused-ABI acceptance; rebuild the minimal bridge before running."""

import subprocess
import sys
from pathlib import Path


def test_fused_topk_request_state_graph_and_host_kv(tmp_path):
    # Run in isolation: the minimal bridge cannot share registrations with vLLM.
    root = next(
        path for path in Path(__file__).resolve().parents if (path / "tools/check_fused_indexer_npu.py").is_file()
    )
    library = root / "build/fused-indexer-test-binding/fused_indexer_test_ops.so"
    assert library.is_file(), "Build tools/build_fused_indexer_test_binding.py first"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/check_fused_indexer_npu.py"),
            "--library",
            str(library),
            "--output",
            str(tmp_path / "result"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
