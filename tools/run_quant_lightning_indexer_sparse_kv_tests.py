#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run only the current fused QLI operator's CPU checks or 950 acceptance."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC_TESTS = (
    "tests/ut/kv_offload/test_fused_block_miss_probe.py",
    "tests/ut/kv_offload/test_quant_lightning_indexer_request_state.py",
    "tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_tiling.py",
    "tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_inline_source.py",
    "tests/ut/kv_offload/test_fused_indexer_entry.py",
    "tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_runner.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("static", "device"), default="static")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.suite == "static":
        command = [sys.executable, "-m", "pytest", "--confcutdir=tests/ut/kv_offload", *STATIC_TESTS, "-q"]
    else:
        if args.library is None or args.output is None:
            parser.error("device acceptance requires --library and --output")
        command = [
            sys.executable,
            str(ROOT / "tools/check_fused_indexer_npu.py"),
            "--device",
            str(args.device),
            "--library",
            str(args.library.resolve()),
            "--output",
            str(args.output.resolve()),
            "--iterations",
            str(args.iterations),
        ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
