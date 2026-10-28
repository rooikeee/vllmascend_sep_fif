#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the isolated fused-indexer test bridge against the active Torch ABI."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)


def main():
    import torch_npu
    from torch.utils.cpp_extension import load

    npu = Path(torch_npu.__file__).parent
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    build = ROOT / "build/fused-indexer-test-binding"
    build.mkdir(parents=True, exist_ok=True)
    load(
        name="fused_indexer_test_ops",
        sources=[
            str(ROOT / "tools/fused_indexer_test_binding.cpp"),
            str(ROOT / "csrc/aclnn_torch_adapter/NPUBridge.cpp"),
            str(ROOT / "csrc/aclnn_torch_adapter/NPUStorageImpl.cpp"),
        ],
        extra_include_paths=[
            str(ROOT / "csrc"),
            str(npu / "include"),
            str(npu / "include/third_party/acl/inc"),
            str(npu / "include/torch_npu/csrc/aten"),
            str(cann / "include"),
        ],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[f"-L{npu}/lib", "-ltorch_npu", f"-L{cann}/lib64", "-lascendcl"],
        build_directory=str(build),
        is_python_module=False,
        verbose=True,
    )
    print(build / "fused_indexer_test_ops.so")


if __name__ == "__main__":
    main()
