// SPDX-License-Identifier: Apache-2.0
#ifndef SPARSE_KV_REQUEST_PLAN_TORCH_ADPT_H
#define SPARSE_KV_REQUEST_PLAN_TORCH_ADPT_H
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "../../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
inline void npu_sparse_kv_request_plan(
    const at::Tensor& reqIds, const at::Tensor& topk, const at::Tensor& stable, const at::Tensor& visible,
    const at::Tensor& rowToReq, const at::Tensor& table, const at::Tensor& active, const at::Tensor& lastIds,
    const at::Tensor& tokens, const at::Tensor& priority, const at::Tensor& freeMask, const at::Tensor& freeStack,
    const at::Tensor& freeCount, const at::Tensor& clock, const at::Tensor& slots, const at::Tensor& missCount,
    const at::Tensor& missTokens, const at::Tensor& missSlots, const at::Tensor& scratch, const at::Tensor& status,
    int64_t maxQueries, int64_t maxToken, int64_t blockSize, int64_t hostBlocks) {
  const at::Tensor inputs[] = {reqIds, topk, stable, visible, rowToReq, table, active, lastIds, tokens,
      priority, freeMask, freeStack, freeCount, clock, slots, missCount, missTokens, missSlots, scratch, status};
  TORCH_CHECK(reqIds.device().is_privateuseone(), "SparseKvRequestPlan requires an Ascend 950 NPU");
  for (int i = 0; i < 20; ++i) {
    const auto type = (i == 0 || i == 7 || i == 9 || i == 13) ? at::kLong : at::kInt;
    TORCH_CHECK(inputs[i].device() == reqIds.device() && inputs[i].scalar_type() == type &&
                inputs[i].is_contiguous(), "SparseKvRequestPlan tensor device/dtype/contiguity mismatch at ", i);
    if (i >= 7) {
      const auto first = reinterpret_cast<uintptr_t>(inputs[i].data_ptr());
      for (int j = 0; j < i; ++j) {
        const auto other = reinterpret_cast<uintptr_t>(inputs[j].data_ptr());
        TORCH_CHECK(first + inputs[i].nbytes() <= other || other + inputs[j].nbytes() <= first,
                    "Mutable planner storage must not overlap other inputs");
      }
    }
  }
  const c10_npu::OptionalNPUGuard guard(reqIds.device());
  EXEC_NPU_CMD(aclnnSparseKvRequestPlan, reqIds, topk, stable, visible, rowToReq, table, active,
      lastIds, tokens, priority, freeMask, freeStack, freeCount, clock, slots, missCount, missTokens,
      missSlots, scratch, status, maxQueries, maxToken, blockSize, hostBlocks);
}
}
#endif
