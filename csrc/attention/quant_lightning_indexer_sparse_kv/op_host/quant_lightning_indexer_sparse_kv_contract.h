// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace optiling::qli_sparse_kv_contract {

constexpr uint32_t TOPK = 2048U;
constexpr uint32_t QLI_UB_SMALL_T_BYTES = 150528U;
constexpr uint32_t QLI_UB_LARGE_T_BYTES = 186880U;
constexpr uint32_t MAX_QUERY_ROWS = 16U;
constexpr uint32_t MAX_CAPACITY = 1U << 24;

constexpr uint32_t DynamicUbBytes(uint32_t tokenCount) {
  return tokenCount > 64U ? QLI_UB_LARGE_T_BYTES : QLI_UB_SMALL_T_BYTES;
}

constexpr uint32_t ResolveStride(int64_t requested, uint32_t defaultStride) {
  return requested == -1 ? defaultStride : static_cast<uint32_t>(requested);
}

constexpr bool PlannerBlockTableShapeIsValid(int64_t rows, int64_t width, int64_t maxRequests) {
  return rows == maxRequests && width > 0;
}

constexpr uint64_t AdditionalWorkspaceBytes(uint64_t sequenceLength, uint32_t aicCount) {
  constexpr uint64_t sequenceTile = 128U;
  constexpr uint64_t scoreRows = 4U;
  constexpr uint64_t scoreElementBytes = sizeof(uint16_t);
  return scoreRows * ((sequenceLength + sequenceTile - 1U) / sequenceTile) * sequenceTile * scoreElementBytes *
         aicCount;
}

}  // namespace optiling::qli_sparse_kv_contract
