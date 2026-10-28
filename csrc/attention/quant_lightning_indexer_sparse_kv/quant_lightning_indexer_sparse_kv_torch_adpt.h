/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef QUANT_LIGHTNING_INDEXER_SPARSE_KV_TORCH_ADPT_H
#define QUANT_LIGHTNING_INDEXER_SPARSE_KV_TORCH_ADPT_H

#include <cstdint>
#include <limits>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"

#include "../../aclnn_torch_adapter/op_api_common.h"
#include "op_host/quant_lightning_indexer_sparse_kv_contract.h"

namespace vllm_ascend::qli_sparse_kv {
namespace detail {

constexpr int64_t kQueryHeads = 32;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kSparseCount = 2048;

inline void CheckTensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t rank,
                        const at::Device& device) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
  TORCH_CHECK(tensor.dim() == rank, name, " must have rank ", rank);
}

inline at::Tensor ValidateAndNormalizeKeyScale(const at::Tensor& keyDequantScale, const at::Tensor& key,
                                               const at::Device& device) {
  TORCH_CHECK(keyDequantScale.defined(), "key_dequant_scale must be defined");
  TORCH_CHECK(keyDequantScale.device() == device, "key_dequant_scale must be on ", device);
  TORCH_CHECK(keyDequantScale.scalar_type() == at::kFloat, "key_dequant_scale must be float32");
  TORCH_CHECK(keyDequantScale.dim() == 2 || keyDequantScale.dim() == 3,
              "key_dequant_scale must be [blocks,block_size] or [blocks,block_size,1]");
  TORCH_CHECK(keyDequantScale.size(0) == key.size(0) && keyDequantScale.size(1) == key.size(1),
              "key_dequant_scale leading dimensions must match key");
  TORCH_CHECK(keyDequantScale.dim() == 2 || keyDequantScale.size(2) == 1, "key_dequant_scale last dimension must be 1");
  TORCH_CHECK(keyDequantScale.stride(1) == 1 && (keyDequantScale.dim() == 2 || keyDequantScale.stride(2) == 1),
              "key_dequant_scale must be contiguous on all axes except axis 0");
  return keyDequantScale.dim() == 2 ? keyDequantScale.unsqueeze(-1) : keyDequantScale;
}

inline void ValidateInputs(const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
                           const at::Tensor& queryDequantScale, const at::Tensor& actualSeqLengthsQuery,
                           const at::Tensor& actualSeqLengthsKey, const at::Tensor& indexerBlockTable,
                           const at::Tensor& reqIds, const at::Tensor& stablePrefixLens,
                           const at::Tensor& visibleSeqLens, const at::Tensor& tokenToReq,
                           const at::Tensor& kvBlockTable, const at::Tensor& activeRows, const at::Tensor& lastReqIds,
                           const at::Tensor& slotToToken, const at::Tensor& priority, const at::Tensor& freeMask,
                           const at::Tensor& freeStack, const at::Tensor& freeCount, const at::Tensor& clock,
                           const at::Tensor& currentSlots,
                           const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots,
                           const at::Tensor& tokenToSlot, const at::Tensor& status, const at::Tensor& residentEnd,
                           const at::Tensor& probeThresholds, const at::Tensor& probeStats,
                           int64_t capacity, int64_t maxToken, int64_t blockSize,
                           int64_t hostNumBlocks, int64_t maxQueries, int64_t probeMode) {
  const at::Device device = query.device();
  TORCH_CHECK(query.is_privateuseone(), "query must be an NPU tensor");
  CheckTensor(query, "query", at::kFloat8_e4m3fn, 3, device);
  TORCH_CHECK(query.size(0) > 0 && query.size(1) == kQueryHeads && query.size(2) == kHeadDim,
              "query must have shape [T,32,128] with T > 0");
  const int64_t rows = query.size(0);

  CheckTensor(key, "key", at::kFloat8_e4m3fn, 4, device);
  TORCH_CHECK(key.size(0) > 0 && key.size(1) == blockSize && key.size(2) == 1 && key.size(3) == kHeadDim,
              "key must have shape [blocks,block_size,1,128]");
  TORCH_CHECK(key.stride(3) == 1 && key.stride(2) == kHeadDim && key.stride(1) == kHeadDim,
              "key must be contiguous on all axes except axis 0");

  CheckTensor(weights, "weights", at::kBFloat16, 2, device);
  TORCH_CHECK(weights.size(0) == rows && weights.size(1) == kQueryHeads, "weights must have shape [T,32]");
  CheckTensor(queryDequantScale, "query_dequant_scale", at::kFloat, 2, device);
  TORCH_CHECK(queryDequantScale.sizes() == weights.sizes(), "query_dequant_scale must have shape [T,32]");

  CheckTensor(actualSeqLengthsQuery, "actual_seq_lengths_query", at::kInt, 1, device);
  CheckTensor(actualSeqLengthsKey, "actual_seq_lengths_key", at::kInt, 1, device);
  TORCH_CHECK(actualSeqLengthsQuery.numel() > 0 && actualSeqLengthsQuery.numel() == actualSeqLengthsKey.numel(),
              "actual sequence length tensors must have the same nonzero length");
  CheckTensor(indexerBlockTable, "indexer_block_table", at::kInt, 2, device);
  TORCH_CHECK(indexerBlockTable.size(0) == actualSeqLengthsQuery.numel(),
              "indexer_block_table request count must match actual sequence lengths");

  namespace contract = optiling::qli_sparse_kv_contract;
  TORCH_CHECK(maxQueries >= 1 && maxQueries <= contract::MAX_QUERY_ROWS &&
              capacity >= maxQueries * kSparseCount && capacity <= contract::MAX_CAPACITY &&
              blockSize == kHeadDim && capacity % blockSize == 0 && maxToken > 0 &&
              maxToken <= std::numeric_limits<int32_t>::max() && hostNumBlocks > 0 &&
              hostNumBlocks <= std::numeric_limits<int32_t>::max(), "invalid request-pool attributes");
  TORCH_CHECK(probeMode >= 0 && probeMode <= contract::PROBE_MAX_MODE, "probe_mode must be 0, 1 or 2");
  const int64_t requests = actualSeqLengthsQuery.numel();
  TORCH_CHECK(requests <= std::numeric_limits<int32_t>::max() && rows <= std::numeric_limits<int32_t>::max(),
              "request/row count exceeds SIMT indexing range");
  const at::Tensor inputs[] = {reqIds, stablePrefixLens, visibleSeqLens, tokenToReq, kvBlockTable,
      activeRows, lastReqIds, slotToToken, priority, freeMask, freeStack, freeCount, clock,
      currentSlots, missCount, missTokens, missSlots, tokenToSlot, status, residentEnd, probeThresholds, probeStats};
  const int64_t sizes[][2] = {{requests, 0}, {rows, 0}, {rows, 0}, {rows, 0}, {requests, -1}, {1, 0},
      {requests, 0}, {requests, capacity}, {requests, capacity}, {requests, capacity}, {requests, capacity},
      {requests, 0}, {requests, 0}, {rows, kSparseCount}, {requests, 0},
      {requests, maxQueries*kSparseCount}, {requests, maxQueries*kSparseCount}, {requests, maxToken},
      {requests, 0}, {requests, 0}, {rows, 0}, {rows, contract::PROBE_STAT_COLUMNS}};
  const at::Tensor indexerInputs[] = {query, key, weights, queryDequantScale, actualSeqLengthsQuery,
                                     actualSeqLengthsKey, indexerBlockTable};
  for (size_t i = 0; i < std::size(inputs); ++i) {
    const auto dtype = i == 20 ? at::kFloat : (i == 0 || i == 6 || i == 8 || i == 12) ? at::kLong : at::kInt;
    const int64_t rank = sizes[i][1] == 0 ? 1 : 2;
    CheckTensor(inputs[i], "request state", dtype, rank, device);
    TORCH_CHECK(inputs[i].is_contiguous() && inputs[i].size(0) == sizes[i][0] &&
        (rank == 1 || (sizes[i][1] < 0 ? inputs[i].size(1) > 0 &&
          inputs[i].size(1) <= std::numeric_limits<int32_t>::max() : inputs[i].size(1) == sizes[i][1])),
        "request state shape/stride mismatch at input ", i);
    if (i >= 6) {
      const auto disjoint = [&](const at::Tensor& other) {
        const auto a = reinterpret_cast<uintptr_t>(inputs[i].data_ptr());
        const auto b = reinterpret_cast<uintptr_t>(other.data_ptr());
        return a + inputs[i].nbytes() <= b || b + other.nbytes() <= a;
      };
      for (size_t j = 0; j < i; ++j) TORCH_CHECK(disjoint(inputs[j]), "mutable request storage overlaps");
      for (const auto& input : indexerInputs) TORCH_CHECK(disjoint(input), "request state overlaps indexer input");
    }
  }
}

}  // namespace detail
}  // namespace vllm_ascend::qli_sparse_kv

namespace vllm_ascend {

inline at::Tensor npu_quant_lightning_indexer_sparse_kv(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights, const at::Tensor& queryDequantScale,
    const at::Tensor& keyDequantScale, const at::Tensor& actualSeqLengthsQuery, const at::Tensor& actualSeqLengthsKey,
    const at::Tensor& indexerBlockTable, const at::Tensor& reqIds, const at::Tensor& stablePrefixLens,
    const at::Tensor& visibleSeqLens, const at::Tensor& tokenToReq, const at::Tensor& kvBlockTable,
    const at::Tensor& activeRows, const at::Tensor& lastReqIds, const at::Tensor& slotToToken,
    const at::Tensor& priority, const at::Tensor& freeMask, const at::Tensor& freeStack,
    const at::Tensor& freeCount, const at::Tensor& clock, const at::Tensor& currentSlots, const at::Tensor& missCount,
    const at::Tensor& missTokens, const at::Tensor& missSlots, const at::Tensor& tokenToSlot,
    const at::Tensor& status, const at::Tensor& residentEnd,
    const at::Tensor& probeThresholds, const at::Tensor& probeStats, int64_t capacity,
    int64_t maxToken, int64_t blockSize, int64_t hostNumBlocks, int64_t maxQueries, int64_t probeMode) {
  qli_sparse_kv::detail::ValidateInputs(
      query, key, weights, queryDequantScale, actualSeqLengthsQuery, actualSeqLengthsKey, indexerBlockTable, reqIds,
      stablePrefixLens, visibleSeqLens, tokenToReq, kvBlockTable, activeRows, lastReqIds, slotToToken, priority,
      freeMask, freeStack, freeCount, clock, currentSlots, missCount, missTokens, missSlots, tokenToSlot,
      status, residentEnd, probeThresholds, probeStats, capacity, maxToken, blockSize, hostNumBlocks, maxQueries, probeMode);
  at::Tensor canonicalKeyScale =
      qli_sparse_kv::detail::ValidateAndNormalizeKeyScale(keyDequantScale, key, query.device());

  for (const auto& state : {lastReqIds, slotToToken, priority, freeMask, freeStack, freeCount, clock,
                            currentSlots, missCount, missTokens, missSlots, tokenToSlot, status, residentEnd, probeStats}) {
    const auto a = reinterpret_cast<uintptr_t>(state.data_ptr());
    const auto b = reinterpret_cast<uintptr_t>(canonicalKeyScale.data_ptr());
    TORCH_CHECK(a + state.nbytes() <= b || b + canonicalKeyScale.nbytes() <= a,
                "request state overlaps key scale");
  }
  const c10_npu::OptionalNPUGuard deviceGuard(query.device());
  at::Tensor sparseIndices =
      at::empty({query.size(0), 1, qli_sparse_kv::detail::kSparseCount}, query.options().dtype(at::kInt));
  int64_t queryQuantMode = 0;
  int64_t keyQuantMode = 0;
  char layoutQuery[] = "TND";
  char layoutKey[] = "PA_BSND";
  int64_t sparseCount = qli_sparse_kv::detail::kSparseCount;
  int64_t sparseMode = 3;
  int64_t preTokens = std::numeric_limits<int64_t>::max();
  int64_t nextTokens = std::numeric_limits<int64_t>::max();
  int64_t keyStride0 = key.stride(0);
  int64_t keyScaleStride0 = canonicalKeyScale.stride(0);

  EXEC_NPU_CMD(aclnnQuantLightningIndexerSparseKv, query, key, weights, queryDequantScale, canonicalKeyScale,
               actualSeqLengthsQuery, actualSeqLengthsKey, indexerBlockTable, reqIds, stablePrefixLens, visibleSeqLens,
               tokenToReq, kvBlockTable, activeRows, lastReqIds, slotToToken, priority, freeMask, freeStack,
               freeCount, clock, currentSlots, missCount, missTokens, missSlots, tokenToSlot, status, residentEnd, probeThresholds, probeStats, queryQuantMode, keyQuantMode, layoutQuery, layoutKey,
               sparseCount, sparseMode, preTokens, nextTokens, keyStride0, keyScaleStride0, capacity, maxToken,
               blockSize, hostNumBlocks, maxQueries, probeMode, sparseIndices);
  return sparseIndices;
}

}  // namespace vllm_ascend

#endif
