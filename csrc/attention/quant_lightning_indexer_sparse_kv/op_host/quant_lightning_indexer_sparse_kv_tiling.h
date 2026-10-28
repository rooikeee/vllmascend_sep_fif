/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QUANT_LIGHTNING_INDEXER_SPARSE_KV_TILING_H_
#define QUANT_LIGHTNING_INDEXER_SPARSE_KV_TILING_H_

#include <cstdint>

#include "quant_lightning_indexer_sparse_kv_contract.h"

#ifndef QLI_SPARSE_KV_FIXED_CONTRACT_ONLY

  #include "exe_graph/runtime/tiling_context.h"
  #include "platform/platform_info.h"
  #include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QuantLightningIndexerSparseKvTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, tSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, maxRows)
TILING_DATA_FIELD_DEF(int64_t, topk)
TILING_DATA_FIELD_DEF(int64_t, capacity)
TILING_DATA_FIELD_DEF(int64_t, maxToken)
TILING_DATA_FIELD_DEF(int64_t, maxRequests)
TILING_DATA_FIELD_DEF(int64_t, maxNumBlocks)
TILING_DATA_FIELD_DEF(int64_t, hostNumBlocks)
TILING_DATA_FIELD_DEF(int32_t, planBlockSize)
TILING_DATA_FIELD_DEF(int32_t, maxQueries)
TILING_DATA_FIELD_DEF(int32_t, probeMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantLightningIndexerSparseKv, QuantLightningIndexerSparseKvTilingData)

struct QuantLightningIndexerSparseKvCompileInfo {};

struct QuantLightningIndexerSparseKvInfo {
  fe::PlatFormInfos* platformInfo = nullptr;
  uint32_t bSize = 0;
  uint32_t tSize = 0;
  uint32_t s2Size = 0;
  uint32_t maxBlockNumPerBatch = 0;
  uint32_t keyStride0 = 0;
  uint32_t keyDequantScaleStride0 = 0;
  int64_t capacity = 0;
  int64_t maxToken = 0;
  int64_t maxRequests = 0;
  int64_t maxNumBlocks = 0;
  int64_t hostNumBlocks = 0;
  int32_t maxQueries = 0;
  int32_t probeMode = 0;
};

ge::graphStatus TilingForQuantLightningIndexerSparseKv(gert::TilingContext* context);

}  // namespace optiling

#endif
#endif
