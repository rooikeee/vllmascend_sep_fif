/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUANT_LIGHTNING_INDEXER_SPARSE_KV_TILING_DATA_H_
#define QUANT_LIGHTNING_INDEXER_SPARSE_KV_TILING_DATA_H_

#include <cstddef>
#include <cstdint>

struct alignas(8) QuantLightningIndexerSparseKvTilingData {
    uint32_t bSize;
    uint32_t tSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t sparseCount;
    uint32_t keyStride0;
    uint32_t keyDequantScaleStride0;
    uint32_t usedCoreNum;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t sparseMode;
    int64_t maxRows;
    int64_t topk;
    int64_t capacity;
    int64_t maxToken;
    int64_t maxRequests;
    int64_t maxNumBlocks;
    int64_t hostNumBlocks;
    int32_t planBlockSize;
    int32_t maxQueries;
};

static_assert(sizeof(QuantLightningIndexerSparseKvTilingData) == 120);
static_assert(offsetof(QuantLightningIndexerSparseKvTilingData, maxRows) == 56);
static_assert(offsetof(QuantLightningIndexerSparseKvTilingData, planBlockSize) == 112);

#endif
