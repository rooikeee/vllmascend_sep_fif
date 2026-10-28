/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_lightning_indexer_sparse_kv_template_tiling_key.h"
#include "request_plan_vf.h"
#if __has_include("../vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_kernel_arch35.h")
#include "../vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_kernel_arch35.h"
#else
#include "vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_kernel_arch35.h"
#endif

using namespace QLIKernel;

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void quant_lightning_indexer_sparse_kv(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR queryScale, GM_ADDR keyScale,
    GM_ADDR actualSeqLengthsQ, GM_ADDR actualSeqLengthsK, GM_ADDR indexerBlockTable,
    GM_ADDR reqIds, GM_ADDR stablePrefixLens, GM_ADDR visibleSeqLens, GM_ADDR tokenToReq,
    GM_ADDR kvBlockTable, GM_ADDR activeRows, GM_ADDR lastReqIds, GM_ADDR slotToToken,
    GM_ADDR priority, GM_ADDR freeMask, GM_ADDR freeStack, GM_ADDR freeCount, GM_ADDR clock,
    GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
    GM_ADDR tokenToSlot, GM_ADDR status, GM_ADDR residentEnd, GM_ADDR sparseIndices, GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(QuantLightningIndexerSparseKvTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantLightningIndexerSparseKvTilingData, tilingDataIn, tiling);
    const QuantLightningIndexerSparseKvTilingData* __restrict tilingData = &tilingDataIn;
    AscendC::TPipe pipe;
    __gm__ uint8_t* userWorkspace = GetUserWorkspace(workspace);

    QuantLightningIndexerKernel<
        QLIType<fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, PAGE_ATTENTION,
                LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), bfloat16_t,
                float32_t, float32_t, uint16_t>> qli;
    qli.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ,
             actualSeqLengthsK, indexerBlockTable, sparseIndices, userWorkspace,
             tilingData, &pipe);
    qli.Process();
    if ASCEND_IS_AIV {
        request_plan::Context context{
            reinterpret_cast<__gm__ int64_t*>(reqIds), reinterpret_cast<__gm__ int32_t*>(sparseIndices),
            reinterpret_cast<__gm__ int32_t*>(stablePrefixLens), reinterpret_cast<__gm__ int32_t*>(visibleSeqLens),
            reinterpret_cast<__gm__ int32_t*>(tokenToReq), reinterpret_cast<__gm__ int32_t*>(kvBlockTable),
            reinterpret_cast<__gm__ int32_t*>(activeRows), reinterpret_cast<__gm__ int64_t*>(lastReqIds),
            reinterpret_cast<__gm__ int32_t*>(slotToToken), reinterpret_cast<__gm__ int64_t*>(priority),
            reinterpret_cast<__gm__ int32_t*>(freeMask), reinterpret_cast<__gm__ int32_t*>(freeStack),
            reinterpret_cast<__gm__ int32_t*>(freeCount), reinterpret_cast<__gm__ int64_t*>(clock),
            reinterpret_cast<__gm__ int32_t*>(currentSlots), reinterpret_cast<__gm__ int32_t*>(missCount),
            reinterpret_cast<__gm__ int32_t*>(missTokens), reinterpret_cast<__gm__ int32_t*>(missSlots),
            reinterpret_cast<__gm__ int32_t*>(tokenToSlot), reinterpret_cast<__gm__ int32_t*>(status),
            reinterpret_cast<__gm__ int32_t*>(residentEnd)};
        request_plan::Shape shape{
            static_cast<int32_t>(tilingData->maxRequests), static_cast<int32_t>(tilingData->maxRows),
            static_cast<int32_t>(tilingData->topk), static_cast<int32_t>(tilingData->capacity),
            static_cast<int32_t>(tilingData->maxToken), static_cast<int32_t>(tilingData->maxNumBlocks),
            static_cast<int32_t>(tilingData->hostNumBlocks), tilingData->planBlockSize, tilingData->maxQueries};
        qli.PlanRequests(context, shape);
    }
}
