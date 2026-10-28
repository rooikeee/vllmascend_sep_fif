// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "request_plan_algorithm.h"
__simt_vf__ __aicore__ LAUNCH_BOUND(request_plan::THREADS) inline void RunRequestPlan(
    __ubuf__ request_plan::Control* control,
    __gm__ int64_t* reqIds, __gm__ int32_t* topk, __gm__ int32_t* stable, __gm__ int32_t* visible,
    __gm__ int32_t* rowToReq, __gm__ int32_t* table, __gm__ int32_t* active,
    __gm__ int64_t* lastIds, __gm__ int32_t* tokens, __gm__ int64_t* priority,
    __gm__ int32_t* freeMask, __gm__ int32_t* freeStack, __gm__ int32_t* freeCount,
    __gm__ int64_t* clock, __gm__ int32_t* slots, __gm__ int32_t* missCount,
    __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* tokenToSlot,
    __gm__ int32_t* status, __gm__ int32_t* residentEnd,
    int32_t requests, int32_t rows, int32_t k, int32_t capacity, int32_t maxToken, int32_t pages,
    int32_t hostBlocks, int32_t blockSize, int32_t maxQueries, int32_t first, int32_t stride) {
  request_plan::Context context{reqIds, topk, stable, visible, rowToReq, table, active, lastIds, tokens,
      priority, freeMask, freeStack, freeCount, clock, slots, missCount, missTokens, missSlots, tokenToSlot, status, residentEnd};
  request_plan::Shape shape{requests, rows, k, capacity, maxToken, pages, hostBlocks, blockSize, maxQueries};
  for (int32_t req = first; req < requests; req += stride) {
    request_plan::Process<request_plan::THREADS>(control, context, shape, req, threadIdx.x);
    asc_syncthreads();
  }
}
