// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "request_plan_algorithm.h"
static_assert(sizeof(request_plan::Control) <= 1152, "Update host localBytes with Control layout");

struct RequestPlanTilingData {
  int32_t requests, rows, topk, capacity, maxToken, pages, hostBlocks, blockSize, maxQueries, hash;
  uint32_t localBytes;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(request_plan::THREADS) inline void RunRequestPlan(
    __ubuf__ request_plan::Control* control,
    __gm__ int64_t* reqIds, __gm__ int32_t* topk, __gm__ int32_t* stable, __gm__ int32_t* visible,
    __gm__ int32_t* rowToReq, __gm__ int32_t* table, __gm__ int32_t* active,
    __gm__ int64_t* lastIds, __gm__ int32_t* tokens, __gm__ int64_t* priority,
    __gm__ int32_t* freeMask, __gm__ int32_t* freeStack, __gm__ int32_t* freeCount,
    __gm__ int64_t* clock, __gm__ int32_t* slots, __gm__ int32_t* missCount,
    __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* scratch, __gm__ int32_t* status,
    int32_t requests, int32_t rows, int32_t k, int32_t capacity, int32_t maxToken, int32_t pages,
    int32_t hostBlocks, int32_t blockSize, int32_t maxQueries, int32_t hash, int32_t first, int32_t stride) {
  request_plan::Context context{reqIds, topk, stable, visible, rowToReq, table, active, lastIds, tokens,
      priority, freeMask, freeStack, freeCount, clock, slots, missCount, missTokens, missSlots, scratch, status};
  request_plan::Shape shape{requests, rows, k, capacity, maxToken, pages, hostBlocks, blockSize, maxQueries, hash};
  for (int32_t req = first; req < requests; req += stride) {
    request_plan::Process<request_plan::THREADS>(control, context, shape, req, threadIdx.x);
    asc_syncthreads();
  }
}

extern "C" __global__ __aicore__ void sparse_kv_request_plan(
    GM_ADDR reqIds, GM_ADDR topk, GM_ADDR stable, GM_ADDR visible, GM_ADDR rowToReq, GM_ADDR table, GM_ADDR active,
    GM_ADDR lastIds, GM_ADDR tokens, GM_ADDR priority, GM_ADDR freeMask, GM_ADDR freeStack, GM_ADDR freeCount,
    GM_ADDR clock, GM_ADDR slots, GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR scratch,
    GM_ADDR status, GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(RequestPlanTilingData);
  GET_TILING_DATA_WITH_STRUCT(RequestPlanTilingData, t, tiling);
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, t.localBytes);
  auto control = reinterpret_cast<__ubuf__ request_plan::Control*>(buffer.Get<int32_t>().GetPhyAddr());
  asc_vf_call<RunRequestPlan>(dim3(request_plan::THREADS), control,
      reinterpret_cast<__gm__ int64_t*>(reqIds), reinterpret_cast<__gm__ int32_t*>(topk),
      reinterpret_cast<__gm__ int32_t*>(stable), reinterpret_cast<__gm__ int32_t*>(visible),
      reinterpret_cast<__gm__ int32_t*>(rowToReq), reinterpret_cast<__gm__ int32_t*>(table),
      reinterpret_cast<__gm__ int32_t*>(active), reinterpret_cast<__gm__ int64_t*>(lastIds),
      reinterpret_cast<__gm__ int32_t*>(tokens), reinterpret_cast<__gm__ int64_t*>(priority),
      reinterpret_cast<__gm__ int32_t*>(freeMask), reinterpret_cast<__gm__ int32_t*>(freeStack),
      reinterpret_cast<__gm__ int32_t*>(freeCount), reinterpret_cast<__gm__ int64_t*>(clock),
      reinterpret_cast<__gm__ int32_t*>(slots), reinterpret_cast<__gm__ int32_t*>(missCount),
      reinterpret_cast<__gm__ int32_t*>(missTokens), reinterpret_cast<__gm__ int32_t*>(missSlots),
      reinterpret_cast<__gm__ int32_t*>(scratch), reinterpret_cast<__gm__ int32_t*>(status),
      t.requests, t.rows, t.topk, t.capacity, t.maxToken, t.pages, t.hostBlocks, t.blockSize, t.maxQueries, t.hash,
      static_cast<int32_t>(AscendC::GetBlockIdx()), static_cast<int32_t>(AscendC::GetBlockNum()));
}
