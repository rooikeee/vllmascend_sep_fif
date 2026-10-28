// SPDX-License-Identifier: Apache-2.0
#ifndef SPARSE_KV_REQUEST_PLAN_TILING_H
#define SPARSE_KV_REQUEST_PLAN_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(RequestPlanTilingData)
TILING_DATA_FIELD_DEF(int32_t, requests);
TILING_DATA_FIELD_DEF(int32_t, rows);
TILING_DATA_FIELD_DEF(int32_t, topk);
TILING_DATA_FIELD_DEF(int32_t, capacity);
TILING_DATA_FIELD_DEF(int32_t, maxToken);
TILING_DATA_FIELD_DEF(int32_t, pages);
TILING_DATA_FIELD_DEF(int32_t, hostBlocks);
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(int32_t, maxQueries);
TILING_DATA_FIELD_DEF(uint32_t, localBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseKvRequestPlan, RequestPlanTilingData)
struct RequestPlanCompileInfo {};
}
#endif
