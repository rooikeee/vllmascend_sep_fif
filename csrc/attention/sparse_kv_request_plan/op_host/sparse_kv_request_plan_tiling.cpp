// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <climits>
#include "sparse_kv_request_plan_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
ge::graphStatus Fail(gert::TilingContext* c, const char* message) {
  OP_LOGE(c->GetNodeName(), "%s", message);
  return ge::GRAPH_FAILED;
}
bool Match(const gert::Shape& shape, int64_t a, int64_t b = -1) {
  return shape.GetDimNum() == (b < 0 ? 1U : 2U) && shape.GetDim(0) == a && (b < 0 || shape.GetDim(1) == b);
}
}
static ge::graphStatus Tile(gert::TilingContext* c) {
  const gert::Shape* shapes[20];
  for (uint32_t i = 0; i < 20; ++i) {
    const auto* shape = c->GetInputShape(i);
    const auto* desc = c->GetInputDesc(i);
    const auto type = (i == 0 || i == 7 || i == 9 || i == 13) ? ge::DT_INT64 : ge::DT_INT32;
    if (!shape || !desc || desc->GetDataType() != type) return Fail(c, "Missing shape or wrong dtype");
    shapes[i] = &shape->GetOriginShape();
    if (i >= 7) {
      const gert::Stride* stride = c->GetRequiredInputStride(i);
      if (!stride) stride = c->GetInputStride(i);
      if (stride && stride->GetDimNum() == shapes[i]->GetDimNum()) {
        int64_t expected = 1;
        for (int64_t d = shapes[i]->GetDimNum() - 1; d >= 0; --d) {
          if (shapes[i]->GetDim(d) > 1 && stride->GetStride(d) != expected) return Fail(c, "Mutable tensor is not contiguous");
          expected *= shapes[i]->GetDim(d);
        }
      }
    }
  }
  const auto* attrs = c->GetAttrs();
  if (!attrs || !attrs->GetInt(0) || !attrs->GetInt(1) || !attrs->GetInt(2) || !attrs->GetInt(3)) return Fail(c, "Missing attributes");
  const int64_t q = *attrs->GetInt(0), maxToken = *attrs->GetInt(1);
  const int64_t block = *attrs->GetInt(2), host = *attrs->GetInt(3);
  if (shapes[0]->GetDimNum() != 1 || shapes[1]->GetDimNum() != 2 || shapes[8]->GetDimNum() != 2 ||
      shapes[5]->GetDimNum() != 2) return Fail(c, "Wrong request/topk/cache/page-table rank");
  const int64_t b = shapes[0]->GetDim(0), r = shapes[1]->GetDim(0), k = shapes[1]->GetDim(1);
  const int64_t capacity = shapes[8]->GetDim(1), pages = shapes[5]->GetDim(1);
  if (b <= 0 || b > INT32_MAX || r <= 0 || r > INT32_MAX || k <= 0 || k > 2048 || q < 1 || q > 16 ||
      capacity < q * k || capacity > (1 << 24) || maxToken <= 0 || maxToken > INT32_MAX ||
      pages <= 0 || pages > INT32_MAX || block <= 0 || block > INT32_MAX || host <= 0 || host > INT32_MAX)
    return Fail(c, "Invalid capacity or dimensions; capacity must cover maxQueries*topk");
  int32_t hash = 32;
  while (hash < 2 * q * k) hash <<= 1;
  for (int i : {2, 3, 4}) if (!Match(*shapes[i], r)) return Fail(c, "Row metadata shape mismatch");
  for (int i : {7, 12, 13, 15, 19}) if (!Match(*shapes[i], b)) return Fail(c, "Request state shape mismatch");
  for (int i : {8, 9, 10, 11}) if (!Match(*shapes[i], b, capacity)) return Fail(c, "Cache state shape mismatch");
  for (int i : {16, 17}) if (!Match(*shapes[i], b, q * k)) return Fail(c, "Union miss shape mismatch");
  if (!Match(*shapes[5], b, pages) || !Match(*shapes[6], 1) || !Match(*shapes[14], r, k) ||
      !Match(*shapes[18], b, 2 * hash + capacity)) return Fail(c, "Output/workspace shape mismatch");
  RequestPlanTilingData t;
  t.set_requests(b); t.set_rows(r); t.set_topk(k); t.set_capacity(capacity); t.set_maxToken(maxToken);
  t.set_pages(pages); t.set_hostBlocks(host); t.set_blockSize(block); t.set_maxQueries(q); t.set_hash(hash);
  // Rounded storage for Control; the kernel checks this bound at compile time.
  constexpr uint32_t localBytes = 1152;
  t.set_localBytes(localBytes);
  if (c->SetDynUBufSize(localBytes) != ge::GRAPH_SUCCESS || !c->GetPlatformInfo()) return Fail(c, "Missing platform/UB");
  const platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
  const uint32_t cores = std::min<uint32_t>(b, platform.GetCoreNumAiv());
  if (!cores) return Fail(c, "No AIV cores");
  c->SetBlockDim(cores);
  c->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
  t.SaveToBuffer(c->GetRawTilingData()->GetData(), c->GetRawTilingData()->GetCapacity());
  c->GetRawTilingData()->SetDataSize(t.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(SparseKvRequestPlan).Tiling(Tile).TilingParse<RequestPlanCompileInfo>(Parse);
}
