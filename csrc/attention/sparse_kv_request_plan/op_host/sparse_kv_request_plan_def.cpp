// SPDX-License-Identifier: Apache-2.0
#include "register/op_def.h"
#include "register/op_def_registry.h"

namespace ops {
class SparseKvRequestPlan : public OpDef {
 public:
  explicit SparseKvRequestPlan(const char* name) : OpDef(name) {
    const char* names[] = {"reqIds", "topk", "stable", "visible", "rowToReq", "table", "active",
        "lastIds", "tokens", "priority", "freeMask", "freeStack", "freeCount", "clock", "slots",
        "missCount", "missTokens", "missSlots", "tokenToSlot", "status", "residentEnd"};
    for (int i = 0; i < 21; ++i) {
      const auto dtype = (i == 0 || i == 7 || i == 9 || i == 13) ? ge::DT_INT64 : ge::DT_INT32;
      auto& input = this->Input(names[i]).ParamType(REQUIRED).DataType({dtype})
          .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
      if (i >= 7) input.IgnoreContiguous();
      else input.AutoContiguous();
    }
    this->Attr("maxQueries").AttrType(REQUIRED).Int();
    this->Attr("maxToken").AttrType(REQUIRED).Int();
    this->Attr("blockSize").AttrType(REQUIRED).Int();
    this->Attr("hostBlocks").AttrType(REQUIRED).Int();
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false).DynamicRankSupportFlag(false)
        .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
        .ExtendCfgInfo("opFile.value", "sparse_kv_request_plan_apt");
    this->AICore().AddConfig("ascend950", config);
  }
};
OP_ADD(SparseKvRequestPlan);
}  // namespace ops
