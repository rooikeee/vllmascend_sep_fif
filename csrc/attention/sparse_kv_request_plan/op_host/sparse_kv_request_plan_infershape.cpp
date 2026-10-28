// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferRequestPlan(gert::InferShapeContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_INFERSHAPE(SparseKvRequestPlan).InferShape(InferRequestPlan);
}
