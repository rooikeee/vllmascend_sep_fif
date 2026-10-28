// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "block_miss_probe.h"

struct BlockMissProbeContext {
  request_plan::Context state;
  request_plan::Shape shape;
  __gm__ const float* thresholds;
  __gm__ int32_t* stats;
  int32_t mode;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(block_miss_probe::THREADS) inline void RunBlockMissProbe(
    __ubuf__ block_miss_probe::Control* control, __ubuf__ const uint16_t* scores,
    __gm__ const float* thresholds, __gm__ int32_t* stats,
    __gm__ int64_t* reqIds, __gm__ int64_t* lastIds, __gm__ int32_t* reverse,
    __gm__ int32_t* stable, __gm__ int32_t* visible, __gm__ int32_t* owners, __gm__ int32_t* active,
    int32_t requests, int32_t maxRows, int32_t capacity, int32_t maxToken, int32_t mode,
    int32_t firstRow, int32_t rows, int64_t firstToken, int32_t scoreStride,
    int64_t firstCausalEnd, bool firstBlock, bool lastBlock) {
  request_plan::Context c{};
  c.reqIds = reqIds; c.lastIds = lastIds; c.tokenToSlot = reverse;
  c.stable = stable; c.visible = visible; c.rowToReq = owners; c.active = active;
  request_plan::Shape s{};
  s.requests = requests; s.rows = maxRows; s.capacity = capacity; s.maxToken = maxToken;
  block_miss_probe::ProcessBlock(control, c, s, scores, thresholds, stats, mode,
      firstRow, rows, firstToken, scoreStride, firstCausalEnd, firstBlock, lastBlock, threadIdx.x);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void ClearBlockProbeRow(__gm__ int32_t* stats, int64_t row) {
  for (int32_t column = 0; column < block_miss_probe::STAT_COLUMNS; ++column)
    stats[row * block_miss_probe::STAT_COLUMNS + column] = 0;
}
