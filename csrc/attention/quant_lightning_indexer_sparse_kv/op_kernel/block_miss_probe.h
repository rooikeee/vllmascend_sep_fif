// SPDX-License-Identifier: Apache-2.0
#ifndef QLI_BLOCK_MISS_PROBE_H
#define QLI_BLOCK_MISS_PROBE_H

#include <cstdint>
#include "request_plan_algorithm.h"

namespace block_miss_probe {
constexpr uint32_t THREADS = 128;
constexpr int32_t MAX_TILE_ROWS = 2;
constexpr int32_t STAT_COLUMNS = 3;  // score blocks, candidates, misses
constexpr int32_t DISABLED = 0;
constexpr int32_t THRESHOLD_ONLY = 1;
constexpr int32_t LOOKUP = 2;
constexpr uint16_t BF16_SIGN = 0x8000;
constexpr uint16_t BF16_ALL_BITS = 0xffff;

struct Row {
  int32_t index, request, limit, stable;
  bool valid, sameRequest;
  float threshold;
};
struct Control {
  Row rows[MAX_TILE_ROWS];
  int32_t blocks;
  int32_t candidates[MAX_TILE_ROWS][THREADS];
  int32_t misses[MAX_TILE_ROWS][THREADS];
};

// QLI stores sortable BF16 keys, not raw FP32 scores. Undo that encoding so
// score > threshold uses exactly the rounded score consumed by final TopK.
__simt_callee__ __aicore__ inline float DecodeScore(uint16_t key) {
  const uint16_t bits = (key & BF16_SIGN) ? key ^ BF16_SIGN : key ^ BF16_ALL_BITS;
  union { uint32_t bits; float value; } decoded;
  decoded.bits = static_cast<uint32_t>(bits) << 16;
  return decoded.value;
}

// This phase only reads persistent planner state. Every threshold candidate
// performs one reverse-table load; no CAS, protection, allocation or prefetch.
// Per-lane counters stay in dead TopK UB and are published once per query row.
__simt_callee__ __aicore__ inline void ProcessBlock(
    __ubuf__ Control* control, const request_plan::Context& c, const request_plan::Shape& s,
    __ubuf__ const uint16_t* scores, __gm__ const float* thresholds, __gm__ int32_t* stats,
    int32_t mode, int32_t firstRow, int32_t rows, int64_t firstToken, int32_t scoreStride,
    int64_t firstCausalEnd, bool firstBlock, bool lastBlock, uint32_t lane) {
  if (mode == DISABLED) return;
  if (firstBlock) {
    if (lane == 0) {
      control->blocks = 0;
      const int32_t active = c.active[0];
      for (int32_t q = 0; q < rows; ++q) {
        auto& row = control->rows[q];
        row.index = firstRow + q;
        row.valid = active >= 0 && active <= s.rows && row.index >= 0 && row.index < active;
        row.request = row.valid ? c.rowToReq[row.index] : -1;
        row.valid = row.valid && row.request >= 0 && row.request < s.requests;
        row.limit = 0;
        row.stable = 0;
        row.sameRequest = false;
        row.threshold = 0;
        if (!row.valid) continue;
        row.valid = c.reqIds[row.request] >= 0;
        row.sameRequest = row.valid && c.reqIds[row.request] == c.lastIds[row.request];
        row.threshold = thresholds[row.index];
        int64_t limit = firstCausalEnd + q;
        if (limit > c.visible[row.index]) limit = c.visible[row.index];
        if (limit > s.maxToken) limit = s.maxToken;
        row.limit = limit > 0 ? static_cast<int32_t>(limit) : 0;
        row.stable = s.maxToken;
        // Match the final request planner's rollback bound without modifying it.
        // Do this once per S1 tile, never inside the per-token lookup loop.
        for (int32_t other = 0; other < active; ++other) {
          if (c.rowToReq[other] == row.request) {
            const int32_t stable = c.stable[other] > 0 ? c.stable[other] : 0;
            if (stable < row.stable) row.stable = stable;
          }
        }
      }
    }
    for (int32_t q = 0; q < rows; ++q) {
      control->candidates[q][lane] = 0;
      control->misses[q][lane] = 0;
    }
    asc_syncthreads();
  }
  if (lane == 0) ++control->blocks;
  const int64_t token = static_cast<int64_t>(firstToken) + lane;
  for (int32_t q = 0; q < rows; ++q) {
    const auto& row = control->rows[q];
    if (!row.valid || token >= row.limit) continue;
    if (!(DecodeScore(scores[q * scoreStride + lane]) > row.threshold)) continue;
    ++control->candidates[q][lane];
    if (mode == LOOKUP) {
      // Volatile keeps even cold/rollback probes as actual reads for this experiment.
      const __gm__ volatile int32_t* reverse = c.tokenToSlot + static_cast<int64_t>(row.request) * s.maxToken;
      const int32_t slot = reverse[token];
      const bool hit = row.sameRequest && token < row.stable && slot >= 0 && slot < s.capacity;
      control->misses[q][lane] += !hit;
    }
  }
  if (lastBlock) {
    asc_syncthreads();
    if (lane == 0) {
      for (int32_t q = 0; q < rows; ++q) {
        int32_t candidates = 0, misses = 0;
        for (uint32_t thread = 0; thread < THREADS; ++thread) {
          candidates += control->candidates[q][thread];
          misses += control->misses[q][thread];
        }
        const auto& row = control->rows[q];
        if (row.index >= 0 && row.index < s.rows) {
          stats[static_cast<int64_t>(row.index) * STAT_COLUMNS] = row.valid ? control->blocks : 0;
          stats[static_cast<int64_t>(row.index) * STAT_COLUMNS + 1] = candidates;
          stats[static_cast<int64_t>(row.index) * STAT_COLUMNS + 2] = misses;
        }
      }
    }
    asc_syncthreads();
  }
}
}  // namespace block_miss_probe
#endif
