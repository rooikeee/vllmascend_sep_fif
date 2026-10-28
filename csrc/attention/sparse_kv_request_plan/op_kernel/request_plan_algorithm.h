// SPDX-License-Identifier: Apache-2.0
#ifndef SPARSE_KV_REQUEST_PLAN_ALGORITHM_H
#define SPARSE_KV_REQUEST_PLAN_ALGORITHM_H

#include <cstdint>
#include <climits>

namespace request_plan {
constexpr uint32_t THREADS = 256;
constexpr int32_t MAX_QUERY_ROWS = 16;
constexpr int32_t INVALID = -1;
constexpr int32_t PENDING = -2;
constexpr int32_t BAD_ROWS = 1;
constexpr int32_t BAD_STATE = 2;
constexpr int32_t NO_CAPACITY = 4;

struct Shape {
  int32_t requests, rows, topk, capacity, maxToken, pages, hostBlocks, blockSize, maxQueries;
};

struct Context {
  __gm__ int64_t *reqIds;
  __gm__ int32_t *topk, *stable, *visible, *rowToReq, *table, *active;
  __gm__ int64_t *lastIds;
  __gm__ int32_t *tokens;
  __gm__ int64_t *priority;
  __gm__ int32_t *freeMask, *freeStack, *freeCount;
  __gm__ int64_t *clock;
  __gm__ int32_t *slots, *missCount, *missTokens, *missSlots, *tokenToSlot, *status, *residentEnd;
};

// One block owns one request. Barriers protect scratch and state between phases.
struct Control {
  int32_t rowIds[MAX_QUERY_ROWS];
  int32_t rowCount, stable, active, reset, failed, freeBefore, misses, evictions, equalCount;
  int32_t residentEnd, nextEnd;
  int32_t histogram[256];
  uint64_t epoch, threshold;
};

__simt_callee__ __aicore__ inline bool Valid(const Context& c, const Shape& s, int32_t req,
                                           int32_t row, int32_t token) {
  if (token < 0 || token >= s.maxToken || token >= c.visible[row]) return false;
  const int32_t page = token / s.blockSize;
  if (page >= s.pages) return false;
  const int32_t physical = c.table[static_cast<int64_t>(req) * s.pages + page];
  return physical >= 0 && physical < s.hostBlocks;
}

template <uint32_t Threads>
__simt_callee__ __aicore__ inline void Process(__ubuf__ Control* w, const Context& c, const Shape& s,
                                             int32_t req, uint32_t thread) {
  const int64_t base = static_cast<int64_t>(req) * s.capacity;
  const int32_t unionSize = s.maxQueries * s.topk;
  const int64_t missBase = static_cast<int64_t>(req) * unionSize;
  auto reverse = c.tokenToSlot + static_cast<int64_t>(req) * s.maxToken;
  if (thread == 0) {
    c.missCount[req] = 0;
    c.status[req] = 0;
    w->rowCount = 0;
    w->stable = s.maxToken;
    w->active = c.active[0];
    w->failed = (w->active < 0 || w->active > s.rows) ? BAD_ROWS : 0;
    if (w->failed) w->active = 0;
    for (int32_t row = 0; row < w->active; ++row) {
      if (c.rowToReq[row] != req) continue;
      if (w->rowCount < s.maxQueries) w->rowIds[w->rowCount] = row;
      ++w->rowCount;
      const int32_t stable = c.stable[row] < 0 ? 0 : c.stable[row];
      if (stable < w->stable) w->stable = stable;
    }
    if (w->rowCount > s.maxQueries || (w->rowCount > 0 && c.reqIds[req] < 0)) w->failed = BAD_ROWS;
    if (req == 0) {
      for (int32_t row = 0; row < w->active; ++row) {
        if (c.rowToReq[row] < 0 || c.rowToReq[row] >= s.requests) w->failed = BAD_ROWS;
      }
    }
    c.status[req] = w->failed;
  }
  asc_syncthreads();
  // Also clear padding and malformed rows; no stale output survives replay.
  for (int32_t row = 0; row < s.rows; ++row) {
    const int32_t owner = c.rowToReq[row];
    if ((row < w->active && owner == req) ||
        (req == 0 && (row >= w->active || owner < 0 || owner >= s.requests))) {
      for (int32_t k = thread; k < s.topk; k += Threads) c.slots[static_cast<int64_t>(row) * s.topk + k] = INVALID;
    }
  }
  asc_syncthreads();
  if (w->failed || w->rowCount == 0) return;
  // All lanes must consume the previous phase's return decision before lane 0
  // can publish a new failure, otherwise some lanes could miss the next barrier.
  asc_syncthreads();
  if (thread == 0) {
    w->reset = c.lastIds[req] != c.reqIds[req];
    w->epoch = w->reset ? 1 : static_cast<uint64_t>(c.clock[req]) + 1;
    w->residentEnd = w->reset ? 0 : c.residentEnd[req];
    w->nextEnd = w->residentEnd < w->stable ? w->residentEnd : w->stable;
    w->misses = 0;
    if (!w->reset && (c.clock[req] < 0 || c.clock[req] == INT64_MAX ||
                     c.freeCount[req] < 0 || c.freeCount[req] > s.capacity ||
                     w->residentEnd < 0 || w->residentEnd > s.maxToken)) {
      w->failed = BAD_STATE;
      c.status[req] = BAD_STATE;
    }
    if (w->reset) c.freeCount[req] = s.capacity;
  }
  asc_syncthreads();
  if (w->failed) return;
  if (w->reset) {
    // The reverse table is initialized once to -1. Request replacement only
    // clears entries belonging to occupied slots, not the full logical context.
    for (int32_t slot = thread; slot < s.capacity; slot += Threads) {
      const int32_t oldToken = c.tokens[base + slot];
      if (oldToken >= 0 && oldToken < s.maxToken) reverse[oldToken] = INVALID;
      c.tokens[base + slot] = INVALID;
      c.priority[base + slot] = 0;
      c.freeMask[base + slot] = 1;
      c.freeStack[base + slot] = s.capacity - 1 - slot;
    }
  }
  asc_syncthreads();
  // Only the previously cached speculative suffix needs invalidation. This
  // bound may overestimate after eviction, but never misses a resident token.
  for (int64_t token = static_cast<int64_t>(w->stable) + thread; token < w->residentEnd; token += Threads) {
    const int32_t slot = reverse[token];
    if (slot >= 0 && slot < s.capacity) {
      reverse[token] = INVALID;
      c.tokens[base + slot] = INVALID;
      c.freeMask[base + slot] = 1;
      const int32_t pos = asc_atomic_add(c.freeCount + req, 1);
      c.freeStack[base + pos] = slot;
    }
  }
  asc_syncthreads();
  for (int32_t q = 0; q < w->rowCount; ++q) {
    const int32_t row = w->rowIds[q];
    for (int32_t k = thread; k < s.topk; k += Threads) {
      const int32_t token = c.topk[static_cast<int64_t>(row) * s.topk + k];
      if (!Valid(c, s, req, row, token)) continue;
      asc_atomic_max(&w->nextEnd, token + 1);
      // CAS also serves as an atomic read of existing/pending mappings. Exactly
      // one occurrence across all MTP rows owns a new miss; nobody spins.
      const int32_t slot = asc_atomic_cas(reverse + token, INVALID, PENDING);
      if (slot == INVALID) {
        const int32_t pos = asc_atomic_add(&w->misses, 1);
        c.missTokens[missBase + pos] = token;
      } else if (slot >= 0 && slot < s.capacity && c.tokens[base + slot] == token) {
        // Duplicate hits can touch the same slot; use an atomic timestamp update.
        asc_atomic_max(c.priority + base + slot, static_cast<int64_t>(w->epoch));
      } else if (slot != PENDING) {
        asc_atomic_max(&w->failed, BAD_STATE);
      }
    }
  }
  asc_syncthreads();
  if (w->failed) {
    for (int32_t m = thread; m < w->misses; m += Threads) reverse[c.missTokens[missBase + m]] = INVALID;
    if (thread == 0) c.status[req] = w->failed;
    asc_syncthreads();
    return;
  }
  if (thread == 0) {
    c.missCount[req] = w->misses;
    w->freeBefore = c.freeCount[req];
    w->evictions = w->misses > w->freeBefore ? w->misses - w->freeBefore : 0;
    w->threshold = 0;
  }
  asc_syncthreads();
  if (w->evictions > 0) {
    // Radix-select the E-th oldest priority, skipping unused high bytes.
    int32_t remaining = w->evictions;
    uint64_t prefixMask = 0;
    for (int32_t shift = 56; shift >= 0; shift -= 8) {
      if ((w->epoch >> shift) == 0 && prefixMask == 0) continue;
      for (int32_t h = thread; h < 256; h += Threads) w->histogram[h] = 0;
      asc_syncthreads();
      for (int32_t slot = thread; slot < s.capacity; slot += Threads) {
        const uint64_t age = static_cast<uint64_t>(c.priority[base + slot]);
        if (!c.freeMask[base + slot] && age != w->epoch && (age & prefixMask) == w->threshold) {
          asc_atomic_add(w->histogram + ((age >> shift) & 255), 1);
        }
      }
      asc_syncthreads();
      if (thread == 0) {
        int32_t before = 0, bucket = 0;
        while (bucket < 256 && before + w->histogram[bucket] < remaining) before += w->histogram[bucket++];
        if (bucket == 256) w->failed = NO_CAPACITY;
        else { w->threshold |= static_cast<uint64_t>(bucket) << shift; w->equalCount = remaining - before; }
      }
      asc_syncthreads();
      if (w->failed) break;
      remaining = w->equalCount;
      prefixMask |= static_cast<uint64_t>(255) << shift;
    }
    if (w->failed) {
      for (int32_t m = thread; m < w->misses; m += Threads) reverse[c.missTokens[missBase + m]] = INVALID;
      if (thread == 0) { c.status[req] = w->failed; c.missCount[req] = 0; }
      asc_syncthreads();
      return;
    }
    // All strictly older slots are selected first; ties use the remaining budget.
    for (int32_t slot = thread; slot < s.capacity; slot += Threads) {
      if (!c.freeMask[base + slot] && static_cast<uint64_t>(c.priority[base + slot]) != w->epoch &&
          static_cast<uint64_t>(c.priority[base + slot]) < w->threshold) {
        reverse[c.tokens[base + slot]] = INVALID;
        c.freeMask[base + slot] = 1;
        c.tokens[base + slot] = INVALID;
        const int32_t pos = asc_atomic_add(c.freeCount + req, 1);
        c.freeStack[base + pos] = slot;
      }
    }
    asc_syncthreads();
    if (thread == 0) { w->equalCount = w->misses - c.freeCount[req]; w->histogram[0] = 0; }
    asc_syncthreads();
    for (int32_t slot = thread; slot < s.capacity; slot += Threads) {
      if (!c.freeMask[base + slot] && static_cast<uint64_t>(c.priority[base + slot]) != w->epoch &&
          static_cast<uint64_t>(c.priority[base + slot]) == w->threshold) {
        const int32_t ticket = asc_atomic_add(w->histogram, 1);
        if (ticket < w->equalCount) {
          reverse[c.tokens[base + slot]] = INVALID;
          c.freeMask[base + slot] = 1;
          c.tokens[base + slot] = INVALID;
          const int32_t pos = asc_atomic_add(c.freeCount + req, 1);
          c.freeStack[base + pos] = slot;
        }
      }
    }
    asc_syncthreads();
  }
  if (thread == 0) {
    w->freeBefore = c.freeCount[req];
    if (w->freeBefore < w->misses) { w->failed = NO_CAPACITY; c.status[req] = NO_CAPACITY; c.missCount[req] = 0; }
    else c.freeCount[req] = w->freeBefore - w->misses;
  }
  asc_syncthreads();
  if (w->failed) {
    for (int32_t m = thread; m < w->misses; m += Threads) reverse[c.missTokens[missBase + m]] = INVALID;
    asc_syncthreads();
    return;
  }
  for (int32_t m = thread; m < w->misses; m += Threads) {
    const int32_t token = c.missTokens[missBase + m];
    const int32_t slot = c.freeStack[base + w->freeBefore - 1 - m];
    c.tokens[base + slot] = token;
    c.priority[base + slot] = static_cast<int64_t>(w->epoch);
    c.freeMask[base + slot] = 0;
    c.missSlots[missBase + m] = slot;
    reverse[token] = slot;
  }
  asc_syncthreads();
  for (int32_t q = 0; q < w->rowCount; ++q) {
    const int32_t row = w->rowIds[q];
    for (int32_t k = thread; k < s.topk; k += Threads) {
      const int64_t pos = static_cast<int64_t>(row) * s.topk + k;
      const int32_t token = c.topk[pos];
      if (Valid(c, s, req, row, token)) c.slots[pos] = reverse[token];
    }
  }
  if (thread == 0) {
    c.lastIds[req] = c.reqIds[req];
    c.clock[req] = static_cast<int64_t>(w->epoch);
    c.residentEnd[req] = w->nextEnd;
  }
  asc_syncthreads();
}
}  // namespace request_plan
#endif
