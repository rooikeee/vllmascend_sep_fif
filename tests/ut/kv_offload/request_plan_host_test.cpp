// SPDX-License-Identifier: Apache-2.0
// Execute the production algorithm with synchronized CPU lanes, not an NPU model.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <set>
#include <thread>
#include <vector>
#define __gm__
#define __ubuf__
#define __simt_callee__
#define __aicore__
thread_local std::barrier<>* blockBarrier;
inline void asc_syncthreads() { blockBarrier->arrive_and_wait(); }
inline int32_t asc_atomic_add(int32_t* p, int32_t value) { return std::atomic_ref<int32_t>(*p).fetch_add(value); }
inline int32_t asc_atomic_cas(int32_t* p, int32_t expected, int32_t value) {
  std::atomic_ref<int32_t>(*p).compare_exchange_strong(expected, value);
  return expected;
}
#include "request_plan_algorithm.h"

struct Fixture {
  request_plan::Shape s{2, 8, 4, 32, 1024, 256, 512, 4, 4, 32};
  std::vector<int64_t> ids{10, 20}, last{-1, -1}, priority = std::vector<int64_t>(64), clock{0, 0};
  std::vector<int32_t> topk = std::vector<int32_t>(32, -1), stable = std::vector<int32_t>(8, 1024);
  std::vector<int32_t> visible = std::vector<int32_t>(8, 1024), owners{0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<int32_t> table = std::vector<int32_t>(512), active{8}, tokens = std::vector<int32_t>(64, -1);
  std::vector<int32_t> free = std::vector<int32_t>(64, 1), stack = std::vector<int32_t>(64), count{32, 32};
  std::vector<int32_t> slots = std::vector<int32_t>(32, -1), misses{0, 0};
  std::vector<int32_t> missTokens = std::vector<int32_t>(32, -1), missSlots = std::vector<int32_t>(32, -1);
  std::vector<int32_t> scratch = std::vector<int32_t>(192), status{0, 0};
  Fixture() { for (int i = 0; i < 512; ++i) table[i] = i; }
  request_plan::Context context() {
    return {ids.data(), topk.data(), stable.data(), visible.data(), owners.data(), table.data(), active.data(),
        last.data(), tokens.data(), priority.data(), free.data(), stack.data(), count.data(), clock.data(),
        slots.data(), misses.data(), missTokens.data(), missSlots.data(), scratch.data(), status.data()};
  }
  void fill(int first, bool duplicates) {
    for (int row = 0; row < 8; ++row)
      for (int k = 0; k < 4; ++k) topk[row * 4 + k] = first + (duplicates ? k : (row % 4) * 4 + k);
  }
  template <int Threads> void run(bool verify = true) {
    auto c = context();
    const auto old = tokens;
    const auto oldLast = last;
    const auto oldPriority = priority;
    for (int req = 0; req < 2; ++req) {
      std::barrier sync(Threads);
      request_plan::Control control{};
      std::vector<std::thread> workers;
      for (int thread = 0; thread < Threads; ++thread) workers.emplace_back([&, thread] {
        blockBarrier = &sync;
        request_plan::Process<Threads>(&control, c, s, req, thread);
      });
      for (auto& t : workers) t.join();
    }
    if (!verify) return;
    assert(status == std::vector<int32_t>({0, 0}));
    for (int req = 0; req < 2; ++req) {
      std::set<int> selected, expectedMiss, actualMiss, resident, freeSlots, allSlots;
      int prefix = 1024, rows = 0;
      for (int row = 0; row < active[0]; ++row) if (owners[row] == req) {
        prefix = std::min(prefix, std::max(0, stable[row]));
        ++rows;
        for (int k = 0; k < 4; ++k)
          if (request_plan::Valid(c, s, req, row, topk[row * 4 + k])) selected.insert(topk[row * 4 + k]);
      }
      if (rows == 0) { assert(misses[req] == 0); continue; }
      for (int token : selected) {
        bool hit = oldLast[req] == ids[req] && token < prefix &&
            std::find(old.begin() + req * 32, old.begin() + (req + 1) * 32, token) != old.begin() + (req + 1) * 32;
        if (!hit) expectedMiss.insert(token);
      }
      for (int i = 0; i < misses[req]; ++i) {
        const int t = missTokens[req * 16 + i], slot = missSlots[req * 16 + i];
        assert(actualMiss.insert(t).second && allSlots.insert(slot).second);
        assert(slot >= 0 && slot < 32 && tokens[req * 32 + slot] == t);
      }
      assert(actualMiss == expectedMiss);
      for (int slot = 0; slot < 32; ++slot) {
        const int token = tokens[req * 32 + slot];
        assert((token == -1) == bool(free[req * 32 + slot]));
        if (token == -1) freeSlots.insert(slot);
        else assert(resident.insert(token).second);
      }
      assert(int(freeSlots.size()) == count[req]);
      std::set<int> stackSlots;
      for (int i = 0; i < count[req]; ++i) assert(stackSlots.insert(stack[req * 32 + i]).second);
      assert(stackSlots == freeSlots);
      for (int token : selected) assert(resident.count(token));
      // No unselected survivor may be older than an evicted valid old token.
      if (oldLast[req] == ids[req]) for (int slot = 0; slot < 32; ++slot) {
        const int token = old[req * 32 + slot];
        if (token < 0 || token >= prefix || selected.count(token) || resident.count(token)) continue;
        for (int other = 0; other < 32; ++other) {
          const int survivor = old[req * 32 + other];
          if (survivor >= 0 && survivor < prefix && !selected.count(survivor) && resident.count(survivor))
            assert(oldPriority[req * 32 + slot] <= oldPriority[req * 32 + other]);
        }
      }
    }
    for (int row = 0; row < 8; ++row) for (int k = 0; k < 4; ++k) {
      const int req = owners[row], token = topk[row * 4 + k], slot = slots[row * 4 + k];
      const bool valid = row < active[0] && req >= 0 && req < 2 && request_plan::Valid(c, s, req, row, token);
      if (valid) assert(slot >= 0 && slot < 32 && tokens[req * 32 + slot] == token);
      else assert(slot == -1);
    }
  }
};

int main() {
  static_assert(sizeof(request_plan::Control) <= 1152);
  Fixture f;
  f.fill(10, true); f.run<256>(); assert(f.misses[0] == 4 && f.count[0] == 28);
  for (int row = 1; row < 4; ++row) for (int k = 0; k < 4; ++k) assert(f.slots[k] == f.slots[row * 4 + k]);
  f.run<32>(); assert(f.misses[0] == 0);
  f.fill(40, false); f.run<32>(); assert(f.count[0] == 12);
  f.fill(60, false); f.run<32>(); assert(f.count[0] == 0);
  for (int step = 0; step < 12; ++step) { f.fill(40 + (step % 5) * 16, step % 3 == 0); f.run<32>(); }
  f.clock = {255, 65535}; f.fill(300, false); f.run<32>();
  f.clock = {65536, 1LL << 40}; f.fill(400, false); f.run<32>();
  f.stable.assign(8, 20); f.fill(10, false); f.run<32>();
  f.stable.assign(8, 1024); f.ids[0] = 99; f.run<32>();
  std::reverse(f.owners.begin(), f.owners.end()); f.run<32>();
  f.active[0] = 3; f.run<32>();
  f.active[0] = 0; f.run<32>();
  f.active[0] = 8; f.table[3] = -1; f.visible[0] = 12; f.topk[5] = -1; f.topk[6] = 5000; f.run<32>();
  f.owners.assign(8, 0); f.run<32>(false); assert(f.status[0] == request_plan::BAD_ROWS && f.misses[0] == 0);
  for (int s : f.slots) assert(s == -1);
  Fixture g; g.fill(10, true); g.run<32>(); g.clock[0] = INT64_MAX; g.run<32>(false);
  assert(g.status[0] == request_plan::BAD_STATE && g.misses[0] == 0);
  Fixture h; h.fill(10, true); h.active[0] = 9; h.run<32>(false);
  assert(h.status[0] == request_plan::BAD_ROWS && h.status[1] == request_plan::BAD_ROWS);
  std::cout << "Production request planner: shared MTP, cold/warm, eviction, radix boundaries, rollback, "
               "reuse, row reordering, padding, invalid pages and error status passed\n";
}
