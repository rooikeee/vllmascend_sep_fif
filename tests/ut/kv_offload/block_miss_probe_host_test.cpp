// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <atomic>
#include <barrier>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>
#define __gm__
#define __ubuf__
#define __simt_callee__
#define __aicore__
thread_local std::barrier<>* blockBarrier;
inline void asc_syncthreads() { blockBarrier->arrive_and_wait(); }
inline int32_t asc_atomic_add(int32_t* p, int32_t v) { return std::atomic_ref<int32_t>(*p).fetch_add(v); }
inline int32_t asc_atomic_cas(int32_t* p, int32_t expected, int32_t v) {
  std::atomic_ref<int32_t>(*p).compare_exchange_strong(expected, v); return expected;
}
template<typename T> inline T asc_atomic_max(T* p, T v) {
  std::atomic_ref<T> ref(*p); T old = ref.load();
  while (old < v && !ref.compare_exchange_weak(old, v)) {}
  return old;
}
#include "block_miss_probe.h"

constexpr int ROWS = 4, REQUESTS = 2, LENGTH = 384, STRIDE = 256, TILE = 128;
uint16_t Encode(float score) {
  auto bits = static_cast<uint16_t>(std::bit_cast<uint32_t>(score) >> 16);
  if (bits == 0x7fc0) bits = 0xffff;
  return bits ^ ((bits & 0x8000) ? 0xffff : 0x8000);
}
struct Fixture {
  request_plan::Shape shape{REQUESTS, ROWS, 4, 16, LENGTH, 3, 6, 128, 4};
  std::vector<int64_t> ids{11,22}, last{11,22};
  std::vector<int32_t> stable{LENGTH,LENGTH,LENGTH,LENGTH}, visible{LENGTH,LENGTH,LENGTH,LENGTH};
  std::vector<int32_t> owners{0,0,1,1}, active{ROWS}, reverse=std::vector<int32_t>(REQUESTS*LENGTH,-1);
  std::vector<float> thresholds{0.5f,1.0f,0.5f,1.0f};
  std::vector<int32_t> stats=std::vector<int32_t>(ROWS*3,-99);
  std::vector<uint16_t> scores=std::vector<uint16_t>(3*2*STRIDE, Encode(1.0f));
  Fixture() { for(int t=0;t<LENGTH;t+=2) reverse[t]=t%16; }
  request_plan::Context context() {
    request_plan::Context c{}; c.reqIds=ids.data(); c.lastIds=last.data(); c.stable=stable.data();
    c.visible=visible.data(); c.rowToReq=owners.data(); c.active=active.data(); c.tokenToSlot=reverse.data();
    return c;
  }
  void run(int mode=2,int row=0,int rows=2,int end=LENGTH,int blocks=3,bool nullReverse=false,int64_t start=0) {
    auto c=context(); if(nullReverse)c.tokenToSlot=nullptr;
    const auto before=reverse; const auto oldStable=stable; const auto oldIds=ids; const auto oldLast=last;
    block_miss_probe::Control control{};
    std::barrier barrier(block_miss_probe::THREADS);
    std::vector<std::thread> threads;
    for(uint32_t lane=0;lane<block_miss_probe::THREADS;++lane) threads.emplace_back([&,lane]{
      blockBarrier=&barrier;
      for(int block=0;block<blocks;++block) {
        block_miss_probe::ProcessBlock(&control,c,shape,scores.data()+block*2*STRIDE,thresholds.data(),stats.data(),
            mode,row,rows,start+block*TILE,STRIDE,end,block==0,block==blocks-1,lane);
        barrier.arrive_and_wait(); // SIMD/SIMT call completion between score blocks
        if(block+1<blocks) assert(stats[row*3]==-99); // no per-block GM counters
      }
    });
    for(auto& thread:threads)thread.join();
    assert(reverse==before && stable==oldStable && ids==oldIds && last==oldLast);
  }
};
int main() {
  for(float v:{-8.0f,-1.0f,-0.0f,0.0f,0.5f,1.0f,8.0f,
               std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})
    assert(std::bit_cast<uint32_t>(block_miss_probe::DecodeScore(Encode(v)))==std::bit_cast<uint32_t>(v));
  assert(std::isnan(block_miss_probe::DecodeScore(Encode(std::numeric_limits<float>::quiet_NaN()))));
  { Fixture f;f.run();assert((f.stats==std::vector<int32_t>{3,384,192,3,0,0,-99,-99,-99,-99,-99,-99})); }
  { Fixture f;f.run(1,0,2,LENGTH,3,true);assert(f.stats[1]==384 && f.stats[2]==0); }
  { Fixture f;f.run(0,0,2,LENGTH,3,true);assert(f.stats==std::vector<int32_t>(ROWS*3,-99)); }
  { Fixture f;f.last[0]=-1;f.run();assert(f.stats[2]==384); }
  { Fixture f;f.stable[1]=128;f.run();assert(f.stats[2]==320); } // min across request rows
  { Fixture f;f.visible[0]=130;f.run();assert(f.stats[1]==130 && f.stats[2]==65); }
  { Fixture f;f.shape.maxToken=257;f.run();assert(f.stats[1]==257 && f.stats[2]==128); }
  { Fixture f;f.run(2,0,2,129);assert(f.stats[1]==129 && f.stats[2]==64); }
  { Fixture f;f.run(2,0,2,-1);assert(f.stats[1]==0 && f.stats[4]==0); }
  { Fixture f;f.active[0]=1;f.run();assert(f.stats[3]==0 && f.stats[4]==0); }
  { Fixture f;f.active[0]=ROWS+1;f.run();assert(f.stats[0]==0 && f.stats[1]==0); }
  { Fixture f;f.owners[0]=-1;f.run();assert(f.stats[0]==0 && f.stats[1]==0); }
  { Fixture f;f.owners[0]=REQUESTS;f.run();assert(f.stats[0]==0 && f.stats[1]==0); }
  { Fixture f;f.ids[0]=-1;f.run();assert(f.stats[0]==0 && f.stats[1]==0); }
  { Fixture f;f.reverse[0]=-2;f.reverse[2]=16;f.run();assert(f.stats[2]==194); }
  { Fixture f;f.thresholds[0]=std::numeric_limits<float>::infinity();f.run();assert(f.stats[1]==0); }
  { Fixture f;f.thresholds[0]=std::numeric_limits<float>::quiet_NaN();f.run();assert(f.stats[1]==0); }
  { Fixture f;f.thresholds[0]=-std::numeric_limits<float>::infinity();f.run();assert(f.stats[1]==384); }
  { Fixture f;f.run(2,2,1,17,1);assert(f.stats[7]==17 && f.stats[8]==17); }
  { Fixture f;f.thresholds[1]=0.5f;f.visible[0]=1;f.visible[1]=2;f.run();
    assert(f.stats[1]==1 && f.stats[2]==0 && f.stats[4]==2 && f.stats[5]==1); }
  { Fixture f;f.run(2,0,2,LENGTH,1,true,static_cast<int64_t>(INT32_MAX)+1);
    assert(f.stats[1]==0 && f.stats[2]==0); }
  std::cout<<"block miss probe: read-only state, masks, thresholds and lane counts passed\n";
}
