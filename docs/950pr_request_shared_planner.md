# 950PR 请求级共享 KV planner 优化

日期：2026-10-28。基线：`wzj_kv_offload_opt@fcf6cd850`。
开发分支：`perf/950pr-request-shared-planner`。

## 目标与决定

MTP3 的四行 query 属于同一请求，历史 KV、缓存映射和淘汰状态按请求/层/rank
共享，TopK 与槽位输出仍按 query 行保留。同一请求的不同 query 选中同一个 token
时只分配、搬运一次。所有行选中项的并集在本次 attention 完成前均不得淘汰。

第一版不增加按完整上下文分配的 `token_to_slot[R,L]` 稠密表。给请求内全部
TopK 建临时 hash，扫描共享 `slot_to_token` 一次，在 hash 中记录命中槽位。
删除新路径中的 `lru_slots` 及其完整分类/重排，改为槽位时间戳、free 标记、
free 栈、free_count。空闲够用时不运行淘汰选择；不够时只补足缺口。

优先级使用 int64 请求轮次，相同轮次的 token 不定义内部 LRU 顺序。
新命中和装入项被标记为当前轮次；淘汰只从未保护的有效槽位中选择最旧项。
这改变替换策略，需要测 miss 率和完整链路，不能仅比较 planner 时间。

## 第一版执行边界

实现单独的 950 AscendC/SIMT `SparseKvRequestPlan`，在全部 TopK 发布后执行。
旧 inline planner 按 query 行运行，不能在它内部直接等待其他 AIV 发布全请求
TopK，否则跨核调度可能死锁。第一版用 kernel 边界提供全请求可见性，保留旧路径
作为对照。后续再研究无死锁的 QLI 融合，不把新方案的分阶段执行称为已融合。

第一版提供算子及 Python 实验入口，不自动替换现有 serving manager。
请求级物理池必须使用 `request * C + slot`；attention 页表的每行必须按
`token_to_req[row]` 指向请求池，禁止沿用 `row * C`。
复用现有 `SparseKvTransfer` 时将 transfer 的行定义为请求，传入 identity
请求索引和请求级 miss 输出，保持其内部地址公式不变。
IndexCache 跨层复用、serving 中 draft 行压缩等接线要在本算子通过 950 验证后完成。

## 数据结构

B 是最大请求槽数；R 是最大 query 行数；Q 是单请求最大行数；K 是每行TopK；
C 是共享热池容量；H 是不小于 `2*Q*K` 的2次幂，至少32。

| 状态 | 类型与形状 | 生命周期 |
| --- | --- | --- |
| slot_to_token | int32 `[B,C]` | 每层持久 |
| priority | int64 `[B,C]` | 每层持久 |
| free_mask / free_stack | int32 `[B,C]` 各一张 | 每层持久 |
| free_count | int32 `[B]` | 每层持久 |
| last_req_ids / clock | int64 `[B]` | 每层持久 |
| current_slots | int32 `[R,K]` | 每轮更新，各行槽号相对请求池 |
| miss_count | int32 `[B]` | 每轮更新，inactive请求置零 |
| miss_tokens / miss_slots | int32 `[B,Q*K]` | 前miss_count项有效 |
| workspace | int32 `[B,2H+C]` | 固定地址GM临时hash及保护标记 |
| status | int32 `[B]` | 每轮错误位，非零不得消费结果 |

无 O(L) 的持久映射。临时 hash 的 H 会随四行并集容量扩大，第一版采用 GM
工作区；这可能增加原子/访存成本，需用 950 数据验证，不能预先承诺加速。
每个 SIMT block 串行负责一个请求，不同请求并行，不存在同一请求多个 allocator
同时修改 free_count。free/miss 压缩和同龄淘汰领取使用原子计数。

## 算法

1. 根据 token_to_req 收集有效行；每行按 visible length、页表和token边界过滤。
2. 请求身份改变时初始化共享池；对所有本请求TopK建立去重hash。
3. 扫描一次C个槽位：失效 speculative suffix，归还free栈；命中项写hash槽位，
   标记保护并更新时间戳。stable prefix取本请求各行的最小值，采用保守失效。
4. 从hash压缩不同miss。free_count足够则直接从栈批量领取。
5. 不足时，对未保护项做按字节的优先级radix选择，找第E小优先级。
   优先领取严格小于阈值项；阈值相等项原子领取剩余名额。只释放恰好E个。
6. 从free栈分配M个槽位，更新slot_to_token、priority、free_mask与hash槽位。
7. 四行各自查询hash，生成current_slots；重复token输出相同槽位。
8. 同stream执行请求级transfer，再执行attention。planner更新的映射是计划，
   不是KV已就绪的证明；任何后续planner不得越过尚未完成的transfer读取该状态。

第一版要求 `C >= Q*K`，保证合法输入的所有选中项可同时放入共享池。
不能把union超过C时的部分结果交给attention。超出每请求Q行等动态错误写status，
该请求miss_count为零、current_slots为-1；请求执行方必须将其视为失败。

临时hash构建平均O(QK)，命中匹配O(C)，各行输出平均O(QK)。淘汰按优先级
实际使用的字节数p扫描，O(pC)，仅缺空位时执行。没有全量LRU顺序写回。
每轮收集请求行另有O(BR)的简单元数据扫描，第一版可见且不隐瞒。

## 预取的后续位置

本轮先解决miss判定和共享分配，不启用候选预取。后续在block score发布后按阈值
筛候选，查询resident状态，预算内写暂存区。不能提前覆盖最终union可能仍使用的
槽位；pending/ready、跨query去重与请求epoch都需要单独验证。

## Graph 与验证

所有持久表、输出和workspace在capture前分配，replay只更新内容。每轮重建hash、
保护标记和miss_count；free栈及priority跨replay持久。padding行输出清-1，inactive
请求miss_count清零。请求替换、行重排、长度变化、rollback、全命中、部分miss、
无空位、空闲充足、重复token、非法页必须覆盖。host buffer生命周期覆盖所有replay。

本机只有910，不能代替950PR验证。CPU测试仅验证算法与同步模型，不证明950编译、
GM可见性、性能或模型正确性。验收需950PR/CANN9.1编译、真实capture与多次replay、
KV内容核验，再比较旧per-row planner与新request-shared planner的实际搬运字节和时延。

## 实现与运行入口

- `csrc/attention/sparse_kv_request_plan/`：Ascend950-only算子、host验证、Torch适配层。
- `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_request_plan.py`：
  持久缓冲、请求级transfer及各query共享attention页表的实验接口。
- `tests/ut/kv_offload/request_plan_host_test.cpp`：执行生产算法，CPU原子及barrier
  替代SIMT原语，验证union miss、保护项、优先级顺序和free栈一致性。
- `tools/check_sparse_kv_request_plan_npu.py`：真实Host注册、请求级KV回搬、capture
  和连续replay，核验KV内容与固定地址。显式拒绝910设备。

```bash
# CPU验证，不需要导入vLLM/NPU。
python -m pytest --confcutdir=tests/ut/kv_offload \
  tests/ut/kv_offload/test_sparse_kv_request_plan.py -q

# 950构建环境，单独编译新算子包；正式使用还需已有SparseKvTransfer包及Torch扩展。
cd csrc
bash build.sh --pkg --ops=sparse_kv_request_plan --soc=ascend950 -j4
cd ..
# 按仓库正常方式重建含新注册入口的Torch扩展后，在950PR执行：
python tools/check_sparse_kv_request_plan_npu.py \
  --device 0 --library /path/to/rebuilt/vllm_ascend_C.so \
  --output test-results/request-plan-950-new
```

也可以在独立测试进程中使用最小Torch桥接，避免重建整个插件（不能与已加载相同
`_C_ascend`注册的正式扩展同时使用）：

```bash
python tools/build_request_plan_test_binding.py
python tools/check_sparse_kv_request_plan_npu.py \
  --library build/request-plan-test-binding/request_plan_test_ops.so \
  --output test-results/request-plan-950-bridge
```

测试前必须加载同时包含`SparseKvRequestPlan`和`SparseKvTransfer`的自定义算子包，
并设置其`ASCEND_CUSTOM_OPP_PATH`及op_api动态库路径。仅编译测试桥接不生成设备算子。

没有修改serving默认后端。旧按行planner与新请求级planner的淘汰顺序不等价，
测试比较有效token集合、槽位/KV一致性及一次搬运语义，不要求两者槽号相同。
部署切换前必须接好请求身份/页表生命周期、IndexCache跨层状态和status失败传播。

## 2026-10-28 验证记录

- 生产算法CPU并发测试通过，启用ASan/UBSan，覆盖256线程冷启动及32线程多轮状态转换。
  包含跨行重复、全命中、部分miss、空闲不足、radix字节边界、回退、请求替换、行重排、
  padding、非法token/page以及错误状态。修复了一处错误标志更新导致的barrier分歧。
- 原有planner oracle、miss选择和warp scan共7个相关测试通过。
- CANN9.1生成950PR_9599的VectorCore设备二进制成功；上传profile原生记录为9589，
  目标机器仍须按其实际950PR型号重新构建/验证。host注册、tiling、ACLNN已编译。
- 最小Torch测试桥接在Torch2.10/torch-npu环境编译并加载成功。
- 本机没有950PR设备，没有执行新算子的设备精度、Graph replay或性能测试。
- `format.sh ci`因环境缺少pre-commit未执行；Python语法和diff空白检查通过。
