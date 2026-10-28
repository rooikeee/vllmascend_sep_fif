# 950PR 请求级共享 KV planner 优化

日期：2026-10-28。基线：`wzj_kv_offload_opt@fcf6cd850`。
开发分支：`perf/950pr-request-shared-planner`。

## 目标与决定

MTP3 的四行 query 属于同一请求，历史 KV、缓存映射和淘汰状态按请求/层/rank
共享，TopK 与槽位输出仍按 query 行保留。同一请求的不同 query 选中同一个 token
时只分配、搬运一次。所有行选中项的并集在本次 attention 完成前均不得淘汰。

采用持久的 `token_to_slot[B,L]` 反向映射，B是请求槽数，L是最大逻辑上下文。
MTP四行共用一张表。直接查询和CAS认领缺失token，不再建立临时hash，
也不再为了确认命中遍历C个缓存槽位。
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
C 是共享热池容量；L是最大逻辑上下文长度。

| 状态 | 类型与形状 | 生命周期 |
| --- | --- | --- |
| slot_to_token | int32 `[B,C]` | 每层持久 |
| token_to_slot | int32 `[B,L]` | 每层持久；-1=未缓存，-2=本轮已认领，非负=槽位 |
| priority | int64 `[B,C]` | 每层持久 |
| free_mask / free_stack | int32 `[B,C]` 各一张 | 每层持久 |
| free_count | int32 `[B]` | 每层持久 |
| last_req_ids / clock | int64 `[B]` | 每层持久 |
| current_slots | int32 `[R,K]` | 每轮更新，各行槽号相对请求池 |
| miss_count | int32 `[B]` | 每轮更新，inactive请求置零 |
| miss_tokens / miss_slots | int32 `[B,Q*K]` | 前miss_count项有效 |
| resident_end | int32 `[B]` | 缓存token编号上界，用于回退后缀失效 |
| status | int32 `[B]` | 每轮错误位，非零不得消费结果 |

反向表大小为4BL字节。例如B16、L131072是8 MiB/层/rank，不乘MTP四行。
反向表在创建时初始化为-1，此后不每轮全量清空。没有hash工作区和逐轮保护数组；
priority等于当前epoch表示本轮保护项。代价是持久映射容量及原子查询/更新，
性能仍需950实测。
每个 SIMT block 串行负责一个请求，不同请求并行，不存在同一请求多个 allocator
同时修改 free_count。free/miss 压缩和同龄淘汰领取使用原子计数。

## 算法

1. 根据 token_to_req 收集有效行；每行按 visible length、页表和token边界过滤。
2. 请求身份改变时扫描旧slot_to_token撤销对应反向映射、重置共享池和free栈，
   不清空整个L。要求调用者第一次使用前初始化双向表，禁止导入不一致状态。
3. stable prefix取本请求各行最小值；只遍历`[stable_prefix,resident_end)`的
   逻辑token反向表，失效缓存的speculative suffix并归还free栈。resident_end
   是保守上界，深回退时范围可较大，但不需要为正常命中查询全扫热缓存。
4. 每个有效TopK token执行`CAS(token_to_slot[token], -1, -2)`。
   返回-1的一方唯一拥有miss并追加输出；返回-2的重复项不重复分配、不自旋；
   返回非负槽位的是hit，原子更新其priority为当前epoch。全部行处理完成后barrier。
   free_count足够则直接从栈批量领取。
5. 不足时，对未保护项做按字节的优先级radix选择，找第E小优先级。
   优先领取严格小于阈值项；阈值相等项原子领取剩余名额。只释放恰好E个。
6. 淘汰时先撤销旧token的反向映射；从free栈分配M个槽位，更新slot_to_token、
   priority、free_mask，将本轮miss的反向表项由-2更新为槽位。失败退出前清理本轮-2。
7. 四行各自读取反向表，生成current_slots；重复token输出相同槽位。
8. 同stream执行请求级transfer，再执行attention。planner更新的映射是计划，
   不是KV已就绪的证明；任何后续planner不得越过尚未完成的transfer读取该状态。

第一版要求 `C >= Q*K`，保证合法输入的所有选中项可同时放入共享池。
不能把union超过C时的部分结果交给attention。超出每请求Q行等动态错误写status，
该请求miss_count为零、current_slots为-1；请求执行方必须将其视为失败。

正常hit/miss判定和各行输出为O(QK)，不需要O(C)匹配。回退失效另有O(S)，S为
上述后缀区间长度。淘汰按优先级实际使用的字节数p扫描，O(pC)，仅缺空位时执行。
请求替换为O(C)，首次分配反向表为O(BL)。没有全量LRU顺序写回。
每轮收集请求行另有O(BR)的简单元数据扫描，第一版可见且不隐瞒。

## 预取的后续位置

本轮先解决miss判定和共享分配，不启用候选预取。后续在block score发布后按阈值
筛候选，查询resident状态，预算内写暂存区。不能提前覆盖最终union可能仍使用的
槽位；pending/ready、跨query去重与请求epoch都需要单独验证。

## Graph 与验证

所有持久表和输出在capture前分配，replay只更新内容。每轮重置miss_count，使用新
epoch保护命中项；双向映射、resident_end、free栈及priority跨replay持久。padding行输出清-1，inactive
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
  替代SIMT原语，验证union miss、保护项、优先级顺序、双向映射和free栈一致性。
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

反向表版变更了ACLNN和Torch ABI：必须同时重编设备包与Torch桥接，不能混用此前
临时hash版的`.so`或算子包。已移走旧csrc/build进行干净重建，核对新生成的ACLNN
包含tokenToSlot和residentEnd，Torch schema为21个tensor加4个属性。
新950PR二进制名为`SparseKvRequestPlan_71ae9c0c8ae1481f46baec0adb928f6b`，
同包包含SparseKvTransfer。增量构建曾复用旧接口，因此README要求旧checkout清理构建目录。

- 生产算法CPU并发测试通过，启用ASan/UBSan，覆盖256线程冷启动及32线程多轮状态转换。
  包含跨行重复、全命中、部分miss、空闲不足、radix字节边界、回退、请求替换、行重排、
  padding、非法token/page以及错误状态。另增加双向映射全量核验、最大token边界、
  pending失败清理、256线程重复命中及40轮随机状态转换。修复了一处错误标志更新
  导致的barrier分歧。
- 原有planner oracle、miss选择和warp scan共7个相关测试通过。
- CANN9.1生成950PR_9599的VectorCore设备二进制成功；上传profile原生记录为9589，
  目标机器仍须按其实际950PR型号重新构建/验证。host注册、tiling、ACLNN已编译。
- 最小Torch测试桥接在Torch2.10/torch-npu环境编译并加载成功。
- 本机没有950PR设备，没有执行新算子的设备精度、Graph replay或性能测试。
- `format.sh ci`因环境缺少pre-commit未执行；Python语法和diff空白检查通过。
