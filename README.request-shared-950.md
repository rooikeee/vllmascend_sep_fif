# 950PR 融合 indexer 优化与验收

> 当前分支新增 score block 后只读查询实验，ABI 增加 `probe_thresholds`、`probe_stats`、
> `probe_mode`；默认关闭。该实验不做搬运，只编译融合算子，入口见
> [block miss 查询实验](docs/950pr_fused_block_miss_probe.md)。下文真实 KV 验收仍需 transfer。

**只优化融合算子 `QuantLightningIndexerSparseKv`，测试和性能验收也只针对融合算子。**
不再以独立 `SparseKvPlan` / `SparseKvRequestPlan` 作为优化或验收入口。
`SparseKvTransfer` 仅作为已有 KV 搬运组件，在融合输出的链路检查中复用。

## 当前实现

一次 mixed AIC/AIV kernel 内执行：

```text
QLI score → 每行 TopK 写出 → 全 AIV SyncAll → 请求级共享 planner
```

planner 读取该次 QLI 产生的 TopK，一个 AIV 独占一个请求。MTP3 的四行共用：

- 持久 `token_to_slot[B,L]` 反向表，直接查询命中，CAS 去重 miss。
- `slot_to_token[B,C]`、free 标记和 free 栈；空位够用时直接分配。
- int64 时间戳优先级；保护所有行选中项，只在空位不足时选择淘汰项。
- 请求身份、rollback 后缀失效及错误状态；每轮清理 padding 输出。

删除融合路径的逐行临时 hash、完整 LRU 重排和 planner GM scratch。
planner 控制区复用已经结束使用的 TopK UB slab，动态 UB 为小 T 150528 字节、
大 T 186880 字节。全 AIV（包括空闲核和空输入核）都参加同步；不在逐行回调内
等待其他核。TopK 此版本经 GM 交接，不再声称 planner 直接读取每行 UB TopK。
全局同步的实际代价和 GM 可见性需要 950 真机验证。

## 接口与生命周期

**融合算子的 ACLNN/Torch ABI 已更新，设备包和 Torch 桥接必须一起重编。**
输入保留 QLI 张量、请求元数据，替换旧 `lru_slots` / `compact_workspace` 为
`priority`、`free_mask`、`free_stack`、`free_count`、`clock`、`token_to_slot`、
`status`、`resident_end`，并新增 `max_queries` 属性。

状态按请求存储，输出槽位按 query 行存储；miss 清单为 `[B,max_queries*2048]`。
MTP3 设置 `max_queries=4`，要求共享容量 `C >= max_queries*2048` 且按 128 对齐。
反向表在创建时初始化为 -1，其他状态由
`vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_fused_indexer.py`
中的 `SparseKVFusedIndexer` 分配。capture 前分配状态，replay 只更新内容。

调用次序为融合算子 → 原有 transfer → attention，同 stream 执行。
`current_slots` 是相对请求池的槽号，物理地址必须按 `request*C+slot` 计算。
`status != 0` 时不得消费当前计划。`check_status()` 仅用于 capture/计时外检查。

本次交付范围是融合算子及其验收入口。原 serving 使用按行热池，其请求池布局、
IndexCache 跨层状态与失败传播尚未迁移；不能把新 ABI 直接接到旧按行调用上。旧 manager 暂不启用此融合路径，继续使用
原按行服务路径；直接调用其旧融合接口会明确报错，避免布局混用。

## 编译

目标：Ascend 950PR、CANN 9.1、匹配的 Torch/torch-npu。先 source CANN 的
`set_env.sh` 并激活 Python 环境。工具链支持目标时，910 主机可以交叉编译 950
二进制；**910 不能执行此算子，交叉编译成功不等于 950 验证通过。**

下面的 `--ops` 列出两个设备算子，各自负责不同步骤：

| 编译目标 | 用途 |
| --- | --- |
| `quant_lightning_indexer_sparse_kv` | 本次优化的融合算子：score、TopK、反向表判定、去重、free 栈分配和淘汰，在一次 kernel 内完成。 |
| `sparse_kv_transfer` | 已有的搬运算子：按融合算子输出的 miss 清单，将真实 KV 从 Host 搬回 NPU。 |

把 transfer 一起打包，是因为下面的设备验收还检查真实 KV 内容；融合算子本身只输出
索引、槽位和搬运计划。这里没有编译或调用独立 `sparse_kv_plan` / `sparse_kv_request_plan`。
若只编译融合算子，可把 `--ops` 改成 `quant_lightning_indexer_sparse_kv`；运行下面的
完整设备验收时，环境仍须提供配套的 `SparseKvTransfer`。

`build_fused_indexer_test_binding.py` 编译的是 Host 侧 Torch 调用桥接，用于调用上述
设备算子，不是另一个设备 kernel。

从旧 checkout 更新时，先将 `csrc/build` 移到一个不存在的备份目录，避免复用旧 ABI。
从仓库根目录执行：

```bash
mkdir -p build/fused-logs
set -o pipefail
(
  cd csrc
  CMAKE_BUILD_PARALLEL_LEVEL=4 bash build.sh --pkg \
    --ops=quant_lightning_indexer_sparse_kv,sparse_kv_transfer --soc=ascend950 -j4
) 2>&1 | tee build/fused-logs/device-build.log

bash csrc/build/cann-ops-transformer-custom_linux-aarch64.run \
  --quiet --install-path="$PWD/build/fused-indexer-opp"
export FUSED_INDEXER_VENDOR="$PWD/build/fused-indexer-opp/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$FUSED_INDEXER_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$FUSED_INDEXER_VENDOR/op_api/lib:$FUSED_INDEXER_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MAX_JOBS=4 python tools/build_fused_indexer_test_binding.py \
  2>&1 | tee build/fused-logs/bridge-build.log
```

vendor 目录以安装器实际生成路径为准。测试桥接只注册融合算子和已有 transfer，
在独立进程加载，不能与注册相同名称的完整插件扩展混用。
本仓库 `ascend950` 生成目标为 `Ascend950PR_9599`，目标机器需确认实际型号兼容性。

## 测试

CPU 测试执行融合算子实际使用的请求状态算法，包含并发原子和 barrier、ASan/UBSan、
双向映射、free 栈、保护及淘汰、请求复用和回退；另外核对 host/device tiling ABI。
这属于融合算子的内部测试，不调用独立 planner。

```bash
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite static
```

950 真机验收只启动融合 QLI 和已有 transfer。用可精确预测的分组编码输入验证真实
score/TopK，检查四行去重和请求级 KV 回搬。覆盖冷启动、全命中、满池淘汰、rollback、
请求替换、行重排、padding、非法页、空输入和动态错误，以及连续 Graph replay。
每轮核对双向映射、free 栈、miss 集合和 KV 内容。错误状态用融合调用测试且不搬运。

```bash
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite device \
  --device 0 \
  --library build/fused-indexer-test-binding/fused_indexer_test_ops.so \
  --output test-results/fused-950-r1
```

输出目录必须不存在。仅全部检查完成后写 `result.json` 的 `passed=true`。
还记录 B2/Q4/K2048/C8192/L32768 全命中融合调用时延（含 Python 入口的页表更新），
不把它冒充完整模型或纯 kernel 性能；不与容量预算不同的旧按行路径计算加速比。
旧逐行 ABI 的历史测试和三路径 benchmark 不属于本次验收入口。

## 本机验证记录（2026-10-28）

- 融合算子 CPU 验收 16 项通过，包含实际生产算法的 ASan/UBSan 并发测试。
- Python 调用兼容保护和 CLI 回归各 8 项通过。
- CANN 9.1 为 `Ascend950PR_9599` 生成 mixed AIC/AIV 二进制和安装包成功；
  kernel 元数据标记 `intercoreSync=1`。这不是设备同步正确性的证明。
- 最小 Torch 桥接编译、加载成功；修改的 Python 文件通过 Ruff 和语法检查。
- 本机为 910，未执行融合算子的 950 精度、Graph 或性能验收。
- `bash format.sh ci` 因缺少 pre-commit 未运行完成，不能声称完整格式检查通过。
