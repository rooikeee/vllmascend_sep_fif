# 950 融合 QLI：score block 后只读 miss 查询

分支：`perf/950pr-fused-block-miss-probe`。基线：`25baece95`。
目标是在每个 AIV score block 的间隙加入阈值筛选和反向表读取，测量额外开销。
本实验只修改 `QuantLightningIndexerSparseKv`；不新增设备算子。

## 执行位置与语义

```text
AIC QK → AIV scale/reduce → score UB → 发起 score GM 写出
                               → 释放当前 QK 缓冲给 Cube
                               → 阈值比较 / 只读反向表查询
继续下一个 score block …
最终 TopK → 原有全 AIV 同步 → 原有请求级规划与分配
```

每个 S2 block 为 128 token，同一个 AIV 本次处理 1 或 2 行 query。
此处“top-p”采用用户指定的 `score > threshold` 候选语义，不做概率累计采样。
阈值是外部提供的 float32 `[T]` 张量，可在 Graph replay 前原地更新。
默认阈值为 `+inf`。本轮不生成或更新阈值，也不复制 ECHO 的预取预算、队列或搬运。

QLI 当前 score UB 保存 BF16 sortable key。查询先还原为 BF16 对应的 FP32 值，再
与阈值严格比较；该分数与最终 TopK 所用分数一致。等于阈值的 token 不查询。
NaN 比较不通过，`+inf` 阈值不产生候选。

候选排除 causal mask、visible length、最大上下文以外 token 和 inactive 行。
对每个候选读取一次 `token_to_slot[request, token]`。映射为合法槽号、请求身份一致、
token 处于该请求各行最小 stable prefix 内，才算 hit；否则算 miss。
在评分阶段尚未做请求替换/回退清理，因此必须检查身份和 stable prefix。
这些元数据每个 S1 tile 读取一次。反向表 load 标记为 volatile，冷请求和回退候选也
实际查表，避免被编译器短路掉，便于测量负载。

此阶段假设上一轮 transfer 已按同 stream 顺序完成，读取调用开始时的缓存快照。
不检查 Host 页表、不做 CAS/去重、不更新 priority、不分配或回收槽位、不产生预取、
不改最终 miss 清单。跨 query 的同一 token 会分别查询并计数，统计不是请求 union。
最终 TopK planner 保留原有页表检查、CAS 去重和分配/淘汰，算法未修改。

## 三档对照与统计

新增属性 `probe_mode`：

| 值 | 内容 |
| --- | --- |
| 0 | 关闭。没有 probe VF 调用，不写 probe_stats；作为同一二进制内的基线。 |
| 1 | 阈值筛选和统计；不读取反向表，用于估算筛选、VF 切换和统计的开销。 |
| 2 | 在 mode 1 基础上查询反向表、统计 miss。 |

新增 `probe_thresholds: float32[T]` 和 `probe_stats: int32[T,3]`。
统计列为本行处理的 score block 数、候选次数、miss 次数。mode 1 的 miss 列为 0。
启用时 padding/空输入行输出 0；关闭时统计内容保持不变，调用者不要读取成新结果。

统计是为了保留可观测结果、避免查表被优化删除。每线程在已空闲的 TopK UB 中累计，
完成本行全部 S2 block 后合并并写出一次，不在每个 token 上做全局 atomicAdd。
最终 TopK 和 planner 随后复用同一 UB，UB 总量不增加。
因此 mode 2 与 mode 0 的差值包含 VF 切换、筛选、元数据、查表和统计；
mode 2 与 mode 1 的差值更接近查表及命中分类的增量，但不能视为纯 load 指令延迟。
同一二进制的 mode 0 也不能自动等同于旧提交的编译产物性能。

## 编译与运行

ABI 已变化，需要同时重编融合设备包和 Torch 桥接。先 source CANN 9.1 环境、激活
匹配的 Python/Torch 环境；从旧 checkout 更新时将 `csrc/build` 移到未占用的备份目录。
本实验不运行 KV 搬运，**只需编译融合算子**：

```bash
mkdir -p build/probe-logs
set -o pipefail
(
  cd csrc
  CMAKE_BUILD_PARALLEL_LEVEL=4 bash build.sh --pkg \
    --ops=quant_lightning_indexer_sparse_kv --soc=ascend950 -j4
) 2>&1 | tee build/probe-logs/device-build.log

bash csrc/build/cann-ops-transformer-custom_linux-aarch64.run \
  --quiet --install-path="$PWD/build/block-probe-opp"
export BLOCK_PROBE_VENDOR="$PWD/build/block-probe-opp/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$BLOCK_PROBE_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$BLOCK_PROBE_VENDOR/op_api/lib:$BLOCK_PROBE_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MAX_JOBS=4 python tools/build_fused_indexer_test_binding.py

python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite static
python tools/bench_fused_block_miss_probe.py \
  --device 0 --library build/fused-indexer-test-binding/fused_indexer_test_ops.so \
  --output test-results/block-probe-b2-q4 \
  --batch 2 --queries 4 --context 32768 --capacity 8192 \
  --thresholds=inf,1,0.5,-1 --iterations 30 --rounds 3 --trace
```

安装目录层级以安装器实际生成路径为准。桥接包含已有 transfer 的注册，但本脚本
只调用融合算子，完全不运行 transfer。不能在同一进程加载重复注册的插件扩展。
本仓库生成的目标为 `Ascend950PR_9599`，执行前确认目标 950 兼容。

输入是精确编码的 score 0/1：默认每行一个 2048-token 高分组。
`inf` / `1` 无候选，`0.5` 只查高分组，`-1` 查询所有可见 token。
可用 `--batch` / `--queries` / `--context` / `--capacity` 改变形状；MTP3 用 queries=4。
建议覆盖 B1/Q1/L2177 的尾块、B2/Q4/L32768，以及 B17/Q4/L32768 的两行/AIV 分支。

脚本先验证三档下最终 TopK 完全相同、最终 miss 集合正确、双向映射和 free 栈一致，
再检查 probe 统计，覆盖冷缓存、热缓存、请求替换、回退、padding、全空/部分空输入。
性能在每档独立 capture 的 Graph 上测量。每次 replay 前恢复相同缓存状态，状态恢复
及统计初始化在 NPU event 计时之外；关闭 Python 页表更新，不计搬运或其他算子。
各轮轮换三档测量顺序，保存原始样本、中位数、相对关闭档的开销和候选/miss 数。
状态恢复会影响 cache 温度，因此结果表示该受控 workload，不直接代替 serving。
脚本中的冷/热指反向表等元数据状态；本轮不分配、搬运或核验真实 KV 热池内容。

`--trace` 输出每档的 Torch NPU trace，帮助确认只有融合 kernel。
融合 kernel 总时延不能直接等同于 AIV 忙时；要判断插入操作是否占用了原本空隙、
是否让 Cube 等待，仍需在 950 上采集指令级时间线。当前 910 主机只能交叉编译和执行
CPU 测试，不能得出 AIV 减速百分比。

## 本机验证（2026-10-28）

- 融合版静态验收 20 项通过；新 probe 使用实际生产头执行 128 线程 CPU 模拟，
  启用 ASan/UBSan，覆盖阈值边界、NaN/无穷、causal/visible mask、请求身份、回退、
  inactive 行和超过 int32 范围的 block 偏移，并核对持久状态未被修改。
- CANN 9.1 为 `Ascend950PR_9599` 交叉编译融合 mixed kernel 和安装包成功。
- 更新后的 Torch 桥接编译、加载成功；Python Ruff、语法和 diff 空白检查通过。
- `format.sh ci` 仍因环境缺少 pre-commit 未完成。
- 950 真机正确性、Graph replay 和 AIV/融合 kernel 性能尚未执行，没有减速数据。
