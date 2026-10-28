# 950PR 请求级共享 miss planner

分支：`perf/950pr-request-shared-planner`。基线：`wzj_kv_offload_opt@fcf6cd850`。

发布位置：`rooikeee/vllmascend_sep_fif` fork；上游为`AmourSec/vllmascend_sep_fif`。

```bash
git clone --single-branch --branch perf/950pr-request-shared-planner \
  https://github.com/rooikeee/vllmascend_sep_fif.git vllmascend-request-shared
cd vllmascend-request-shared
```

这版新增实验算子 `SparseKvRequestPlan`，供950PR上编译、正确性检查和后续性能测量。
MTP3四行共享请求级持久`token_to_slot[B,L]`反向表，直接查hit/miss，用CAS去重，
同一个token只搬一次。已经删除临时hash及为匹配命中执行的全缓存扫描，
用free表/栈和时间戳优先级替换完整LRU顺序维护。

**反向表版更改了算子ABI，拉取更新后必须同时重编设备包和Torch桥接。**
此前临时hash版的二进制不能与新代码混用。

**目前是 TopK 后的独立 planner，不是已替换 serving 的融合 QLI。** 原服务仍走旧
路径，直接启动原服务不会测到这版。第一版先验证请求级状态、分配和真实KV回搬；
QLI融合、IndexCache跨层接线、服务集成和预取尚未完成。

## 环境

- Ascend950PR / CANN9.1，使用目标机原有匹配的Torch、torch-npu环境。
- Python3.11+、C++20编译器（CPU测试）、C++17工具链（Torch桥接）、cmake、ninja、patch。
- CPU测试需pytest；设备测试需torch和torch-npu。不需要先安装整个vLLM插件。
- 所有命令从本仓库根目录执行，构建之前先source目标机CANN的`set_env.sh`并激活Python环境。
- 本仓库`--soc=ascend950`构建脚本当前生成`Ascend950PR_9599`二进制；原profile写的是
  `9589`，请核对目标机型号与CANN兼容性，不把交叉编译成功当成目标卡运行成功。

## 1. 编译两个设备算子

同时编译planner和回搬算子，避免只安装planner后找不到transfer。

如果是在旧版本checkout上更新，先移走旧`csrc/build`再构建。本仓库增量构建可能
保留旧op-info/ACLNN接口和设备`.o`，日志显示成功也不代表接口已经更新。例如：

```bash
mkdir -p build
# 仅在csrc/build存在、备份目标不存在时执行：
mv csrc/build build/csrc-before-direct-map
```

```bash
set -o pipefail
mkdir -p build/request-plan-logs
(
  cd csrc
  bash build.sh --pkg --ops=sparse_kv_request_plan,sparse_kv_transfer --soc=ascend950 -j4
) 2>&1 | tee build/request-plan-logs/device-build.log
```

成功后生成 `csrc/build/cann-ops-transformer-custom_linux-aarch64.run`。
安装到独立目录；不要覆盖正在使用的正式算子包：

```bash
bash csrc/build/cann-ops-transformer-custom_linux-aarch64.run \
  --quiet --install-path="$PWD/build/request-plan-opp"

export REQUEST_PLAN_VENDOR="$PWD/build/request-plan-opp/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$REQUEST_PLAN_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$REQUEST_PLAN_VENDOR/op_api/lib:$REQUEST_PLAN_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果安装器在目标环境中产生不同层级，以实际包含`op_api/`和`op_impl/`的vendor目录
作为`REQUEST_PLAN_VENDOR`。保留生成kernel JSON和编译日志，便于核对目标芯片与源码。

## 2. 编译最小Torch桥接

```bash
MAX_JOBS=4 python tools/build_request_plan_test_binding.py \
  2>&1 | tee build/request-plan-logs/bridge-build.log
```

产物为 `build/request-plan-test-binding/request_plan_test_ops.so`。
桥接使用生产适配头，注册planner与transfer两个Torch入口。它应在独立测试进程中
使用，不能与已经注册相同`_C_ascend`算子的正式扩展同时加载。

## 3. CPU算法检查

```bash
python -m pytest --confcutdir=tests/ut/kv_offload \
  tests/ut/kv_offload/test_sparse_kv_request_plan.py -q
```

该测试编译并执行生产算法头，CPU线程/原子/barrier模拟SIMT执行，带ASan/UBSan。
覆盖四行共享、重复token、全命中、空位充足/不足、优先级淘汰、请求替换、MTP回退、
行重排、padding及非法页。它不替代950设备正确性。

## 4. 950真实Graph和KV测试

选择一张空闲950PR：

```bash
python tools/check_sparse_kv_request_plan_npu.py \
  --device 0 \
  --library build/request-plan-test-binding/request_plan_test_ops.so \
  --output test-results/request-plan-950-r1 \
  2>&1 | tee build/request-plan-logs/device-check.log
```

输出目录必须尚不存在。测试明确拒绝910，使用2个请求、每请求4行、TopK2048、
共享容量12288、32K逻辑上下文；K/V来自真实注册Host缓冲。
先warmup和capture，再连续10次replay，改变输入、请求身份、行顺序、stable prefix
及active rows。检查以下内容：

- 请求内重复token只产生一次miss，四行指向同一请求池的槽位。
- miss集合匹配CPU参考，所有选中token的槽位有效。
- 从Host回搬的K/V逐元素完全一致。
- padding槽位输出为-1、inactive请求miss_count为0，Graph引用地址不变。
- 双向映射互逆，planner完成后没有残留pending；回退修改Host内容后仍能精确回搬。

成功会产生 `test-results/request-plan-950-r1/result.json`，其中`passed=true`。
失败保留完整日志，不把部分replay通过当作整体通过。
这个测试没有执行真实模型的score、TopK或attention，因此尚不能证明模型精度或加速。

## 实现约束与后续性能测试

`max_queries`表示每请求实际query行上限，MTP3设置4。第一版要求共享容量
`C >= max_queries * TopK`，不允许union放不下却继续attention。
状态按请求池位置归属，请求身份改变会清空该请求状态；跨调度位置移动会冷启动。
缓存token唯一、free栈不重复由本算子维护，不能直接导入旧按行LRU表。

反向表每个请求/层一份int32 `[L]`，B16、128K上下文为8 MiB/层/rank。
初次创建时填-1；请求替换仅撤销旧缓存槽位对应项。MTP回退沿反向表清理
`[stable_prefix,resident_end)`后缀，普通命中查找只遍历各行TopK。
表内-2只表示本轮miss已经被某个线程认领，不表示KV已就绪。

调用顺序是`TopK -> request plan -> request transfer -> attention`，同stream执行。
planner更新的slot映射是计划状态，transfer完成后KV才可读取。每轮必须刷新
active rows和输入元数据。`status`非零表示本次计划不可用，必须拒绝消费。
生产部署仍需接入失败传播；当前Python的`check_status()`只用于capture/计时外检查。

性能对照必须记录B、四行TopK重叠、union miss、C、冷/热状态及真实搬运字节。
不要用“各行独立缓存的旧路径”和“请求共享的新路径”的不同容量预算隐瞒差异。
目前没有950PR性能数据；先确认正确性，再测planner、transfer和完整Graph链路。

详细设计、表大小、复杂度、Graph状态协议见
[设计文档](docs/950pr_request_shared_planner.md)。本机已完成CPU测试、950PR交叉编译
和Torch桥接编译加载，设备执行留给950PR机器验证。
