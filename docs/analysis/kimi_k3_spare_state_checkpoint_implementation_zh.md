# Kimi-K3 空闲 STATE checkpoint：实现与 CPU UT 验证

日期：2026-10-10。工作分支：`yajizhan/k3-prefill-warmup-state-index`，基于 `8abca55c7e38ab8827b7e02ba1bfe6c6a4600362` 的工作区修改。

本次实现把空闲 STATE slots 用于保存完整 KDA checkpoint，增加现有 STATE/PAGE/CPU 层次的有效容量。保持 max_num_seqs 和 STATE 大张量大小；只对新产生的 checkpoint 选择存储位置，不主动搬迁已经存在的 PAGE 图像。实际性能尚未测量，之前报告中的百分点和加速数字仍为条件估算。

实现默认关闭。当前支持 K3、prefix caching、PP1、显式关闭 ReplaySSM，以及无 connector 或单机 `lmcache_offload` 的 `offload` / `kv_both` 角色。运行模式可为无投机（每请求 1 slot），或 DSpark3 且 `state_checkpoint_interval_tokens=-1`（每请求 4 slots，仅 prefill checkpoint）。其他投机配置、ReplaySSM、PP/P-D、RapidServe 等不支持的组合，显式启用时会报错。没有修改原有 [ATOM 架构报告](/mnt/raid0/mengqing/ATOM/docs/analysis/kimi_k3_prefix_cache_analysis_zh.md) 和 [vLLM 架构报告](/mnt/raid0/mengqing/vllm/reports/kimi_k3_hybrid_prefix_cache_analysis_zh.md)。

## 1. 启用与回退

在现有 K3 server 启动环境中设置：

```bash
export ATOM_ENABLE_REPLAYSSM=0
export ATOM_KDA_SPARE_STATE_CHECKPOINTS=1
export ATOM_KDA_SPARE_STATE_RESERVE=8
```

保留 `--enable_prefix_caching` 和 `max_num_seqs=96`。DSpark3 另设置 `--method dspark --draft-model Inferact/Kimi-K3-DSpark --num-speculative-tokens 3 --state-checkpoint-interval-tokens -1`。关闭新特性可将 `ATOM_KDA_SPARE_STATE_CHECKPOINTS` 设为 `0` 后重启；这是启动时配置，不是热切换接口。

reserve=8 是初始策略值，尚未经过 MI355X 性能调优，与 TP8/DCP8 的并行度没有数值推导关系。单位是物理 STATE slots：DSpark3 下 8 个 vacant slots 可容纳 2 个新请求的运行状态。

存储位置和 checkpoint 生成位置是两个独立策略：

- 仅验证存储收益：保持原有 `interval=-1`、`ATOM_STATE_CHECKPOINT_DEMAND=0`，anchor 优先存入空闲 STATE。
- 希望增加已观察到的共享分叉恢复点：设置 `ATOM_STATE_CHECKPOINT_DEMAND=1`，与现有 demand/cut 路径配合。Agentic recipe 内部原本写死 `export ATOM_STATE_CHECKPOINT_DEMAND=0`，需要修改该行，避免覆盖外部环境。
- 无投机模式的周期边界仍由原有 `--state-checkpoint-interval-tokens` 控制。DSpark3 模式要求 `interval=-1`，不支持正周期或 decode checkpoint；没有增加 KDA midstep kernel。

新增 UT 验证了这一具体流程：旧请求只有 36-token anchor；另一分支共享前 16 tokens，首次访问缺少 KDA 恢复点，demand 把 forward 切到 16；该状态保存到 STATE 后，下一分支可联合 MLA+KDA 命中这 16 tokens。

## 2. 存储、命中与回收

`PageUnitCheckpointStore` 继续拥有唯一的 hash→checkpoint ID 索引、generation、COPYING/READY/EVICTING 状态以及 pin 计数。每个 record 的存储为 PAGE units 或一个完整 STATE slot。`BlockManager.state_caches` 仍只有一个 KDA gate：slot、PAGE、CPU 是同一状态的替代来源，与 MLA 的联合约束保持不变。

新图像的放置顺序为：

1. 空闲 STATE 数量高于 reserve 时，借一个 slot；复制在现有 maintenance 阶段、下一次 forward 之前执行。
2. STATE 借用空间用尽而 PAGE 仍有足够 free units 时，继续使用 PAGE，保留更多唯一图像。
3. PAGE free units 不足时，可以替换最旧的未 pin slot 图像；无法借用时沿用原 PAGE 分配/eviction，最终无空间则丢弃本次保存。

第 2 步避免了一个容量倒退：如果每次都替换旧 slot 图像，整个 GPU 状态缓存会被限制为 N−A−R 幅，浪费原有 PAGE 存储能力。本实现允许两种存储容量叠加。

STATE 的空闲列表中，READY、未 pin 的借用图像仍可被 admission 回收：先用 vacant，再在不足时回收缓存 slot。回收时先删除 coordinator 记录和 hash，再将 slot 交给运行请求。槽位保存中、restore 中、CPU D2H 中均不可提前复用。最后一个 reader 完成才返回可分配集合。

reserve 是新 checkpoint 的借用余量，也参与 CPU slot source 的 pin 预算；它不会把最大请求数永久减小。新请求可以消耗 vacant reserve。因此，并发突增后实际 vacant 数量可能暂时小于 8；下一次保存会重新检查空间，不能把 8 解释为任何时刻都保证空闲的硬分区。

恢复时必须先保护 source，再分配 destination，避免 admission 自己淘汰命中的 checkpoint。多 reader 分别恢复到自己的主 slot，共享不可变 source；DSpark3 每个 reader 还会分配另外 3 个私有投机 slots。如果可分配数量刚好只够本请求的完整 slot 集合，且其中包含未 pin 的 source，则直接将 source 接管为主 slot，删除缓存记录并省去复制；同时记录精确的已接管 hash，让联合 CPU-KV 加载的状态校验仍成立。已经被其他 reader 或 CPU 保存 pin 的 source 不可接管，也不计入可分配数量。

## 3. CPU state 保存

CPU 保存新增显式的 `StateSlotSource(slot_id)`，保持原有 PAGE tuple 格式兼容。worker 将类型传到 `StateByteCodec`：PAGE 源读取 `page_unit_views`；slot 源读取 `state_entry_views`。二者生成相同顺序、相同 layout ID 的完整 checkpoint 字节，所以 CPU key、恢复格式、32 GiB retention cap 均不变。

slot 图像只有在 maintenance batch 完成并发布 READY 后，才可提交给 CPU 保存。提交时 pin，原有跨 rank completion 聚合确认 source 释放或终态后才归还 slot。source 释放和 CPU put 的最终结果仍是两个阶段；迟到的旧 generation 通知不能释放新图像。

slot source 的并行 CPU 保存数额外限制为 `max(1, reserve)`，且 free slots 不超过 reserve 时推迟新的 slot 保存。未提交的候选留在有界 backlog。PAGE 保存仍受原有 max_inflight 限制。

**正在 CPU 读取的 slot 不按超时强制回收。** 超时不能证明 DMA 已停止；必须等待 source-safe 或 terminal completion。若 worker 永久失联，这些 slot 会继续受保护，需要原有故障处理或重启；本实现选择避免覆盖仍可能在读取的状态。外部 P/D 的 PAGE-source lease API 不接受 slot source，启用范围也排除了 P/D。

## 4. 完整状态复制与资源边界

生产路径调用新的 torch-only `copy_kda_checkpoint_slots`，使用 `torch._foreach_copy_` 复制每个 `(layer, slot)` 的 conv 和 SSM 视图。不会从最终状态推导中间状态，也不更改 dtype；槽位的层间 stride 保持原布局。

执行前检查 layout、image size、实际 tensor 字节数、slot 范围、PAGE/slot 源互斥，以及 source/destination 重叠。spec 必须满足 image_bytes=slot_bytes；关闭 ReplaySSM 时，无投机与 DSpark3 的 K3 均使用完整 conv+SSM 图像。DSpark3 的 conv 窗口包含 `num_spec=3` 扩展，复制函数和 CPU codec 均使用运行时 layout/字节数，不沿用无投机模式的固定尺寸。PAGE 图像继续使用原有 descriptor copy。

容量 UT 覆盖两个独立场景：

| 场景 | 物理 STATE slots | 20 个活跃请求占用 | reserve | 理论可借用 checkpoint slots |
|---|---:|---:|---:|---:|
| 无投机，max_num_seqs=96 | 96 | 20 | 8 | 68 |
| DSpark3，max_num_seqs=96，ReplaySSM=0 | 384 | 80 | 8 | 296 |

两种场景的 STATE 保存均不消耗 PAGE free units。296 是 `96×4−20×4−8` 的容量上限，假设没有额外 restore/CPU-source pin；并非同时能保证的 READY 数，更不能直接转换为命中百分点。checkpoint 按需产生，`demand=0 / interval=-1` 仍只保留 prompt anchor。真实每幅图像字节数以 runtime spec 为准，原先无投机模式的 1.839 GiB/rank 估算不能直接套给 DSpark3。

本次不改变池子分配大小，不减少 CPU cap，也没有承诺某个命中率或吞吐提升。尚未实现 PAGE→STATE 热图像迁移、slot→PAGE 异步降级、DSpark decode checkpoint 或 ReplaySSM 状态支持。

### DSpark3 prefill 的正确性边界

K3 prefill 的 causal convolution 和最终 SSM 均写入 `state_slots[0]`；maintenance 在后续 forward 覆盖运行状态前，将该完整图像复制到借用 slot。恢复也只填充主 slot，额外 3 个 slots 属于后续 speculative verify 的私有运行空间，不属于 checkpoint。原有 decode 根据接受长度选择 SSM 状态、回滚 conv 窗口的逻辑继续运行。

启动 guard 要求 `interval=-1`。`checkpointers_at` 对该模式额外拒绝 unaimed/decode 以及到达 prompt 末尾之后的状态保存；decode 的 MLA block hashing 继续进行。因此，本次并未通过简单放开开关来序列化 speculative decode 中的任意状态。

## 5. 诊断指标

新增字段进入 checkpoint funnel 和 `/debug/cache_stats` 聚合：

| 字段 | 含义 |
|---|---|
| `slot_checkpoint_stores` | 已排入 maintenance 的 slot 保存次数 |
| `slot_checkpoint_restores` | 已排队的 slot restore 次数，包含随后取消的操作 |
| `slot_checkpoint_adoptions` | 可分配 slots 刚好够一个请求时，直接接管未 pin source 的次数 |
| `slot_checkpoint_evictions` | slot 图像被替换或 admission 回收的次数，不含 reset/orphan |
| `slot_checkpoint_ready` | READY slot 图像数，包含被 pin 的 READY 图像 |
| `slot_checkpoint_pinned` | 有在途 reader 的 slot 图像数，可能与 READY 重叠 |
| `slot_checkpoint_copying` | 尚未发布的 slot 图像数 |

以上三个瞬时数量不能直接相加。`slots_held` 的底层 occupancy 也已包含未 pin 的借用图像。命中提升仍需用同一请求集合的实际复用 token 差来计算，不能将 slot restore 次数转换成新增命中 tokens。

## 6. 静态与 UT 验证

测试使用已有 `rocm/atom-dev:sglang-latest` 镜像的新建临时容器，不挂 GPU、不连接网络、不加载模型；未使用或改动现有服务容器。Python 3.12.3、pytest 9.0.3、torch 2.10.0+ROCm 7.2.4，torch 仅运行 CPU 操作。

新增覆盖包括：96/20/8 容量、PAGE fallback、slot/PAGE/CPU 联合 lookup、COPYING 不可见、多个 reader、取消和挂起 restore、独占接管、reset/orphan、CPU 源释放、迟到 generation、offload pin 上限、不按超时回收、真实 BlockManager admission、demand 分叉生成与命中，以及 800 步混合操作后的全局 ownership 不变量。

新增 DSpark3 UT 覆盖 STATE/PAGE/CPU 恢复时完整 4-slot 分配、多 reader、取消、精确容量下 source 接管、CPU 保存与 reader 并行 pin、未来 admission 的 4-slot 空间预算、禁止 decode state checkpoint 但继续 MLA hashing、384/80/8 容量，以及不支持配置的启动拒绝。benchmark UT 验证 C48 配置展开和环境变量进入 recipe fingerprint。

字节 UT 分别使用 `num_spec=0` 和 `num_spec=3` 的 conv 几何，对 8 组独立模拟 rank 数据执行生产复制函数，逐字节检查所有层的 BF16 conv/FP16 SSM：保存后覆盖原运行 slot，再恢复到两个 reader，并检查其他 slots 未被修改。这是分片数据的 CPU 验证，不是真实 TP8 通信或 GPU kernel 验证。

最终相关 CPU 套件：**829 passed、3 skipped、0 failed**，包括 checkpoint/offload、BlockManager、benchmark catalog 和 benchmark results 回归。3 项跳过均明确要求 GPU，分别位于 lmcache connector（1 项）与 PAGE copy planner（2 项）。针对本次 DSpark3 和 catalog/metadata 的目标套件为 **126 passed**，已包含在总数内，不重复相加。Ruff、Black --check、Python 3.12 py_compile、JSON/YAML 解析、离线 C48 matrix 展开和 git diff --check 均通过。

[JUnit 结果](/mnt/raid0/mengqing/ATOM/.runtime/k3_spare_state_validation/dspark-cpu-unit.xml)、[本次测试命令](/mnt/raid0/mengqing/ATOM/.runtime/k3_spare_state_validation/run_dspark_cpu_ut.sh)、[静态检查脚本](/mnt/raid0/mengqing/ATOM/.runtime/k3_spare_state_validation/run_dspark_static.py)及[离线 C48 matrix](/mnt/raid0/mengqing/ATOM/.runtime/k3_spare_state_validation/dspark-agentic-matrix.json)保存在工作区 .runtime 目录。静态工具为临时目录中的 Black 26.10.0、Ruff 0.16.10，没有修改服务容器依赖。测试容器显式设置 TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor，以适配宿主 UID 在容器中没有 passwd 条目的情况。

原有 `test_kda_checkpoint_slot_copy.py` 和 `test_gdn_state_relocation.py` 在无 GPU 的镜像中因 AITER 导入时调用 rocminfo/JIT 而无法收集；未伪造 GPU/AITER 模块来使其通过。它们不计入最终 CPU 套件的通过数。新增 torch-only UT 覆盖本特性的实际 slot copy，原 PAGE planner 和签名 UT 也纳入回归。

## 7. 主要代码

- [开关与支持范围](/mnt/raid0/mengqing/ATOM/atom/model_engine/block_manager.py)、[环境变量](/mnt/raid0/mengqing/ATOM/atom/utils/envs.py)。
- [borrowed slot 空闲列表与回收](/mnt/raid0/mengqing/ATOM/atom/model_engine/state_pool.py)。
- [统一索引、恢复、generation 与 CPU pin](/mnt/raid0/mengqing/ATOM/atom/model_engine/page_unit_checkpoint.py)。
- [实际 slot 复制函数](/mnt/raid0/mengqing/ATOM/atom/model_ops/attentions/pool_layout/slot_checkpoint.py)、[KDA backend 调用](/mnt/raid0/mengqing/ATOM/atom/model_ops/attentions/gdn_attn.py)。
- [CPU 源类型](/mnt/raid0/mengqing/ATOM/atom/kv_transfer/disaggregation/types.py)、[worker 转发](/mnt/raid0/mengqing/ATOM/atom/kv_transfer/offload/hybrid/kimi_k3/connector.py)、[CPU codec](/mnt/raid0/mengqing/ATOM/atom/kv_transfer/offload/hybrid/kimi_k3/state_object.py)。
- [控制面与分叉 UT](/mnt/raid0/mengqing/ATOM/tests/test_kda_spare_state_checkpoints.py)、[逐字节 UT](/mnt/raid0/mengqing/ATOM/tests/test_kda_spare_state_copy.py)。


## 8. `.github` 单机混部 Agentic 用例

配置位于 [models_agentic.json](/mnt/raid0/mengqing/ATOM/.github/benchmark/models_agentic.json)，prefix 为 `kimi-k3`，variant suffix 为 `-agentic-dspark3-tp8-dcp8-lmcache-state`。在 **ATOM Agentic Benchmark** 中选择包含当前修改的分支、`profile=test`、`models=kimi-k3`；concurrency / duration 留空即运行 C48、3600 秒。`dry_run=true` 只展开矩阵，不分配 GPU。留空 models 会同时选中 test catalog 中的 DeepSeek 与 K3；nightly catalog 不变。

新用例保留 TP8/DCP8、DP1、EP1（不启用 expert parallel）、关闭 DP attention、max_num_seqs=96、max_num_batched_tokens=8192、DSpark3 / AL=3.00、FP4 权重 / FP8 KV、LMCache DRAM 128 GiB/rank。STATE 开关为 1、reserve=8、ReplaySSM=0、interval=-1、demand=0；其余 ATOM/AITER/LMCache 参数参考提供的 2026-09-29 环境日志。

该入口通过现有 benchmark template 直接启动一个 ATOM server，不调用 ATOMesh，不需要历史 Dynamo/ETCD 配置或 IP。Agentic 数据集沿用 `semianalysis_cc_traces_weka_062126`，context cap=1,048,576；请求长度来自真实 trace，不强制随机生成 1M/1k 的请求。AIPerf commit 固定为参考用例的 `754356e9a39acc6cc6afb242d123bb57c3fb6f75`。

对比实验可在相同配置下仅将 `ATOM_KDA_SPARE_STATE_CHECKPOINTS` 改为 0 后重启。开关、reserve、ReplaySSM、LMCache 容量等已加入 benchmark metadata 的性能参数集合，避免两次运行因遗漏环境变量而得到相同的 recipe 标识。本地未执行真实 benchmark；其他服务器运行前需同步当前分支中的代码修改和 catalog。
