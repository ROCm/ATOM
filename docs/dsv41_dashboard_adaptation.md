# DSV4.1：迁移到 ATOM random TP4/FP8 配置

2026-10-29 代码归并更新：已将通过完整服务运行的FP8 owner迁移收进可选生产路径，`run_dashboard.py --execution fp8-owner`显式选择EP1/full；默认native。实现、依赖、CPU测试及证据边界见[归并说明](dsv41_fp8_owner_result.md#代码归并与运行入口)。以下带时间戳段落均为历史过程，其中“生产未改／加载中”不表示当前状态。

2026-10-29 10:53 UTC 更新：旧owner＋retained-prefill已迁移至FP8 KV、EP1、full M1536，并完成同工作负载C256/8K1K/n2560整网。迁移版6281.294tok/s，对保留主线5269.811为1.191939×（+19.19%），TPOT46.187→38.588ms；真实AL2.488826→2.471299，decode行数略增。每rank正式4个新Triton特化，且为新单臂对保留基线，尚未稳定性能／任务精度验收。完整配置、实际路由、成本及下一步见[FP8 owner迁移整网结果](dsv41_fp8_owner_result.md)。该次运行时生产atom/保持f2d3，EP4和packed仍独立比较。


2026-10-29 10:23 UTC：**FP8 / full M1536 的真实模型迁移诊断已完成**。新 [R2 review](../../validation/dashboard_whole_20261029/fp8_owner_service_r2/REVIEW.json.gz) 核验42个源绑定，512条C256请求全部完成，四rank全部40层TP4/EP1、7种owner容量各40层FP8 attention捕获；每rank M1536实际219步、owner target286步、retained-prefill 3,776,827行、compact draft context286次。组件数值/状态证据与真实链路完成证据分别保留，不代表新GSM精度接受。

这组诊断为random8K/output256、无请求预热，记录1469.303tok/s、81.733ms TPOT、真实AL2.167512，含首次owner图捕获与编译；**不能与8K/1K/n2560基线计算倍率**。容器rc0并退役，8卡空闲已核验；原生退出未显式释放owner图，未宣称完整图/通信池生命周期通过。启动辅助Python误触发service bootstrap的调用栈已无损导出，未重跑模型。

新的 [R3完整性能运行](../../validation/dashboard_whole_20261029/fp8_owner_service_r3) 于10:23启动，保留相同六文件迁移算术；只新增owner图注册/释放、native退出前释放所有注册图，及限定server/spawn/显式entry的bootstrap范围。CPU角色隔离、退出图reset/clear幂等、AST差异检查及Black/Ruff通过。C256 warm593后执行random8K/output1K/n2560，仍为TP4/EP1、FP8 KV、真实DSpark5、owner full；冻结48个绑定，生产atom/未改。正式记录前后AL、执行行数、owner/prefill计数及cold；复用已完成同节点main基线，明确时间间隔，不重跑旧arm或旧review。当前加载中，尚无这次完整性能结果。EP4与packed政策仍分别比较。



2026-10-29 10:01 UTC：用户明确将主线改为**先迁移旧优化到 FP8 KV / 完整验证行数，再单独比较 EP4 部署**。旧1.324404976×证据有效，但其分母为 Spec0/BF16，候选含DSpark5/owner packed/EP4，不能拿新对照中未启用owner路径的1.00542×解释为旧收益消失。

已完成第一版六文件私有 overlay：[fp8_owner_r1](../../validation/dashboard_whole_20261029/fp8_owner_r1)。decode采用原生V4.1 FP8 query/scale及四个H16分片；owner prefill接原生FP8缓存行；draft context将512个value byte与16个E8M0 scale byte一起gather并恢复全局行序。保留原生TP4/EP1 GEMM完整M/K；full模式C256执行M1536，不调用packed的192行前缀裁剪。生产atom/源码未修改。

[新组件review](../../validation/dashboard_whole_20261029/fp8_owner_r1/REVIEW.json)已完成一次：四rank M384/H64 attention与四次原生H16调用逐位一致，独立FP32参考relative L2为0.002064–0.002100；full1536与ragged883有效行（capacity1536）的draft context values/scales逐字节一致，两次graph更新一致；56个STATE/PAGE区域共14,095,360B全部字节一致。CPU验证六种原生draft writer分支及EP1/FP8准入。这是组件证据，尚不包含真实权重、完整FFN/目标链或任务精度。

真实模型诊断[fp8_owner_service_r2](../../validation/dashboard_whole_20261029/fp8_owner_service_r2)已启动，C256/random8K/output256/n512/真实DSpark5/owner full，模型初始化中。此前R1在无可见GPU的导入预检中触发AITER架构探测失败，未加载权重；新R2只修正预检设备可见性及1536GiB主机内存配置，沿用组件review的同六份overlay。未重复已完成组件或旧整网实验。本诊断仍不是正式吞吐对照；完整源合同、实际接受长度、执行行数及40层路由通过后，再安排正式同配置性能对照。EP4与packed分别作为后续变量。

2026-10-29 09:20 UTC 更新：同节点锁定main与当前分支四臂各2560请求已完成，见[完整整网基线](dsv41_c256_baseline.md)。合并吞吐5269.81→5298.38tok/s（+0.54%），两对方向相反，尚无稳定加速；原始冷编译和两次管理中断均保留。


2026-10-29。按照“先适配新的模型配置”的顺序，本轮首先落地 ATOM random dashboard 的 C256 配置。实现是独立运行入口和 profile，原 EP4/BF16 recipe、冻结实验与模型算术保持原状。

配置及真实 CPU 预检已经通过。2026-10-29 后续已启动整网适配，补齐 GPU 可见环境中的预检隔离、1536 GiB 主机内存、旧版 ATOM cache 常量绑定，以及旧 AITER 缺失的 FP8 MLA 依赖。实际 C256 测点的 TP4/EP1/DP1、无 DPA 已由运行包验明，见 [C256 配置核对](dsv41_c256_verified_config.md)。当前尚无完成的整网性能对照，不能标为新模型性能接受。

05:54 UTC 更新：R7 主线 A1 已完成真实模型初始化、593 条请求预热，进入2560条正式请求；A1/B1/B2/A2 串行同配置对照正在运行。四 rank 实际加载的40层 MoE 均为 TP4/EP1、384专家、H5120、每卡 I640、MXFP8 activation，`_comm_fused_moe=None`。预热中的 C256 target 执行 M1536，不能继续把 M768 当作新主线固定形状；正式阶段的行数分布另行报告。

实际加载与静态接入证据见 [C256_WINDOW_STATIC.json](../../validation/dashboard_whole_20261029/C256_WINDOW_STATIC.json)。容器实际合并的 FMOE 表在 `H5120/I640/E384/A8W4` 下有16条原生配置，`comm_fused_configs` 均为 `0`。这是继布局开关之后的另一道缺口：仅启用 `ATOM_MOE_GU_ITLV` 仍没有该形状的 Window winner。I640 对 Window G2 要求 K128；G1 日志中的 N256→N128 不代表 G2 也只能采用 N128。现成 Window 还会改变 G2 route epilogue 和通信 payload，不能当作保持 BF16 数值不变的透明替换。

旧 M768 的来源也已追到代码：[prefix_lengths](../atom/models/deepseek_v41/owner_service/owner_serving_plan.py) 默认每 owner `capacity=192`；[OwnerServing](../atom/models/deepseek_v41/owner_service/owner_serving.py) 在 `mode=packed` 时调用它并重建验证前缀。DSpark 仍产生5列 proposal，但 target 执行的 prefix 被预算截短。这与本次 `confidence_schedule=false`、没有 owner bootstrap、C256 target 运行6×256行的合同不同。旧吞吐差异包含调度/接受长度/工作量变化，不只是 TP/EP GEMM 效率；新方案必须同时核算少做 target 工作后增加的验证轮数、draft 工作和状态处理，不能仅按行数减半声称2×。

## 配置与入口

- 唯一配置：[dashboard_profile.json](../tools/dsv41/dashboard_profile.json)。对照 public ATOM `bdec681249254846f73b833d1a5112f86730ca34` 的 `models.json` 和 `atom_test.sh`，使用本分支实现。
- 运行入口：[run_dashboard.py](../tools/dsv41/run_dashboard.py)。支持 `plan / preflight / serve / benchmark`。
- 原研究入口：[run.py](../tools/dsv41/run.py) 和 `r13_profile.json` 保留；新配置不用其 Spec0 对照和 owner bootstrap。当前受控整网 harness 另行使用593条独立请求预热，两侧相同，并明确区别于 public profile 的512次重复 prompt 预热。

| 合同 | 新配置 |
|---|---|
| 模型 | `deepseek-ai/DeepSeek-V4.1-Flash`；可显式指定相同模型的本地挂载路径 |
| 并行 | attention/expert/shared TP4，DP1，关闭 EP 和 DPA |
| KV / index | FP8 / FP4 |
| 投机 | DSpark5，真实 target verification，无 synthetic acceptance |
| 服务容量 | max-num-seqs512，max-batched-tokens16384，attention chunk16384 |
| 编译/图 | level3、FULL，原生 capture ladder |
| memory / prefix | .9，prefix caching 开启 |
| 客户端 | random8192/1024、range .8、C256、2560 formal、512 warm、seed0、chat template、ignore-eos |
| 模型执行行数 | 由实际 target/draft/batch/graph 决定，没有固定 M768 假设 |

CPU 中由实际 Config 解析得到 `kv_cache_block_size=128`、模型上下文上限1048576、40 backbone 层及含 draft 的43层 topology。CLI 的 block-size 默认16并不是该模型最终运行的 page 大小。最终 capture sizes 为 `[1,2,3,4,5,6,7,8,16,32,48,64,128,256,512]`，这仍不是每个 shape 已完成图捕获的证据。

入口会清除子进程继承的 `DSV41_* / ATOM_* / AITER_*` 实验选择以及 `LD_PRELOAD`，然后设置 profile 环境和请求的 cache 路径。锁定版本 `atom.utils.backends.VLLM_CACHE_ROOT` 是硬编码模块常量；整网 harness 额外将它绑定到本次 cache，单独设置同名环境变量不会生效。原生 Mono 默认行为由选定源码决定；没有强制关闭小 M Mono，也没有扩大其行数范围。`PYTHONPATH` 只包含显式 ATOM/AITER 源，旧 acceptance 的 `sitecustomize` 不进入子进程。可见 GPU、ROCm 路径等基础环境保留；父进程环境不修改。

`serve` 先运行真实 `EngineArgs → Config → validate_runtime_config`，成功后通过 `exec` 进入原生 `python -m atom.entrypoints.openai_server`，保留 spawn 的模块语义。没有复制模型、MoE 或 benchmark forward。新的 preflight 不要求 GPU，也不假造 Config。配置探针的子进程显式隐藏 GPU，实际服务保留原设备选择；这修复了 GPU 节点上原生 Config 查询设备触发初始化、进而被 CPU-only 断言拒绝的问题。

每次写入使用独占创建，保留命令、profile、关键入口文件摘要和源提交，拒绝覆盖旧结果。这是配置溯源，不是完整服务 source freeze；正式实验仍需保存实际依赖、模型 revision、loaded contract、graph keys 和产物。

## 使用

在具备该模型所需 ROCm/Python 依赖的环境，从 ATOM 根目录执行。每次尝试使用新的输出目录；示例 `/tmp` 路径在正式 GPU 任务前必须纳入该任务的存储预算。

```bash
# 只生成可审阅的服务/客户端命令，不导入 torch 或启动服务。
python tools/dsv41/run_dashboard.py plan \
  --model /models/DeepSeek-V4.1-Flash \
  --aiter-source /home/yifehuan/dsv41_origin_aiter_reference \
  --output /tmp/dsv41-fp8-plan

# 真实解析参数、模型配置和运行时准入，不加载权重。
python tools/dsv41/run_dashboard.py preflight \
  --model /models/DeepSeek-V4.1-Flash \
  --aiter-source /home/yifehuan/dsv41_origin_aiter_reference \
  --output /tmp/dsv41-fp8-entry

# 前台服务。自动进行一次自己的 preflight，使用独立结果/cache 目录。
python tools/dsv41/run_dashboard.py serve \
  --model /models/DeepSeek-V4.1-Flash \
  --aiter-source /home/yifehuan/dsv41_origin_aiter_reference \
  --output /tmp/dsv41-fp8-server

# 服务及实际 GPU/model 就绪后，在另一终端运行原生客户端。
python tools/dsv41/run_dashboard.py benchmark \
  --model /models/DeepSeek-V4.1-Flash \
  --aiter-source /home/yifehuan/dsv41_origin_aiter_reference \
  --output /tmp/dsv41-fp8-client
```

默认地址为127.0.0.1:19126，可显式调整 `--host/--port`。`--model-source` 指定 ATOM 源，`--python` 指定子进程解释器。`benchmark` 连接已有服务，不负责分配节点、等待权重加载或停止服务；HTTP health 本身不能证明 GPU 模型就绪。

## 迁移后哪些优化依据需要重建

当前 TP4 MoE 的原生路径是 replicated input → shared/router/routed experts → shared combine + TP all-reduce。旧 owner192 → BF16 AG → EP4 → owner RS 不是这个部署的基线。旧 EP4 每 rank96个完整专家与新 TP4 每 rank持有所有专家的分片，也有不同的 GEMM/权重读取成本。

因此，[大并发方案](dsv41_wide_window_next.md) 中的 owner 一次供料 A 候选仍保留为旧研究部署的方向，不直接移植到此次 public TP4/FP8 主线。更不能通过放宽 owner/BF16 guards 假装完成 FP8 owner 状态适配。旧 G2 完成曲线、M768 cost shares、1.3244×历史组合和33.78%历史缺口也不是新配置的实测预算或成绩。

本次静态核对发现一个与 Window 接入直接相关的边界：

- 当前 `envs.ATOM_MOE_GU_ITLV` 默认是 false，public DSV4.1 random catalog 没有设置该开关。
- `create_comm_fused_moe_backend` 在该开关为 false 时直接拒绝，所以本轮保持默认的 profile **在当前源码下不会启用 comm-fused Window backend**。
- `Mxfp4MoEMethod.is_guinterleave` 的条件是 `ATOM_MOE_GU_ITLV or fp8_activations`，DSV4.1 则明确使用 FP8 activations。方法支持的布局和 wrapper 的准入条件并不是同一个判断。
- 这是一项值得继续核对的静态接入差异，不是已证实的错误，也不是可以直接删除 guard 的证明。即使满足该开关，还需要 actual loaded weight layout、ROCm/MoRI capability、完整 ShapeKey、`AITER_CONFIG_FMOE_FILE` 的 winner 及实际 bucket 命中；不同 payload 还涉及独立精度合同。
- R7 已证实上述实际几何和无 winner 状态。完整时间预算仍未测得，不创建虚构 winner，不以调优表内历史 `us` 作为本次完整窗口耗时。

相关代码：[backend eligibility](../atom/model_ops/fused_moe/comm_fused_moe.py)、[MoE method](../atom/model_ops/moe.py)、[V4 shared/routed 与输出归约](../atom/models/deepseek_v4/__init__.py)、[V41 activation/router](../atom/models/deepseek_v41/moe.py)、[环境默认值](../atom/utils/envs.py)。本次未改变开关或 guard。

迁移后的下一步按以下顺序进行：

1. 新 TP4/FP8 配置的一次实际加载诊断，记录 loaded expert/shared geometry、KV/index格式、Mono/普通路由、实际 target/draft M 和 graph keys。服务接受长度保留原始计数及统计范围，不预设 M768。
2. 从这个新基线建立完整 `attention-post → FFN → next consumer` 窗口的成本表；先核对已有量化共享、原生 G1/G2、TP AR 及 mHC。旧 EP4 的 AG/RS 时间不填入新表。
3. 对 Window 接入先做上述布局/eligibility/winner 的 CPU/静态证明，再核算 native G2 与输出通信可隐藏时间、重复 MID 读取、shared join、数值变化、fill/drain。只准备一个有正向净预算的完整链候选，不扫参数。
4. 完整链通过后才做新配置内的服务 screen；A/B 都保持 FP8/TP4/DSpark5。新原生路径本身先取得基线，不把配置迁移带来的变化冒称 kernel 加速。

原生 benchmark 的 `/debug/mtp_stats` 是从服务启动起累计的统计，包含 warmup，不是 formal-only AL；profile 明确保留这个限制。target/draft 实际执行行与正式阶段图/编译计数尚需上述新诊断接入。当前 `benchmark` 命令适用于配置复现，不能单凭其聚合结果完成 kernel 归因或优化验收。

## 已完成的验证与资源记录

- [4项入口合同测试](../tests/tools/test_dsv41_dashboard_profile.py)通过：catalog 的 DSpark/C256请求协议、旧环境隔离、原生 module entry/投机保留、纯标准库 plan 与防覆盖。Black 与 Ruff 通过。
- [真实 CPU preflight](../../validation/dashboard_config_20261029/r1/output/preflight.json)通过。镜像 `32c81e56…`，ATOM `f2d3ee47…`，AITER reference `729feefb…`，本地模型 config 已记录摘要；没有声称这套镜像就是 public main 的镜像。
- [执行结果](../../validation/dashboard_config_20261029/r1/RESULT.json)：rc0、容器已退役、GPU device mounts为空，输出13824B；CUDA未初始化、权重未加载。raw 摘要与当前入口/profile 对应的冻结副本核对通过。
- 本轮只在借用的 job244395/node155 上运行这个 CPU 容器；重新核对了用户、RUNNING、8卡、Exclusive=1、原结束时间2026-10-29 06:51:29。没有新增 allocation 或 GPU 工作负载，保留借用资源。
- 新 CPU tmpfs64MiB；新源码/文档/输出预留600000B。准入时旧 whole-join146598912B + 保留 spool405504B + 新预留小于148MB，沿用总保守上界9999758848B。后续 GPU 加载需独立重新准入。

CPU 配置验证不代替完整服务生命周期、GSM8K、L2 或性能接受；这些结果在本轮均未产生。已完成的 G2 ready、预取、owner stage 和输入重叠不重跑。
