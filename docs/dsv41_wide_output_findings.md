# 大 M Mono 输出链：生产接入、整网结果与架构决策

2026-10-28。基于 `yifehuan/dsv41_fusedmoe` 的 `e51f5be03`，主线截点仍为用户确认的 `ebd5f84f3`。输出融合现已接入真实 prequantized FFN，生产实现为 `atom/model_ops/deepseek_v41/ffn_combine.py`。整网、精度和编译产物位于仓外 `validation/whole_join_20261028/`。历史 C5 相对主线为 **1.324404975757×**；本轮在新节点上对 C5 的增量测量独立报告，不相乘。

## 当前验收结果

最终算子把 shared 相加和 ragged padding 合成一次写入，直接交给同一个 RCCL reduce-scatter；下一层继续使用原生 Mega-mHC。TP4、H5120、连续同形 BF16、M≥4096 且 M%4≠0 才走新路径。对齐或不支持的输入保留 `v41_mhc_reduce_scatter(routed + shared)`。G1/G2、专家布局、router、shared 数学、RCCL 归约顺序均未改变。

运行时行数不作为 constexpr：ELEMENTS/PADDED 使用 i32 参数，避免每个 ragged M 产生新 specialization。H5120 恰为 BLOCK1024 的整数倍，因此每个 tile 全部有效或全部为 padding；使用 tile 级谓词使内存访问向量化。实际生产产物有 2 条 `buffer_load_dwordx2`、1 条 `buffer_store_dwordx2`，12 VGPR，LDS/private/spill 为 0。五个变化 M 共用一个 specialization。仅加 alignment hint 的旧候选仍生成标量访存，已停止。

真实 layer8 完整 FFN，向量实现相对动态标量 join：

| M | 标量 ms | 向量 ms | 比率 |
|---:|---:|---:|---:|
| 4097 | 1.039817274 | 1.033574700 | 1.006040× |
| 7835 | 1.557199180 | 1.535347104 | 1.014233× |
| 16315 | 2.867434740 | 2.824273348 | 1.015282× |

隔离输出段各 rank、五个 M、三次变化输入的相对 L2=0。完整 FFN 的 owner L2 约 0.0002–0.00048，与原生自重复同量级；没有为它新造统一阈值。

节点155 R3 整网 ABBA（每组2560请求、593预热，A1复用同节点紧邻的有效 C5 A2）：C5 pooled **7782.992051 → 7825.518820 tok/s**，**1.005464064×（+0.5464%）**；mean TPOT **30.900477 → 30.777207 ms**。两对为 **1.010770831 / 1.000148327×**；baseline A2/A1 漂移 **+1.2329%**，候选重复 **+0.1690%**。不能据此声称稳定1%提升，也不能将漂移归因于某个 kernel 或接受率。P99 TPOT/ITL 按各组原值保留，未声称所有尾延迟改善。

四组请求及输入/输出 token 数完全对齐；四 rank 正式阶段无新增 JIT、trace、编译或捕图，完整图键和输入地址绑定不变，40层 EP 与18层 router/packet 路径均核对。正式编译 hash `39fb988d82ede3953f9d23347469e1f9ed83c109cddbdd76f68597e9d3b3ded2` 与探针 hash 不同，但规范化的47条实际指令及资源相同；B2产物与B1相同。

全量 GSM8K：**1212/1319 = 0.9188779378316907 ≥ 0.91**。采用仓库 DSpark 对应的3-shot/local-completions/flexible-extract、并发65协议，样本题号及原始结果复计分一致。每个 rank 的18个目标层各有6次实际 fused 调用，40层 EP 加载合同一致。本部署为 BF16 KV/EP4，与 catalog 的 FP8 KV 不同：结论是本部署在所列协议下通过同一任务门槛。首次验收入口的无保护 runpy 在 spawn 时重入，尚未加载权重即失败；原始证据保留，入口已改为原生 exec，并用CPU spawn对照及完整 GSM 验证。

权威凭据：`node155/r3/PERFORMANCE_RESULT.json`、`EXTENDED_RESULT.json`、`B1_KERNEL_REVIEW.json`、`node155/gsm_r4/GSM_REVIEW.json`。以下章节保留此前原型的决策依据；合成输出段收益不是新的整网倍率。

## 1. 大、小 M 的设计目标不同

| 维度 | 小 M Mono | 大 M 输出/专家链 |
|---|---|---|
| 主要成本 | launch、细粒度依赖、低 occupancy、单 token 权重读取 | 大 GEMM 效率、连续 HBM/跨卡搬运、整体输出物化 |
| 计算组织 | 合法范围内的 token task 与持久 CTA | expert tile、完整 K、权重复用；保留原生 G1/G2 |
| 借鉴内容 | ownership、ready、最后读者、固定 ABI | 同样的责任与生命周期；不复制小 token 调度 |
| 优化判断 | 完整小步延迟 | 包含 shared、通信、padding 和真实消费者的完整区间 |

并发 C 与执行行数 M 必须分开。M768 是先前观测到的宽 decode 形状之一，M7835/16315 是实际大 prefill 形状；它们不应直接套用小 Mono 的阈值。EP4 的原生 G1/G2、量化、专家映射和 shared TP4 保持原有合同。

当前 `finish_ffn` 已经把 pending FFN 输出递交下一层；无 Engram 强制 settle 时，原生 Mega-mHC 已融合 delayed post/pre/norm，并可输出 MXFP8。重新拆出 post 会丢失已有融合。

## 2. 实际尝试和停止的方案

第一版使用现有 Mono `PeerBuffer` 的真实 IPC 地址表，每个连续 32 KiB packet 配 ready/ack：BF16 相加时直接写 owner，owner 收齐四个 source 后以 FP32 rank order 归约，再交给原生 Mega-mHC。该实现没有修改 G1/G2，没有声称 G2 内部已经重叠。

五个 M 的变化输入、ragged padding 和重复 graph replay 通过了传输合同检查；但完整输出段性能拒绝晋级：

| M | native 输出段 ms | 直接写 owner ms | 加速比 |
|---:|---:|---:|---:|
| 768 | 0.080876 | 0.089328 | 0.9054× |
| 7835 | 0.568975 | 0.968256 | 0.5876× |
| 16315 | 1.201355 | 2.480642 | 0.4843× |

新一次 eager 分段诊断使用同一份设备指令，仅增加 host 阶段入口。M16315 逐 rank 中位数：相加并远端发布 1.945–2.222 ms，owner 归约及 ack 0.177–0.219 ms，原生 Mega-mHC 0.106–0.108 ms。发布段明显最重；尚未把 payload store、system fence 和逐包标记的费用进一步独立分解。该诊断包含 host launch 间隙，不能替代上表 graph ABBA，也不能把不同 rank 的最大值相加。

第二版保持算术和 packet 协议，改为本地连续发布、owner 拉取四卡贡献。M7835/16315 的完整输出段仅为 **0.5240× / 0.5137×**，同样停止。所有完成的实验均已退役；没有重跑已完成的 unchanged 方案。

这两个方案的 owner 相对 native L2 约 0.00293，norm 约 0.00323；同输入 native 自重复为 0。rank-order FP32 与原生 RCCL 的舍入顺序不同，不能以“只是通信”宣称数值等价。没有自定阈值或 GSM8K 验收。

**决策：当前整 G2 完成的接口下，逐包 IPC 没有兑现跨 producer 的流水，却增加了传输/同步成本。它不是本轮可部署的大 M 优化。** 后续恢复这条方向需要可证明的 producer completion，以及完整链路收益，不能只调整包大小或扩大小 Mono guard。

## 3. 有收益的候选：一次写出 RCCL 消费布局

真实现有尾段：

```text
routed[M,H] + shared[M,H]
  → combined[M,H]
  → cat/pad[4*ceil(M/4),H]
  → native RCCL reduce-scatter
  → native Mega-mHC post/pre/norm/MXFP8
```

候选 `native_join/combine.py`：

```text
routed[M,H] + shared[M,H]
  → 一个 kernel 写 combined[4*ceil(M/4),H]，同时把尾行置零
  → 同一个 native RCCL reduce-scatter
  → 同一个 native Mega-mHC post/pre/norm/MXFP8
```

它复用 Mono 的“消费者决定布局、buffer 有明确所有者”原则，保持原生归约顺序。没有新增 peer 传输或归约算法。每个实例绑定固定 plan、stream 和持久输入/输出地址；graph reset 后释放，热路径不重建对象。

四 rank 的 graph ABBA，所有 transport/shared/padding 和原生消费者费用均包含在计时内：

| M | native ms | candidate ms | 比率 | 每次输出段节省 |
|---:|---:|---:|---:|---:|
| 768 | 0.080250 | 0.080103 | 1.0018× | 0.15 μs，无可靠收益 |
| 4096 | 0.283722 | 0.287425 | 0.9871× | −3.70 μs，保留 native |
| 4097 | 0.319534 | 0.299887 | 1.0655× | 19.65 μs |
| 7835 | 0.574491 | 0.527687 | 1.0887× | 46.80 μs |
| 16315 | 1.198247 | 1.092829 | 1.0965× | 105.42 μs |

M7835 两个 pair 为 1.08446/1.09292；M16315 为 1.09662/1.09630。各 rank、五个 M、三次变化输入下，owner、residual、norm、next-pre、post、comb，以及量化输出的相对 L2 均为 **0**，native 自重复也为 0。还完成了 128 次 replay 后再次变化输入的验证。新 Triton kernel 编译产物、ISA 和资源已保存，无 spill。

这些历史数据是**合成 producer/权重、真实 native consumer 的输出段测试**，当时未加载真实 FFN 权重，本身不构成整网 GSM8K、服务吞吐或正式 zero-JIT/capture 证据。后续验收见本文开头。aligned M 的负例必须保留，不能把该路径推广到所有形状。

搬运账本：对 ragged M，删除 cat 对完整 combined 的一次读取和一次写出，减少 `2*M*H*sizeof(BF16)` 本地 HBM 流量。M7835 每 rank 为 160,460,800 B；M16315 为 334,131,200 B。RCCL 网络 payload、原生 routed/shared 输出以及 owner pending 均仍存在，不宣称已经实现 G2→mHC 的单 kernel。

## 4. 接入合同及剩余边界

1. **真实 prequantized FFN 出口已接入。** 在 shared/G2 join 后调用具名 guarded op，支持域及回退见当前验收结果。完整单层、整网和 GSM 已完成；正式服务维持原有 side-stream 配置，不把合成双流测试等同于新的整网配置。
2. **参数和生命周期只维护一份。** 由 owner/graph 持有最终 padded send buffer 与 owner output，Dynamo 用具名 guarded op 表达二输入归约；fake shape 保持 `ceil(M/TP)`。input update 原位进行，完整 graph key 的地址、bands、q 与布局信息全部保留。
3. **进一步研究 G2 的 caller-owned destination。** 当前 pinned AITER `fused_moe(..., output=...)` 确实存在，其合同要求 contiguous `[M,H]`，不允许与输入重叠。可研究从 padded arena 取 `[:M]` view 作为原生输出；ATOM 量化方法/调用者的转发、sort 清零、dual-stream reader 和最后消费者尚未接入验证。它不能直接证明 G2 内部可以发布任意块。
4. **归约与下一层消费的更深融合仍需新的原生接口。** 大 M 应围绕完整 expert tile 和消费者行统计组织；在全部本地专家贡献与 shared 到齐后才发布。保留大 GEMM，按实测费用决定粗粒度 completion；不能用逐包 marker 代替真正可重叠的 producer 工作。Mega-mHC 的原有 post/pre 融合必须保住。
5. **服务证据分开记录。** 本轮 R3 和 GSM 只验收已接入的输出融合。G2 caller-owned destination、块完成发布、G2 内部通信重叠及 RS→mHC 单 kernel 仍未实现，不能从本轮结果推断。

## 5. 最初无权重原型的启动成本和证据

最初原型的环境准备暴露了缓存路径、冷加载临时空间、NCCL `/dev/shm` 等问题，以及测试代码的 workspace 参数顺序错误；原始失败保留在同一验证目录。稳定入口恢复同版本 17 个 native SO（逐文件 SHA 校验），复用已经验收的 COMGR 串行化 guard；后者只串行启动期 native action，不改参数/返回值。该阶段实测临时目录采样峰值约 193.5 MB，所有探针无模型加载；后续真实 FFN、整网与 spawn 入口修复另有独立记录。

本轮保守计费峰值 9,955,192,722 B，包括保留旧节点账单、550 MB tmpfs、256 MiB SHM 及全轮 32 MB 输出额度。没有扩大 10 GB 限额。

主要证据均位于项目的 `validation/wide_output_20261028/`：

- `RESULT_REVIEW.json`：第一版 push 的协议、L2、完整输出段结果。
- `STAGE_DIAGNOSIS.json`：同设备指令的独立分段诊断。
- `OWNER_PULL_FINAL_REVIEW.json`：第二种传输方向的结果。
- `NATIVE_JOIN_REVIEW.json`：有效布局候选的五个 M、四 rank、L2 和 ABBA。
- `native_join/combine.py`、`native_join/probe.py`：唯一有收益候选及独立验证入口。
- 各实验的 `SOURCE.json`、`ADMISSION.json`、`output/`：源码绑定、隔离额度、原始日志、产物与普通退役。

## 6. 缩短迭代周期与输入链的实际路由

R3每次独立服务的总墙钟约748秒，其中正式生成约301秒，约447秒属于启动、加载、预热和收尾。不能将全部447秒都称为“权重加载”。四组重复这些阶段接近半小时；继续重复已完成ABBA不能解释约0.5%的增量。

新的仓外 `validation/whole_join_20261028/single_load/` 在启动前安装 native/vector 两条输出路径。EngineCore 在无 running/waiting/prefill-waiting 请求时，使用原有串行 utility RPC 向四 rank 下发具名 variant 和单调 epoch；四 rank 同步并确认后，清除 block manager 的 KV 与 state-checkpoint 前缀索引。权重不重载、源文件不热更新，decoder 图及完整地址/bands键保留。预热两条路径后一次 seal，每一阶段独立检查实际路径、worker身份、权重指针、图绑定以及零 cold 操作。只允许 eager FFN 出口切换；遇到捕获立即拒绝。

768请求/组用于较快筛选，保留原始结果、固定输入与并发，并与最终2560请求/组明确区分。减少请求数和减少加载次数是两种节省，分别计费。完整性能验收仍支持2560。

单次加载流程已实测完成：总墙钟 **861.355秒（14分21秒）**，四组正式生成 **383.408秒**，其余 **477.947秒**。对照历史四次独立启动、每组2560请求的3008.283秒（50分8秒），本轮筛选周期为 **3.4925×**；其中单独减少重复启动/预热/收尾约 **1311秒（21分51秒）**，剩余差异主要来自请求数缩短，不能全部算作加载优化。保持2560请求/组、采用已测固定开销的粗估约28.2分钟，**尚未实测该完整单次加载工作量**。

本轮短测 pooled native **7339.979433 → vector 7409.985543 tok/s，1.009537644×**，两对1.007917547/1.011163540×。四 rank 的PID、权重指针摘要和39个原生图绑定跨四阶段不变；每阶段清空前缀索引，正式新增JIT/编译/捕获为0，18层实际路径符合选择。短测排空尾段占比更高，因此该0.954%用于方向筛选，不替代R3完整2560请求的0.546%结果。

复核与耗时账本：`single_load/REVIEW.json`、`single_load/TIME_SAVINGS.json`；实际入口 `single_load/launch.py`。当前脚本冻结本轮768参数；后续新实验必须使用新的输出和source manifest，完整driver可选2560，不覆盖旧结果。

适用边界是相同模型/TP/EP/DSpark配置下预先安装的eager FFN路径。改变权重布局、TP/EP或投机配置、修改捕获图内部算术时，仍需独立启动与对应图验收；该流程不用于在线重载源文件，也不把主线Spec0与候选Spec5简单切换成同一实例。

输入链首个 `quant_destination` 探针核实了 dispatch 假设：M4097/7835/16315全都命中 native custom AllGather，因而受保护的RCCL原位接收分支**一次也没执行**。原始FFN近乎持平的计时只属于原生回退，不能接受原位API、归因收益或继承GSM。`ROUTE_CORRECTION.json`保留这一边界，不重跑相同探针。

实际 pinned AITER 的 eager `all_gather_unreg` 先 `hipMemcpyAsync` 到已经注册的IPC input pool，再调用同一个AllGather。后者有原生 `end_sync` 收齐远端读者。因此新的 `quant_pool` 探针研究原生量化直接写这个pool，再走原生 registered AG；router BF16、scale编码、shared/G1/G2保持原样。只允许真实torch-backed的原生IPC pool，拒绝raw/VMM/symm等未验证存储，不构造虚假的Tensor owner。必须测实际eager完整FFN；graph原生registered路径本来可能省掉复制，不能用graph计时替代eager收益。该候选的整网集成另行验收。

`quant_pool` 已完成真实layer8 eager完整FFN、三次变化输入和128次graph重放后的再更新。每rank确认152次直接使用真实注册pool指针，三个M的FP8输入相对L2=0、scale字节一致。结果如下；没有把这个单层结果加进服务倍率：

| M | 原生 eager FFN ms | 量化直写 pool ms | 比率 | 每层节省 |
|---:|---:|---:|---:|---:|
| 7835 | 1.574008107 | 1.570170283 | 1.002444× | 3.84 μs |
| 16315 | 2.841735005 | 2.827955484 | 1.004873× | 13.78 μs |

两对均正向；owner相对L2分别为0.000245–0.000325与0.000196–0.000267，与Native自重复同量级。`quant_pool/REVIEW.json`保存逐rank与原始记录。该实现仍是严格部署下的验证代码，直接依赖communicator私有pool，尚无所有生产调用者的显式借用合同或整网验收，因此**不晋级生产**，也不继承上一版GSM结果。下一架构边界应将norm→量化→通信暂存的owner、stream和最后读者统一，而不是把私有pool指针散布到各层；大GEMM及原生同步保持原样。本轮不再重复整网验收这个仅单层微收益的原型。
