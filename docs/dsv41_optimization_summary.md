# DeepSeek-V4.1-Flash 优化汇总与交付说明

2026-10-30。面向 AMD MI355X 上的真实模型推理，当前 FP8 / TP4 / EP1 / C256 配置已有完整服务样本 **5269.811 → 6281.294 output tok/s，1.191939×，吞吐提高19.19%**；同输出量的正式时间减少16.10%。这个结果来自一次新迁移运行与保留的同节点基线，存在时间间隔和正式冷编译，尚未成为稳定性能及任务精度验收。

最近几轮 Window 实验没有带来新的整网收益。保留的供料优化在局部完整窗口快3.97%，三种分列 Window 均已停止。下一项 attention 去复制候选已编写，**GPU 未运行，收益未知**。本次提交整理代码与证据并 rebase 主线，不产生新的性能成绩。

## 1. 针对什么场景

| 项目 | 当前目标与实际测量范围 |
|---|---|
| 模型 | DeepSeek-V4.1-Flash，真实 checkpoint；隐藏维5120、384个 routed experts、top-6 |
| 硬件 | AMD MI355X / gfx950；进程使用4卡，测量节点8卡独占 |
| 并行 | attention / routed / shared TP4，EP1，DP1，无 DPA / DEP |
| 数值与推测解码 | FP8 KV、FP4 index、真实 DSpark5、完整验证行数；无合成接受率 |
| 服务工作负载 | C256；random input8192 / output1024，range0.8，seed0；2560条正式请求 |
| 受控实验预热 | 593条独立预热；公开 dashboard 入口默认512条，两种协议不能混称 |
| 引擎 | max-num-seqs512、max-batched-tokens16384、attention chunk16384、level3 / FULL graphs、原生 prefix cache 与 bounded replay |
| 主要 decode 形状 | 全局 M1536，每 owner384行；专家 intermediate 有效576、物理 padding640 |
| 主要 prefill 代理 | M7550、M16291；单层测量不能代替全部层与请求轨迹 |

优化目标是减少分布式推理中的重复供料、投影、归约和状态写读，同时保持原生大矩阵 GEMM 的执行方式。当前 C256 不是小 M Mono 的适用范围；没有扩大小 M guard，也没有用旧 packed M768 代替 full M1536。

部署核对见[实际公开配置](dsv41_c256_verified_config.md)和[原生同配置基线](dsv41_c256_baseline.md)。历史 BF16 / EP4 / packed 的结果单列在下文，避免混用分母。

## 2. 优化了多少

以下“加速比”均为 baseline time / candidate time，或相同输出量下的 candidate throughput / baseline throughput。“时间变化”和“吞吐变化”分别计算。

| 范围 | 对照 → 候选 | 结果 | 证据与采用状态 |
|---|---|---|---|
| 当前配置原生整网 ABBA | 5269.8108 → 5298.3810 tok/s | 1.00542148×，+0.542%吞吐 | 两对倍率0.99462092 / 1.01619355，方向不一致，无稳定独立内核收益 |
| FP8 owner + retained-prefill 整网 | 5269.811 → 6281.294 tok/s | **1.191939×，+19.19%吞吐** | 新单臂对保留同节点主线；可选路径，未提升为默认 |
| 同上正式耗时 | 446.783019 → 374.837073秒 | **减少71.945946秒，−16.10%时间** | TPOT46.1870 → 38.5883ms；相对原生分支为1.185512× |
| 新供料：layer8完整FFN/RS → layer9 mHC/norm/QKV | 0.75551 → 0.72667ms | **1.03969×，−3.817%时间，约28.84µs/次** | 三组输入、四rank、24个配对全部正收益；尚未集成服务 |
| Attention A0 去复制 | 保留四个H16，直接使用Q/scale view及调用方输出 | **未测** | 只有CPU源码/布局准备检查；无权重加载、GPU或服务成绩 |

FP8 迁移正式区间，两侧每臂均有18,926,022输入tokens、2,354,462输出tokens。接受长度从2.488826降为2.471299，decode行数/rank从5,804,691增为5,835,366（+0.528%），因此不能把收益解释成接受长度升高或裁掉验证行。实际覆盖全部40层TP4/EP1、3724个M1536 FFN步骤、1846次 retained-prefill / 18,623,230行；四rank均完成图释放和原生退出。

正式阶段每rank仍出现4个新 Triton 特化，零冷编译检查失败，耗时完整保留，没有扣掉“估计编译时间”。本 FP8 配置没有新的 GSM8K 或完整模型数值验收。代码归并后的入口有CPU检查，历史性能仍绑定原始运行版本；**rebase 后的提交未做GPU复测**。

供料优化保留原生G1/G2 callable、参数与实际M/K，G1输入FP8 values/scales逐字节相同。固定 full-M gate 调度1765解决 owner384普通选表会选择不同 split-K 的问题；logits、top-6 IDs/weights与full-M原生逐字节相同。完整链的差异与原生atomic自比较同量级，这不是完整链逐位等价或任务精度验收。按旧调用数和38层代理外推仅约4.08秒 / 1.09%整网时间，**该外推不是实测服务收益，不能乘到1.191939×上**。

详细结果：[FP8 owner整网](dsv41_fp8_owner_result.md)、[供料局部完整链](dsv41_fp8_owner_supply_result.md)。

## 3. 最近哪些方案没有收益，为什么停止

每行只与同次运行的对照比较。decode对照已含供料候选，prefill对照为原生owner路径；表中正数表示时间增加。

| 候选 | Decode M1536 | Prefill M7550 | Prefill M16291 | 决定 |
|---|---:|---:|---:|---|
| BF16十列 + RCCL reduce-scatter | +33.92% | +7.90% | −0.30% | 停止 |
| FP8设备内 Window | +4.987% | +10.470% | +8.891% | 停止，新增压缩数值未验收 |
| 紧凑BF16 + chunk-ready | +27.19% | +30.42% | +43.33% | 停止，72个有效净配对全部负收益 |

最终紧凑BF16完整窗口分别为0.728004 → 0.925934ms、2.033622 → 2.652151ms、3.883588 → 5.566313ms。同算术的chunk等待相对逐列等待确实收回35.927 / 88.932 / 180.401µs，但不足以抵消新装配的成本。最终产物无spill、无FP8中间转换，也已删除逐列全rank barrier，因此不能继续把退化全部归因于其中某一项。

能确定的是：分列执行、partial物化与就绪管理增加的总成本超过回收时间。尚无足够证据精确分摊原生G2效率损失、物化、重复flag acquire、计算CTA与服务CTA竞争各占多少；不为解释旧失败再次开参数扫描。

两项前期错误独立记录：M16291布局错误复用了旧形状；BF16x8 / BN128 / 32lane组合产生零次store。最终R5 decode修正为16lane，R4 decode成绩作废，两档prefill保留有效R4。最终三形状结果用了两次真实权重加载，不写成一次全成功。即使修复后，净退化仍然成立。

详见[BF16列RCCL](dsv41_fp8_column_rs_result.md)、[FP8设备Window](dsv41_fp8_device_window_result.md)、[紧凑BF16最终结果](dsv41_bf16_chunk_window_result.md)。

## 4. 时间花在哪里，流程如何收紧

现有记录能说明主要工作类型，不能提供整晚精确的人工/GPU时间占比。以下时间来自各自记录，不能相加成同一预算。

| 时间投入 | 已有证据 | 下一步处理 |
|---|---|---|
| 环境、加载、编译、图准备 | 真实模型预热曾约301秒，新迁移约104秒；缓存复用的旧服务ready约165秒，较旧362秒缩短。这些是启动成本 | 固定依赖、复用加载器与代码缓存；不计入正式吞吐收益 |
| 完整服务反复验证 | 每个正式2560请求臂约375–447秒，另需启动和预热；ABBA需要四臂 | 只有新候选完整局部窗口净收益成立，才做新组合服务验证 |
| 小范围Window反复装配 | 三种方案均退化；完整FFN代理仅约49%当前wall，G2/RS又只是其中一段 | 停止这套分列路线，保留全M/K/N原生大GEMM，优先删除实际工作 |
| ABI、布局、实际写回与生命周期 | 先后遇到图内事件API、多形状布局、零写回、退出/导出及容量问题 | 加载权重前做接口、步长、非零写回、图释放与容量检查；失败单独留证 |
| 观察工具本身 | 旧decode profiler约2倍扰动，不能用于生产预算；稀疏enclosing估计仍高于wall约3.54%–3.64% | 完整窗口用无标记A/B；阶段成本独立采样并测观察扰动，不把分项相加冒充精确整网积分 |
| 日志、证据与资源 | 历史记录无损压缩及缓存清理回收约1.079GB；节点申请曾等待Resources | 每次新运行重新做容量准入；新增记录只绑定本次变化，不重建整套历史评审 |

因此，“没有收益”准确适用于最近的 Window 尝试；整个项目已有历史整网收益，当前FP8迁移也有19.19%的样本提升。问题在于后续多轮投入仍集中于覆盖有限且新增成本高的路径，完整attention的净收益与阶段成本一直缺少直接测量。

完整FFN的当前成本代理来自真实layer8权重，四rank范围如下，单位ms。观察阶段使用独立图，不能直接求和。

| 全局M | 完整FFN | G1 | G2 + 本地归约 | RS | 下一层mHC/norm/QKV |
|---|---:|---:|---:|---:|---:|
| 1536 | .7070–.7083 | .2413–.2581 | .1539–.1584 | .1008–.1158 | .0351–.0357 |
| 7550 | 1.9172–1.9217 | .5690–.5747 | .3987–.4150 | .4301–.4405 | .0812–.0833 |
| 16291 | 3.6255–3.6375 | 1.0319–1.0433 | .7702–.7736 | .9022–.9172 | .1431–.1484 |

达到原主线2×需要223.391510秒，相对当前374.837073秒还须减少 **151.445563秒 / 40.403%时间**。以49.06%的完整FFN代理覆盖估算，即使FFN整体快2×，其他范围仍需减少约59.5秒。这是条件预算，不是2×可达承诺，也不是硬件极限结论。详见[成本结果](dsv41_fp8_owner_window_cost_result.md)与[重排优化计划](dsv41_c256_2x_replan.md)。

## 5. 当前代码做了什么，如何启用

| 代码范围 | 作用与状态 |
|---|---|
| [owner服务实现](../atom/models/deepseek_v41/owner_service/) | 模型专用owner执行、FP8上下文传输、attention、retained-prefill与生命周期；可选接入 |
| [owner attention适配](../atom/models/deepseek_v41/owner_service/owner_head_attention.py) | 当前运行路径仍为四个H16，含Q/scale复制与输出拼接 |
| [完整owner attention链](../atom/models/deepseek_v41/owner_service/owner_cost_attention_chain.py) | QKV、索引/缓存、attention、投影及原生状态合同 |
| [部署准入](../atom/models/deepseek_v41/owner_service_contract.py) | 限定已知部署，FP8 EP1与旧BF16 EP4分别处理 |
| [统一dashboard入口](../tools/dsv41/run_dashboard.py)及[profile](../tools/dsv41/dashboard_profile.json) | 默认native；显式fp8-owner启用迁移组合，旧EP4/packed不混入 |
| [attention A0源码快照](../tools/dsv41/experiments/attention_stride_r1/README.md) | 本次保存的未验证候选与fixture/probe/launcher；不安装到运行路径 |

在已准备兼容ROCm/AITER的环境中，可生成新配置供审阅：

```bash
python tools/dsv41/run_dashboard.py plan --execution fp8-owner \
  --model /models/DeepSeek-V4.1-Flash \
  --aiter-source /path/to/compatible/aiter --output /path/to/new-plan
```

按入口帮助使用`preflight`核验实际依赖，使用`serve`启动服务，输出目录须为新目录。公开profile默认512预热，不会自动复现本报告593预热的受控实验，也不是零冷编译/任务精度验收器。AITER须含V4.1 FP8 MLA及相应原生接口；[依赖快照](dsv41_results_20261030/FP8_DEPENDENCY.json)记录实际源码SHA。模型权重、容器、缓存、共享库及仓外完整验证目录不随Git提交。

## 6. 历史结果的适用范围

| 历史配置/候选 | 性能/精度结果 | 边界 |
|---|---|---|
| BF16 C5组合：Spec0主线 → DSpark5 + owner packed + EP4 + prefill | 5741.094198 → 7603.533722 tok/s，**1.324404976×**；TPOT42.541209 → 31.636498ms | 测量候选fc4da1513，对照ebd5f84f3；正式零冷。与当前FP8/EP1/full分母不同，不能连乘 |
| 旧输出融合精度 | GSM8K 1212/1319，91.8878% | 仅对应已测BF16/EP4配置 |
| 早期registered-input原型 | 短测表观+6.54%，GSM1137/1319，86.20%，失败 | 不算可交付收益 |
| 修正registered-input | GSM1207/1319，91.5087%；完整ABBA约+0.1218% | 无稳定净收益，生产恢复，原型归档 |
| Shared G2 / send-buffer | 7794.957890 → 7817.986205 tok/s，1.002954258×；GSM1217/1319，92.2669% | 弱收益，未提升默认；精度不外推到当前FP8配置 |

架构与旧证据见[融合架构](dsv41_fusedmoe_architecture.md)、[输出融合](dsv41_wide_output_findings.md)、[输入融合](dsv41_prefill_input_fusion_plan.md)、[prefill流水研究](dsv41_prefill_pipeline_plan.md)。旧长臂GPU故障并未因后续未复现就被证明修复；不把诊断运行算作性能或精度验收。

## 7. 下一次实验的明确交付

1. 保留供料局部收益。对实际层型/形状核验固定gate，适用时带入新组合，其余保持原生路径。不为它单独开启多轮整网测试。
2. 测完整attention A0：保留四H16算术，删除Q/scale复制与output concat；一次真实权重加载内完成数值、自比较、graph更新、完整窗口A/B及必要阶段成本。
3. 用测得成本选择下一项：本体足够大才试H64 KV复用；准备或投影更大则直接优化对应阶段。去copy尚无收益承诺。
4. FFN decode研究G1 epilogue直接生成与当前BF16舍入后相同的FP8 values/scales。prefill已融合这一步，不能重复计益，应看route物化、归约输入、投影和层间状态读写。新G2方向保持全N计算。
5. 一个候选、一个完整窗口、一次加载内完成必要数值与A/B；只有局部净收益成立，才启动新组合服务验证。历史冻结运行与review不重跑。

A0主case为owner384/global1536，数值补充为owner容量221/global883，验证layer8 FULL → layer9 REUSE的选择依赖。fixture加载真实attention/mHC权重，但hidden状态及历史缓存是构造输入，layer9使用独立residual，**两窗口之间未计FFN**。阶段标记要求对完整窗口的扰动不超过1%。该设计能筛选候选，不能代替真实服务轨迹与最终精度验收。

## 8. 文档、证据和版本

本页是入口；上文链接的详细结果保留各自原始日期、已失败/已停止的决策与资源状态。历史页中的`../../validation/...`指向共享工作区仓外目录，GitHub单独checkout不会包含它们。为便于审阅，本次另保存约220KB关键原始结果快照及[SHA清单](dsv41_results_20261030/MANIFEST.json)，没有重新执行旧评审：

| 可随Git审阅的证据 | 内容 |
|---|---|
| [结果快照说明](dsv41_results_20261030/README.md) | 来源、压缩读取方法、完整证据与摘要的边界 |
| [基线ABBA review](dsv41_results_20261030/r9/ABBA_REVIEW.json.gz) | 四臂合并、漂移与冷事件 |
| [FP8迁移review](dsv41_results_20261030/fp8_owner_service_r3/REVIEW.json.gz)、[成本/行数](dsv41_results_20261030/FP8_MIGRATION_COST_RESULT.json) | 当前19.19%样本的证据 |
| [供料review](dsv41_results_20261030/owner_supply_r1/REVIEW.json.gz) | 局部24配对与数值 |
| [完整FFN成本](dsv41_results_20261030/OWNER_WINDOW_COST_RESULT.json) | 三种M的阶段成本及外推范围 |
| [BF16最终聚合](dsv41_results_20261030/OWNER_BF16_WINDOW_RESULT.json) | R4 prefill / R5 decode、无效项排除与净退化 |
| [A0源码映射](../tools/dsv41/experiments/attention_stride_r1/SOURCE_SNAPSHOT.json) | 14个文件原始字节、路径、SHA；GPU未运行 |

版本关系：旧同配置主线为`ebd5f84f3`，原生分支与FP8私有overlay的基底为`f2d3ee47f`。六份已测迁移文件在`f723af820`归入可选owner路径，原归并后HEAD为`96405d8609abdb0e9be594b47dcd4a492e81c6d4`。本次以抓取的主线`124c3b0d1636919f03513c33d3e414d2112d2cb0`为rebase目标，性能仍绑定旧测量版本。

本次范围仅ATOM：提交原有七份待提交文档、本汇总、关键结果副本和A0源码快照，保留已有项目提交；aiter仓库的调优CSV按用户要求留在本地。远端旧提交`f12ea9a63777d05c9f5a53fd8bb2b8a144a11c9b`的MoE padding修复及测试保留，dashboard沿用当前FP8配置。原本地HEAD及原远端tip各有本地backup分支可回溯。

原A0 launcher严格锁定旧HEAD96405d860与历史工作区/依赖；rebase后不能仅替换HEAD断言直接运行。下一次须针对新原生接口准备新的实验版本并重新做加载前检查，旧快照保持不变。CPU检查的范围及本次Git交付记录见[发布检查](dsv41_results_20261030/PUBLISH_CHECKS.md)。
