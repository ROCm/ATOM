# MI355X 官方 FP8 Native AgentX + MTP 基线与 Mono 接入方案

2026-10-16。本报告完成的是当前 ATOM Native、官方 FP8 checkpoint、MTP k=3 的 AgentX 基线，覆盖 TP2/EP2 与 TP8/EP8、根会话 C1/C4。Mono 接入及性能目标是后续实施方案，尚未实现或验收。

## 1. 已测基线与结论

四点均使用全部 393 条公开 trace 作为候选 corpus，每点重新启动服务，默认预热，正式计时 900 秒，均 `submission_valid=true`，已完成请求错误率为 0。有限时间内执行的请求子集随速度不同；不声称 393 条轨迹全部执行完。

| Native 配置 | 成功请求 | 输出 tokens/s | tokens/s/GPU | TPOT p50 / p95，ms | TTFT p50 / p95，s | profiling 内部采样 AL |
|---|---:|---:|---:|---:|---:|---:|
| TP2/EP2 k3 C1 | 94 | 81.42 | 40.71 | 5.85 / 17.67 | 2.72 / 8.98 | 1.953 |
| TP8/EP8 k3 C1 | 113 | 99.14 | 12.39 | 5.19 / 9.42 | 1.77 / 5.21 | 2.013 |
| TP2/EP2 k3 C4 | 176 | 155.91 | 77.96 | 7.60 / 54.19 | 4.32 / 8.26 | 2.199 |
| TP8/EP8 k3 C4 | 181 | 170.21 | 21.28 | 5.88 / 24.54 | 2.30 / 4.85 | 1.803 |

TPOT 使用 AgentX 导出的逐请求平均 inter-token latency 的分位数，不是逐 SSE chunk 间隔。MTP 一次可以发送多个 token；AL=平均每次目标验证的输出 token 数，包含保底 token，k=3 最大为4。AL取独立30秒计数在 profiling 内部的差分，未覆盖两端所有秒数；C4 TP8区间也可能包含部分drain。完整原始计数保留。

TP8 相对 TP2：C1 总吞吐 +21.8%，C4 +9.2%；C4 p95 TPOT 降低54.7%，p95 TTFT降低41.3%。两种配置消耗 GPU 数差4倍。当前结果支持把 TP2 作为每 GPU 吞吐优化的首个接入目标，把 TP8 作为低尾延迟对照；它不是完整TP搜索，TP1/TP4及k1/k2尚未测。

TP8 C4 的默认30秒drain到期时还有2个长请求，客户端取消了它们；181为成功请求数，取消未计入已完成请求错误率，harness仍标记有效。其他三点没有该取消记录。优化后的评估必须同时列出取消量，增加drain的补测要与本协议结果另列，不能静默改变口径。

所有点 scheduler preemptions=0，无 OOM、context overflow、OSL mismatch。最大同时运行请求数：C1=3，C4=7；C表示根会话，不能当作服务器最大batch。KV池最高使用率分别约2.80%、1.99%、5.75%、4.19%，本次瓶颈不是KV容量耗尽。

| 配置 | 输入长度均值 / p95 | 输出长度均值 / p95 | 活跃解码 tokens/s | 活跃 prefill tokens/s |
|---|---:|---:|---:|---:|
| TP2 C1 | 86,452 / 236,238 | 780 / 4,527 | 149.82 | 25,044 |
| TP8 C1 | 87,826 / 235,752 | 791 / 3,920 | 190.78 | 46,672 |
| TP2 C4 | 70,895 / 140,319 | 804 / 4,126 | 208.47 | 28,542 |
| TP8 C4 | 74,146 / 170,509 | 884 / 4,369 | 232.05 | 43,158 |

活跃吞吐排除了部分闲置区间，与全程输出吞吐是不同指标。总体结果包含trace中的工具等待、SPAWN/JOIN依赖和回放调度。不同速度会改变900秒内执行到的轨迹，必须结合分箱与固定输入诊断评估收益。

上下文分箱结果（TTFT/TPOT均为p50；样本数较小的分箱只作描述）：

| 配置 | <32K：n / TTFT s / TPOT ms | 32–64K | 64–128K | 128–256K |
|---|---|---|---|---|
| TP2 C1 | 2 / .756 / 5.674 | 46 / 1.943 / 6.265 | 31 / 3.020 / 5.588 | 15 / 8.609 / 5.580 |
| TP8 C1 | 3 / .135 / 4.130 | 48 / 1.098 / 5.495 | 47 / 1.987 / 4.742 | 15 / 5.192 / 5.299 |
| TP2 C4 | 19 / .722 / 6.704 | 75 / 3.994 / 7.712 | 70 / 4.882 / 9.341 | 12 / 6.177 / 5.546 |
| TP8 C4 | 19 / .425 / 5.813 | 75 / 1.652 / 6.256 | 71 / 2.672 / 5.888 | 16 / 3.688 / 4.732 |

## 2. 固定配置与适用范围

- 硬件：作业217242，crsuse2-m2m-080，8×MI355X/gfx950；每点只使用对应TP数量GPU。
- checkpoint：`Qwen/Qwen3.8-Flash-Next-FP8`，官方block FP8；BF16 KV，保留checkpoint量化排除和PLE存储策略。
- ATOM `0617ed48bbd4b81dd1933947cf31e23d3767138f`，AITER `ea3d22a304114dfac30b2a6c1c466a451d6d1b05`；测量时两仓clean。
- 服务Torch2.10、ROCm7.2.4、Transformers5.16.1；client独立venv，agentx-harness1.0.11。
- TP与EP同组，max-model-len262144，max-seqs32，max-batched8192，memory utilization.90，chunked prefill。
- Native MTP k3，单个draft layer复用，target FULL graph capture [1,2,4,8,16,32]，verify最大q=4。mRoPE draft mid-steps没有draft graph replay。
- ReplaySSM开启，cache_len16；每请求一个GDN状态checkpoint。Mono关闭。
- 当前Native MTP配方关闭prefix caching。本报告不声称是cache-enabled的最佳AgentX后端。
- AgentX scenario：streaming chat、ignore_eos、seed42、轨迹起点.25–.75、first_turn_prefix cache bust、10 additional warmup requests/lane、trace idle cap300秒。
- dataset `semianalysisai/cc-traces-weka-062126-256k`，revision `8fecd2fc56694469f758f0afbbb6335ad3043740`，SHA256 `e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`；393/393载入，筛选观察最大context255,999。
- InferenceX protocol参考固定到 `77a21c0aa849d1bd3cdcb23ee33554ebeb4d88e4`。

未显式传采样参数，Native chat API默认temperature1/top_p1/top_k-1。测量真实接受率，没有注入golden AL。上游golden AL来自其他coding corpus/后端及不同采样设置，不能作为本次接受率。

AgentX回放固定请求/输出长度，不实际执行生成代码，成功率不代表SWE正确率或模型数值精度。旧GSM8K整网精度任务仍暂停，长上下文模型精度未因此验收。没有电力采集/audit，本结果也不是完整InferenceX leaderboard submission。

本地独立AITER cache修复了旧二进制缺少Relu2枚举导致的启动失败，没有修改模型计算语义。NVFP4、PTPC模型均未用于替代官方FP8。

## 3. Profiler证据与目标选择依据

正式点结束后，另跑120秒AgentX诊断，在TP2 C4配置上开启10秒profiler。两rank各导出约13MB trace，实际GPU活动窗口约6.05秒，rank0 kernel总时间5.98秒。该窗口包含21个长prefill chunk、6个decode verify，以及draft步骤，主要代表长prefill路径，不能外推为全程性能占比。

rank0按kernel名称近似分类：跨卡归约约25.7%，MoE/路由/采样约24.4%，QSA/index约16.4%，其他dense/MoE GEMM约15.2%，HC/norm/quant约7.5%，GDN recurrence/conv约3.9%。分类不能分离所有target/draft GEMM，通信kernel也包含peer等待，不能全部算有效传输时间。

主要单kernel累计时间：cross_device_reduce_1stage约1.536秒，FP8 fmoe约1.394秒，QSA sparse paged GQA约.681秒，QSA index logits约.257秒。rank1结果接近。GDN并不是这个窗口最大的开销，因此只优化GDN不能覆盖场景主要成本。

CPU scope：6次decode target提交平均约2.2ms；draft step0约3.1ms、step1/2各约2.7ms。这些是profiler下的CPU inclusive scope，不能相加当作无profiler GPU TPOT。它们说明draft eager路径也需优化，单独加速target受Amdahl上限限制。

优先方向：完整MTP状态/输出契约 → TP2/EP2多query target → draft融合与可正确replay的graph → peer归约/MoE/QSA → 长前缀复用和prefill调度。性能预算以无profilerCUDA event/固定输入测量补充，本次profile用于确定方向。

## 4. Mono接入架构

### 4.1 服务和调度边界

继续使用Native HTTP、AgentX兼容chat模板、scheduler、block/state allocator、rejection sampler。prefill初期保留Native路径。接入应由step metadata选择Mono verify/Mono draft/Native，所有fallback计数可观察；不允许通过删除speculative拒绝条件就宣布支持。

当前代码仅生产启用TP1/S1；TP8通过显式bootstrap实验。`config.py`拒绝speculative_config，`dispatch.py`拒绝drafter，`runner.py`拒绝return_hc_state，`step_meta.py`拒绝spec decodes和ReplaySSM；TP2没有kernel ABI。需逐项实现并通过契约后开放。

### 4.2 多token target输出和metadata

新增明确的VerifyStepView：request ids、cu_query_start、actual/padded lengths、每请求accepted count、logical positions、block/state slots、active mask、ReplaySSM write_pos/slot_idx。B请求×(k+1)候选token按请求有序排列，不能把S=4当作4个独立S1请求。

target保持完整HC bundle，形状遵循Native `[T,hc_count,hidden]`，同时保留Native logits路径和draft共享embedding/head。普通decode输出与spec target输出分别定义，避免现有final mixer提前合并HC。PLE是target的一部分，draft没有PLE层；draft仍需正确消费对应target HC行。

### 4.3 GDN / ReplaySSM / PLE 提交语义

GDN卷积候选窗口、recurrent checkpoint和ReplaySSM records必须与Native相同。verify可计算全部候选，但只使接受前缀及保底token成为下一轮可见状态；拒绝tail不能改变下一轮write_pos。chunk checkpoint保存、切换state slot、fork源/目标与slot回收遵循Native metadata。

实现首选直接接Native ReplaySSM buffer/write_pos契约，保留Native恢复路径。若为性能改用临时候选state，也要与ReplaySSM共同设计桥接，不能隐式转换历史状态。状态写入不依赖host同步或复制整个每层矩阵。

PLE ngram和dilated conv窗口也存候选tail；下一轮按accepted count选择窗口。需覆盖跨chunk、跨接受长度、冷slot重用及prefix restore。QSA恢复正确而PLE历史错误仍会造成整网偏差。

### 4.4 QSA多query及长上下文

对每个candidate维护独立logical position、causal边界、compressed slot、raw/compressed key及rope position plane；accepted count修正draft下一轮paged view。候选压缩块越界或拒绝后重算时，不可把未提交压缩记录纳入未来索引。

初期沿用Native QSA验证，先让Mono接管HC/GDN/MoE并明确标记partial。完整Mono验收必须实现q2/q3/q4 causal索引与注意力，并通过上下文边界和页表测试。context buckets建议4K/16K/32K/64K/128K/256K，最终根据实际shape/time分布裁剪；不能把旧256token direct路径当作AgentX覆盖。

### 4.5 TP2/EP2 与 TP8/EP8

扩展Dims、weight binding、peer layout和kernel编译key。dense/head projection按TP切分；MoE按完整expert归属，保留I=640及128列scale block，不切成不合法80/320列的FP8块。TP2每rank256个expert、TP8每rank64个expert，router top10和shared expert语义一致。

peer buffer按B×q宽度、layer parity与step epoch标记；不同graph width及accepted长度切换不能误读旧mailbox。覆盖rank对称、全rank共识、零owned-route、padding和所有rank参与归约，避免persistent CTA deadlock。先实现稳定边界，再把归约与HC/MoE收尾融合。

### 4.6 Draft接入

阶段一保留Native draft，先验收Mono target。阶段二实现一层draft的embedding/HC norm、fc_embedding/fc_hidden、QSA、MoE、final mixer/head，并共享权重而非另存大表。

step0处理target整个token流，prefill和decode形状不同；steps1..k-1每请求一行且串行依赖采样结果。支持mRoPE `[3,T]`固定graph输入缓冲并原位更新正确的最后一维，不能复用误以为batch在第一维的graph staging。动态accepted长度、slot和positions全部更新后才replay。Graph bucket以B和q分别建key，q长度与request维度不可混淆。

保持原采样与rejection sampler，head/sampling可融合但需证明对应随机数与概率契约。k1/k2/k3按profile测量选取，接受率不能通过跳过验证或伪造AL提高。

### 4.7 Prefix caching与prefill

本次实际cache hit为0，而trace理论可复用前缀约88%–96%。这是另一条大收益路径，需要先完成Native MTP下的跨轮状态restore契约，再与Mono复用同一cache。不能把未验证cache打开后的性能收益归到Mono kernel。

缓存对象必须联合包含paged QSA、GDN checkpoint/ReplaySSM可恢复tail、PLE ngram/conv、logical position与draft所需状态；key只对应已提交token前缀。SPAWN共享前缀、冷/热恢复、eviction后重算、分叉后独立写均需正确。遵循ATOM现有state checkpoint/allocator，初期不引入当前Qwen不支持的外部KV offload。

prefill优化采用Native大矩阵kernel，先优化实际重算token量，再调整chunk和decode优先级。8192 chunk对TP2尾延迟的影响需用2048/4096/8192对照；保留总吞吐并限制最大decode间隔，不能只缩chunk换更好p95却隐藏吞吐损失。Mono不应全程占满全部CU而阻断混合批次的调度公平性。

## 5. 性能目标（开发验收值，非已达成结果）

### 5.1 无prefix caching的同配置目标

以本次四点为起始基线，第一版完整target+draft接入的目标为p50 TPOT减少20%，aggregate输出吞吐提高15%。C4同时要求p95 TPOT减少30%；TTFT p95允许最多5%回退。它们是规划目标，需要重复对照证明，并非profile可推导出的保证收益。

| 配置 | p50 TPOT目标，ms | p95 TPOT目标，ms | 输出tokens/s目标 | tokens/s/GPU目标 | p95 TTFT上限，s |
|---|---:|---:|---:|---:|---:|
| TP2 C1 | ≤4.68 | ≤17.67 | ≥93.64 | ≥46.82 | ≤9.43 |
| TP8 C1 | ≤4.15 | ≤9.42 | ≥114.01 | ≥14.25 | ≤5.47 |
| TP2 C4 | ≤6.08 | ≤37.93 | ≥179.30 | ≥89.65 | ≤8.68 |
| TP8 C4 | ≤4.70 | ≤17.18 | ≥195.74 | ≥24.47 | ≤5.09 |

不能只择一个指标达标：稳定性、上下文覆盖、采样/state正确性和主要性能指标需同时满足。某点fallback过高时只能报告partial结果。

为什么先设20%而不是沿用历史1.5×：历史4.60ms是短prompt/noMTP/S1实验，当前有draft三步、q4验证、长prefill和子代理。单独target加速的总收益受draft/prefill/runtime限制。target+draft+runtime预算按 `T_decode ≈ (T_verify + ΣT_draft + T_sampler/state/runtime) / AL` 跟踪，混合场景还需加prefill排队/插入时间，不能只除AL预测用户TPOT。

阶段预算：相同输入无profiler的verify GPU时间目标≤Native75%，draft三步总GPU及提交时间≤Native70%，state/metadata预算≤总decode迭代5%；以独立events/trace确认，不将CPU inclusive scope当作GPU执行时间。若这两部分在真实decode总时间中权重不足以支持20%TPOT，先更新测量拆分和实现优先级，不降低验收口径掩盖问题。

### 5.2 Prefix caching后的系统目标

另立Native cache-enabled与Mono cache-enabled两组基线，独立于上述无cache成绩。建议系统目标：多轮长前缀实际重算prompt token量至少减少80%，热轮次p95 TTFT相对本次无cache至少减少50%，p95 TPOT不回退，完整回放吞吐目标≥1.5×本次对应点。

这些目标依赖cache支持及trace实际命中，不可只用理论hit推算：checkpoint恢复、prefix key、内存驻留/驱逐、trace等待和draft成本都影响结果。要求按cold/hot轮次、上下文分箱与实际cache hit分别报告；首次冷prefill不适用热轮次目标。

### 5.3 更高并发与规模

完成C1/C4验收后扩C8/C16和TP4，记录最大实际子代理并发、admission/queue及每GPU吞吐。TP4和k1/k2在未测前不承诺绝对目标。全节点多副本TP2×4也未测，不能把77.96 tokens/s/GPU乘8当作已实现的整节点吞吐。

## 6. 正确性与性能验收

1. host契约与GPU局部差分：q1/q2/q3/q4、TP2/TP8，state read/write、候选position、compressed boundary、graph padding和HC输出形状。BF16逐阶段使用独立CPU参考/Native同输入检查误差，不能只比较tensor shape。
2. 强制接受长度0/1/2/3、全拒绝/全接受/交替序列，逐轮校验GDN checkpoint/ReplaySSM尾、conv、PLE、QSA页内容及下一轮logits。跨slot回收、chunk续算、prefix fork/restore和长context边界必须覆盖。
3. 同输入target logits与draft logits统计：max/RMS/relative误差、cosine、top-k/routing差异及低margin分歧，分层设置容差。容差由Native重复性和当前独立参考误差测定，禁止事后放宽来通过失败样本。
4. sampler/rejection契约：正确接入Native概率和随机数，greedy定序用固定token前缀逐步验证；随机采样不要求两个完整随机答案完全相同，需检查概率/接受分布与state不污染。AL变动必须结合相同输入和采样设置解释。
5. 场景稳定性：393条全部载入、native262144窗口、SPAWN/JOIN无悬挂、context overflow=0、OSL mismatch=0、无新增OOM/抢占/请求错误，报告所有cancel和fallback。
6. 性能采用新服务、相同版本/参数，至少ABAB各两次900秒；候选稳定后关键点用默认3600秒。按root trace bootstrap置信区间，AL和ISL/OSL分布一起报告。单次这四点是开发基线，尚无重复性置信区间。
7. profiler诊断与正式计时分离，记录warmup、steady-state、drain范围。发布表需包含有效性、吞吐/GPU、TPOT/TTFT/E2E p50/p95/p99、分箱样本数、实际batch/cache hit、draft depths、fallback比例和取消量。

此处正确性工作属于未来接入验收方案，旧GSM8K暂停状态不变。本次benchmark没有证明整网质量或长context模型精度。

## 7. 实施顺序与交付

| 阶段 | 主要改动 | 完成条件 |
|---|---|---|
| P0 契约/计量 | VerifyStepView、HC输出、TP2形状、fallback/accepted-state计数 | GPU同输入契约可审查，无spec门禁误开 |
| P1 Target | TP2/EP2多query HC/GDN/PLE/MoE；初期Native QSA/draft | q1–q4状态差分通过；partial覆盖明确 |
| P2 完整verify | QSA causal多query、ReplaySSM提交、TP8 peer | 所有目标层Mono，长context/拒绝tail正确 |
| P3 Draft | 单层MTP融合、mRoPE正确graph、共享head/embedding | 完整target+draft路径，达到同配置性能表 |
| P4 系统路径 | Native MTP cache restore、Mono cache绑定、prefill chunk调度 | cold/hot正确，达到系统目标并分别归因 |
| P5 扩展 | TP4、k深度、C8/C16及多副本 | 实测Pareto与重复置信区间，按结果选默认 |

PR按阶段保持可审查commit；核心实现与本地kernel放ATOM，保留独立数值参考，不修改AITER全局CSV来携带仅此模型的实验调参。每阶段先验收再开生产gate。

## 8. 复现与artifact

复现脚本已收录在ATOM `tools/qwen_agentx/`；本次执行副本位于workspace `data/`，包含`start_qwen_agentx_native.sh`、`run_qwen_agentx.py`、`sweep_qwen_agentx.py`、`summarize_qwen_agentx.py`、`profile_qwen_agentx.py`、`analyze_qwen_agentx_profile.py`。脚本固定本次环境路径；在其他环境需调整root/checkpoint/client及scripts路径。

GPU容器 `yifehuan_qwen_tp8_native_217225`，通过已授权作业执行脚本。例：

```bash
spur exec 217242 docker exec yifehuan_qwen_tp8_native_217225 \
  bash /home/yifehuan/glm5_tileRT/data/start_qwen_agentx_native.sh 2 3 UNIQUE_TAG
spur exec 217242 docker exec yifehuan_qwen_tp8_native_217225 \
  /var/tmp/qwen_agentx_client/bin/python \
  /home/yifehuan/glm5_tileRT/data/run_qwen_agentx.py \
  --tag UNIQUE_TAG --concurrency 4 --duration 900
```

启动前检查当前GPU任务，前一服务应完成并正常退出；TAG不可覆盖已完成结果。脚本本身不申请/切换GPU作业。

artifact根目录 `/home/yifehuan/data/qwen38_agentx_native_20261016/`。每正式点有manifest、server/client log、profile export JSON/JSONL、server metrics及time slices、MTP/cache快照、summary.json；profiler诊断在`tp2_k3_c4_profile`，trace保存在最后服务的`tp2_k3_c4_formal/profiles/rank_0|rank_1`，正式计时已结束后才开启。

现有Mono PR：<https://github.com/ROCm/ATOM/pull/2610>。本方案不声称该PR当前已经支持AgentX MTP。
