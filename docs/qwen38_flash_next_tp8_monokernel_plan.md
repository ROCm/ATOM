# Qwen3.8-Flash-Next TP8 MonoKernel 状态与后续方案

更新日期：2026-10-16。

### 当前冻结点（P15 探针后）

用户要求先锁最优解、停止 C1 调优，再研究 conc1--16 与 TP1/2/4/8 泛化。
冻结接受态仍为 P12（含 P8、P11a、P11b），源码 commit
`5b9f12907975b861d872b7624ba317e35c97d4ad`。最新同口径 7 轮 FULL-graph client
wall 中位数 **4.581315 ms**（4.575074--4.586181 ms），相对 native 6.8652 ms 为
**1.4985x**。1000-token 请求完成，wall 4.869284 s。正式 TP8 production gate 仍关闭，
整网重验通过数据目录 bootstrap 显式启用 probe，不能只依赖未注册的环境开关。

P15 尚无新的接受候选：真实 shared I=80 shard pad 到 block-FP8 后只节省 19.98%
权重字节；weight-only、direct FP8 和混合 FP8 的已测实现均慢于 BF16。
BF16 active task table 没有稳定典型收益；256-CTA G2 单任务和 chunk2 queue 均回退。
这些结论只适用于已测实现，不构成对所有统一任务图方案的否定。

真正 P12 fresh trace 的 gather+argmax endpoint 为 21.899 us，达到 20 us 探针门槛，
但 8-rank endpoint 候选最好只回收约 0.753 us，暂不接入生产。完整资源、数值、
graph replay、失败中间版和验证边界记录在
`/home/yifehuan/data/qwen38_flash_tp8_baseline_20261015/p15_shared_task_graph/report.md`。

下一阶段先建立 conc=1/2/4/8/16 × TP=1/2/4/8 的 native / Mono / fallback 矩阵，
以普通 decode 为基线，记录实际 batch width、TPOT、总吞吐和每 GPU 吞吐。
当前 Mono 声明 TP1/TP8、S=1/2/4/8，实际构建覆盖见下文，production 只启用 TP1/S1；
TP2/TP4 与 S16 需扩展 specialization 并逐项验收，不能将 native fallback 当 Mono 泛化。
MTP verify width 在普通并发矩阵之后单独评估。

### 泛化基线首轮（2026-10-16）

源码覆盖更正：`runner._build_width` 的 TP8 分支明确只接受 S1；
`SUPPORTED_WIDTHS=(1,2,4,8)` 不能解释为 TP8 支持这四个宽度。
TP1 实际支持 S1/2/4；S8 仍会被 kernel 拒绝，需扩展并单独验收。
本轮完成 TP8 native conc1/2/4/8：client decode TPOT 为
6.8831 / 7.1369 / 7.8757 / 8.4369 ms；Mono-enabled 对照为
4.6044 / 7.2940 / 8.0422 / 8.5599 ms，只有 C1 是 Mono，其他为 native fallback。
实际 Mono-enabled decode width 已由 worker 计数确认；native 首轮缺 worker 快照，需复核。
TP1 native conc1/2/4/8 已在后继 Job 217242（080）完成，TPOT 为
7.7858 / 8.2965 / 9.1861 / 9.5697 ms。
原始数据、显存、hash、口径和验证边界见
`/home/yifehuan/data/qwen38_generalization_20261016/report.md`。
TP8 Mono S1 部分重复的 text hash 变化，本轮不是 production 数值验收，gate 保持关闭。

### 泛化修复与验收边界（2026-10-16）

修复 QSA pre 缺失的 HC `up_group_ready` mailbox，以及 QSA post 动态分支
携带 `Mailbox` context 导致的 FlyDSL IR 提取错误；后者改用分支外绑定和既有
raw output-combine helper，算术与发布顺序保持不变。Qwen 的 lazy runner 显式
按宽度隔离 build refusal，避免较大宽度失败禁用所有已可用宽度；其他部署默认行为不变。

TP1 Mono 在仅数据目录 bootstrap 放宽 SGPR spill 上限的资源审计模式下，
C1/2/4 TPOT 为 6.3089 / 8.8054 / 11.7079 ms；S2/S4 分别慢于 native
约 6.1% / 27.5%，不能作为正式接受态。源码资源阈值和 production gate 均未放宽。
trace 显示 GDN K0 的宽度扩展成本为主要瓶颈，TP8 S2/S4 则尚无真实 Mono 实现。

两组 host 回归测试共 113 项通过。独立 QSA 随机权重、零 expert probe 的
S1 direct、S2/S4 direct 与 full 路径相对 L2 为 0.0029--0.0032，
各 1000 次 graph replay 输出一致且 finite；这不替代真实 checkpoint 整网精度验收。
下一阶段优先 TP8 C1/2/4/8，逐项区分 native、真实 Mono 与 fallback，暂不评估 MTP。

本文记录已经验收的 TP8/EP8 C1 路径、被整网结果否决的候选，以及继续优化必须遵守的边界。
逐轮脚本和原始 JSON 保存在
`/home/yifehuan/data/qwen38_flash_tp8_baseline_20261015/`，不进入正式源码树。

## 1. 冻结场景与性能口径

- 硬件：单机 8 张 MI355X / gfx950。
- 模型：`Qwen/Qwen3.8-Flash-Next-FP8`，48 层，其中 36 个 GDN、12 个 QSA。
- 模型结构：Hidden 2560、HC4、512 routed experts、top-10、expert intermediate 640。
- C1 / S=1 decode，BF16 KV cache，greedy decode，FULL CUDA Graph。
- dense 与 attention 使用 TP8；MoE 使用 EP8 expert ownership，每个 rank 持有 64 个完整 expert。
- 不启用 speculative decode、MTP、TBO、DP、PP 或 CP。

`I=640` 不能按 FP8 block-N 128 连续切成 TP8 的 I=80，因此 EP8 只承担 MoE 权重归属。
C1 时每个 rank 已有完整 token activation，不增加 token dispatch 或 all-to-all。

统一整网口径是 19-token prompt、生成 128 token、预热后测 7 轮，报告 FULL-graph client
decode wall 中位数。服务端 TPOT 只用于毫秒级交叉检查。

| 版本 | wall | 相对 native | 状态 |
|---|---:|---:|---|
| Atom native TP8 + EP8 MoE | 6.8652 ms | 1.000x | 冻结基线 |
| P8 QSA K2 hybrid | 4.8324 ms | 1.421x | 接受 |
| P11a scalar routed-G2 slot | 4.8120 ms | 1.427x | 接受 |
| P11b static routed-G1 | 4.7857 ms | 1.435x | 接受 |
| P12 compact router | **4.577454 ms** | **1.4998x** | 当前接受态 |
| P14 front fusion | 5.109785 ms | 1.3435x | 拒绝并移除 |

当前 TP8 实现仍关闭 production gate。正式启用前需要固定 hidden/logits/argmax 阈值、长 replay
和主线 CI；源码保留可编译、可显式探测的 TP8 specialization。

阶段目标：

| 目标 | wall | 相对 native | 当前还需回收 |
|---|---:|---:|---:|
| R1 release 性能线 | <=4.60 ms | >=1.492x | 性能已达到 |
| R2 C1 主目标 | 4.10--4.25 ms | 1.615--1.674x | 0.327--0.477 ms |
| R3 stretch | <=3.433 ms | 2.000x | 1.144 ms |

模型冻结逻辑权重流量约 11.136 GB/token。MI355X 实测 copy roofline 为 4.756 TB/s，纯流量
下限约 2.34 ms；该下限未包含 GDN recurrence、QSA sparse attention、量化、peer protocol
和 EP 负载尾部。R2 有现有 trace 支撑；R3 只有在同 shape 各阶段组合下限不高于 3.20 ms 时
才作为工程目标。

## 2. 当前接受结构

```text
上一层 K2 发布 MoE rank partial
  -> standalone deferred TP8 pull
  -> native HC pre/mix
  -> GDN: Mono K1 -> Mono K2
  -> QSA: Mono K1 -> Mono K2 composite
     长上下文回退 native QSA K2 + standalone finish
```

TP8 GDN/QSA specialization 使用 256 CTA x 512 threads 的 persistent launch。attention 和 MoE
partial 通过对称 peer region 发布；消费端按固定 rank 顺序求和，并保持 BF16 endpoint。MoE
router/top-k 在各 rank 一致，`expert_map` 将 global expert id 映射到本 rank 的 64 个 expert。

正式代码保持单一 Qwen package：

```text
atom/models/qwen4_exp_mono/
├── config.py          # profile 与 production gate
├── dims.py            # TP1/TP8 rank-local shape
├── plan.py            # GDN/PLE-GDN/QSA build plan
├── contract.py        # ABI 与资源 contract
├── weights.py         # binder 与 expert ownership
├── state.py           # graph-stable scratch/peer/seam
├── step_meta.py       # GDN/QSA runtime views
├── peer_layout.py     # attention/MoE peer packet layout
├── runner.py          # build、选择、launch 与 fallback
└── kernels/
    ├── gdn_tp8.py
    ├── qsa_tp8.py
    ├── collective.py
    ├── finish.py
    ├── hc.py
    ├── router.py
    ├── quant.py
    ├── g1.py
    └── g2.py
```

TP 差异进入 build key 和 `Dims`。热路径不使用 runtime `if tp == 8` 选择矩阵 shape。探针源码
留在数据目录；只有通过完整整网验收的候选才能进入上述 package。

## 3. P12 trace 与剩余空间

P12 精简 trace：

```text
/home/yifehuan/data/qwen38_flash_tp8_baseline_20261015/p13_current_trace/
```

稳定 decode window 的 wall 为 4512.811 us，kernel busy union 为 4500.786 us，未覆盖
13.065 us。完整 token cycle 的 wall 为 4640.931 us，busy union 为 4545.098 us，未覆盖约
100.511 us。因此近期收益必须来自层内计算、通信和任务图，host/runtime 不能承担 R2。

| 阶段 | 总时间 | 次数 | 每层约 |
|---|---:|---:|---:|
| GDN K2 | 1964.422 us | 36 | 54.57 us |
| QSA K2 | 798.959 us | 12 | 66.58 us |
| GDN K1 | 654.524 us | 36 | 18.18 us |
| deferred pull | 421.729 us | 48 | 8.79 us |
| native HC pre | 283.314 us | 49 | 5.78 us |
| native HC mix | 269.560 us | 49 | 5.50 us |
| QSA K1 | 124.188 us | 12 | 10.35 us |

front chain 合计约 1.75 ms/token，K2 合计约 2.76 ms/token。两者都有结构空间，但所有收益
必须以 FULL graph A/B/A/B 为准，不能按 isolated layer 线性外推。

## 4. P14 结论：standalone 有效，整网无效

P14 尝试将 deferred pull、attention HC front 和 GDN K1 合成 projection-owned front。

正式 8-rank probe：

| 路径 | max-rank 中位数 |
|---|---:|
| pull + HC + GDN K1 | 39.081 us/layer |
| P14 composite | **32.526 us/layer** |
| isolated 收益 | **6.554 us/layer** |

8 个 rank 的 reduced row、HC streams/mixed/raw、GDN partial、conv state 和 recurrent state
逐位一致。资源为 65 VGPR、82 SGPR、29,952 B LDS、无 spill。该结果证明 HC-up group-ready
到 GDN projection 的细粒度交接在单层边界内可行。

完整整网 A/B 却得到：

| 路径 | FULL-graph wall |
|---|---:|
| P14 off | 4.588928 ms |
| P14 on | 5.109785 ms |
| 差值 | **+0.5323 ms / +11.63%** |

所以 P14 没有进入接受态。正式 runner、workspace、环境开关和 `front.py` 已删除。probe 与 JSON
保留用于说明编译和数值可行性，不作为整网收益证据。若以后重新研究 front fusion，必须先用
整网 trace 解释 graph 内调度和跨层资源耦合导致的 0.532 ms 回退，再写 production 接线。

原始结果：

```text
/home/yifehuan/data/qwen38_flash_tp8_baseline_20261015/p14_gdn_front_full/benchmark_a1.json
/home/yifehuan/data/qwen38_flash_tp8_baseline_20261015/p14_gdn_front_full/benchmark_b1.json
/home/yifehuan/data/qwen38_flash_tp8_baseline_20261015/p14_gdn_front_fusion_probe/p14_formal_multirank.json
```

## 5. PR #2529 复现与可借鉴结构

PR #2529 的公开 1.6x 不是 TP8 C1 MonoKernel：它测的是 TP4+EP4、MTP2、并发 4--16，
每个 request 生成 891 token 且 draft acceptance 为 100%。其实现仍是多个专用 kernel，核心收益
来自减少约 980 个 decode launch，而非把完整 layer 合成 persistent K1/K2。

在当前 ROCm 7.2.4、PyTorch 2.10、AITER `5d36825e` 上，使用 PR head `9e346de6` 和相同
891-token workload 做 flags-off/on/off A/B/A：

| 并发 | off A1 | on B | off A2 | B / 两次 A 中心值 |
|---:|---:|---:|---:|---:|
| 4 | 1076.37 tok/s | 1081.76 tok/s | 1085.12 tok/s | **1.001x** |
| 8 | 2031.90 tok/s | 2089.22 tok/s | 2034.00 tok/s | **1.028x** |
| 16 | 3824.80 tok/s | 3808.63 tok/s | 3722.10 tok/s | **1.009x** |

全部输出 hash 一致并以 `stop` 结束。本环境没有复现 1.51--1.61x。PR 文档中的 native HC
`mix_fused` 约 26--27 us；当前 AITER 实测已经是 12.93--17.21 us，而 PR FlyDSL HC 为
13.88--27.70 us。其最大历史收益依赖较弱的 native/AITER 基线，不能直接作为我们的目标。
完整记录位于 `/home/yifehuan/data/pr2529_repro/report.md`。

逐项对照：

| PR 思路 | PR 的作用 | 当前 TP8 Mono 状态 | 结论 |
|---|---|---|---|
| FlyDSL/MFMA HC | 3--48 rows 减少 HC launch | C1 仍用 native HC；P14 整网回退 | 当前 C1 不移植；S>1/MTP 按 shape 重测 |
| QKV + index projection | 共享 activation quant 和 GEMM | QSA K1 已在一个 launch 中并行完成两路 BF16 projection | 所有权已覆盖；可另测 FP8 activation/weight format |
| fused index prep | 九个 index cache launch 合一 | QSA K1 已包含 index norm、mRoPE、pool、cache write | 已覆盖 |
| fused Q/K prep | norm、mRoPE、KV write 合一 | QSA K1 已覆盖 | 已覆盖 |
| router buckets | router 直接生成 expert task table | C1 top-10 静态 slot task；无独立 sort/plan | C1 收益小；S>1 expert union 时复用 |
| small-batch MoE | BF16 activation 直接消费 FP8 weight | 当前 routed G1/G2 使用 MXFP8 activation 和量化 scratch | 值得做格式 probe，不能直接替换 |
| FP8 shared expert | shared expert 作为额外 expert 进入同一路径 | shared G1/G2 仍是 BF16 独立路径 | **当前未覆盖，优先 probe** |
| ReplaySSM fused gate | verify loop 内联 gate，删一次 launch | 当前 Mono 不拥有 MTP verify | MTP 阶段直接采用 |
| per-rank greedy argmax | 避免全 vocab logits all-gather | LM head/采样仍在 Mono 外 | **系统尾部优先 probe** |
| fused MTP layer | draft 按 target layer 构造并复用融合路径 | 当前无 Mono draft runner | C1 收口后最高优先级扩展 |

PR 最有价值的方法是先消除**格式、计划和结果只被下一阶段消费的临时边界**：router 直接产出
消费任务，cache prep 直接写最终 cache，greedy verify 直接产出 token id。这个原则应进入 Qwen
Mono 的任务图，但不照搬它当前已落后于 native 的 HC kernel。

## 6. 新的最大优化点

需要区分当前 C1 延迟和生产吞吐。

### 6.1 当前 TP8 C1：shared expert 与 K2 task graph

C1 剩余最大的层内块仍是 2.76 ms/token 的 K2。PR 暴露了一个此前没有单独量化的结构：将
BF16 shared expert 在加载时转为 block-FP8，并作为 routed task graph 的固定 extra expert。当前
shared G1/G2 独立读取 BF16 权重、维护 `shared_mid` 和专用 ready 链；它既增加权重流量，也让
160 个 G2 CTA 串行执行 shared-G2 后再处理 routed slots。

目标结构：

```text
HC output / input quant
  -> compact router
      -> fixed shared slot + active local routed slots
          -> unified G1 task queue
          -> per-slot Mid ready
          -> unified (slot, output-part) G2 queue
          -> ordered BF16 merge
          -> peer publish
```

先做三个 probe，禁止直接改整网：

1. **P15-A shared-FP8 format lower bound**：只比较真实 checkpoint shared G1/G2 的 BF16 与
   block-FP8 weight 路径，保持输入和 BF16 endpoint；记录精度、每层 wall 和权重字节。门槛为
   GDN/QSA common finish 至少回收 2 us/layer，48 层投影至少 0.096 ms/token。
2. **P15-B unified shared/routed task table**：即使保留 BF16 shared weight，也让 router 发布固定
   shared slot 和本 rank active routed slots，删除 shared 专用全局 ready。验证任务图收益与格式收益。
3. **P15-C 256-CTA G2 queue**：所有 CTA 领取 `(slot, output-part)`，按 part/slot 局部 ready；
   merge 保持 top-k 顺序。禁止 `all shared ready` 或 `all G1 ready` 全局屏障。门槛为 >=3 us/layer。

P15-A 是 lossy candidate。必须比较固定 hidden/logits/argmax 和长生成；GSM8K “统计无回退”不能
替代数值契约。若精度不接受，保留 P15-B/C，shared weight 继续 BF16。

### 6.2 整网固定尾部：local argmax endpoint

当前 LM head、TP logits 汇聚和 greedy sampling 在 Mono 外。PR 的 verify-only 实现可以推广到普通
C1 greedy decode：每 rank 只计算本地 vocab shard 的 `(max_value, global_token_id)`，然后对 8 个
pair 做固定顺序 reduce，直接输出 token id。这样避免完整 `[1, vocab]` logits all-gather、全量 logits
materialization 和通用 sampler。

先抓 trace 量化 LM-head 后半段；只有其每 token 成本 >=20 us 才实现。首个 probe保持 LM-head GEMM
不变，只替换 gather+argmax endpoint。目标回收 10--30 us/token。需要 logprobs、非 greedy、beam 或
调试 logits 时回退原路径。

### 6.3 生产吞吐：MTP-aware Mono width

PR 真正放大的场景是 MTP2：每个 request 的 target verify 宽度为 3，C4/C8/C16 对应 S=12/24/48。
这不应作为 C1 小修，而应作为独立 specialization：

- target verify build keys：S=3/6/12/24/48；
- draft layer复用 common finish 与 EP ownership；
- ReplaySSM gate 在 recurrent loop 内计算；
- greedy verify 使用 local argmax endpoint；
- 接受判定和 accepted-count commit 保持 GPU resident；
- router 直接生成跨 token expert-union task table，共享 expert weight load。

这是 C1 达到 R2 后最大的吞吐扩展，潜力高于继续压单个 C1 stage，但收益必须使用真实 acceptance
分布报告，不能只用 100% acceptance 的整数列表。

## 7. 修订后的执行顺序

1. 冻结 P12，完成 P15-A shared-FP8 lower-bound probe。
2. 无论 P15-A 数值是否接受，都分别验证 P15-B task table 和 P15-C 256-CTA G2 queue。
3. 重抓整网 trace，量化 LM-head gather+sampling；达到 20 us 门槛才做 local argmax endpoint。
4. 三项合计若能稳定回收 0.25--0.35 ms，进入 R2 的 4.10--4.25 ms 区间。
5. R2 后建立 MTP2 target/draft build key 和 ReplaySSM/argmax GPU endpoint。
6. S=2/4/8 普通并发与 MTP width 共用 expert-union/task-table primitive，不再维护两套调度。

已关闭的路线包括当前 P14 production 接入、dual-owner expert、128-column ownership、G1
K-group partial、带全局屏障的 G2 split、active-slot paired G2、router K-shard、HC weight ownership
和顺序 one-launch 拼接。除非新 trace 推翻根因，不重复实现。

## 8. 验收与资源约束

每个候选按以下顺序验收：

1. fresh compile，记录 symbol、ABI、容器和 FlyDSL 版本；
2. ISA/资源检查，private/VGPR spill 必须为 0；
3. 单 rank 分阶段数值对照；
4. 8-rank peer 顺序和 BF16 endpoint 对照；
5. GDN state 或 QSA cache 对照；
6. graph capture 和 1000 replay；
7. phase A/B/A/B；
8. 48 层 FULL graph A/B/A/B 和 1000-token wall；
9. 固定 hidden、logits、argmax；
10. 通过后才收敛正式 ABI、测试和文档。

活动目录预算：candidate JIT <=1 GiB，ATT <=2 GiB，单轮目录 <=6 GiB。默认关闭完整 IR dump。
rejected candidate 删除 JIT、IR、大 trace 和重复日志，只保留复现脚本、小 JSON、资源摘要与根因。
不创建实验 branch、worktree 或 stash。

## 9. TP8 普通多并发：PR 2586 结构审查与快速验证（2026-10-16）

用户当前顺序覆盖旧 §7：C1 冻结，优先普通 decode S2/S4，再 S8；暂不开展 MTP。
参考 ROCm/ATOM PR 2586，固定审查 HEAD `5c40eed50ea99e63a23ba9eb450fdcdfbb616ad2`。
DSV4 的 pure TP8 与本模型 TP8+EP8 不同，不能直接照搬 weight packing 或数值路径。

优先借鉴的结构：

- MFMA B 列承载多 sample；shared weight tile 一次加载服务所有行。
- routed UG 使用 `(sample, slot, tile)` 静态 CTA-stride，输入 staging 可复用；
  shared 工作并入专家任务，而非额外 scheduler。
- down 的 task 数按输出 tile 定义，不随 S 成倍增长；wave 的 K 工作覆盖
  `(sample, slot, Kchunk)`。不同 sample 必须有 B 列 mask，不能共享错误的 expert id。
- route 完整 K 求和、加权与 BF16 舍入边界保持模型契约；route coefficient 使用
  VGPR/LDS broadcast，防止多行 uniform SGPR 扩张。
- 小批次 software pipeline：提前发下一批权重读取；attention/MoE 分阶段、
  同一 peer/epoch 生命周期，不做 S1 全层顺序复制。

PR 文档 ordinary C1/C4/C8 的 2.31×/1.67×/1.23× 是作者公开声明，未本地复现；
不是 Qwen 验收成绩，也不承诺所有宽度达到 C1 的倍率。

快速 probe 使用 Job 217242、080、8×MI355X，同一随机输入与 7×100 graph replay 计时。
下表为各 rank 中位数的均值，包含 step-begin + finish + peer pull，不是单 G2：

| S2 候选 | finish 链路 µs | 判定 |
| --- | ---: | --- |
| paired HC/shared G1，G2 逐行（A） | 84.397 | 当前对照 |
| TP8 shared G2 三轮裁成一轮 | 84.730 | 无收益，移除 |
| shared G2 双 MFMA 列，共用 W2（B） | 83.900 | 约 0.6%，仅 probe 候选 |
| B + routed W2 提前到 mid-ready 等待前 | 84.002 | 无额外收益，移除 |

B 保持 S1 路径不变，S2 四类输出相对原 S1 逐行参考 relL2=0；8-rank
BF16 peer 顺序与 1000 replay 均 exact。LDS 27136 B、VGPR55、SGPR106、
SGPR spill26、private scratch/VGPR spill0，未放宽资源门禁。
不同 token 的不同 route、真实 checkpoint hidden/logits/argmax 与完整整网 ABAB
仍未验收；没有放开生产 width gate，不将 partial-Mono 算作泛化通过。

先前 partial-Mono paired finish 的 C2 为 6.986 ms，对旧 native 7.137 ms 仅约
1.022×，不是 matched-startup 验收；串行扩宽 GDN K1 接入后为 7.268 ms，退化，
不接受。这两项解释了下一步应改 routed down 多行任务几何，而不是继续顺序拼接。
最终 A/B 复测为 84.059/83.993 µs，差异仅约 0.08%，双列 shared G2 也没有
稳定可辨认收益，默认整网不启用。复现操作见
[整网 Mono 复现说明](qwen38_flash_next_mono_reproduction.md)。
shared 局部改动当前收益太小，不接整网；下一候选优先 S2 batched routed down，
先保持逐 route BF16 边界，再验证 S4/S8，最后同配置 native/Mono 整网对照。
