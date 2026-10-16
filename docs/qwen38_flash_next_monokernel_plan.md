# Qwen3.8-Flash-Next C1 MonoKernel 优化方案

日期：2026-10-15

目标硬件：AMD MI355X / gfx950

当前生产范围：真实 FP8 checkpoint、TP1、BF16 activation/KV、decode C1（每步一个 token）、FULL graph。

本文只描述 C1 接受态、与 `origin/yifehuan/dsv4-mono` 的结构差距、下一候选和验收门槛。S=2/4/8、TP、MTP 暂停；历史实验保存在
`/home/yifehuan/data/qwen38_flash_main_baseline_20261012/`。

## 1. 目标和当前事实

性能目标只由 Qwen 自身的流量、依赖图和 MI355X 实测决定。Kimi-K3、GLM、DSV4 和 TileRT 只用于参考调度方法，不能用于限制 Qwen 的目标，也不能证明 2.0× 已经或尚未可达。

| 路径 | FULL graph / TPOT | 相对 9.0–9.2 ms native | 定位 |
|---|---:|---:|---|
| Qwen native | 9.0–9.2 ms | 1.00× | 配对基线 |
| 当前 Qwen C1 Mono | 6.65–6.68 ms | 1.35–1.38× | 接受态 |
| 第一里程碑 | ≤5.10 ms | 1.76–1.80× | 证明层内 DAG 已经明显接近 roofline |
| C1 主目标 | 4.50–4.60 ms | 1.96–2.04× | 约 2.0×，当前继续保留 |
| 乐观依赖下限 | 约 3.9–4.3 ms | 2.09–2.36× | 研究下限，不是性能承诺 |

当前距 2.0× 还需降低约 2.05–2.15 ms，即 Mono wall 的约 31%。5.10 ms 是中间放行线，不再作为最终目标。只有同 shape 阶段探针和 PMC/ATT 证明依赖下限已经高于 4.6 ms，才按第 2.4 节重新设定目标。

服务端 TPOT 只有约 0.1 ms 精度；候选结论使用配对 FULL-graph client wall、GPU trace/event 和 ISA/resource。

当前 C1 结构：

```text
GDN layer
  native attention-HC front
  → qwen_gdn_layer [GDN body → GDN/HC handoff → MLP-HC → MoE]

QSA layer
  qwen_qsa_pre  [attention-HC front → QKV/index → norm/mRoPE → cache]
  → qwen_qsa_post [selection → sparse attention → o_proj → MLP-HC → MoE]
```

生产 gate 保持：

```python
SUPPORTED_WIDTHS = (1, 2, 4, 8)
ENABLED_WIDTHS = (1,)
SUPPORTED_TP = (1,)
```

`SUPPORTED_WIDTHS` 只表示 ABI 已定义。C1 达到目标前不继续 width/TP 功能或调优。

## 2. Qwen 自身的理论和工程下限

### 2.1 不可避免的逻辑流量

下面按真实 checkpoint ABI 计算 C1 每 token 至少需要读取或更新一次的数据。S=1 的 top-10 专家天然互异；FP8 scale 已计入。实现产生的 global scratch 往返不属于理论下限。

| 部分 | 逻辑流量 |
|---|---:|
| 48 层两组 BF16 HC + router + shared/routed top-10 MoE | 4.226 GB |
| 36 层 BF16 GDN 权重 | 4.173 GB |
| 36 层 GDN recurrent/conv state 最低读写 | 0.231 GB |
| 12 层 BF16 QSA 权重 | 1.235 GB |
| BF16 LM head `248320 × 2560` | 1.271 GB |
| 合计 | **11.136 GB/token** |

当前 19-token prompt、生成 128 token 的基线始终处于 QSA direct-context 范围。KV/index 读写、hidden seam 和采样还有少量流量，但相对 11.136 GB 很小；长上下文目标需要另外加入对应 cache 流量。

MI355X 上 512 MiB `copy_` 的实测读写总流量中位数为 4.756 TB/s。因此纯流量下限是 2.34 ms；它没有包含分阶段依赖、低占用小矩阵、GDN recurrence、QSA selection/attention、量化、mailbox 和采样，不能作为可实现目标。

| wall | 等效逻辑带宽 | 占实测 copy roofline |
|---:|---:|---:|
| 6.66 ms | 1.672 TB/s | 35.2% |
| 5.10 ms | 2.184 TB/s | 45.9% |
| 4.60 ms | 2.421 TB/s | 50.9% |
| 4.55 ms | 2.447 TB/s | 51.5% |
| 4.50 ms | 2.475 TB/s | 52.0% |

同 shape 的原生 BF16 GDN 大投影已经达到约 3.3–3.5 TB/s。由此可知 4.5–4.6 ms 没有违反 HBM 或 MFMA 上限；难点是让 HC、MoE G1/G2、QSA K2 和 stage handoff 在串行依赖下把整网平均效率提高到 copy roofline 的约一半。

### 2.2 当前真实关键路径

接受态短 trace 的正确分解为：

| 部分 | 每 token 合计 | 每层 |
|---|---:|---:|
| 35 个普通 GDN + 1 个 PLE-GDN Mono body | 3.960 ms | 110.0 µs |
| GDN 外部 attention-HC pre/mix | 0.416 ms | 11.6 µs |
| 12 个 QSA K1 | 0.753 ms | 62.8 µs |
| 12 个 QSA K2 | 1.165 ms | 97.1 µs |
| PLE、head、sampling 和其他 | 0.366 ms | — |
| 合计 | **6.660 ms** | — |

因此 GDN 含 front 共 4.376 ms，QSA K1+K2 共 1.918 ms。trace 的 kernel sum 基本等于 wall，当前不存在可直接删除的 1–2 ms host 空洞。

### 2.3 两级性能预算

| 部分 | 5.10 ms 里程碑 | 4.55 ms / 2.0× 主目标 | 当前 |
|---|---:|---:|---:|
| GDN 含 front，36 层 | 3.240 ms，90 µs/层 | 2.880 ms，80 µs/层 | 4.376 ms |
| QSA K1+K2，12 层 | 1.452 ms，121 µs/层 | 1.248 ms，104 µs/层 | 1.918 ms |
| 其余 | 0.408 ms | 0.422 ms | 0.366 ms |

80 µs 的 GDN 层对应约 2.63 TB/s，约为 copy roofline 的 55%；104 µs 的 QSA 层对应约 1.84 TB/s，约为 39%。这个分配比统一要求所有阶段达到 51.5% 更符合模型：GDN 由大 BF16 投影主导，QSA 还承担低并行度的 selection 和 sparse attention。

### 2.4 何时调整 2.0× 目标

不能因为其他模型只取得某个倍率而下调目标。只有完成以下 Qwen 专用下限测量后才能调整：

1. 分别测 HC front、GDN in/out projection、GDN state、router、shared/routed G1、G2、QSA K1 projection、selection/attention、o_proj 和 LM head 的同 shape 最优探针。
2. 每个探针保持正式数值契约、权重布局和任务规模；纯 copy 或删掉依赖的 toy kernel 只能给物理下限。
3. 按真实 DAG 组合阶段下限，并只计入有证据能够重叠的区间。
4. 若组合下限 `L_dep ≤ 4.25 ms`，继续保留 4.50–4.60 ms 主目标。
5. 若经过 PMC/ATT 复核后 `L_dep > 4.25 ms`，新工程目标取 `ceil_0.05ms(1.08 × L_dep)`，同时记录限制来自带宽、MFMA issue、依赖还是资源驻留。

当前已有证据给出的乐观依赖下限约为 3.9–4.3 ms，所以 2.0× 仍是合理研究目标，尚无依据提前降为 1.8×。

## 3. DSV4 C1 参考

参考分支：`origin/yifehuan/dsv4-mono`，当前提交 `ce4aede4532b00723b1cb788eb37b5b16a97a534`。

参考目录：

```text
atom/models/deepseek_v4_mono/
├── contract.py
├── runner.py
├── state.py
├── weights.py
└── kernels/
    ├── layer.py
    ├── attention.py
    ├── attn_pre.py
    ├── indexer.py
    ├── mhc.py
    └── moe.py
```

DSV4 的关键结构：

- 每层通常是两个 `256 CTA × 512 threads` persistent launch：K1 `attn_pre`，K2 `layer_post`。
- K2 在同一 launch 内串起 selection/sparse attention/o_proj、FFN seam、MoE 和 TP8；Qwen 的 QSA 两 launch 结构因此本身没有偏离 DSV4。
- 没有 grid-wide barrier。每个 consumer 只等待自身需要的 tagged mailbox 数据。
- 数值与 tag 放在同一个 mailbox pair 中，producer/consumer 交接只做一次 coherent round trip。
- GEMV/MFMA task 在等待输入前先发出冷 weight load，等待时间由权重访存覆盖。
- stage task 由统一 `stage_tasks/start/run_units` 几何生成；CTA 完成早期任务后继续领取后续任务。
- `C1Continuation` 把连续 token 的 slot、position、block-table frontier 保留在 host mirror。稳定 C1 不重复 D2H 验证或 TP object collective。
- runner 在 `ModelRunner` 层拥有 graph 选择、publication、greedy token 和 cache generation，而不只是在模型内部替换 forward。

DSV4、Kimi-K3、GLM 和 TileRT 的收益都不进入 Qwen 目标公式。它们只提供局部 ready、weight-before-wait、动态尾部队列、核内 collective 和 runtime ownership 等实现方法。

## 4. Qwen 与 DSV4 的结构差距

| 维度 | DSV4 C1 | 当前 Qwen C1 | 影响 |
|---|---|---|---|
| 每层边界 | K1 attn_pre + K2 attn_post/FFN | GDN 前 HC 在核外；QSA 也是 K1/K2 | GDN front 是明确缺口；QSA launch 数不是缺口 |
| DAG 依赖 | tagged value，局部 producer/consumer | 多个 plain global tensor + ready tag；部分 group collapse | 增加写回、重读和粗粒度尾部等待 |
| weight/ready 顺序 | weight load 先于 input poll | 多个 Qwen stage 先等 ready，再加载 weight | 无法用冷权重访存覆盖 producer 尾部 |
| stage 调度 | 统一 task geometry，CTA 连续推进 | 各 emitter 独立映射，组合后仍有阶段性 owner | 容易出现空闲 CTA 和 fan-in 尾部 |
| GDN handoff | stage consumer 读取自身输入 | 160 output tile 折叠为两个 group 后 HC norm | 早完成 tile 只能在 80-tile group 内等待 |
| QSA handoff | K1 发布 K2 必需 cache/scratch，K2 内部细粒度推进 | K1/K2 边界合理，但 K2 内仍有 materialize、重读和尾部等待 | 优化 K2 内部 DAG，是否合并 launch 由资源决定 |
| runtime ownership | publication + C1 continuation + device token | 普通 model wrapper，复用 Atom 通用 token path | TP1 当前影响待测；不是已证实的 1.6 ms 主因 |
| TP | TP8 peer protocol 在 kernel 内 | TP1 | 当前 C1 不因 TP 缺失扣分 |

生产 runner 中的三处 `.item()` 都受 `ATOM_MONO_CHECK` 保护，正常 C1 不执行。后续不得把诊断分支误判为生产同步热点。

### 4.1 最终结构项与当前阶段

本文的当前执行范围是 C1/S=1/TP1 性能收敛，不等于最终 Mono runtime 只支持这一范围。以下状态用于区分已经存在但未放行的实验路径、C1 后必须补齐的结构，以及只有 trace 证明有收益才提前做的外层融合。

| 结构项 | 当前状态 | 进入顺序 |
|---|---|---|
| S=1 GDN/QSA hybrid | 已接入生产 gate | 当前优化对象 |
| S=2/S=4 | 功能实现和 probe 已存在，性能未达标，生产禁用 | finish-v2 与 C1 达标后重做 specialization |
| S=8 | 未实现 | 复用通过验收的 S=2/S=4 task geometry |
| TP peer reduce | TP1 ABI 已稳定，多 rank peer reduce 未实现 | 先做实际部署需要的 TP2，再扩 TP4/TP8 |
| LM head 与采样 | 仍走 Atom 原生图 | 层内达到 5.10 ms 后接入；若 trace 提前显示为主尾部则上调 |
| MTP/draft/接受判定 | 未实现 | width、TP、KV/index ownership 稳定后实现 |
| token-to-token runtime | 未实现 | 当前 kernel sum 约等于 wall；层内和 MTP 完成后再接管 |
| PLE/embedding/final mix | 核外或原生路径 | 最终必须纳入 graph ownership；只有 trace 证明阻塞层内流水时提前融合 |

因此，TP、width、MTP 和最终 runtime ownership 是完整方案的必做项；它们暂停是为了避免把尚未解决的 common-finish DAG、资源和数值问题复制到更多 specialization。当前阶段不再把“生产禁用”写成“完全未实现”。

## 5. C1 最终代码结构

保留当前模块边界，但把顶层 composition 改成 DSV4 风格的统一 stage DAG：

```text
atom/models/qwen4_exp_mono/
├── config.py          # 模型/硬件常量，只保留生产 gate
├── contract.py        # ABI、资源和 ISA contract
├── plan.py            # C1 layer/build key 和 stage geometry
├── dispatch.py        # native fallback；后续只转发到 session
├── runner.py          # layer launch/binding，不放 kernel 数学
├── session.py         # C1 publication/graph/cache generation（达到层内目标后再接）
├── state.py           # graph-stable scratch/state views
├── step_meta.py       # GDN/QSA device metadata views
├── weights.py         # 零拷贝权重 ABI
└── kernels/
    ├── layer_common.py # mailbox、stage start、task mapping、shared context
    ├── gdn_layer.py    # attention HC front + GDN + finish 的顶层 composition
    ├── gdn.py          # GDN stage emitter
    ├── qsa_layer.py    # 最终 K1/K2 composition；当前仍由 pre/post 实例化
    ├── qsa_pre.py
    ├── qsa_post.py
    ├── finish.py       # MLP HC + MoE stage emitter
    ├── hc.py
    ├── quant.py
    ├── router.py
    ├── g1.py
    └── g2.py
```

当前不为了目录整齐重命名文件。`layer_common.py` 和 `session.py` 只有在对应结构候选通过性能门槛后才创建；失败候选不进入正式代码。

## 6. 优化路线

### M36：common finish-v2 任务图

这是当前最高优先级，因为它同时作用于 36 个 GDN 层和 12 个 QSA 层。现有源码暴露了两个结构瓶颈：

1. router 的 33 个 task 和 shared-G1 的 40 个 task 都先等待全部 160 个 `hc_up_ready`，然后才读取权重；HC→MoE 因而存在一次完整向量 fan-in。
2. G2 只有 `H/16 = 160` 个 output task。每个 CTA 串行执行 shared-G2 和 10 个 routed slot，最长尾部只使用 160/256 个 CU；routed W2 也在等待 Mid 后才开始加载。

finish-v2 原计划改为：

```text
HC up group 0 ready ─┬─ router/shared-G1 K[0:1280]
HC up group 1 ready ─┴─ router/shared-G1 K[1280:2560]
                           ↓
                        route/top-k
                           ↓
G1 slot/group ready → dynamic (slot, output-part) G2 queue
                           ↓
                 per-part ordered BF16 merge
```

其中 routed-G2 slot/part queue 已由独立 probe 证伪，不再进入正式实现。保留的实现范围只包括 HC→router/shared-G1 的局部 K 流水：

- router 和 shared-G1 使用两个 K1280 group；每组先发 1–2 个冷 weight step，再等待对应 HC group。
- routed G1 保持 400 个动态 task；expert id 就绪后先发 W13 load，再等待对应 `xq_ready`。
- expert id 已知后先读 W2，再等待该 slot 的五个 Mid group；prefetch 必须有界，不能把整块 W2 跨 poll 保活。

第一轮先用独立 S=1 G2 probe 决定任务粒度，不修改正式 `g2.py`。probe 复用生产布局、MFMA、scale 和 BF16 舍入顺序，对比当前 256-CTA launch 中仅 160 CTA 工作的 chain 与以下候选：

```text
task(part, kind), kind = shared 或 slot-pair 0..4
  → 写 shared 到最终 output 暂存，或写 BF16 routed_partial[slot, part]
  → agent-scope release arrival[part]++
  → 第 6 个 arrival acquire 后按 slot 0..9 合并并覆盖 output
```

候选共 960 个 producer task。每个 CTA 先执行 `bid`，再用一次 agent-scope atomic 取得 arrival rank，执行 `256+rank`、`512+rank`，前 192 个 rank 再执行 `768+rank`。这与 DSV4 的 bounded extra-task 结构一致，避免在 FlyDSL 中使用带 CTA barrier 的无界 queue loop。rank counter 按 256 取模，因此 graph replay 不需要 host 清零；生产 step 仍沿用 scratch 生命周期。最后一个到达者完成本 part 的 ordered merge，因此没有独立 merge CTA、polling CTA 或 grid-wide phase barrier。额外常驻数据只有 51.2 KiB routed partial、160 个 arrival word 和一个 rank word；shared partial 暂存在最终输出。

G2 probe 满足了正确性和稳定性条件，但性能明确失败：

- 三组变化输入和 1000 graph replay 逐 BF16 bit 相等，arrival 状态每轮归零；两侧均无 spill。
- 直接 kernel trace 中 current-chain 中位数为 13.08 µs，slot/part candidate 为 59.68 µs；50 replay/event 配对中 candidate 回退 30.95 µs，仅为 0.78×。
- MFMA 数保持 55,040 不变，但总指令增加 46.1%、SALU 增加 131.1%、VMEM write 墠加 1960%、LDS wait 增加 100.8%、总等待增加 87.7%。
- 根因是 current-chain 把 shared 和十个 routed contribution 留在同一 CTA 的 LDS/register 中；拆成 960 个跨 CTA producer 后，51.2 KiB partial 往返、960 次 arrival 和 global ordered merge 的代价远高于多使用 96 个 CU 的收益。
- 固定 ROCm 7.2.4 镜像缺少 `rocprof-trace-decoder`，ATT 无法解码；已保留失败日志、trace、PMC、ISA 和资源信息。

因此正式 G2 继续使用 same-CTA chain。

HC group-ready 已完成 probe 和正式组合验收。两个 K1280 group 保留原 CTA 映射；每组 HC-up tile 完成后，由一个已有 producer CTA 汇聚 80 个 tile 并发布，router/shared-G1 按原 MFMA 顺序分两段消费。没有新增 grid-wide barrier，也没有改 top-k、routed G1、G2 或 BF16 merge 顺序。

在 `217241 / crsuse2-m2m-059`、固定 ROCm 7.2.4 镜像上的结果：

| 组合 kernel | all-160 wait | group-ready | 单层变化 |
|---|---:|---:|---:|
| GDN S=1 | 129.922 µs | 117.201 µs | -12.721 µs，1.109× |
| QSA K2 direct256 S=1 | 104.571 µs | 94.501 µs | -10.070 µs，1.107× |

36 个 GDN 与 12 个 QSA K2 的层级投影约节省 0.579 ms/token。GDN/PLE-GDN 的输出、conv/recurrent state 和全部数值 scratch 与既有接受态哈希逐位一致；QSA S=1/2/4 的正式输出和全部 QSA/HC/MoE scratch 逐位一致，token isolation 与每次 1000 graph replay 通过。

资源变化是 GDN `137→139 VGPR`、`40→41 SGPR spill`，QSA K2 保持 93 VGPR、`90→113 SGPR spill`；两者仍为 0 private/VGPR spill、21,776 B LDS、一个 persistent CTA/CU。资源有所增加，但同节点组合 wall 的约 10% 收益稳定复现，因此作为 TP1 接受态提交。FULL graph 仅完成旧结构 A1（client decode wall 中位数 6.618 ms/token）；候选 B 在模型加载阶段按方向调整中止，不能据此声明整网收益。

### M37：GDN 改为模型专用 K1/K2

当前 GDN 路径实际是两个 native HC kernel 加一个组合 kernel。组合 kernel 同时包含 BF16 GDN 和 FP8 finish，资源为 137 VGPR、40 个 SGPR spill；隔离编译时 GDN+precombine 是 110 VGPR、0 SGPR spill，standalone finish 是 129 VGPR。继续把 front 塞进同一个大 kernel 会进一步扩大编译期资源并集。

推荐边界：

```text
GDN K1:
  attention HC front
  → GDN in projection / conv / recurrence
  → GDN out projection
  → publish attn_out + residual seam

GDN K2:
  common finish-v2
  → MLP HC / router / G1 / G2 / merge
```

这样生产路径从“两个 native HC + 一个巨型 kernel”变成两个专用 persistent launch，launch 数减少一个，同时在 K1/K2 边界重置寄存器、SGPR 和 LDS 生命周期。已有 M36 GDN-front 原型只作为 K1 的数值和 group-ready 探针，不再直接并入旧的组合 kernel。

K1 目标为 45–50 µs，K2 目标为 30–35 µs；连同 seam 后 GDN 总预算为 80–85 µs/层。若拆分后 K1+K2 不低于接受态 `native HC front + qwen_gdn_layer`，立即恢复接受态，并用 ISA/PMC 判断限制是否仍在 finish-v2。

### M38：QSA K1/K2 共享同一套 stage runtime

QSA 已经具有正确的两 launch 外部边界，重点是复用 finish-v2 并改内部 task DAG：

- K1 的 HC front、QKV 和 index projection 使用同样的 group-ready + bounded weight-before-wait。
- K1 只发布 K2 selection/attention 必需的 query/index/cache packet，不保活诊断 scratch。
- K2 的 selection、sparse attention、o_proj output tile 和 finish-v2 按局部 ready 推进。
- QSA K1/K2 是否合并不作为目标；只有两个 kernel 的总资源和 seam 测量证明合并更快时才尝试。

主目标预算是 K1+K2 约 104 µs/层。common finish-v2 通过后，再分别把 K1 从约 63 µs 压到 35–40 µs、K2 从约 97 µs 压到 64–69 µs。

### M39：GDN scratch 和 handoff 收缩

在 K1/K2 边界稳定后，才处理 GDN 内部 scratch：

1. `gdn_projected` 的 recurrence consumer 使用 tagged payload 或 producer-owned compact packet。
2. `gdn_core/gdn_normed` 只保留真实跨 CTA 的字段。
3. output tile 直接形成 K1 seam，不再维护供旧组合 finish 使用的完整中间视图。
4. split output partial 保持现有数值顺序，并按 part 发布。

任何候选都必须证明减少了 VMEM transactions 或关键路径等待。

### M40：C1 continuation runtime

只有层内 kernel wall 接近 5.1 ms、host/graph seam 成为可见比例后，才移植 DSV4 的 runtime ownership：

- `C1Continuation(owner_generation, slot, slot_generation, position, blocks)`；
- anchor/transition 做完整 device-value 验证；
- stable C1 使用 scheduler 的 CPU mirror，不做 D2H；
- TP1 不做无意义 consensus；未来 TP 才接固定长度 tensor consensus；
- device-resident sampled token 继续复用 Atom 现有 deferred token path。

当前 Atom 已经把上一 token 保存在 GPU 并用 `fill_deferred_decode_ids` 写入下一步 input buffer，因此 runtime 候选必须先用 trace 证明仍有可删除的 recurring seam。

## 7. 每轮验收

每个候选按以下顺序执行：

1. 保存接受态源码 SHA256、kernel symbol、JIT key 和环境。
2. fresh compile，检查 LDS/VGPR/SGPR/private、spill 和 occupancy。
3. 固定输入逐阶段对照：GDN output、conv state、recurrent state、HC streams/raw、MoE output、最终 hidden/logits/argmax。
4. token isolation 和 1000 graph replay。
5. 单组件 accepted/candidate/accepted GPU event。
6. accepted/candidate trace 或 ATT；ATT 不可用时保留工具失败证据、trace/PMC、ISA/resource delta。
7. 投影 ≥0.1 ms/token，且没有 private/VGPR spill 或 occupancy 降级时进入 FULL graph A/B/A/B；SGPR/VGPR 数增加必须由同节点组合 wall 的稳定收益覆盖。
8. FULL graph 通过后才更新接受态。

硬门槛：

- private/VGPR spill = 0；
- 一个 persistent CTA/CU；
- 不增加未经证明的 grid-wide/all-ready barrier；
- 生产路径无 `.item()`、`.cpu()` 或 host device-value validation；
- graph node 数与设计边界一致；
- rejected candidate 只保留 `summary.json`、`diagnosis.md`、小型 diff、结果 JSON 和压缩 ISA。

## 8. 已关闭路线

- GDN front 160→1 all-ready：每层多 20–23 µs，FULL graph 回退约 0.39–0.49 ms。
- G1 gate/up 串行 weight：减少部分寄存器，但破坏 load/MFMA overlap。
- G2 W2 跨 poll 长生命周期 prefetch：提高 VGPR/SGPR spill 和 LDS wait。
- G2 static partition：删除 atomic 后失去 arrival-rank 的负载均衡。
- paired shared-G2 body：复用 shared W2，却把两个 routed 链串行化。
- G2 slot/part queue + ordered merge：逐位正确且无 spill，但跨 CTA partial、arrival 和 merge 令 kernel trace 从 13.08 µs 增至 59.68 µs；PMC 显示 SALU、VMEM write、LDS wait 和总等待显著增加。
- m35 S=4 split finish：36 层等效仅约 42.5 µs，低于 0.1 ms gate，已拒绝。

## 9. 当前下一步

1. G2 slot/part queue 已关闭；正式 G2 保留 same-CTA chain。
2. HC group-ready 已通过独立 probe、正式 GDN/QSA 数值、资源和 A/B/A 验收，作为 TP1 接受态提交；整网候选 B 尚未完成，因此不更新 6.65–6.68 ms 的既有接受基线。
3. 暂停 TP1/C1 局部优化和未完成的 FULL graph A/B/A/B，不继续实现 M37/M38。
4. 下一阶段转向真实部署主路径 TP8：先冻结 Atom 原生 TP8 的 rank-local shape、collective、FULL graph wall 和 trace，再单独设计 Qwen TP8 K1/K2、peer-buffer 和 rank-order reduce。
5. TP8 的结构与里程碑记录在
   [`qwen38_flash_next_tp8_monokernel_plan.md`](qwen38_flash_next_tp8_monokernel_plan.md)，
   避免继续扩张本 TP1/C1 文档。
