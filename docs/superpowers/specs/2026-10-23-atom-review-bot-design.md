# atom-bot：ATOM PR 自动评审流水线 — 设计

日期：2026-10-23
分支：`feat/atom-review-bot`
工作目录：`/mnt/raid0/zufa/atom-bot-work`（ATOM worktree，基于 `origin/main` @ e914a0937）

> 本设计文档用中文书写，因为它是本次工作的内部规划文档。
> **最终落入 `.claude/skills/` 的 skill 文件、脚本注释、workflow 注释一律用英文**，
> 与 ROCm/ATOM 和 ROCm/aiter 上游保持一致。

---

## 1. 背景

ROCm/aiter 已有一套成熟的 PR 自动评审流水线，由 `@aiter-bot review` 评论触发：

- `.claude/skills/review-pr/` — 24 个文件、约 7000 行（SKILL.md 仅占 500 行）
- `.claude/skills/validate-kernel-pr/` — 14 个文件
- `.github/workflows/aiter-review-bot.yml` — `issue_comment` 原生触发，四个 job

ROCm/ATOM 目前只有 `.claude/skills/review-pr/SKILL.md` 一个文件、582 行纯 prose。
它是 aiter 的早期分支：Step 1–8 骨架一致，Step 6 的六条结构化检查几乎逐字相同
（文中仍残留 "The sibling aiter skill's P5..." 的痕迹），但停在了 aiter 引入
artifact + gate 之前的那一代。

**目标**：在 ATOM 建立等价的 `atom-bot` 流水线，并借这次移植把 ATOM 的评审规则
本身升级一遍。

## 2. 目标与非目标

### 目标

1. `@atom-bot review` 评论触发，在 ATOM PR 上产出评审卡。
2. ATOM 的 review-pr skill 从"纯 prose"升级为"artifact + gate"结构，
   使"跳过某一步"和"执行某一步"在产物上可区分。
3. ATOM 的评审规则从真实 ATOM 评审历史中重新生长，每条规则挂真实 PR 编号作为证据。
4. 提供最小化的端到端确定性验证（`validate-atom-pr`），结论与 advisory 评审严格分离。

### 非目标

- 不把评审变成 merge gate。review 线永远 advisory；只有带可复现 reproducer 的
  deterministic 结论才可能 gate，且本期不启用。
- 不重造 ATOM 已有的 accuracy / benchmark 基建。bot 复用
  `.github/benchmark/models_accuracy.json` 的既有 catalog 结构。
- 不在本期解决 box308 的 GPU 占用问题（见 §5.3）。

## 3. 总体架构

形状完全照搬 aiter，因为那套结构已被 597 个 PR 的实践校准过，没有理由重新发明。

```
┌─ gate (ubuntu-latest, 免费) ────────────────────────────────┐
│  引号感知触发检测 + authorized.txt 白名单 + queue_gate 队列深度 │
│  未授权评论在此被挡住，绝不触达自建机                          │
└────────────────┬───────────────────────────────────────────┘
                 │ ok == true
     ┌───────────┴────────────┬──────────────────────┐
     ▼                        ▼                      ▼
┌─ watchdog ─────┐  ┌─ review (self-hosted) ──┐  ┌─ report ──────┐
│ ubuntu         │  │ claim 👀                │  │ ubuntu        │
│ 20min 无 runner │  │ run_one.sh              │  │ review 整个    │
│ 接单就报告      │  │   → fetch.sh            │  │ 消失时兜底     │
│                │  │   → worker  (GLM)       │  │               │
│                │  │   → refuter (跨家族)     │  │               │
│                │  │   → _gates.py           │  │               │
│                │  │   → _collect.py         │  │               │
│                │  │ _publish.py / _notify.py│  │               │
└────────────────┘  └─────────────────────────┘  └───────────────┘
```

三个 job 跑在 GitHub 托管 infra 上，正好在自建机不可用时仍然活着——这是 aiter
踩过的坑（runner 掉线后 job 停在 `queued`，job 内的通知器永远不会运行，
GitHub 也不发邮件，那台机器静默离线了数天）。

### 3.1 仓库布局

```
.claude/skills/review-pr/
  SKILL.md              ≤500 行，硬预算
  rules.md              规则正文，按 family 派生后注入
  MAPPING.md            family → rule 映射，由 triage.py mapping 生成
  RUNNER-SETUP.md       部署文档
  authorized.txt        触发白名单
  fetch.sh              Step 1：抓 PR + 建 worktree + 取证
  triage.py             family 派生器 + gate 子命令
  run_one.sh            headless 编排
  render.sh             prompt 模板替换
  check_prompts.py      prompt 与 SKILL.md 漂移检测
  prompts/{worker,refuter}.md
  _gates.py _collect.py _lib.py _publish.py _notify.py _apply_refutation.py
  queue_gate.py watchdog.py lost_review.py
  preflight.sh selftest.sh
  tests/

.claude/skills/validate-atom-pr/
  SKILL.md
  validate_pr.sh
  pick-idle-gpu.py
  report_schema.json

.github/workflows/atom-review-bot.yml
.github/benchmark/models_accuracy.json    新增 test_level: "bot" 条目
```

**命名不沿用 `validate-kernel-pr`**：ATOM 验的不是 kernel，是端到端推理行为。
名字错了会诱导后来者按 kernel 的思路加东西。

### 3.2 命名与配置原则

标识符改名：

| aiter | ATOM |
|---|---|
| `@aiter-bot review` | `@atom-bot review` |
| `.aiter-review-status` | `.atom-review-status` |
| `<!-- aiter-bot review pr=N head=SHA -->` | `<!-- atom-bot review pr=N head=SHA -->` |

### 配置分类原则

aiter 有 15 个 `AITER_*` bot 配置变量（另 2 个 grep 命中是 aiter 自身的运行时宏，
不属 bot 配置）。ATOM 不照单全收，按**三类三去处**重新归置：

| 类别 | 去处 | 判据 | ATOM 的内容 |
|---|---|---|---|
| **秘密** | GitHub secrets → job env | 提交即泄露 | `ATOM_BOT_TOKEN` |
| **机器事实** | runner `.env`，不进版本控制 | 值因机器而异，换台机器就得改 | agent 入口、refuter 入口、`TMPDIR`、`GH_CONFIG_DIR`、`PATH`、GPU 子集（§5.2） |
| **策略** | **提交进仓库** | 对所有机器相同，且改动应当可审计 | 5 个 failure-class → owner 映射、3 个时间预算、探活超时 |

`ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` / `TMPDIR` / `PATH` / `GH_CONFIG_DIR`
的**名字**是 Anthropic SDK、POSIX、gh 的既定约定，不由本项目决定；
它们的**值**是机器事实，归第二类。

**可直接删除的 2 个**：`*_BOT_NAME` / `*_BOT_REPO`，均可从 `GITHUB_REPOSITORY` 推出。

**时间预算必须提交，有一条硬理由**：`check_prompts.py` 中有断言要求
`RUN_BUDGET < workflow 的 timeout-minutes`，而后者在版本控制里。一个**被提交的
断言**约束着一个**没被提交的值**——这正是漂移的源头。两者一起提交，断言才有意义。

ATOM 的预算值（aiter 对照：run budget 100 分钟 / job 120 分钟）：

| 项 | ATOM | aiter | 说明 |
|---|---|---|---|
| `RUN_BUDGET` | **120 分钟** | 100 | 整轮 run，由 fetch / worker / refuter / validate 共享 |
| workflow `timeout-minutes` | **150** | 120 | 必须严格大于上行，`check_prompts.py` 断言之 |
| `AGENT_TIMEOUT` | 40 分钟 | 40 | 单次 agent 尝试；超时不重试 |

ATOM 给到 2 小时而非 aiter 的 100 分钟，是因为 §6 的 validate 要在同一轮里起两次
服务、跑两次 eval（见 §6.6）。留 30 分钟的 job 余量而非 aiter 的 20 分钟，
同理——一个在 job 上限被硬杀的 run 死时没有任何分类状态，比超时更难排查。

**owner 映射提交进仓库**（与 `authorized.txt` 同级）的权衡：aiter 把默认值写在
`_notify.py` 里、用 repo var 作覆盖，好处是改 owner 不需要 committer 权限。
ATOM 当前只有一名 owner，这个好处用不上，而"换 owner 是一个可审计的 PR 而非
repo settings 里无人看见的一次修改"的好处是实的。若将来 on-call 轮转由非
committer 调整，再加回覆盖通道。

**曾经考虑并否决的方案**：把 GPU 子集也提交进仓库。用一个场景即可证伪——那将
意味着**改我们自己这台机器用哪几张卡要发一个上游 PR**。GPU 子集是机器事实，
与路径同类，归第二类。记录此否决，以免日后重提。

**aiter 的问题不是"用了环境变量"，是把策略也放进了 `.env`。** 其
RUNNER-SETUP.md 自己写了后果：

> `.env` is not in version control -- so a value left here outlives the reason
> it was added. Remove it once the default covers the case.

与其事后清理，不如一开始就按类别放对地方。

## 4. 组件分类

### 4.1 可直接移植（改名即可，约 1200 行）

`run_one.sh`、`render.sh`、`check_prompts.py`、`_gates.py`、`_collect.py`、
`_lib.py`、`_apply_refutation.py`、`_publish.py`、`_notify.py`、`queue_gate.py`、
`watchdog.py`、`lost_review.py`、`atom-review-bot.yml`。

其中 `_apply_refutation.py`、`_collect.py`、`_gates.py` 零处提及 aiter。

移植时**保留原注释**。那些注释记录的是真实事故（"十八次评审产生三十六次自我竞争的
run"、"这台机器静默离线数天"），删掉就等于把教训丢了，下一个人会重新踩。

### 4.2 必须重写

**`triage.py` 的 `derive()`** — aiter 的 45 个 family 正则全是 aiter 路径
（`csrc/kernels/`、`aiter/ops/triton/`、`flydsl`、`@compile_ops`、
`aiter/__init__.py` 导入链）。ATOM 的轴完全不同，见 §7。

**SKILL.md + rules.md 的拆分与升级** — 见 §7。

**`fetch.sh` 的取证器** — 947 行中，FlyDSL bounds、AOT pairing、struct ABI
对 ATOM 无意义，删除；guards / siblings / symbols / twins / test_quality /
ci_coverage / perf_claims 保留；新增 ATOM 特有的（cudagraph 捕获区内的动态分配、
跨 repo 的 aiter 符号解析）。

### 4.3 新增

`validate-atom-pr` 的 bot 级 case 执行器，见 §6。

## 5. 机器与 GPU 共享

### 5.1 两台机器的实测底数（2026-10-23）

| | box308 = hjbog-srdc-8 | srdc-24（开发/测试机） |
|---|---|---|
| GPU | 8× MI308X gfx942 (DID 0x74a2) | 8× MI308X gfx942 (DID 0x74a2) |
| GPU 占用 | **全满**：GLM-5.3 server 占 VRAM 98–99%，已运行两周 | **全空**：VRAM 0% |
| review agent 模型 | 本地 GLM（`localhost:30000`），吃掉全部 8 卡 | 开发期走 AMD 网关（`claude-glm`），不吃本机卡 |
| 盘 | `/mnt/raid0` 剩 9.4T | `/mnt/raid0` **100% 满**（剩 126G）；`/mnt/md0` 剩 9.6T |
| 已有 runner | `/mnt/raid0/zufa/aiter-runner/actions-runner`（标签 `box308`） | 无 |
| 共享性 | — | 多用户共享（ljin1 3.7T、yifehuan、anqfu、felix… 十余人） |

### 5.1b 两阶段的模型后端，与由此而来的 GPU 账

**开发期（srdc-24）**：review agent 走 AMD 网关（`claude-glm`）。本机 8 张卡全空，
validate 不受影响。代价是网关配额——一次 review 要喂 SKILL.md 全文 +
`rules_expanded.txt` + 完整 diff + 十余个 artifact，worker 与 refuter 各一轮且可重试，
而网关限流为 2M uncached input token / 60s 并会发递增 Retry-After。
**这是开发期风险，不是架构风险**：它影响迭代速度，不影响最终形态。
P0 第一次真实 review 时应顺手记录 token 量，作为后续判断依据。

**实测状态（2026-10-26，Task 9）：仍未测。** `run_one.sh` 要在 auto-mode 的 Claude
会话里启动一个带 `--dangerously-skip-permissions` 的 headless 子 agent，而本环境的
auto-mode 分类器把这个模式判为 "Create Unsafe Agents" 并拦下——这不是误报，`run_one.sh`
脚本头部本身就写着同样的警告。因此 Task 9 只能跑完全部静态检查（pytest/selftest/
check_prompts/actionlint/preflight 五项，见 task-9-report.md），**无法产出第一次真实
token 用量与墙钟耗时**。这笔数字之所以重要：一次 review 要把 SKILL.md 全文、
`rules_expanded.txt`、完整 diff 和十余个 artifact 喂给模型，worker 与 refuter 各一轮
且可重试，而网关限流是 2M uncached input token / 60s——配额够不够、要不要降级重试
策略，都要等这个数字才能判断。第一次在普通终端（非 auto-mode 会话）里跑通
`run_one.sh` 就是产出这个数字的动作，留给 `P0-HANDOFF.md` 里描述的验收跑。
**这里不编造估算值**——没测出来就留空，比一个看着像实测的估算更诚实。

**最终形态（box308）**：review agent 指向该机已在运行的 GLM-5.3
（`localhost:30000`），零配额问题。

**但由此产生一笔必须算清的 GPU 账**：box308 的 8 张卡正被那个 GLM 占死
（VRAM 98–99%），而它同时是最终目标机。也就是说**最终形态下 validate 在 box308
上没有卡可用**。这不是"以后再说"的问题，它决定 §6 的 validate 能不能在生产形态
下成立。候选解见 §5.3，必须在 P3 之前定。

设计约束（现在就要守住）：**不要写出任何假设"review 与 validate 必须同机"的代码**。
`validate_pr.sh` 本就是独立可调用脚本，保持这一点，则"review 在 box308、validate
在 srdc-24"始终是可选项。

runner 装 `/mnt/md0`（剩 9.6T），不装 `/mnt/raid0`（已 100% 满）。
RUNNER-SETUP 明确警告过根盘满会让 review 跑到一半失败。

### 5.2 GPU 共享策略（子集优先 + 可降级）

aiter 的 `pick-idle-gpu.py` + `validate_pr.sh` stage 2 已经解决了大部分问题：

1. **采样窗口判空闲**，不是瞬时快照：10 次 × 1s，阈值 `peak gfx ≤2%` /
   `used ≤2GiB` / `free ≥16GiB`。
2. **"未知"绝不当"空闲"**：活动查询失败时降级为 VRAM-only，并输出
   `idleness-basis: vram-only`。该脚本注释中已记录
   *"MI308X on ROCm 7.0 raises AMDSMI_STATUS_UNEXPECTED_DATA"* ——
   作者正是在 MI308X 上踩的，与本项目用的卡完全一致。
3. **`--all` 返回完整候选排序**：第一名的锁被抢时可往下落，
   避免"明明有空闲卡却报没有"。
4. **`flock -n /tmp/gpu-<N>.lock` 全程持锁**，全部被锁时报
   "all N verified-idle GPUs were already locked by another validator"，
   明确归类为环境事实而非 PR 缺陷。
5. base 与 head 在**同一张锁定的卡**上背靠背跑。

`flock` 只约束同样使用 flock 的进程。srdc-24 是多用户共享机，同事直接
`HIP_VISIBLE_DEVICES=3 python train.py` 不会看这把锁。因此新增：

- `pick-idle-gpu.py --only <list>`：默认只在划定子集内选卡。
- `pick-idle-gpu.py --fallback-all`：子集全忙时才放开到其余卡。
- 流程：**子集内采样 + flock → 全忙则按完整排序采样其余 → 再全忙则
  `NOT RUN — all verified-idle GPUs locked`**。
- 报告强制记录 `gpu_pool: subset|fallback` 与 `idleness-basis`。

默认子集：`ATOM_BOT_GPUS=6,7`（末尾两张，与人工从 0 开始的习惯错开）。
**此值待与机器上其他使用者确认后固定。**

### 5.3 box308 的未决问题

box308 是最终目标机，但它的 8 张卡被 GLM-5.3 server 占死。迁移到 box308 时必须
正面解决，候选方案（**本期不设计，只留接口**）：

- 把 GLM 缩到 TP4，腾出 4 张卡；
- 接受 box308 上 validate 恒为 `NOT RUN — no idle GPU`（gate 支持此状态）；
- validate 留在 srdc-24，review 在 box308，两机分工。

本期的设计约束是：**不要写出任何假设"review 和 validate 必须同机"的代码**。
`validate_pr.sh` 已经是独立可调用的脚本，保持这一点。

## 6. validate 最小 case 设计

### 6.1 复用 ATOM 既有基建

ATOM 已有机器可读的 accuracy catalog，不需要重造：

- `.github/benchmark/models_accuracy.json` — 41 条，带 `test_level`
  （`pr` 15 / `main` 7 / `nightly` 19）、`accuracy_threshold`、
  `accuracy_baseline`、`runner`、`extraArgs`、`env_vars`。
- `.github/workflows/atom-test.yaml` 已按 event 过滤 level：PR 事件只跑 `pr` 级。
- `atom-test.yaml` 中的 `offline-smoke-test`（Llama-3-8B 单模型、45 分钟上限）
  注释原文：*"one small model exercises that code path as well as thirteen do"*
  —— ATOM 的最小化哲学已经成文并落地。
- `scripts/run_gsm8k_eval.sh` 已有 `LIMIT` 旋钮与 `CHAT` 模式。

借鉴 sglang 的做法：sglang 没有 bot 流水线（其 `cookbook-review-pr` 是扁平
checklist），但它把最小化 e2e 的正确性约束写成了 repo 级 rule
（`.claude/rules/prefer-sgl-eval.md`：统一 harness、`--num-examples 200` 小样本、
sampling 必须显式传、**只比同 harness 同 settings 的分数并注明用了哪套**）。
核心思想是：**最小化不是"少跑点"，是"明确说清这个小 case 能抓住什么、抓不住什么"。**

### 6.2 新增 `test_level: "bot"`

一条小模型 case + GSM8K `LIMIT=200`。

**首选候选：`meta-llama/Meta-Llama-3-8B-Instruct`**，即 `atom-test.yaml` 的
`offline-smoke-test` 已在用的那个模型（其设置 `--kv_cache_dtype fp8
--gpu-memory-utilization 0.3` 也沿用）。选它的理由：权重已在机器上、单卡可起、
非 thinking 模型、ATOM 自己已经用它守着 offline 入口这条路径。

**选型约束（按优先级）**：① 非 thinking 模型；② 单卡可起（TP=1），
以便与 §5.2 的子集策略兼容；③ 权重已在本地，不引入下载时间；
④ 走 aiter 算子路径，否则 validate 对 ATOM 最常见的 PR 类型没有覆盖。
候选若与 ④ 冲突，需在 P2 重新选型并记录理由。

**为什么必须避开 thinking 模型**： `run_gsm8k_eval.sh` 的注释已记录：
completion 模式下 thinking 开不开是硬币翻面（在 Qwen3.5 上实测 220/300 不思考、
80/300 思考，后者耗时 8 倍且常截断），带来约 **2.5pp** 的分数抖动。
小样本下这个抖动会淹没信号。

### 6.3 核心决策：比 delta，不比绝对分

base 与 head 在**同一张锁定卡**上、同 harness、同 `LIMIT`、同 seed、同 sampling
背靠背跑，报告 `head − base`。

论证：catalog 中 baseline 为 1319 samples、`±0.0059`。降至 200 samples 时
σ ≈ 0.0059 × √(1319/200) ≈ 0.015，±2σ ≈ **3pp**；而现有 threshold 余量仅 2pp
（如 `thr 0.94 / base 0.96`）。**小样本的绝对分无法与 catalog baseline 比较**，
配对跑才能把共模噪声抵消掉。

这与 aiter validate 的 base/head 背靠背结构同构，不是新发明。

### 6.4 前置任务：实测噪声地板

delta 阈值**必须由实测决定，不得拍脑袋**。上线前先跑 N 次 base-vs-base
（同一 commit 跑两遍），测出配对 delta 的分布，取其尾部定阈值。
跳过这一步等于重新引入假阳性——而假阳性正是整个流水线最想避免的东西。

验收标准：记录 N、每次的 delta、分布摘要，写入 `validate-atom-pr/SKILL.md`
作为该阈值的依据，连同测定日期与所用 commit。

### 6.5 时长账 — 决定 validate 能否进 review job

validate 不是跑个 pytest，而是起服务再打 eval，且 base / head 各来一遍：

```
base: 起服务 + 加载 8B 权重 + 200 题 GSM8K + 停服务
head: 同上
```

两次模型加载 + 两次 eval，全部要塞进 §3.2 的 `RUN_BUDGET`（120 分钟），
且与 fetch / worker / refuter 共享该预算。

**这个数未实测。** 它决定 §6 的形态：

- 若两轮总计可在约 60 分钟内完成 → validate 进 review job，卡上直接出
  `Validation` 行（本设计的现有假设）。
- 若超出 → 退回独立 job：review 卡先出，`@atom-bot validate` 另行触发，
  结果事后补评论。

**P2 的第一个动作就是实测这个数**，而不是先写接入代码。先写代码再发现塞不进，
等于白写一轮接入逻辑。实测方法：在 srdc-24 上手工跑一遍完整的 base+head 流程并计时。

### 6.6 报告行

`Validation (deterministic)` 行照 aiter 的五态原样写：

- 有精确 head 匹配的报告 → 结论 + target/runner/arch/skipped stages
- triage 判定无运行时面 → `N/A — no runtime surface changed`
- 需要但无 target → `NOT RUN — <triage reason>`
- 需要且有 target 但跑不了（无空闲 GPU / validator 缺失 / 开关关闭）
  → `NOT RUN — <reason>`（环境缺口，非 PR 缺陷）
- 在任何 `NOT RUN` / `N/A` 状态下，findings 不得把运行时行为当事实断言

## 7. ATOM family 轴与规则升级

### 7.1 这不是搬运，是重新生长

ATOM 现有 582 行 SKILL.md 的规则没有真实 ATOM PR 编号作为证据
（aiter 的 `rules.md` 每条规则都挂着真实 PR）。P1 的交付不是"把文件拆成两个"，
而是"规则从真实 ATOM 评审历史里重新长一遍"。

### 7.1b 语料：过去半年 ATOM 所有公开渠道的评论

**取数范围**：近 6 个月（2026-04-23 起）ROCm/ATOM 的全部公开评论渠道——
PR 顶层评论、PR inline review comments、review 正文、issue 评论。不设 reviewer
白名单：外部协作者抓到的真问题与 maintainer 的同样算数；噪声在下一步筛掉。

**排除**：机器人账号（copilot、各类 CI bot）的评论。

**从评论到规则的路径**（三步，前两步可半自动，第三步必须人工）：

1. **抽取**：按渠道拉全量评论，附带所在 PR 号、文件路径、改动类型。
2. **聚类**：按"评论指向的缺陷类别"归并，而非按文字相似度。
   一簇达到可辨认的规模即为候选 rule。
3. **成文**：人工写 rule body，**必须挂至少一个真实 ATOM PR 编号作为证据**，
   并指派给 `triage.py` 中某个 family。无 family 承载的 rule 永远不会被读到，
   `triage.py expand` 对此会失败——这是 aiter 的既有约束，照搬。

**验收**：每条新 rule 都能回答"若删掉它，什么样的真实 ATOM PR 会漏检"，
并给出那个 PR 的编号。答不出的不收——这条判据借自 sglang 的
`unit-test-admission.md`（"what future diff would turn this case red?
If the only answer is editing the test itself, delete it"）。

### 7.2 候选 family

从 ATOM 现有 SKILL.md 的 Step 3 清单与 Step 4 的 Q1–Q4 判据提炼：

| family | 触发信号 | 关注点 |
|---|---|---|
| `tier1-every-request` | `model_runner.py` / `config.py` / `scheduler.py` | 每次 forward 都执行 |
| `tier1-model-base` | `deepseek_v2.py` 等多模型族基类 | 子类是否覆写；"Kimi-only fix" 实际影响四个模型族 |
| `tier2-shared-op` | `moe.py` / `attention_mla.py` / `linear.py` / `layernorm.py` / `aiter_mla.py` | 跨模型族共享算子 |
| `aiter-dep` | 新 `from aiter import X`、aiter 调用新 kwarg | 跨 repo 同步 |
| `plugin-bridge` | `atom/plugin/vllm/*`、`atom/plugin/sglang/*` | 桥接 API 破坏 |
| `cudagraph-capture` | 捕获区内新增 Python 控制流 / 动态分配 / 属性查找 | 静默破坏 cudagraph |
| `ep-tp-shapes` | `num_experts / world_size`、TP/EP 分母 | 用错并行度 → 路由到错误专家 |
| `dispatch-change` | `if dtype/arch/layout/quant/backend` | 静默绕过 |
| `compile-contract` | `@compile_ops` / `torch.library.custom_op` | fake impl 的 dtype/shape |
| `env-var-registry` | `os.environ.get("ATOM_...` | 是否在 `atom/envs.py` 注册并写进文档 |
| `fp8-quant` / `weight-variant` | fp8 / fnuz / `_preshuffled` | dtype 判定应看 tensor.dtype 而非 arch 名 |
| HK 系列 | 临时脚本 / 无关文件 / TODO / sys.path | 通用卫生 |

### 7.3 验收标准

aiter 的派生器是在 597 个开放 PR 上量化过的：中位数命中 15/49 条规则（31%），
每个 PR 至少命中一个 family，没有任何 family 命中超过半数 PR。

ATOM 的 `derive()` 必须做等价标定，拿真实 ATOM PR 历史跑：

- 每个 PR 至少命中一个 family（否则派生器有洞）
- 没有 family 命中超过半数 PR（否则该 family 是噪声）
- 记录中位数命中率
- `triage.py mapping` 生成的 `MAPPING.md` 与代码不得漂移（由 tests 保证）

### 7.4 SKILL.md 的 500 行预算

照搬 aiter 的硬预算与理由：该文件曾在两个月内从 487 → 632 → 1372 行，
而指令内容几乎没动（1210 行时，825 行是 shell、299 行是 prose）。
条件性内容进 `rules.md` 经 `triage.py expand` 到达评审者；
可执行内容进脚本被调用。提高预算必须在 commit message 里说明理由。

## 7.5 与 ATOM 现有 CI 的关系：各走各的专用 lane

ATOM 现有 `atom-test.yaml` 在 PR 上会跑 accuracy matrix（`ci-gate` 之后），
与 bot 的 validate 跑的是同一套 GSM8K。不理清关系会出三个问题：重复跑、
结论冲突、抢同一批 runner。

照 aiter 的做法：**bot 自带专用 CI，与仓库主 CI 完全分离**。
aiter 有 `.github/workflows/review-bot-selftest.yml`，只负责守 bot 自己的代码。

ATOM 对应地建 `.github/workflows/atom-review-bot-selftest.yml`：

- **触发**：仅当 `.claude/skills/review-pr/**` 或
  `.claude/skills/validate-atom-pr/**` 变化时。
- **内容**：`selftest.sh`（守 `run_one.sh` 的各道 guard）+ `check_prompts.py`
  （prompt 与 SKILL.md 的漂移）+ `triage.py mapping` 与 `MAPPING.md` 的一致性
  + `triage.py` 的单元测试。
- **跑在哪**：`ubuntu-latest`。全是纯逻辑检查，不需要 GPU、不碰自建机。

三个问题的答案：

| 问题 | 答案 |
|---|---|
| 重复跑？ | 不重复。主 CI 的 accuracy 是**跨 PR 的回归门禁**（绝对分 vs catalog baseline）；bot 的 validate 是**单 PR 的 base/head 配对比较**（delta）。量的是两件事。 |
| 结论冲突听谁的？ | **主 CI**。它是 merge 门禁，样本量完整；bot 的 validate 是 advisory 证据，样本量小。冲突时卡上如实并列两者，不替人类做裁决。 |
| 抢 runner？ | 不抢。主 CI 跑在 ROCm/ATOM 的 org runner（`linux-atom-do-mi350x-8` 等）；bot 跑在自带标签的专用 runner（`atom-bot-24`）。标签不重叠是硬要求（§8 P0）。 |

## 8. 分期与验收

### P0 — 链路跑通（srdc-24 + zufayu/ATOM fork）

1. 在 srdc-24 的 `/mnt/md0` 安装 runner，注册到 `zufayu/ATOM` fork
   （注册 token 自助可得，不需要 ROCm admin —— 照
   `/mnt/raid0/zufa/setup-fork-runner.sh` 的既有路径）。
   标签：`atom-bot-24`（**不复用 `box308`**，否则 job 会被 aiter 的 runner 抢走）。
2. 移植 §4.1 的 1200 行，改名。
3. `atom-review-bot.yml` 四个 job。
4. 在 fork 上造 PR，跑通评审链路至产出评审卡。

**验收**：`card.md` 产出且 7 道 gate 全绿；`preflight.sh` 全绿；`selftest.sh` 通过。

**注意验收标准不是"PR 上出现评论"**。bot 账号的 PAT 尚未发放，而 token 在 aiter
那套里**只被发布链路用到** —— claim 👀、`_publish.py`、`_notify.py`、
`queue_gate.py`、`watchdog.py`、`lost_review.py`。评审本身
（fetch → worker → refuter → gates → collect）一行 token 都不用，而且是**故意的**：
review step 跑的是 `--dangerously-skip-permissions` 的 headless agent、处理不可信的
PR 内容，aiter 专门把 token 挡在那一步之外。

因此无 token 不阻塞 P0，只需把验收落在评审质量上。相应降级：

- `_publish.py` **默认即 dry-run**（其 docstring: *"Dry-run by default; --post
  actually posts"*），不传 `--post` 正是所需形态。
- claim 👀 步骤在无 token 时跳过。
- `_notify.py` 无 token 时返回 0 且不发帖；`report` job 本就是为"review 说了但没
  说出口"兜底而存在，该路径已有设计。

### P0.5 — 发布链路（拿到 bot token 当天）

配 `ATOM_BOT_TOKEN` secret，给 `_publish.py` 加 `--post`，恢复 claim 👀。

**验收**：fork PR 上出现 atom-bot 发的评审卡；同一 head SHA 重复触发被去重。

这样拆的收益是**评审质量与发布权限解耦**：前者不等 token 即可做完做透。

### P1 — skill 升级（重点）

1. 拆 SKILL.md（≤500 行）+ `rules.md`。
2. 规则从真实 ATOM 评审历史重新生长，每条挂真实 PR 编号。
3. 写 ATOM 版 `derive()`，按 §7.3 标定。
4. 裁剪并新增 `fetch.sh` 的取证器。
5. 补齐 artifact + gate：`answers.txt` / `core_files.txt` / `verdicts.txt` /
   `ai_diagnostic.txt` / `refutations.txt` / `independent.txt`。

**验收**：§7.3 的四条标定指标；gate 能挡住一次人为制造的"跳步"评审。

### P2 — validate

1. `pick-idle-gpu.py` 加 `--only` / `--fallback-all`。
2. 新增 `test_level: "bot"` case。
3. 实测噪声地板，定 delta 阈值（§6.4）。
4. 接进 review job，产出 `Validation (deterministic)` 行。

**验收**：噪声地板实测数据入档；一次人为引入的精度回归能被 delta 抓到。

### P3 — 上游

1. 解决 box308 的 GPU 占用（§5.3）。
2. runner 注册到 ROCm/ATOM（需 repo admin）。
3. `authorized.txt` 填 ATOM reviewer roster。
4. 配 `ATOM_BOT_TOKEN` 等 secrets / vars。

## 9. 风险与未决项

| 项 | 状态 |
|---|---|
| 默认 GPU 子集 `6,7` 是否与同事冲突 | **待确认**：srdc-24 上是否有卡已被口头认领 |
| atom-bot 账号 PAT | **待发放**。不阻塞 P0（评审链路不用 token），发布链路拆为 P0.5 |
| box308 的 GPU 占用 | **已升级为必答项**：box308 既是最终目标机、其 8 卡又被 GLM-5.3 占死，故最终形态下 validate 无卡。P3 前必须在 §5.3 三个候选里选定 |
| 开发期网关配额 | **仍未测（2026-10-26）**：auto-mode 分类器拦下了 `run_one.sh` 的 headless 子 agent（判为 "Create Unsafe Agents"，非误报），Task 9 没能跑验收。AMD 网关限流 2M uncached input token/60s，而一次 review 要喂 SKILL.md 全文 + `rules_expanded.txt` + 完整 diff + 十余个 artifact，worker/refuter 各一轮且可重试——配额够不够只有实测知道。首次在普通终端跑通 `run_one.sh` 就是产出这个数字的动作，不是精度问题而是"从未测过"；此处不写估算值。仅影响开发迭代速度，不影响最终形态（box308 用本地 GLM，零配额问题）|
| validate 两轮总时长 | **未实测**，决定 §6 能否进 review job。P2 的第一个动作就是实测（§6.5） |
| ~~maintainer 是否接受自建机 workflow~~ | 已消解：本项目作者即 ATOM team 成员。注意这与下一行的 repo admin 是两回事 |
| refuter 跨家族 | aiter 自己承认 worker/refuter 同为 GLM 时对抗性偏弱；ATOM 新建时应一开始就把 refuter 指向 Opus（`ATOM_BOT_REFUTER_AGENT`） |
| `authorized.txt` roster | 待定，P3 |
| ROCm/ATOM repo admin 权限 | 待确认，P3 阻塞项 |
| `/mnt/raid0` 已 100% 满 | runner 装 `/mnt/md0` 规避；但该盘是共享盘，长期有被填满的风险 |
