# atom-bot P0 — 评审链路落地 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 srdc-24 上把 ATOM 的 PR 自动评审链路跑通，对一个真实 ATOM PR 产出 `card.md` 且七道 gate 全绿——不依赖 bot token，不依赖 GPU。

**Architecture:** 移植 ROCm/aiter 已被 597 个 PR 校准过的 `review-pr` 流水线：`fetch.sh` 收集证据 → headless agent 写评审卡 → refuter 独立反驳 → 七道 gate 验证"每一步确实做了" → `_collect.py` 归档。P0 只做评审链路，发布链路（需 token）属 P0.5，ATOM 专属规则（family 轴）属 P1。

**Tech Stack:** bash、python3（标准库 + `jsonschema`）、gh CLI ≥2.24、GitHub Actions（`issue_comment` 触发）、self-hosted runner、headless Claude（`claude-glm`，走 AMD 网关）

**Spec:** `docs/superpowers/specs/2026-10-23-atom-review-bot-design.md`

## Global Constraints

- **工作目录**：`/mnt/raid0/zufa/atom-bot-work`（ATOM worktree，分支 `feat/atom-review-bot`）。
- **aiter 移植源**：`/mnt/raid0/zufa/aiter`，**固定在 `rocm/main` 的 `c1c482e00`**。每次取文件用 `git -C /mnt/raid0/zufa/aiter show c1c482e00:<path>`，不要用工作区（工作区在 fork 分支上，内容不同）。
- **语言**：所有 `.claude/skills/` 下的文件、脚本注释、workflow 注释**一律英文**，与 ROCm/ATOM 及 ROCm/aiter 上游一致。只有 `docs/superpowers/` 下的 spec 与本计划用中文。
- **bot 账号名**：`rocm-atom-bot`。（`atom-bot` 已被 2012 年注册的无关账号占用，不可用——触发词若写 `@atom-bot` 会 @ 到真人。）账号尚未注册，但 P0 不需要它存在；若最终注册了别的名字，改名命令见 Task 9 Step 5。
- **SKILL.md 硬预算 500 行**，且无超过 30 行的 code fence。提高预算必须在 commit message 里写明理由。
- **配置三类三去处**：秘密→GitHub secrets（仅 `ATOM_BOT_TOKEN`）；机器事实→runner `.env`（agent 入口、`TMPDIR`、`GH_CONFIG_DIR`、`PATH`、GPU 子集）；策略→提交进仓库（owner 映射、时间预算、探活超时）。**不要引入 `*_BOT_NAME`、`*_BOT_REPO`**（从 `GITHUB_REPOSITORY` 推导）。
- **时间预算（提交进仓库，写死在脚本默认值里）**：`RUN_BUDGET=7200`（120 分钟）、`AGENT_TIMEOUT=2400`（40 分钟）、workflow `timeout-minutes: 150`。`check_prompts.py` 断言 `RUN_BUDGET < timeout-minutes`，两者必须一起改。
- **runner 标签 `atom-bot-24`**，严禁复用 `box308`——否则 ATOM 的 job 会被 aiter 的 runner 抢走执行。
- **保留 aiter 的原注释**。那些注释记录的是真实事故（"十八次评审产生三十六次自我竞争的 run"、"这台机器静默离线数天"），删掉等于丢掉教训。移植时只改标识符，不删解释性注释。
- **P0 不需要 token**：`_publish.py` 默认 dry-run，claim 👀 步骤跳过，`_notify.py` 无 token 时返回 0。
- **每个 Task 结束必须 commit**，commit message 用英文，结尾加 `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`。

---

### Task 1: 基础设施层（零 aiter 耦合的四个文件）

`_lib.py`、`_gates.py`、`_collect.py`、`_apply_refutation.py` 是其余一切的地基。前三者定义了"七道 gate"和"必需 artifact"的契约；`_apply_refutation.py` 把 refuter 杀掉的 finding 从卡上摘掉。这四个文件在 aiter 里几乎零处提及 aiter（`_lib.py` 1 处，其余 0 处）。

**Files:**
- Create: `.claude/skills/review-pr/_lib.py`
- Create: `.claude/skills/review-pr/_gates.py`
- Create: `.claude/skills/review-pr/_collect.py`
- Create: `.claude/skills/review-pr/_apply_refutation.py`
- Test: `.claude/skills/review-pr/tests/test_gates_contract.py`

**Interfaces:**
- Produces: `_lib.REQUIRED`（必需 artifact 列表：`card.md`、`verdicts.txt`、`answers.txt`、`ai_diagnostic.txt`、`core_files.txt`、`refutations.txt`、`independent.txt`、`rules.txt`、`pr_meta.json`）、`_lib.OPTIONAL`（`late_findings.txt`）、`_lib.run_gates(work, proj) -> [(name, rc, first_line, full)]`、`_lib.nonempty(path) -> bool`、`_lib.pr_meta(work) -> (number, title)`
- Produces: `_gates.py <WORK_DIR> [PROJECT_ROOT]`，七道全绿 exit 0，任一红 exit 1
- Consumes（由 Task 3 提供）：`triage.py` 的七个 gate 子命令

- [ ] **Step 1: 建目录并取回四个文件**

```bash
cd /mnt/raid0/zufa/atom-bot-work
mkdir -p .claude/skills/review-pr/tests
A=/mnt/raid0/zufa/aiter
PIN=c1c482e00
for f in _lib.py _gates.py _collect.py _apply_refutation.py; do
  git -C "$A" show "$PIN:.claude/skills/review-pr/$f" > ".claude/skills/review-pr/$f"
done
wc -l .claude/skills/review-pr/_*.py
```

预期：`_lib.py` 185、`_gates.py` 40、`_collect.py` 100、`_apply_refutation.py` 51 行。

- [ ] **Step 2: 改掉 skill 路径引用**

`_lib.py` 的 `run_gates()` 里写死了 `.claude/skills/review-pr/triage.py` 的路径，目录名不变，所以**只需确认不需要改**。检查唯一一处 aiter 引用：

```bash
grep -n -i aiter .claude/skills/review-pr/_lib.py .claude/skills/review-pr/_gates.py \
  .claude/skills/review-pr/_collect.py .claude/skills/review-pr/_apply_refutation.py
```

预期：只在 `_lib.py` 命中 1 行。按上下文改成 ATOM 对应物；若它只是注释里的举例，改成 ATOM 的等价例子，不要删整条注释。

- [ ] **Step 3: 写契约测试（先失败）**

这个测试锁住"七道 gate 的名字和顺序"以及"必需 artifact 清单"。它存在的理由：后续任务会改 `triage.py`，而一个被悄悄改名或漏掉的 gate 不会让任何东西报错，只会让一道检查静默消失。

```python
# .claude/skills/review-pr/tests/test_gates_contract.py
"""Pin the gate roster and the required-artifact list.

A gate that is silently renamed or dropped does not fail anything -- the
review simply stops being checked on that axis. This test is the only thing
that notices.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import _lib

EXPECTED_GATES = [
    "answers", "diagnostic", "corefiles", "ledger",
    "card", "refutations", "independent",
]
EXPECTED_REQUIRED = [
    "card.md", "verdicts.txt", "answers.txt", "ai_diagnostic.txt",
    "core_files.txt", "refutations.txt", "independent.txt",
    "rules.txt", "pr_meta.json",
]
PROJ = str(HERE.parent.parent.parent)


def test_required_artifacts_are_pinned():
    assert _lib.REQUIRED == EXPECTED_REQUIRED
    assert _lib.OPTIONAL == ["late_findings.txt"]


def test_gate_roster_is_pinned(tmp_path):
    (tmp_path / "pr.diff").write_text("", encoding="utf-8")
    (tmp_path / "rules.txt").write_text("", encoding="utf-8")
    res = _lib.run_gates(str(tmp_path), PROJ)
    assert [name for name, _, _, _ in res] == EXPECTED_GATES


def test_empty_work_dir_fails_closed(tmp_path):
    """An empty WORK dir means no step ran. Every gate must go red, not green."""
    (tmp_path / "pr.diff").write_text("", encoding="utf-8")
    (tmp_path / "rules.txt").write_text("", encoding="utf-8")
    res = _lib.run_gates(str(tmp_path), PROJ)
    assert all(rc != 0 for _, rc, _, _ in res), \
        "a gate went green on an empty WORK dir -- it fails open"
```

- [ ] **Step 4: 跑测试，确认它失败**

```bash
cd /mnt/raid0/zufa/atom-bot-work && python3 -m pytest .claude/skills/review-pr/tests/test_gates_contract.py -v
```

预期：`test_required_artifacts_are_pinned` PASS（`_lib.py` 已经带着这些常量）；另外两个 FAIL，因为 `triage.py` 还不存在，`run_gates` 调不起来。这是预期的——Task 3 补上。

- [ ] **Step 5: Commit**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git add .claude/skills/review-pr/_lib.py .claude/skills/review-pr/_gates.py \
        .claude/skills/review-pr/_collect.py .claude/skills/review-pr/_apply_refutation.py \
        .claude/skills/review-pr/tests/test_gates_contract.py
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "review-pr: vendor the gate infrastructure from aiter

_lib.py, _gates.py, _collect.py and _apply_refutation.py carry no
repo-specific logic, so they port unchanged apart from one comment example.
The contract test pins the gate roster and the required-artifact list: a gate
that is silently renamed or dropped fails nothing, it just stops checking.

Two of its three cases fail until triage.py lands in Task 3.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: SKILL.md 骨架 + prompts + 漂移守卫

prompt 逐字引用 SKILL.md 的两段（Step 8 输出规则、Step 7.7），`check_prompts.py` 断言这个逐字关系。这一对必须一起落地，否则守卫守的是空气。

P0 的 SKILL.md 直接以 ATOM 现有 582 行为基础做**结构改造**（加 artifact 契约、gate 调用、500 行预算），**不动规则内容**——规则重写是 P1。

**Files:**
- Create: `.claude/skills/review-pr/SKILL.md`（≤500 行，覆盖现有 582 行版本）
- Create: `.claude/skills/review-pr/rules.md`
- Create: `.claude/skills/review-pr/prompts/worker.md`
- Create: `.claude/skills/review-pr/prompts/refuter.md`
- Create: `.claude/skills/review-pr/prompts/README.md`
- Create: `.claude/skills/review-pr/render.sh`
- Create: `.claude/skills/review-pr/check_prompts.py`

**Interfaces:**
- Produces: `render.sh <worker|refuter> <WORK_DIR> [findings-file]` → stdout 为替换好占位符的 prompt；占位符为 `{{PR}}`、`{{TITLE}}`、`{{REPO}}`、`{{WORK}}`、`{{PROJ}}`、`{{SKILL}}`、`{{FINDINGS}}`
- Produces: `check_prompts.py` exit 0 表示 prompt 与 SKILL.md 无漂移
- Produces: SKILL.md 中两个 verbatim 锚点（`check_prompts.py` 的 `CHECKS` 依赖它们逐字存在）：
  - `prompts/refuter.md`：起 `## Step 7.7 — Independent refutation`，止 `` `rules.md` § Refutation has why. ``
  - `prompts/worker.md`：起 `**Output rules (strictly enforced):**`，止 `` - `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion ``

- [ ] **Step 1: 把 ATOM 现有 SKILL.md 的规则正文搬进 rules.md**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git show origin/main:.claude/skills/review-pr/SKILL.md > /tmp/atom-skill-orig.md
wc -l /tmp/atom-skill-orig.md   # 预期 582
```

把 `### A — Coverage Gaps` 到 `### Housekeeping (quick scan)` 表格结束之间的内容，连同各自标题，原样写入 `rules.md`——**不改写**，P1 才重写。文件头加：

```markdown
# ATOM review rules

Rule bodies, injected into a review by `triage.py expand` according to the
families triage derived for the diff. They are NOT in SKILL.md: the median PR
needs a fraction of them, and shipping all of them into every review costs
attention that the derived subset is supposed to buy back.

Every rule must cite a real ATOM PR as evidence. The rules below were carried
over verbatim from the pre-pipeline SKILL.md and most do not yet carry one --
that is the work of P1, not a licence to add more without evidence.
```

**还要补一段 `STEP4` 的正文**，否则 `triage.py expand` 会在 Task 4 就失败——
它对"能被派生、却在 rules.md 里找不到正文"的 rule id 是硬失败：

```markdown
### STEP4 — Full backbone assessment required

Not a defect pattern: a pseudo-rule that says this diff touches a Tier 1
backbone file, so Step 4's assessment is mandatory rather than optional, and
`core_files.txt` must carry a line for every backbone file in the diff.

ATOM's Tier 1 is not an import chain -- there is no `import atom` to break.
It is "executes on every forward-pass request regardless of model"
(`model_runner.py`, `config.py`) or "base class for more than two production
model families" (`deepseek_v2.py` backs DSv2 / V3 / V3-0324 / Kimi, so a
change there reaches all four even when the PR says it is Kimi-only).

The reason must name what THIS PR changed. "core file, large blast radius" is
equally true of every PR ever opened against that file, and the corefiles
gate rejects it as UNANCHORED.
```

- [ ] **Step 2: 写 SKILL.md，加入 artifact 契约与 gate 调用**

以 `/tmp/atom-skill-orig.md` 为基础，做六处结构改造（规则正文已移出，SKILL.md 里只留指针）：

1. Step 2 末尾加：**把五个答案写入 `$WORK/answers.txt`，每行一条，`Q1:` … `Q5:`**
2. Step 4 开头加：**每个 backbone 文件一行写入 `$WORK/core_files.txt`**，格式 `<path> TIER1|TIER2|TIER3 COVERED|GAP|N/A -- <reason>`
3. Step 5 开头加：**逐条裁决 `$WORK/rules.txt` 写入 `$WORK/verdicts.txt`**，格式 `<RULE-ID> FIRE|CLEAR|N/A — <reason>`；规则正文指向 `$WORK/rules_expanded.txt`
4. Step 6 末尾加：**六条结构化检查各一行写入 `$WORK/ai_diagnostic.txt`**，`1:` … `6:`
5. 新增 Step 7.6（Refutation，写 `$WORK/refutations.txt`）与 Step 7.7（Independent refutation，写 `$WORK/independent.txt`），照 aiter SKILL.md 对应段落
6. Step 8 开头加七道 gate 的调用块，照 aiter SKILL.md Step 8（`triage.py` 路径保持 `.claude/skills/review-pr/triage.py`）

末尾加预算声明：

```markdown
**Nothing new goes in this file.** SKILL.md is budgeted at 500 lines and holds
only the prose every review needs, with no code fence over 30 lines.
Conditional content goes in `rules.md` and reaches the reviewer through
`triage.py expand`; executable content goes in a script and gets called.
Raising the budget is a decision to make this file less likely to be read in
full — make it in a commit that says why.
```

检查预算：

```bash
cd /mnt/raid0/zufa/atom-bot-work
wc -l .claude/skills/review-pr/SKILL.md
awk '/^```/{n++; if(n%2==1){s=NR} else if(NR-s>31){print "fence too long starting line " s}}' \
  .claude/skills/review-pr/SKILL.md
```

预期：行数 ≤500，无 fence 超长告警。

- [ ] **Step 3: 取回 render.sh / check_prompts.py / prompts 并改名**

```bash
cd /mnt/raid0/zufa/atom-bot-work
A=/mnt/raid0/zufa/aiter; PIN=c1c482e00
mkdir -p .claude/skills/review-pr/prompts
for f in render.sh check_prompts.py prompts/worker.md prompts/refuter.md prompts/README.md; do
  git -C "$A" show "$PIN:.claude/skills/review-pr/$f" > ".claude/skills/review-pr/$f"
done
chmod +x .claude/skills/review-pr/render.sh
sed -i 's|ROCm/aiter|ROCm/ATOM|g; s/aiter-bot/rocm-atom-bot/g' \
  .claude/skills/review-pr/render.sh .claude/skills/review-pr/prompts/*.md
grep -rn -i aiter .claude/skills/review-pr/render.sh .claude/skills/review-pr/prompts/
```

最后一条 grep 的残留逐条人工判断：**提到 aiter 作为 ATOM 的上游依赖是正确的**（ATOM 消费 aiter 算子），只有"把 aiter 当作被评审的仓库"才要改。

- [ ] **Step 4: 把 prompts 的 verbatim 块换成 ATOM SKILL.md 的对应段落**

用脚本抽取替换，不要手抄——手抄就是漂移的来源：

```bash
cd /mnt/raid0/zufa/atom-bot-work && python3 - <<'PY'
import pathlib, re
S = pathlib.Path(".claude/skills/review-pr/SKILL.md").read_text(encoding="utf-8")
CHECKS = [
    ("prompts/refuter.md", "## Step 7.7 — Independent refutation",
     "`rules.md` § Refutation has why."),
    ("prompts/worker.md", "**Output rules (strictly enforced):**",
     '- `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion'),
]
for rel, start, end in CHECKS:
    i = S.find(start); j = S.find(end, i)
    assert i >= 0 and j >= 0, f"anchor missing in SKILL.md for {rel}: {start!r} / {end!r}"
    block = S[i:j+len(end)]
    p = pathlib.Path(".claude/skills/review-pr") / rel
    t = p.read_text(encoding="utf-8")
    m = re.search(r"```\n(.*?)\n```", t, re.S)
    assert m, f"no fenced verbatim block in {rel}"
    p.write_text(t[:m.start(1)] + block + t[m.end(1):], encoding="utf-8")
    print(f"{rel}: replaced {len(block)} chars")
PY
```

- [ ] **Step 5: 跑漂移守卫**

```bash
cd /mnt/raid0/zufa/atom-bot-work && python3 .claude/skills/review-pr/check_prompts.py && echo "NO DRIFT"
```

预期：输出 `NO DRIFT`，exit 0。

`check_prompts.py` 还有个 `check_fail_classes()`，断言 `run_one.sh` 能发出的每个 failure class 都被 `_notify.py` 认识。两者 Task 5/6 才落地，此时应当**跳过**。若它报错而非跳过，让它对**缺失文件**宽容（缺文件 = 还没到检查的时候），但**绝不要**对"文件存在但 class 不匹配"宽容——那正是它存在的理由。

- [ ] **Step 6: 验证守卫真的抓得到漂移**

```bash
cd /mnt/raid0/zufa/atom-bot-work
cp .claude/skills/review-pr/prompts/worker.md /tmp/worker.bak
sed -i '0,/Output rules/s//Output ruIes/' .claude/skills/review-pr/prompts/worker.md
python3 .claude/skills/review-pr/check_prompts.py; echo "exit=$?"
cp /tmp/worker.bak .claude/skills/review-pr/prompts/worker.md
python3 .claude/skills/review-pr/check_prompts.py && echo "restored, NO DRIFT"
```

预期：改动后 `exit=1`，还原后通过。**抓不到人为漂移的守卫等于没有守卫**，这步就是验证它不是摆设。

- [ ] **Step 7: Commit**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git add .claude/skills/review-pr/SKILL.md .claude/skills/review-pr/rules.md \
        .claude/skills/review-pr/prompts .claude/skills/review-pr/render.sh \
        .claude/skills/review-pr/check_prompts.py
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "review-pr: split SKILL.md, add the artifact contract and the drift guard

Rule bodies move to rules.md so the derived subset, not all of them, reaches
each review. SKILL.md keeps the prose every review needs and gains what it
lacked: each step now writes an artifact, so a skipped step and a performed
one are no longer textually identical.

The prompts quote SKILL.md verbatim and check_prompts.py asserts that
relation. Both land together -- a guard shipped without the thing it guards is
an assertion about nothing. Verified by introducing a one-character drift and
confirming the guard goes red.

Rule CONTENT is untouched; rewriting it from real ATOM review history is P1.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: triage.py 的七道 gate

`triage.py` 2872 行里，gate 子命令与 family 派生器是两块独立的东西。P0 只需要 gate 能跑；ATOM 的 family 轴是 P1。

**Files:**
- Create: `.claude/skills/review-pr/triage.py`
- Create: `.claude/skills/review-pr/MAPPING.md`
- Test: `.claude/skills/review-pr/tests/test_gate_behavior.py`

**Interfaces:**
- Consumes: Task 1 的 `_lib.run_gates()` 以这些子命令名调用它
- Produces: `triage.py answers|diagnostic|corefiles|ledger|card|refutations|independent <args...>`，通过 exit 0，不通过 exit 非 0 并说明是哪条、为什么
- Produces: `triage.py expand <rules.txt> <rules.md>` → stdout 为派生规则正文
- Produces: `triage.py mapping` → stdout 为 family→rule 的 markdown 表
- Produces: 模块级常量 `TIER`（ATOM backbone 文件元组）、`derive(files, title, raw_diff)`

- [ ] **Step 1: 取回 triage.py**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git -C /mnt/raid0/zufa/aiter show c1c482e00:.claude/skills/review-pr/triage.py \
  > .claude/skills/review-pr/triage.py
wc -l .claude/skills/review-pr/triage.py   # 预期 2872
```

- [ ] **Step 2: 替换 Tier 表为 ATOM 的 backbone**

第 11 行的 `TIER` 是 aiter 的 backbone 清单。换成 ATOM 的（取自 spec §7.2 与 ATOM 原 SKILL.md 的 Step 4 表）：

```python
TIER = ("atom/model_engine/model_runner.py", "atom/config.py",
        "atom/models/deepseek_v2.py", "atom/model_ops/moe.py",
        "atom/model_engine/scheduler.py", "atom/model_ops/attention_mla.py",
        "atom/model_ops/attentions/aiter_mla.py", "atom/model_ops/linear.py",
        "atom/model_ops/attention_mha.py", "atom/model_ops/layernorm.py",
        "atom/models/deepseek_v4.py")
```

`corefiles` gate 里判定 Tier 1 的逻辑（aiter 用 `aiter/__init__.py` 与 `aiter/jit/core.py`）改为 ATOM 的 Tier 1 三件：`atom/model_engine/model_runner.py`、`atom/config.py`、`atom/models/deepseek_v2.py`。

```bash
grep -n "aiter/__init__\|aiter/jit/core" .claude/skills/review-pr/triage.py
```

逐处按上述映射替换。注意 ATOM 的 Tier 1 判据不是"import 链"，而是"每次 forward 都执行"或"多个模型族的基类"——改注释说明这一点。

- [ ] **Step 3: 把 `derive()` 换成最小但真实的版本**

P0 不做完整 family 轴（P1 的工作），但**空的 `rules.txt` 会让 ledger gate fail closed**，所以 `derive()` 必须返回非空。整体替换为：

```python
def derive(files, title="", raw_diff=""):
    """P0 minimal deriver. The full ATOM family axis is P1 -- see the spec, S7.

    Deliberately small: a family that fires on everything teaches the reviewer
    that the list is noise, which is worse than a list that is short. Each
    family here is one a path or an added line can decide on its own.
    """
    paths = list(files)
    add = "\n".join(l for f in files.values() for l in f["add"])
    t = []
    def hit(name, rules): t.append((name, rules))

    INFRA = (".github/", "docs/", "3rdparty/", ".gitignore", "requirements",
             "README", "CONTRIBUTING", ".gitmodules", "setup.py", "pyproject.toml")
    if paths and all(p.startswith(INFRA) or p.endswith((".md", ".txt", ".cfg", ".toml"))
                     or "/docs/" in p for p in paths):
        return [("infra-only", "NONE")]

    if any(p in TIER[:3] for p in paths):
        hit("tier1-every-request", "STEP4 A2 D4")
    elif any(p in TIER for p in paths):
        hit("tier2-shared-op", "A1 A2 B6")
    if any(p.startswith("atom/plugin/") for p in paths):
        hit("plugin-bridge", "E2 E3")
    if re.search(r"from aiter import|import aiter\b|aiter\.", add):
        hit("aiter-dep", "E1 B5")
    if re.search(r"^\s*(el)?if .*(dtype|arch|layout|quant|backend)", add, re.M):
        hit("dispatch-change", "B1 B2 B6")
    if re.search(r"\b(fp8|e4m3|e5m2|fnuz|mxfp[48]|fp4)\b|448\.0|240\.0", add, re.I):
        hit("fp8-quant", "C1 C2")
    if re.search(r"cuda_?graph|capture", add, re.I):
        hit("cudagraph-capture", "D1 D4")
    if re.search(r"num_experts|\btp_size\b|\bep_size\b|world_size", add):
        hit("ep-tp-shapes", "A3 B2")
    if re.search(r'os\.environ\.get\(\s*["\']ATOM_', add):
        hit("env-var-registry", "HK5")
    if any(p.endswith(".sh") or re.search(r"(^|/)(runperf|test_local_)", p) for p in paths):
        hit("temp-script", "HK1")
    if re.search(r"#\s*(TODO|FIXME)|raise NotImplementedError", add, re.M):
        hit("incomplete-code", "HK4")
    if len({p.split("/")[0] for p in paths}) >= 3:
        hit("scattered-diff", "HK2")

    if not t:
        hit("generic-code", "A1 B1 D4")   # never empty: the ledger gate fails closed
    return t
```

**HK 编号用 ATOM 的，不是 aiter 的——两边同号不同义。** ATOM 原 SKILL.md 的编号是：
`HK1` 临时脚本、`HK2` 无关文件、**`HK3` bug fix 无测试**、`HK4` TODO/stub、
`HK5` 未注册的新环境变量、`HK6` 测试参考实现 dtype 提升、`HK7` 新依赖。
照 aiter 的编号映射会产出一批**指向错误规则正文**的裁决，而 gate 发现不了——
它只检查"每条派生规则有裁决"，不检查正文是否对题。

aiter 有而这里**故意不要**的两个 family：

- `syspath-mutation` —— ATOM 的 sys.path 检查在 Step 6 的 AI 诊断表里，没有 HK 编号。
  派生一个无正文的 rule id 会让 `triage.py expand` 直接失败。P1 给它编号时再加回来。
- Triton 系列（T1–T8）—— ATOM 不直接写 Triton kernel。

同步更新两处模块级常量：

- `ALL_RULES` 改为上面出现过的 rule id 全集：
  `A1 A2 A3 B1 B2 B5 B6 C1 C2 D1 D4 E1 E2 E3 HK1 HK2 HK4 HK5 STEP4`
- `UNREACHABLE_BY_DESIGN` 改为空 dict，加注释说明 P0 阶段 ATOM 没有 scanner-backed
  规则、P1 会填

**`STEP4` 必须在 `rules.md` 里有正文**（Task 2 Step 1 已要求补）。它是伪规则，含义为
"对此 diff 做完整的 Step 4 backbone 评估"；`triage.py expand` 对"能派生却无正文"的
rule id 会失败——**这会在 Task 4 跑 fetch.sh 时就炸，而不是等到 Task 9**。

- [ ] **Step 4: 跑 Task 1 的契约测试，确认转绿**

```bash
cd /mnt/raid0/zufa/atom-bot-work && python3 -m pytest .claude/skills/review-pr/tests/test_gates_contract.py -v
```

预期：三个用例全 PASS。特别确认 `test_empty_work_dir_fails_closed` 通过——空 WORK 目录下七道 gate 必须全红。

- [ ] **Step 5: 写 gate 行为测试（该拒的拒、该过的过）**

```python
# .claude/skills/review-pr/tests/test_gate_behavior.py
"""Each gate must reject what it exists to reject, and accept a real answer.

A gate tested only on the green path is a gate nobody has seen say no.
"""
import pathlib
import subprocess
import sys

SKILL = pathlib.Path(__file__).resolve().parent.parent
TRIAGE = str(SKILL / "triage.py")


def run(*args):
    return subprocess.run([sys.executable, TRIAGE, *args],
                          capture_output=True, text=True)


def test_answers_rejects_formulaic(tmp_path):
    p = tmp_path / "answers.txt"
    p.write_text("Q1: n/a\nQ2: n/a\nQ3: n/a\nQ4: n/a\nQ5: n/a\nBLIND: no\n",
                 encoding="utf-8")
    r = run("answers", str(p))
    assert r.returncode != 0, "formulaic answers passed the answers gate"


def test_answers_accepts_substantive(tmp_path):
    p = tmp_path / "answers.txt"
    p.write_text(
        "Q1: moves the rmsnorm scale computation ahead of the quant step so the "
        "intermediate never reaches HBM\n"
        "Q2: gfx942 and gfx950, bf16 and fp8, decode only\n"
        "Q3: no public ATOM API changes; the new kwarg is internal to linear.py\n"
        "Q4: one fewer HBM round trip per layer, measured on the fused path\n"
        "Q5: the description explains the mechanism, not only the action\n"
        "BLIND: looked for a cudagraph interaction in the changed region and found "
        "none; the branch sits outside the captured block\n",
        encoding="utf-8")
    r = run("answers", str(p))
    assert r.returncode == 0, f"substantive answers rejected: {r.stdout}{r.stderr}"


def test_ledger_rejects_unadjudicated_rule(tmp_path):
    rules = tmp_path / "rules.txt"; rules.write_text("A1\nB1\n", encoding="utf-8")
    verdicts = tmp_path / "verdicts.txt"
    verdicts.write_text("A1 CLEAR — no sibling variant of the changed function "
                        "exists in atom/model_ops/\n", encoding="utf-8")
    diff = tmp_path / "pr.diff"; diff.write_text("", encoding="utf-8")
    r = run("ledger", str(rules), str(verdicts), str(diff))
    assert r.returncode != 0, "a rule with no verdict passed the ledger gate"
    assert "B1" in (r.stdout + r.stderr)


def test_diagnostic_rejects_bare_clean(tmp_path):
    p = tmp_path / "ai_diagnostic.txt"
    p.write_text("1: clean\n2: clean\n3: clean\n4: clean\n5: clean\n6: clean\n",
                 encoding="utf-8")
    r = run("diagnostic", str(p))
    assert r.returncode != 0, "'clean' with no evidence passed the diagnostic gate"
```

- [ ] **Step 6: 跑行为测试**

```bash
cd /mnt/raid0/zufa/atom-bot-work && python3 -m pytest .claude/skills/review-pr/tests/test_gate_behavior.py -v
```

预期：四个用例全 PASS。若 `test_answers_accepts_substantive` 失败，说明 gate 过严——读它打印的拒绝理由再调，**不要为了让测试过而放宽 gate**。

- [ ] **Step 7: 生成 MAPPING.md 并确认与代码一致**

```bash
cd /mnt/raid0/zufa/atom-bot-work
python3 .claude/skills/review-pr/triage.py mapping > .claude/skills/review-pr/MAPPING.md
python3 .claude/skills/review-pr/triage.py mapping | diff - .claude/skills/review-pr/MAPPING.md \
  && echo "MAPPING in sync"
head -20 .claude/skills/review-pr/MAPPING.md
```

- [ ] **Step 8: Commit**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git add .claude/skills/review-pr/triage.py .claude/skills/review-pr/MAPPING.md \
        .claude/skills/review-pr/tests/test_gate_behavior.py
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "review-pr: port the seven gates, with a minimal ATOM deriver

The gates are what make a skipped step visible, so they come first. The
backbone tier table is ATOM's: Tier 1 here means 'runs on every forward pass'
or 'base class for several model families', not 'import atom breaks' -- ATOM
has no such import chain.

derive() is deliberately minimal for P0. The full family axis is P1, built
from real ATOM review history. It still never returns empty: an empty
rules.txt means Step 1b did not run, and the ledger gate fails closed on it.

Each gate is tested both ways. A gate tested only on the green path is a gate
nobody has seen say no.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: fetch.sh — Step 1 的取证器

aiter 的 `fetch.sh` 947 行，其中 FlyDSL bounds、AOT pairing、struct ABI 三个取证器对 ATOM 无意义。

**Files:**
- Create: `.claude/skills/review-pr/fetch.sh`

**Interfaces:**
- Consumes: `triage.py`（Task 3）的 `derive` / `expand` / `citest` 等子命令；`gh` CLI
- Produces: 一个 `$WORK` 目录，stdout 打印 `WORK=<path>`；目录内含 `pr.diff`、`pr_meta.json`、`rules.txt`、`rules_expanded.txt`、`applies.txt`、`merge_target.txt`、`guards.txt`、`siblings.txt`、`symbols.txt`、`twins.txt`、`test_quality.txt`、`ci_coverage.txt`、`perf_claims.txt`、`comment_only.txt`、`evidence.txt`，以及 git worktree `merge-target/` 与 `head/`

- [ ] **Step 1: 取回并裁剪**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git -C /mnt/raid0/zufa/aiter show c1c482e00:.claude/skills/review-pr/fetch.sh \
  > .claude/skills/review-pr/fetch.sh
chmod +x .claude/skills/review-pr/fetch.sh
grep -n "flydsl\|aot_pairing\|structabi\|struct_abi" .claude/skills/review-pr/fetch.sh
```

删掉 grep 命中的三个取证器的调用块与产物说明。**保留**其余全部，特别是 `guards`、`siblings`、`symbols`、`twins`、`test_quality`、`citest`、`perf_claims`、`comment_only`、`evidence`。

- [ ] **Step 2: 改默认仓库与 validate skill 路径**

```bash
cd /mnt/raid0/zufa/atom-bot-work
sed -i 's|ROCm/aiter|ROCm/ATOM|g; s|validate-kernel-pr|validate-atom-pr|g;
        s|REVIEW_AUTO_VALIDATE:-1|REVIEW_AUTO_VALIDATE:-0|g' \
  .claude/skills/review-pr/fetch.sh
grep -n "validate-atom-pr\|REVIEW_AUTO_VALIDATE" .claude/skills/review-pr/fetch.sh
```

P0 没有 `validate-atom-pr/`，默认关掉自动验证。读 validate 缺失那段代码确认它写出的是 `NOT RUN — <reason>` 而**不是** exit 非 0——缺 validator 是环境缺口，不是 PR 缺陷，更不该让整轮评审失败。

- [ ] **Step 3: 对一个真实 ATOM PR 跑一遍**

```bash
cd /mnt/raid0/zufa/atom-bot-work
env -u GH_TOKEN -u GITHUB_TOKEN gh pr list --repo ROCm/ATOM --state merged --limit 10 \
  --json number,title,changedFiles \
  --jq '.[]|select(.changedFiles<10)|"\(.number) \(.title)"'
```

从输出里选一个代码改动（非纯文档）的 PR，记为 `$PR`：

```bash
cd /mnt/raid0/zufa/atom-bot-work
PR=<上一步选的编号>
bash .claude/skills/review-pr/fetch.sh "$PR" ROCm/ATOM 2>&1 | tee /tmp/fetch.log
W=$(grep -oE 'WORK=[^ ]+' /tmp/fetch.log | tail -1 | cut -d= -f2)
echo "WORK=$W"; ls -la "$W"
```

- [ ] **Step 4: 验证产物齐全且非空**

```bash
for f in pr.diff pr_meta.json rules.txt rules_expanded.txt applies.txt merge_target.txt; do
  if [ -s "$W/$f" ]; then echo "  OK      $f ($(wc -l < "$W/$f") lines)"
  else echo "  MISSING $f"; fi
done
echo "--- derived families ---"; cat "$W/rules.txt"
```

预期：六个文件全部存在且非空；`rules.txt` 至少一行。

**若 `rules.txt` 为空**，回 Task 3 Step 3 检查兜底分支，不要往下走——空 rules.txt 会让 ledger gate 恒红，后面全是假失败。

- [ ] **Step 5: 清理 worktree 并 commit**

```bash
cd /mnt/raid0/zufa/atom-bot-work
for wt in "$W/merge-target" "$W/head"; do
  [ -d "$wt" ] && git worktree remove --force "$wt" 2>/dev/null
done
rm -rf "$W"; git worktree prune
git add .claude/skills/review-pr/fetch.sh
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "review-pr: port fetch.sh, dropping the three aiter-only forensics

FlyDSL bounds, AOT pairing and struct ABI answer questions ATOM does not
have. Everything else stays: the guard, sibling, symbol, twin, test-quality,
CI-coverage and perf-claim collectors all describe defects ATOM ships too.

Auto-validation defaults off because validate-atom-pr does not exist yet. A
missing validator must read as an environment gap on the card, never as a
finding against the PR and never as a failed review.

Verified against a real merged ATOM PR: every required artifact lands
non-empty.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 策略配置落地（owner 映射 + 白名单 + 发布侧脚本）

按 spec §3.2，把 aiter 放在 `.env` 和 repo vars 里的**策略**改成提交进仓库的文件。

**Files:**
- Create: `.claude/skills/review-pr/owners.txt`
- Create: `.claude/skills/review-pr/authorized.txt`
- Create: `.claude/skills/review-pr/_notify.py`
- Create: `.claude/skills/review-pr/lost_review.py`
- Create: `.claude/skills/review-pr/queue_gate.py`
- Create: `.claude/skills/review-pr/watchdog.py`
- Create: `.claude/skills/review-pr/_publish.py`
- Test: `.claude/skills/review-pr/tests/test_owners.py`

**Interfaces:**
- Produces: `owners.txt`，每行 `<failure-class> <github-login> <what is wrong>`，`#` 起注释
- Produces: `_notify.OWNERS`（dict: class → (login, description)）与 `_notify.owner_for(cls) -> str`，未知 class 归 `flow`
- Produces: `authorized.txt`，每行一个 GitHub login，`#` 起注释

- [ ] **Step 1: 写 owners.txt**

```
# atom-bot failure-class -> owner. One line per class: <class> <login> <what is wrong>.
# '#' starts a comment. This file is the single source of truth; _notify.py reads it.
#
# It lives in version control, not in repo variables, on purpose: changing who gets
# paged is then an auditable PR rather than an edit in repo settings that nobody sees.
# If on-call rotation ever has to be changed by someone without commit rights, add an
# override channel back -- but add it deliberately, not by default.
flow     zufayu   the review pipeline itself (bot logic / prompts)
env      zufayu   the runner environment (gh, token, git, disk)
glm      zufayu   the model backend serving the review agent
atom     zufayu   the ATOM serving framework
machine  zufayu   the machine (GPU, disk, network, OS)
```

五类都指向同一人是**正确的现状**，不是偷懒——分类本身仍有价值，它让失败原因出现在通知里。等有第二个人接手某一类再改那一行。

- [ ] **Step 2: 写 authorized.txt**

```
# atom-bot trigger allowlist -- only the GitHub logins here may use @rocm-atom-bot review.
# One login per line (case-insensitive). '#' starts a comment. Edit this file (a maintainer
# commit) to change who is allowed -- version-controlled and auditable. Reviews run on a
# shared box; keep it curated.
#
# P0 is a staging run on a fork, so this starts at one login. The ATOM reviewer roster
# goes in at P3, when the bot moves to ROCm/ATOM.
zufayu
```

- [ ] **Step 3: 取回发布/通知侧五个脚本并改名**

```bash
cd /mnt/raid0/zufa/atom-bot-work
A=/mnt/raid0/zufa/aiter; PIN=c1c482e00
for f in _notify.py lost_review.py queue_gate.py watchdog.py _publish.py; do
  git -C "$A" show "$PIN:.claude/skills/review-pr/$f" > ".claude/skills/review-pr/$f"
done
sed -i 's/AITER_BOT_TOKEN_FILE/ATOM_BOT_TOKEN_FILE/g; s/AITER_BOT_TOKEN/ATOM_BOT_TOKEN/g;
        s/AITER_MAX_WAIT_HOURS/ATOM_MAX_WAIT_HOURS/g; s/aiter-bot/rocm-atom-bot/g;
        s|ROCm/aiter|ROCm/ATOM|g' \
  .claude/skills/review-pr/_notify.py .claude/skills/review-pr/lost_review.py \
  .claude/skills/review-pr/queue_gate.py .claude/skills/review-pr/watchdog.py \
  .claude/skills/review-pr/_publish.py
grep -rn "AITER_\|aiter-bot" .claude/skills/review-pr/_notify.py \
  .claude/skills/review-pr/lost_review.py .claude/skills/review-pr/queue_gate.py \
  .claude/skills/review-pr/watchdog.py .claude/skills/review-pr/_publish.py
```

最后一条 grep 应当无输出。再手工处理 `_publish.py`：删掉 `AITER_BOT_REPO` / `AITER_BOT_NAME` 的读取，改为仓库从 `GITHUB_REPOSITORY` 推导、bot 名写死常量（spec §3.2 要求直接删除这两个变量）。

- [ ] **Step 4: 把 `_notify.py` 的 owner 映射改成读 owners.txt**

把内嵌的 `class -> (repo-var, default, description)` 字典整体替换为：

```python
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def _owners():
    """Read owners.txt: '<class> <login> <description>' per line, '#' comments.

    The map is in version control, not in repo variables: changing who gets
    paged should be an auditable commit, not an invisible settings edit.
    """
    out = {}
    for line in (HERE / "owners.txt").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(None, 2)
        if len(parts) >= 2:
            out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return out


OWNERS = _owners()


def owner_for(cls):
    """An unknown class routes to the bot owner rather than to nobody."""
    row = OWNERS.get(cls) or OWNERS.get("flow")
    return row[0] if row else "zufayu"
```

文件内其余对旧字典的引用（描述文本等）一并改为从 `OWNERS[cls][1]` 取。

- [ ] **Step 5: 写 owners 测试**

```python
# .claude/skills/review-pr/tests/test_owners.py
"""Every failure class run_one.sh can emit must have an owner.

A class with no owner routes a real failure to nobody, silently: the review
died, someone needs to look, and no one is told.
"""
import pathlib
import re
import sys

SKILL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL))
import _notify


def test_every_emitted_class_has_an_owner():
    run_one = (SKILL / "run_one.sh").read_text(encoding="utf-8")
    emitted = set(re.findall(r"\bfail\s+(\w+)\s+\d+", run_one))
    assert emitted, "found no `fail <class> <code>` calls in run_one.sh"
    missing = sorted(c for c in emitted if c not in _notify.OWNERS)
    assert not missing, f"run_one.sh emits classes with no owner: {missing}"


def test_unknown_class_routes_to_flow():
    assert _notify.owner_for("no-such-class") == _notify.owner_for("flow")
```

- [ ] **Step 6: 跑测试**

```bash
cd /mnt/raid0/zufa/atom-bot-work && python3 -m pytest .claude/skills/review-pr/tests/test_owners.py -v
```

预期：`test_unknown_class_routes_to_flow` PASS；`test_every_emitted_class_has_an_owner` FAIL（`run_one.sh` 还没落地，Task 6 补）。这是预期的。

- [ ] **Step 7: Commit**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git add .claude/skills/review-pr/owners.txt .claude/skills/review-pr/authorized.txt \
        .claude/skills/review-pr/_notify.py .claude/skills/review-pr/lost_review.py \
        .claude/skills/review-pr/queue_gate.py .claude/skills/review-pr/watchdog.py \
        .claude/skills/review-pr/_publish.py .claude/skills/review-pr/tests/test_owners.py
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "review-pr: move policy into version control

The failure-class owner map and the trigger allowlist are policy: identical on
every machine, and changing them should leave a trace. They become files next
to the skill rather than repo variables, so reassigning who gets paged is a
reviewable commit instead of a settings edit nobody sees.

ATOM has one owner today and all five classes point at them. The classes still
earn their keep -- they put the cause in the notification.

BOT_NAME and BOT_REPO are dropped outright; GITHUB_REPOSITORY already says.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: run_one.sh + selftest.sh — 编排与 guard

**Files:**
- Create: `.claude/skills/review-pr/run_one.sh`
- Create: `.claude/skills/review-pr/selftest.sh`

**Interfaces:**
- Consumes: `fetch.sh`、`render.sh`、`check_prompts.py`、`_apply_refutation.py`、`_gates.py`、`_collect.py`；`_notify.py` 的 class 名
- Produces: `run_one.sh <PR> [owner/repo]`；失败时写 `$GITHUB_WORKSPACE/.atom-review-status`，内容 `<class>\t<message>`
- Produces: 可发出的 failure class 集合，必须是 `owners.txt` 的子集（Task 5 的测试断言之）

- [ ] **Step 1: 取回并改名**

```bash
cd /mnt/raid0/zufa/atom-bot-work
A=/mnt/raid0/zufa/aiter; PIN=c1c482e00
for f in run_one.sh selftest.sh; do
  git -C "$A" show "$PIN:.claude/skills/review-pr/$f" > ".claude/skills/review-pr/$f"
  chmod +x ".claude/skills/review-pr/$f"
done
sed -i 's/AITER_RUN_BUDGET/ATOM_RUN_BUDGET/g; s/AITER_AGENT_TIMEOUT/ATOM_AGENT_TIMEOUT/g;
        s/AITER_REVIEW_RETRIES/ATOM_REVIEW_RETRIES/g;
        s/AITER_REVIEW_REFUTER_AGENT/ATOM_REVIEW_REFUTER_AGENT/g;
        s/AITER_REVIEW_AGENT/ATOM_REVIEW_AGENT/g;
        s/AITER_GLM_PROBE_TIMEOUT/ATOM_GLM_PROBE_TIMEOUT/g;
        s/\.aiter-review-status/.atom-review-status/g; s/aiter-bot/rocm-atom-bot/g;
        s|ROCm/aiter|ROCm/ATOM|g' \
  .claude/skills/review-pr/run_one.sh .claude/skills/review-pr/selftest.sh
grep -n "AITER_\|aiter" .claude/skills/review-pr/run_one.sh
```

残留逐条判断：提到 aiter 作为上游依赖是对的，提到"被评审的仓库是 aiter"要改。

- [ ] **Step 2: 改时间预算默认值**

```bash
cd /mnt/raid0/zufa/atom-bot-work
sed -i 's/ATOM_RUN_BUDGET:=6000/ATOM_RUN_BUDGET:=7200/' .claude/skills/review-pr/run_one.sh
grep -n "ATOM_RUN_BUDGET:=\|ATOM_AGENT_TIMEOUT:=" .claude/skills/review-pr/run_one.sh
```

预期：`ATOM_RUN_BUDGET:=7200`、`ATOM_AGENT_TIMEOUT:=2400`。同时把那段注释里的 "100 min, under the job's timeout-minutes of 120" 改成 "120 min, under the job's timeout-minutes of 150"，并补一句为什么比 aiter 宽：validate 要在同一轮里起两次服务（P2）。

- [ ] **Step 3: 确认 GLM 深探仍是条件执行**

开发期用 `claude-glm` 走网关，不设 `ANTHROPIC_BASE_URL`，探针应自动跳过；迁到 box308 时它会自动生效。**确认这个条件判断还在，不要删**：

```bash
grep -n 'ANTHROPIC_BASE_URL' .claude/skills/review-pr/run_one.sh
```

预期：能看到 `if [ -n "${ANTHROPIC_BASE_URL:-}" ]; then`。

- [ ] **Step 4: 跑 selftest.sh**

```bash
cd /mnt/raid0/zufa/atom-bot-work && bash .claude/skills/review-pr/selftest.sh; echo "exit=$?"
```

预期：`exit=0`，末行汇总全绿。

`selftest.sh` 守的是 `run_one.sh` 的三条行为：超时路由到 `flow` 而非 page 模型 owner；预算不够时拒绝起新尝试；refuter 超时仍只在卡**承认**的前提下发布。**这些 guard 失败说明改名改坏了逻辑，不是测试太严。**

- [ ] **Step 5: 跑 Task 5 的 owners 测试，确认转绿**

```bash
cd /mnt/raid0/zufa/atom-bot-work && python3 -m pytest .claude/skills/review-pr/tests/test_owners.py -v
```

预期：两个用例全 PASS。若报某 class 没 owner，把它加进 `owners.txt`——**不要从 `run_one.sh` 里删掉那个 class**，它存在是因为那种失败真的会发生。

- [ ] **Step 6: 跑 check_prompts.py 的完整检查**

```bash
cd /mnt/raid0/zufa/atom-bot-work && python3 .claude/skills/review-pr/check_prompts.py && echo "OK"
```

预期：exit 0。此时 `run_one.sh` 与 `_notify.py` 都在了，Task 2 Step 5 里暂时跳过的 fail-class 子检查现在应当真正生效。

- [ ] **Step 7: Commit**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git add .claude/skills/review-pr/run_one.sh .claude/skills/review-pr/selftest.sh
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "review-pr: port the orchestrator and its guards

run_one.sh carries the failure taxonomy: every exit, including a signal kill,
leaves a classified status, so a dead review names someone instead of dying
quietly. selftest.sh holds it to that.

Budget is 120 minutes inside a 150 minute job, up from aiter's 100/120,
because validate will serve a model twice in the same run (P2). The GLM deep
probe stays conditional on ANTHROPIC_BASE_URL: unset while we are on the
gateway, live the moment this moves to box308's local GLM.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: 两个 workflow

`atom-review-bot.yml` 是触发链路；`atom-review-bot-selftest.yml` 是 bot 的专用 CI lane（spec §7.5），只跑纯逻辑检查、不碰自建机。

**Files:**
- Create: `.github/workflows/atom-review-bot.yml`
- Create: `.github/workflows/atom-review-bot-selftest.yml`

**Interfaces:**
- Consumes: `.claude/skills/review-pr/` 下全部脚本
- Produces: 触发词 `@rocm-atom-bot review`；runner 标签 `atom-bot-24`

- [ ] **Step 1: 取回触发 workflow 并改名**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git -C /mnt/raid0/zufa/aiter show c1c482e00:.github/workflows/aiter-review-bot.yml \
  > .github/workflows/atom-review-bot.yml
sed -i "s/aiter-review-bot/atom-review-bot/g; s/aiter-review-/atom-review-/g;
        s/aiter-bot/rocm-atom-bot/g; s/AITER_BOT_TOKEN/ATOM_BOT_TOKEN/g;
        s/AITER_REVIEW_AUTHORIZED/ATOM_REVIEW_AUTHORIZED/g;
        s/AITER_MAX_WAIT_HOURS/ATOM_MAX_WAIT_HOURS/g;
        s/\.aiter-review-status/.atom-review-status/g" \
  .github/workflows/atom-review-bot.yml
```

- [ ] **Step 2: 三处结构性改动**

```bash
cd /mnt/raid0/zufa/atom-bot-work
sed -i 's/self-hosted, box308/self-hosted, atom-bot-24/; s/timeout-minutes: 120/timeout-minutes: 150/' \
  .github/workflows/atom-review-bot.yml
grep -n "_OWNER\|OWNER_OVERRIDE" .github/workflows/atom-review-bot.yml
```

1. runner 标签 → `atom-bot-24`（上面 sed 已做）
2. review job 的 `timeout-minutes` → `150`（上面 sed 已做）
3. **删掉 owner 的 repo-var 注入**：`_notify.py` 现在读 `owners.txt`，所以 notify step 的 `env:` 里五个 `AITER_*_OWNER` 以及 gate / watchdog job 里的 `OWNER_OVERRIDE` 全删。把 grep 命中的行逐条删除，连同因此只剩空 `env:` 的情况一并收拾。

- [ ] **Step 3: P0 降级 —— claim 👀 与 publish 的 token 依赖**

claim 步骤加条件，让它在无 token 时跳过而不是失败。**不要直接在 `if:` 里读 `secrets`**
——该上下文在 step 级 `if` 表达式里不可靠。把 secret 绑到 step 的 `env`，再测那个变量：

```yaml
        env:
          TOK: ${{ secrets.ATOM_BOT_TOKEN }}
        if: env.TOK != ''
```

这样写的好处是 P0.5 把 secret 配上的当天，这一步自动变活，不用再改 workflow。

publish 步骤去掉 `--post`（`_publish.py` 默认 dry-run），并在其上方加注释说明这是刻意降级：

```yaml
      # P0 has no bot token: _publish.py is dry-run by default, so the card is
      # rendered and logged but not posted. P0.5 adds --post together with the
      # ATOM_BOT_TOKEN secret. The review itself never uses the token -- the
      # agent runs --dangerously-skip-permissions over untrusted PR content.
      - name: Render the review card (dry-run until P0.5)
        if: success()
        run: python3 "${{ steps.rl.outputs.dir }}/_publish.py" "$PR"
```

- [ ] **Step 4: 写专用 selftest CI**

```yaml
# .github/workflows/atom-review-bot-selftest.yml
# The bot's own CI lane. Pure logic checks only -- no GPU, no self-hosted box --
# so it can gate every change to the skill without competing with ATOM's main CI
# for runners. The main CI gates merges on absolute accuracy across PRs; this one
# only asserts that the bot's own machinery still holds together.
name: atom-review-bot selftest

on:
  pull_request:
    paths:
      - '.claude/skills/review-pr/**'
      - '.claude/skills/validate-atom-pr/**'
      - '.github/workflows/atom-review-bot*.yml'
  push:
    branches: [main]
    paths:
      - '.claude/skills/review-pr/**'
      - '.claude/skills/validate-atom-pr/**'

permissions:
  contents: read

jobs:
  selftest:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
        with:
          fetch-depth: 0          # check_prompts.py resolves the repo root via git
          persist-credentials: false

      - name: Install test deps
        run: python3 -m pip install --quiet pytest jsonschema

      - name: Prompts have not drifted from SKILL.md
        run: python3 .claude/skills/review-pr/check_prompts.py

      - name: MAPPING.md matches the deriver
        run: |
          python3 .claude/skills/review-pr/triage.py mapping > /tmp/mapping.md
          diff /tmp/mapping.md .claude/skills/review-pr/MAPPING.md

      - name: SKILL.md stays inside its 500-line budget
        run: |
          n=$(wc -l < .claude/skills/review-pr/SKILL.md)
          echo "SKILL.md is $n lines"
          [ "$n" -le 500 ] || { echo "::error::SKILL.md is $n lines, budget is 500"; exit 1; }

      - name: run_one.sh guards hold
        run: bash .claude/skills/review-pr/selftest.sh

      - name: Unit tests
        run: python3 -m pytest .claude/skills/review-pr/tests/ -v
```

- [ ] **Step 5: actionlint 两个 workflow**

**必须在 repo root 跑**，否则会误报（在子目录跑会把 workflow 当孤立文件，报 21-inputs 之类的假错）：

```bash
cd /mnt/raid0/zufa/atom-bot-work && actionlint .github/workflows/atom-review-bot.yml \
  .github/workflows/atom-review-bot-selftest.yml; echo "exit=$?"
```

预期：`exit=0`，无输出。

- [ ] **Step 6: Commit**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git add .github/workflows/atom-review-bot.yml .github/workflows/atom-review-bot-selftest.yml
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "ci: add the atom-bot trigger workflow and its own selftest lane

The trigger workflow keeps aiter's four-job shape, and the reasons in its
comments are kept with it: the gate job exists so an unauthorized comment never
starts a job on the shared box, and the watchdog and report jobs run on hosted
infra precisely so they are alive when the box is not.

Three ATOM changes: the runner label is atom-bot-24 so ATOM jobs are never
claimed by aiter's runner, the job cap rises to 150 minutes to sit above the
120-minute run budget, and the owner repo-variables are gone now that the map
is a file in the repo.

The selftest lane is ubuntu-only and gates every skill change without competing
for ATOM's CI runners.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: runner 安装与 preflight

**Files:**
- Create: `.claude/skills/review-pr/preflight.sh`
- Create: `.claude/skills/review-pr/RUNNER-SETUP.md`

**Interfaces:**
- Produces: `preflight.sh`，exit 码 = 红项数量，全绿 exit 0

- [ ] **Step 1: 取回 preflight.sh 并改名**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git -C /mnt/raid0/zufa/aiter show c1c482e00:.claude/skills/review-pr/preflight.sh \
  > .claude/skills/review-pr/preflight.sh
chmod +x .claude/skills/review-pr/preflight.sh
sed -i "s/aiter-review-bot/atom-review-bot/g; s/aiter-bot/rocm-atom-bot/g;
        s/AITER_REVIEW_AGENT/ATOM_REVIEW_AGENT/g;
        s/AITER_BOT_TOKEN_FILE/ATOM_BOT_TOKEN_FILE/g; s/AITER_BOT_TOKEN/ATOM_BOT_TOKEN/g;
        s/actions\.runner\.\*aiter\*/actions.runner.*ATOM*/g" \
  .claude/skills/review-pr/preflight.sh
grep -n -i aiter .claude/skills/review-pr/preflight.sh
```

- [ ] **Step 2: 安装 runner 到 /mnt/md0**

`/mnt/raid0` 已 100% 满（剩 126G），而每次评审要 checkout 仓库并建两个 worktree。装到 `/mnt/md0`（剩 9.6T）：

```bash
sudo mkdir -p /mnt/md0/atom-runner && sudo chown "$(whoami)" /mnt/md0/atom-runner
cd /mnt/md0/atom-runner
curl -o actions-runner.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz
tar xzf actions-runner.tar.gz && rm actions-runner.tar.gz
```

拿 fork 的注册 token（自助，不需要 ROCm admin）：

```bash
env -u GH_TOKEN -u GITHUB_TOKEN gh api -X POST \
  repos/zufayu/ATOM/actions/runners/registration-token --jq .token
```

用它注册，标签 `atom-bot-24`：

```bash
cd /mnt/md0/atom-runner
./config.sh --url https://github.com/zufayu/ATOM --token <上一步的 token> \
  --labels atom-bot-24 --name srdc24-atom-review --work _work --unattended --replace
```

- [ ] **Step 3: 写 runner .env（只放机器事实）**

```bash
mkdir -p /mnt/md0/atom-runner/tmp /mnt/md0/atom-runner/gh-config
cat > /mnt/md0/atom-runner/.env <<'EOF'
# Machine facts only. Policy (budgets, owner map, allowlist) lives in the repo --
# see the skill's RUNNER-SETUP.md and section 3.2 of the design spec.
TMPDIR=/mnt/md0/atom-runner/tmp
ATOM_REVIEW_AGENT=/usr/local/bin/claude-glm
GH_CONFIG_DIR=/mnt/md0/atom-runner/gh-config
PATH=/usr/local/bin:/usr/bin:/bin
EOF
```

**不要**在这里设 `ATOM_RUN_BUDGET` / `ATOM_AGENT_TIMEOUT` / 任何 `*_OWNER`——它们是策略，已在仓库里。

- [ ] **Step 4: 配 gh 认证（不走环境变量）**

```bash
GH_CONFIG_DIR=/mnt/md0/atom-runner/gh-config gh auth login --with-token < <token-file>
GH_CONFIG_DIR=/mnt/md0/atom-runner/gh-config gh api repos/ROCm/ATOM --jq .full_name
```

预期：输出 `ROCm/ATOM`。用 `GH_CONFIG_DIR` 而不是导出 `GH_TOKEN`，这样 headless 评审 agent 不会从环境里继承到 GitHub token。

- [ ] **Step 5: 装成 systemd 服务（带 Restart）**

```bash
cd /mnt/md0/atom-runner
sudo ./svc.sh install "$(whoami)"
sudo ./svc.sh start
U=$(systemctl list-unit-files 'actions.runner.*ATOM*' --no-legend | awk '{print $1; exit}')
[ -n "$U" ] || { echo "no runner unit -- svc.sh install did not take"; exit 1; }
sudo mkdir -p "/etc/systemd/system/$U.d"
printf '[Service]\nRestart=always\nRestartSec=10\n' | sudo tee "/etc/systemd/system/$U.d/restart.conf"
sudo systemctl daemon-reload && sudo systemctl restart "$U"
systemctl is-enabled "$U"; systemctl show -p Restart "$U"
```

预期：`enabled`，`Restart=always`。

**不要手工 `./run.sh` 或写 keeper 循环**：手启的 runner 下次重启就没了，而没有 runner 时 job 停在 `queued`——不失败、不通知、从 PR 上看像"评审还在进行中"，永远。aiter 那台机器就这么静默离线过数天。

- [ ] **Step 6: 写 RUNNER-SETUP.md**

照 aiter 的章节结构（注册 / .env / gh 认证 / systemd / verify）改写，但**开头必须有这两条 ATOM 特有的前提**：

```markdown
# Runner setup — atom-review-bot

## Why this box, and what must not be copied from aiter's setup

aiter's box serves its own GLM locally, which takes all eight GPUs. Copying
that arrangement here would leave nothing for validation. On this box the
review agent reaches its model over the network (the AMD gateway during
development, box308's already-running GLM-5.3 in the end), and the eight local
GPUs stay free. If you move this to a box that serves its own model, say so in
this file and redo the GPU arithmetic -- do not discover it at run time.

## Install on a data volume, not the root fs

/mnt/raid0 on this box is 100% full. The runner lives on /mnt/md0. Each review
checks out the repo and builds two worktrees; a box whose volume fills up fails
mid-review.

## What belongs in .env, and what does not

.env holds machine facts only: TMPDIR, ATOM_REVIEW_AGENT, GH_CONFIG_DIR, PATH,
and the GPU subset once validation lands. Budgets, the owner map and the
allowlist are policy and live in the repo -- a value parked in .env outlives the
reason it was added, and nothing reviews it.
```

- [ ] **Step 7: 跑 preflight**

```bash
cd /mnt/raid0/zufa/atom-bot-work && bash .claude/skills/review-pr/preflight.sh; echo "exit=$?"
```

预期：全绿 `exit=0`，但 publish identity 一节会打一条 ⚠（没有 `ATOM_BOT_TOKEN`）——那是警告不是红项，符合 P0 预期。有红项按它打印的 hint 逐条修。

- [ ] **Step 8: Commit**

```bash
cd /mnt/raid0/zufa/atom-bot-work
git add .claude/skills/review-pr/preflight.sh .claude/skills/review-pr/RUNNER-SETUP.md
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "review-pr: add preflight and runner setup for srdc-24

RUNNER-SETUP.md leads with the thing that is easy to get wrong by copying
aiter: that box serves its own GLM on all eight GPUs, and reproducing the
arrangement here would leave none for validation. Here the model is reached
over the network and the local GPUs stay free.

The runner installs on /mnt/md0 because /mnt/raid0 is full, and as a systemd
service with Restart=always because a hand-started runner dies on reboot and
reports nothing: the job just sits queued, looking like a review in progress.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: 端到端验收

**Files:**
- Modify: `docs/superpowers/specs/2026-10-23-atom-review-bot-design.md`（回填实测数据）

- [ ] **Step 1: 本地手工跑一遍 run_one.sh**

先绕开 GitHub Actions 直接跑，失败时能直接看到输出：

```bash
cd /mnt/raid0/zufa/atom-bot-work
export ATOM_REVIEW_AGENT=/usr/local/bin/claude-glm
PR=<Task 4 Step 3 里选的那个 PR 号>
time bash .claude/skills/review-pr/run_one.sh "$PR" ROCm/ATOM 2>&1 | tee /tmp/run_one.log
echo "exit=${PIPESTATUS[0]}"
```

**必须在普通终端里跑，不要在 auto-mode 的 Claude 会话里跑**——auto-mode 的分类器会把带 `--dangerously-skip-permissions` 的 headless 子 agent 判为 "Create Unsafe Agents" 并拦下。runner 是普通 shell，线上没这个问题。

- [ ] **Step 2: 确认七道 gate 全绿**

```bash
grep -A 12 "SEVEN GATES" /tmp/run_one.log
```

预期：七行全 `✅`，末行 `======== 7 green / 0 red ========`。

**若有红**：读它打印的拒绝理由。红 gate 意味着评审被**故意扣下**——通常重跑一次 worker 就过（模型这次没按格式写）。同一道连续三次红，说明 gate 或 SKILL.md 的指令有问题，回对应 Task 修，**不要放宽 gate 让它过**。

- [ ] **Step 3: 读卡**

```bash
cat .claude/skills/review-pr/reports/PR-$PR/card.md
```

人工判断三件事：
1. "What it does" 是不是一句话、写给没读过 diff 的人看；
2. 每条 finding 是否都有**问题 / 影响 / 行动**三段，且以动词短语收尾；
3. `Validation (deterministic)` 行是否为 `N/A — no runtime surface changed` 或 `NOT RUN — <reason>`——**绝不能是编造的结论**，P0 没有 validator。

- [ ] **Step 4: 记录 token 用量与耗时**

spec §9 把"开发期网关配额"列为风险，这是第一次有数据的机会：

```bash
grep -iE "token|usage|elapsed|real" /tmp/run_one.log | tail -20
```

记下 worker + refuter 的输入 token 总量与整轮墙钟时间。

- [ ] **Step 5: 若 bot 账号名最终不是 `rocm-atom-bot`**

一条命令改完全树（`<新名>` 换成实际注册的账号名）：

```bash
cd /mnt/raid0/zufa/atom-bot-work
grep -rl 'rocm-atom-bot' .claude .github | xargs sed -i 's/rocm-atom-bot/<新名>/g'
grep -rn 'rocm-atom-bot' .claude .github   # 预期无输出
bash .claude/skills/review-pr/selftest.sh && python3 .claude/skills/review-pr/check_prompts.py
```

- [ ] **Step 6: 跑全部检查做最后确认**

```bash
cd /mnt/raid0/zufa/atom-bot-work
python3 -m pytest .claude/skills/review-pr/tests/ -v
bash .claude/skills/review-pr/selftest.sh
python3 .claude/skills/review-pr/check_prompts.py
actionlint .github/workflows/atom-review-bot.yml .github/workflows/atom-review-bot-selftest.yml
bash .claude/skills/review-pr/preflight.sh
```

预期：pytest 全 PASS；selftest、check_prompts、actionlint 均 exit 0；preflight 仅剩 token 那条 ⚠。

- [ ] **Step 7: 把实测数据写回 spec 并 commit**

编辑 `docs/superpowers/specs/2026-10-23-atom-review-bot-design.md`：
- §5.1b 填入 Step 4 的 token 用量与整轮耗时
- §9 风险表把"开发期网关配额"一行从"未测"改为实测值

```bash
cd /mnt/raid0/zufa/atom-bot-work
git add docs/superpowers/specs/2026-10-23-atom-review-bot-design.md
git -c user.email=8138883+zufayu@users.noreply.github.com -c user.name=zufayu commit -m "docs: record the first real review's token cost and wall clock

P0 acceptance ran end to end on a real ATOM PR: card produced, seven gates
green. The gateway-quota risk now carries a measured number instead of a
question mark.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## P0 完成的标志

- 对一个真实 ATOM PR，`run_one.sh` 产出 `card.md`，七道 gate 全绿
- `pytest`、`selftest.sh`、`check_prompts.py`、`actionlint`、`preflight.sh` 全部通过
- runner 作为 systemd 服务常驻，`Restart=always`
- spec §5.1b 填入了实测的 token 用量与耗时

## 明确不在 P0 范围内

- **发布链路**（claim 👀、`_publish.py --post`、真实 PR 评论）→ P0.5，需 bot token
- **ATOM family 轴与规则重写**（近半年全渠道评论语料）→ P1
- **validate-atom-pr**（GSM8K base/head delta、GPU 子集锁、噪声地板实测）→ P2
- **迁到 box308、注册到 ROCm/ATOM**（需 repo admin，需先解决 box308 的 GPU 占用）→ P3
