---
name: review-pr
description: AI code review for ATOM PRs. ATOM consumes aiter kernels and integrates with vLLM/SGLang plugins. Reviews check perf claims, aiter cross-repo deps, model coverage, dispatch correctness, and AI-generated code patterns. Invoke with a PR number.
argument-hint: <PR number>
---

# ATOM PR Review

ATOM is a ROCm/AMD GPU kernel optimization layer (MI300X/MI355X) that:
- Consumes aiter ops (attention, MoE, GEMM, norm, quant)
- Integrates with vLLM and SGLang as a plugin/backend
- Provides custom MLA, sparse attention, TBO, and quantization fusion

A change here can break inference for all models using the affected kernel path.

---

## Step 1 — Fetch

```bash
PR=$1
REPO="ROCm/ATOM"

gh pr view $PR --repo $REPO --json title,body,number,labels,files,author,reviews,comments > /tmp/pr_meta.json
gh pr diff $PR --repo $REPO > /tmp/pr.diff

# Linked issue
ISSUE=$(cat /tmp/pr_meta.json | python3 -c "
import json,re,sys
body = json.load(sys.stdin).get('body','') or ''
m = re.search(r'(?:fix|close|resolve)[s]?[: ]*#(\d+)', body, re.I)
print(m.group(1) if m else '')
")
[ -n "$ISSUE" ] && gh issue view $ISSUE --repo $REPO --json title,body > /tmp/pr_issue.json
```

```bash
# Prior reviewer comments (top-level)
cat /tmp/pr_meta.json | python3 -c "
import json,sys
d = json.load(sys.stdin)
for r in d.get('reviews',[]):
    b = (r.get('body','') or '').strip()
    if b: print(f'[REVIEW {r[\"author\"][\"login\"]}] {b[:200]}')
for c in d.get('comments',[]):
    b = (c.get('body','') or '').strip()
    if b: print(f'[COMMENT {c[\"author\"][\"login\"]}] {b[:200]}')
"

# Inline review comments (line-level — often more specific than top-level)
gh api "repos/$REPO/pulls/$PR/comments" | python3 -c "
import json,sys
comments = json.load(sys.stdin)
for c in comments:
    author = c.get('user',{}).get('login','')
    body = (c.get('body','') or '').strip()
    path = c.get('path','')
    line = c.get('line') or c.get('original_line','')
    if body and 'copilot' not in author.lower() and 'bot' not in author.lower():
        print(f'[INLINE {author}] {path}:{line}')
        print(f'  {body[:250]}')
" 2>/dev/null
```

Read the diff and PR body before proceeding.

**Cross-file verification — before reporting any kernel/dispatch finding.** The diff shows changed lines, not the whole story. Grep the entire symbol family (`.cu` + `.cuh` + `.h`, or the whole module), not just files in the diff — sync/fence/atomics or the "other half" of a scatter often live in a header, and dispatch/else-branch completeness must be read in the full function, not the hunk. A "no synchronization" or "missing branch" finding based only on the diff is how false positives happen.

**Classify every CI failure before blaming the PR.** A red check is not automatically the PR's fault:
- Read the failed step/service. An external `docs/readthedocs.com` build failing on a `.claude/`-only or code-only change is unrelated infra (those files do not feed the RTD sphinx docs) — do not treat it as a content failure (ATOM#1549).
- Compare against main: if main fails the same job in the same window, it is baseline/flaky, not a regression introduced here.
- Expired CI logs (`HTTP 410 Gone`) on old runs mean the failure is months-stale and meaningless against today's main — ask for a rebase + fresh run instead of quoting it.

---

## Step 2 — Semantic Understanding (answer before rules)

**Write the five answers into `$WORK/answers.txt`, one line each, `Q1:` … `Q5:`.**
Step 8 will not let you write a card without them — an inline `_Answer:_` blank
left nothing behind, so a review that skipped this step read as identical to
one that did it.

**Q1 — What specifically changed computationally?**
Not "improves MLA" — which kernel path, what data flow, what formula?
_Answer:_

**Q2 — Hardware + model scope: which arch(es), which model family/families?**
gfx942 / gfx950? DeepSeek V3 / V3-0324 / Kimi K2.5 / GPT-OSS / GLM? TP config?
_Answer:_

**Q3 — Does this introduce or modify aiter op usage?**
New `from aiter import X`? New kwargs on existing aiter calls? Removed aiter calls?
_Answer:_

**Q4 — Performance claim: what is the mechanism?**
Not "faster" — WHY? (fewer kernel launches, fused allreduce+norm, better tiling?)
_Answer:_

**Q5 — Does the description explain WHY or only WHAT?**
"Enable ar+norm+quant fusion" = surface. "Eliminates 2 intermediate HBM round-trips between
allreduce, rmsnorm, and quant by calling the aiter fused kernel" = understanding.
_Answer:_

---

## Step 3 — PR Type Classification

- [ ] **Performance optimization** → P1 (benchmark numbers), P5 (setup cost excluded?), P2 (production shapes), P3 (reproducible)
- [ ] **New aiter op usage / aiter API change** → E1 (aiter dep), B5 (param added/removed/renamed propagation), B6 (new kwarg unhandled by all ATOM dispatch branches?), E3 (dead param)
- [ ] **API signature change** (param added / removed / renamed in an ATOM function or base class, default or return type changed) → B5 (propagation to all receivers), E1 (aiter dep if aiter-side), E2 (plugin bridge sync)
- [ ] **New constexpr / routing flag / new attribute key added** → B6 (do ALL dispatch branches handle the new value, or assert on it?), C3 (new arch string literal?)
- [ ] **Model integration** (`atom/models/*.py`) → A2 (shared backbone coverage), Step 4 risk
- [ ] **Dispatch logic change** → B1 (silent bypass), B2 (phase proxy), B6 (new value unhandled?), A3 (scope too broad)
- [ ] **cudagraph / capture path** → B3 (cudagraph compatibility)
- [ ] **Plugin / vLLM / SGLang interface** → E2 (bridge sync)
- [ ] **Config / tuning** → A2 (other GPU archs), A3 (scope too broad)
- [ ] **Bug fix** → A1 (sibling variant), HK3 (regression test); root cause explained?
- [ ] **FP8 / quantization** → C1 (fnuz by dtype), C2 (dtype hardcoded)
- [ ] **Weight transform / new weight attr** → F1 (double HBM pin)
- [ ] **Async / multi-stream / weight prep** → G1 (stream sync missing), G1b (blocking queue.get without timeout in serving code)
- [ ] **New if/elif dispatch with variable assignment** → D1b (UnboundLocalError on uninitialized path)
- [ ] **New `@compile_ops` / `torch.library.custom_op`, or change to an op's return dtype/arity** → D7 (fake/abstract impl exists?), D6 (fake dtype/shape matches real op?)
- [ ] **Kernel launcher / buffer-offset or index arithmetic (long-context or large-batch path)** → D9 (int32 overflow at production scale)
- [ ] **New aiter / C-extension kernel call** → D8 (contiguous check)
- [ ] **Removes or reverses a zero-init / assert / `.contiguous()` / documented invariant** → D4 (invariant reversal cited?)

---

## Step 4 — Backbone File Risk Assessment

**Write one line per backbone file this diff touches into `$WORK/core_files.txt`
— Step 8 gates on it.** Format: `<path> TIER1|TIER2|TIER3 COVERED|GAP|N/A -- <reason>`.
This step was prose only, so a review that skipped it was textually identical to
one that performed it. Touching no Tier 1/2 file: write one `NONE -- <reason>`
line naming the Tier 3 files touched instead.

**What makes an ATOM file "backbone"?** Apply these questions to any file in the diff.

```
Q1 — Tier 1 test: Is this file executed on EVERY forward-pass request,
     regardless of which model is being served?
     (model_runner, engine_core, scheduler, config = YES)
     → YES → Tier 1 (system-critical: every inference is affected)

Q2 — Tier 1 alt: Is this file a base class inherited by >2 production model
     implementations, so a bug here affects all of them even if the PR says
     "model-specific fix"?
     (deepseek_v2.py is base for DSv2/V3/V3-0324/Kimi = YES)
     → YES → Tier 1

Q3 — Tier 2 test: Does this file implement an op (attention, linear, norm, MoE)
     that is shared across >1 model family, where a correctness bug
     silently produces wrong results for all users of that op?
     (attention_mla.py, linear.py, moe.py = YES)
     → YES → Tier 2

Q4 — Tier 3 note: Is this a plugin bridge file (vllm/*.py, sglang/*.py)?
     → Tier 3 by blast radius, but HIGH VISIBILITY — only plugin users
       are affected, but those users see the API break immediately.

Otherwise → Tier 3 (model-specific or kernel-specific).
```

Key difference from aiter: ATOM has no `import atom` — Tier 1 is defined by
"executes on every request" or "base class for multiple model families", not by import chain.

The table below is the current snapshot; Q1–Q4 classify new files not yet listed.

Backbone files ranked by git commit frequency (2025–2026) and blast radius:

| Tier | File | Git commits | Blast radius | Common failure mode |
|------|------|-------------|-------------|---------------------|
| **1** | `atom/model_engine/model_runner.py` | 158 | **ALL** inference — every forward pass | OOM, cudagraph break, wrong batch assembly |
| **1** | `atom/config.py` | 91 | All models — config drives dispatch | Wrong model config silently changes kernel path |
| **1** | `atom/models/deepseek_v2.py` | 68 | DSv2/V3/V3-0324/Kimi base class | Wrong MLA, OOM, accuracy drop for all DSv* |
| **2** | `atom/model_ops/moe.py` | 69 | All MoE models | Wrong expert routing, double weight pinning |
| **2** | `atom/model_engine/scheduler.py` | 68 | Request batching for all models | Stall, wrong decode/prefill split |
| **2** | `atom/model_ops/attention_mla.py` | 54 | All MLA attention paths | Wrong KV, accuracy drop, crash |
| **2** | `atom/model_ops/attentions/aiter_mla.py` | 52 | aiter MLA dispatch | Wrong kernel, wrong dtype |
| **2** | `atom/model_ops/linear.py` | 49 | All linear layers (every model) | Wrong GEMM dispatch, wrong quant |
| **2** | `atom/model_ops/attention_mha.py` | 46 | All MHA models | Wrong attention output |
| **2** | `atom/model_ops/layernorm.py` | 32 | Norm + quant fusion path | Wrong scale, wrong dtype |
| **2** | `atom/models/deepseek_v4.py` | 34 | DSv4 / Kimi-K2.5 specific | Wrong sparse MLA, SWA layout break |


**Tier-1 special rule**: When `model_runner.py` or `config.py` is touched, ask: does the change
interact with cudagraph capture? Any new Python control flow, dynamic tensor allocation, or
attribute lookup inside the captured region will silently break cudagraph.

**`deepseek_v2.py` special rule**: Base class for DSv2, DSv3, DSv3-0324, and Kimi.
A bug here affects all four model families even if the PR says "Kimi-only fix".
Check: is the changed method overridden in subclasses? If not, all variants are affected.

**Mandatory backbone checks — must be answered before writing the verdict:**

For **Tier 1** files (model_runner, config, deepseek_v2, scheduler):
- [ ] List every function/method changed. Grep for callers: `grep -r 'def <name>' atom/models/ atom/model_engine/`. If any caller not mentioned in the PR exists, flag it.
- [ ] For `deepseek_v2.py`: run `grep -rn 'def <changed_method>' atom/models/` — does any subclass override it differently? If not overridden → all DSv2/V3/V3-0324/Kimi are affected.
- [ ] Is there an integration test (full forward pass, not unit test alone) exercising this path after the change? If not → `📝 HK3`
- [ ] State explicitly: if this change is wrong, what breaks? (crash / silent wrong value / OOM / cudagraph break) and how would it be detected?

For **Tier 2** files (moe, attention_mla, linear, aiter_mla, scheduler):
- [ ] Which model families use this op/file? List them. Is at least one from each family tested?
- [ ] Are production shapes covered? (TP=4, TP=8, decode single-token, prefill ISL≥4096)
- [ ] Does the change affect the FP8 path, the BF16 path, or both? If both, are both tested?

**AI code red flag — verbatim duplication across backbone files:** If the same algorithmic block appears in 2+ backbone files with only variable names changed (same formula, same comments, same structure), ask: was each file's invariant verified independently, or was the fix copy-pasted? See D5.

---

## Step 5 — Rule Checklist

Six failure categories — work all six in order. Severity: 🔴 block / ⚠️ should fix / 📝 note.

**🔴 gate — before firing any 🔴, write down the concrete input that triggers it.** Name the specific shape / scale / dtype / arch / value that makes the finding fire (e.g. "at `token_id` > 16M with H=32, D=128 the int32 product exceeds 2^31", or "when the new attribute is absent the `getattr` default silently drops shared-expert slots"). If you cannot state a concrete triggering case, the 🔴 is unproven — **downgrade to ⚠️ ("worth checking") or drop it.** A 🔴 that reads as a definite blocker but names no demonstrable triggering input is exactly how a false positive lands on a maintainer's PR. This gate applies to every rule below — including those whose own text omits an explicit FP self-check (e.g. D9): the same index expression is safe in a capped/small-batch path and unsafe only at a scale you must actually exhibit.

| Category | Core question | Key triggers |
|---|---|---|
| **A. Coverage gaps** | Same bug elsewhere? Shared path other models? | `_opt`, `_prefill_opt`, `_v2`; shared backbone; broad condition |
| **B. Silent bypass** | Every input reaches the right branch? | gated-off param; phase proxy (`max_q`); string alias |
| **C. Hardcoded arch/dtype** | Does the constant break on another GPU or config? | `bf16` fixed; `fp8_e8m0` fixed; `gfx942` assumed |
| **D. Uninitialized state** | Is the buffer clean before kernel launch? | `::empty()`+`atomic`; cudagraph dynamic allocation |
| **E. Cross-repo sync** | Does the consumer know? | new aiter symbol; new param nobody passes; plugin bridge |
| **F. Resource duplication** | Does the change double HBM silently? | new `_preshuffled`/`_quantized` weight alongside original |

---

**Adjudicate every rule in `$WORK/rules.txt`, one line each, into `$WORK/verdicts.txt`.**
Format, one per rule id:

```
<RULE-ID> FIRE|CLEAR|N/A — <the specific reason, naming file:line, symbol, or the condition>
```

`CLEAR` means you looked and it does not apply *to this diff*; it is a claim, and the
reason is what makes it checkable. "ok", "n/a", "fine" are not reasons — Step 8's gate
rejects them. Step 8 will not let you write a verdict card until every derived rule has
a line with a reason.

**The rule bodies are not in this file.** `$WORK/rules_expanded.txt`, written by
`triage.py expand`, holds the full text of exactly the rules this diff derives — read
that. All of them live in `rules.md`; the median PR needs a fraction of their lines,
so keeping them here loaded irrelevant rule text into every review, on top of a skill
that is already long enough that "read all of it" is a hope rather than a guarantee.

---

## Step 6 — AI Code Diagnostic

| Question | Warning sign |
|----------|-------------|
| Description explains mechanism (WHY) not just action (WHAT)? | WHAT only → elevated risk |
| Perf numbers clean integers? (2.0x, 1.5x) | Cherry-picked or fabricated |
| Perf claims only trace screenshots with no numeric values? | Screenshots ≠ numbers; reviewer will ask |
| Test only shows token_num ≤ 256 or only M=1? | AI default toy shapes |
| Dispatch gate: are gated-off params asserted or silent? | Silent → B1 |
| New default path without env-var? | → D2 |
| Unrelated files committed? | AI commit artifact → HK2 |
| Root cause of bug fix explained? | "fix the issue" without mechanism → AI guess |
| "Test Plan" / "Test Result" section left as template placeholder? | Untested PR, AI-generated description |
| `sys.path.insert` or `sys.path.append` at module level? | Global state leak — use relative imports |
| PR description footer says "🤖 Generated with Claude Code" or similar AI attribution? | Author may not understand the change — elevate review priority on dispatch logic and test coverage |

**Structural verification — the table above is a cheap pre-filter; a clean description does not make the code correct.** When the diff touches code, AI fails in specific *structural* ways. Run these checks and report each as a finding tagged `[verified]`/`[inferred]`, ending in an action verb (per the finding format below):

1. **Hallucinated-symbol sweep.** List every symbol NEW to this diff — function name, kwarg, enum/constant, attribute, import — and grep each against its real definition (aiter API, ATOM module, torch). AI invents plausible names and signatures that do not exist or do not match. Any symbol you cannot locate is a defect until proven real.
   → `🔴 [symbol] on [line] not found in [module] / signature mismatch — confirm it exists or it is a hallucinated API`
2. **Twin divergence (copy-paste half-adapted).** Identify mirrored code — fwd/bwd, v2/v3, prefill/decode, per-arch branches. Compare field by field; any asymmetry (one side int64 the other int32, one masked the other not, one stride order flipped) is an unfinished copy. This is the signature AI bug (cf. D9's fwd-int32/bwd-int64 case).
   → `🔴/⚠️ [detail] differs between [twin A] and [twin B] — copy-paste left [side] unadapted`
3. **Claim/comment ↔ code, and number provenance.** Does the code actually enforce the invariant the description or a comment asserts? Then take the single most impressive number in the PR and trace it to its source (script output / log line). A number or PR/issue citation you cannot trace is `[unverified]` — never repeat it as fact. (The sibling aiter skill's P5 once shipped a fabricated "1.14x" for aiter#4166 that the PR never claimed — verify, do not trust.)
   → `⚠️ [claim/comment] asserted but code does [X]; or [number] not traceable to any output — mark [unverified] and ask for the source`
4. **Safety theater.** For each new `if`/`try`/`assert` guard: is it reachable, will it ever fire, does `except: pass` swallow a real error? AI adds defensive code that is unreachable or silently hides failures.
   → `⚠️ guard on [line] is [unreachable / swallows errors] — remove it or make it actually enforce the invariant`
5. **Test calibrated to pass, not to falsify.** Is the reference impl structurally a twin of the kernel (the same bug lives in both, so they always agree)? Is `atol`/`rtol` loosened with no justification? Does it assert against the kernel's own output? AI writes tests that pass rather than tests that could catch a regression.
   → `⚠️ test [name] cannot fail because [mirrored ref / loose tol / self-comparison] — replace with an independent oracle`
6. **Magic constant without derivation.** A new tile size / threshold / epsilon / literal — is there a stated derivation or tuning basis, or does it merely look plausible?
   → `📝 constant [value] on [line] has no stated derivation — ask for the tuning/source basis`

**Write one line per check into `$WORK/ai_diagnostic.txt`** — `1:` … `6:`, each naming
what you looked at and what you found (`clean` alone is not an answer). Step 8 will not
let you write a card without them.

If 3+ table signs OR any structural check above fires: note "elevated AI code risk — recommend thorough review of dispatch logic and full test coverage." Regardless of the table count, when the diff changes code the structural checks are mandatory — a clean, well-written description is itself something AI produces easily.

---

## Step 7 — Free-Form Review

- Does the algorithmic approach make sense for MI300X/MI355X memory hierarchy?
- For MLA changes: does paged KV vs. dense KV distinction hold throughout?
- For MoE changes: expert routing correctness under different EP configurations?
- For quantization fusion: scale computation order, dtype promotion path?
- Memory size calculations: look for missing multiplicative factors. Real example (ATOM#1423): `swa_pages = self.model_runner.num_swa_blocks` missing `* self.block_size` — size was off by block_size factor.
- Tensor copy vs. in-place: `.copy_()` into an existing buffer when an `out=` parameter is available wastes an allocation. Real example (ATOM#1493): reviewer suggested inplace write instead of `copy_`.
- FP8 dtype check: `_is_fnuz` should check `tensor.dtype == fp8_fnuz`, not infer from arch name. Real example (aiter#4073): valarLip: "check _is_fnuz by tensor's DType instead of arch".
- EP vs TP shape math: EP (expert parallelism) shards experts — each device holds `num_experts / EP` experts. TP (tensor parallelism) shards weight matrices — each device holds a column/row slice. A dispatch path that divides `num_experts` by `world_size` when `world_size` is the TP degree silently routes to wrong experts. Always check: is the denominator the right parallelism degree for the dimension being split?
- LDS budget for aiter Triton kernels: gfx942 / gfx950 = 64KB LDS per CU. An ATOM PR that changes tile size, blocking factor, or adds a new Triton kernel should verify the aiter kernel's LDS allocation doesn't exceed 64KB — the compiler falls back to VGPR spilling causing perf regression or compile failure.

---

## Step 7.5 — Blind-Spot Check

Before writing the verdict, answer this one question in full:

**"Is there any correctness risk, resource hazard, or behavioral edge case in this diff that none of Steps 1–7 above caught?"**

If the answer is yes, add it to the findings. If the answer is no, proceed.

---

## Step 7.6 — Refutation

**Try to kill each finding before reporting it.** One line per attempt in
`$WORK/refutations.txt`: `RED|WARN|NOTE SURVIVED|KILLED -- what you opened and what it
said`. A killed finding never reaches the card.

## Step 7.7 — Independent refutation

**Hand `$WORK/card.md`, the diff and the PR metadata to a reader who has not seen your
reasoning** — a second agent, or a person — with the findings false until defended. One
line per finding in `$WORK/independent.txt`: `SURVIVED|KILLED -- what they opened`; a
killed finding comes off the card. With no such reader, write `NONE AVAILABLE -- <reason>`
and put `not independently refuted` on the review line. `rules.md` § Refutation has why.

---

## Step 8 — Verdict

**Before writing the card, run the seven gates below. A red gate means that step did not
happen; go back to it rather than reporting.**

```bash
.claude/skills/review-pr/triage.py answers "$WORK/answers.txt" || {
  echo "Step 2 / 7.5 not answered — the card must not be written yet" >&2
  exit 1
}
.claude/skills/review-pr/triage.py diagnostic "$WORK/ai_diagnostic.txt" || {
  echo "Step 6 structural checks not recorded — the card must not be written yet" >&2
  exit 1
}
.claude/skills/review-pr/triage.py corefiles "$WORK/core_files.txt" "$WORK/pr.diff" \
  "$PROJECT_ROOT" || {
  echo "Step 4 not performed for every backbone file — the card must not be written yet" >&2
  exit 1
}
.claude/skills/review-pr/triage.py ledger "$WORK/rules.txt" "$WORK/verdicts.txt" \
  "$WORK/pr.diff" || {
  echo "rule pass incomplete — the card must not be written yet" >&2
  exit 1
}

# Then write the card to $WORK/card.md and run the last two gates against it.
.claude/skills/review-pr/triage.py card "$WORK/card.md" "$WORK/verdicts.txt" \
  "$WORK/ai_diagnostic.txt" "$WORK/answers.txt" "$WORK/pr.diff" "$WORK/late_findings.txt" || {
  echo "a finding in the card is not nailed down — fix or drop it before reporting" >&2
  exit 1
}
.claude/skills/review-pr/triage.py refutations "$WORK/refutations.txt" "$WORK/pr.diff" \
  "$WORK/card.md" || { echo "a finding reached the card unattacked" >&2; exit 1; }
.claude/skills/review-pr/triage.py independent "$WORK/independent.txt" "$WORK/card.md" \
  || { echo "a finding was never put to a reader who had not seen the reasoning" >&2
       exit 1; }
```

`answers` names a Step 2 or Step 7.5 question left unanswered or answered with a formula;
`diagnostic` does the same for Step 6's six structural checks. `corefiles` names every
Tier 1/2 backbone file with no assessment line, and every reason that is formulaic or
anchored in nothing this PR changes; it derives the backbone set from the diff, so a
`NONE` declaration does not get past it. `ledger` names each derived rule with no
verdict, each verdict with no reason, and each `FIRE` citing no file this PR changes.
`card` checks that each finding on the card is nailed down and that nothing `FIRE` is
quietly missing from it. `refutations` requires that you tried to kill each finding;
`independent` requires that someone who had not seen your reasoning tried too, or that
the card admits nobody did.

Write the card to `$WORK/card.md` before running the gate — a missing file is a hard
failure, because not writing it was the cheapest way past this check.

**Output rules (strictly enforced):**
- Run Steps 1–7 internally. Do NOT narrate steps, do NOT show checklists, do NOT show which rules fired.
- Output ONLY the card below. Nothing before it, nothing after it.
- If there are no findings, the findings section is omitted entirely.
- "What it does" must be one sentence, written for a reviewer who hasn't read the diff.

```
## ATOM PR #NNN — [title]

**[One sentence: what this PR does, in plain terms. E.g. "Wraps the per_group emit_bf16 AllReduce path in a torch.library op so Dynamo can trace it."]**

[✅ LGTM | ⚠️ NEEDS WORK | 🔴 BLOCK]

🔴 [specific finding — what, where, why it matters]
⚠️ [specific finding]
📝 [note]
```

Each finding must have **three parts**:
1. **Problem** — what exactly is wrong, with file/line if relevant
2. **Impact** — what goes wrong at runtime if this is not fixed (wrong output / crash / OOM / perf regression)
3. **Action** — end with a verb phrase: "**Author must** [do X]" or "**Reviewer should ask** [Y]" — no verb = incomplete finding, do not include

**Tag every finding [verified] or [inferred], and never ship a root cause you only inferred.**
- `[verified]` — traced to the actual code/evidence chain.
- `[inferred]` — plausible but unconfirmed; say so and downgrade to "worth checking," do not assert it as the cause.
A finding that stops at "likely / probably the root cause" without an evidence chain is not shippable — either trace it to [verified] or label it [inferred] and frame it as a question.

Do NOT use rule codes (P1, D2, A1…) in output — they are internal labels only.

Examples of good findings:
- `🔴 deepseek_v2.py:1245 changes torch.zeros → torch.empty, but the old comment explicitly says "trailing pad must be zero for asm reader" and the new comment claims "never read" — if padding IS read, every quantized token in the batch gets corrupted output silently. **Author must** cite the asm spec or a test proving padding is not read.`
- `⚠️ PR claims 1.3–1.5x speedup but the benchmark starts after shuffle_weight() completes — users pay that cost on every cold start. **Author must** re-run with shuffle_weight included in the timing window and confirm the speedup holds.`
- `⚠️ Chunked indexer logic is copy-pasted verbatim into deepseek_v2.py and deepseek_v4.py with only variable names changed — if v4's variable semantics differ, the formula silently produces wrong KV offsets for all v4 callers. **Author must** confirm correctness was verified independently under v4's variable layout.`
- `📝 No corresponding ATOM consumer PR mentioned. **Reviewer should ask** who will pass emit_bf16=True to activate this path.`

Examples of bad findings (too vague, missing action verb):
- `⚠️ Missing perf numbers` — no impact, no action
- `🔴 D2 violation` — rule code means nothing to a reviewer
- `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion

---

## Adding New Rules

When a human reviewer (valarLip or others) catches something real that this skill missed:
1. Add to Step 5 with the real PR number as evidence
2. Commit: `review-pr: add R[NAME] from ATOM#[NNN] — [one line]`

**Nothing new goes in this file.** SKILL.md is budgeted at 500 lines and holds
only the prose every review needs, with no code fence over 30 lines.
Conditional content goes in `rules.md` and reaches the reviewer through
`triage.py expand`; executable content goes in a script and gets called.
Raising the budget is a decision to make this file less likely to be read in
full — make it in a commit that says why.
