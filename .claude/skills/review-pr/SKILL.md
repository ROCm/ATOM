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

### A — Coverage Gaps
_"Fixed one path; the same bug lives in a sibling."_

**A1 — Sibling function/kernel not fixed** ⚠️ (🔴 if Tier-1/2 backbone)
Fix changes address calc, bounds check, or data layout: scan same file for variants named `_opt`, `_prefill_opt`, `_decode`, `_v2`.
Real example (aiter#3841): strided q_nope fix on decode kernel; `_prefill_opt` in same file had same bug.
→ `⚠️ A1: same bug may exist in [variant] — check function family in this file`

**A2 — Change covers one model/GPU, shared path affects others** ⚠️
PR labeled "[MI308]" or "DSv4-only" but touches a backbone file shared with Kimi/DSv3/gfx950:
- Special: `deepseek_v2.py` is the base class for DSv2, DSv3, DSv3-0324, and Kimi — a "Kimi-only fix" here affects all four.
- If benchmark only shows one GPU arch, ask about the other.
Real example (ATOM#1498): "[MI308]" backbone change still affects gfx950 (MI355X).
→ `⚠️ A2: [change] labeled [scope] but shared backbone [file] also affects [other models/archs]`

**A3 — Activation condition broader than validated scope** ⚠️
New dispatch enables kernel for model family X, tested only on subcase Y.
Real example (vLLM#16435): FusedMoE activated for wrong families → follow-up restrict PR needed.
→ `⚠️ A3: activation condition enables [X] but only [Y] was tested`

---

### B — Silent Bypass
_"The code looks complete but certain inputs silently take the wrong path."_

**B1 — Dispatch gate with unchecked parameter** 🔴
New `if/elif/else`: for each parameter gated off — is it **asserted** or **forwarded**?
Trigger params: `block_table`, `alibi_slopes`, `window_size`, `logits_soft_cap`, `dropout_p`.
Real example (aiter#3576): `block_table is not None` False-branch computed dense attention silently.
→ `🔴 B1: [param] silently ignored in [branch] — assert or forward`

**B2 — Decode/prefill phase classified by size proxy** ⚠️
Code checks `max_q`, `q.shape[0] == 1`, or `seq_len <= 1` to decide decode vs. prefill:
short prefill / chunked prefill with small `max_q` → misclassified as decode → wrong kernel path.
Use explicit phase flag from batch metadata (`is_prefill`, `attn_metadata.is_prefill`).
Real example (ATOM#1372): vLLM plugin used `max_q` heuristic; zejunchen: "short prefill wrongly treated as DECODE."
→ `⚠️ B2: phase classified by [proxy] — chunked/short prefill may be misclassified`

**B3 — cudagraph capture compatibility** 🔴
New Python `if`/`else`/loop, dynamic shape computation (`tensor.item()`, `.tolist()`), or tensor allocation (`torch.empty`, `torch.zeros`) inside a cudagraph-captured region breaks capture — graph replays the original trace, any branching or dynamic allocation at replay time raises or silently corrupts.
Static changes are safe: constant arithmetic, dtype cast on an existing tensor, forwarding kwargs without branching.
Trigger: new control flow or allocation inside `model_runner.py` forward path, a model's `forward()` method, or any function decorated with cudagraph context. NOT a trigger: new fixed-shape kernel call with no branching.
Real example (ATOM#1321): root cause of cudagraph fix unclear — reviewer asked for explanation.
→ `🔴 B3: [new if/allocation] inside cudagraph-captured region — explain why capture remains valid`

**B4 — Triton `tl.constexpr` safety check disabled without invariant proof** ⚠️
A `tl.constexpr` bool that gates a validity check (e.g., `CHECK_NEG_ONE_SENTINEL`, `CHECK_BOUNDS`) can be set `False` by a caller to skip the check. If the invariant the check enforces is not independently guaranteed on that path, illegal memory access or silent wrong values result.
Trigger: new `tl.constexpr` bool in a Triton kernel that disables a bounds/sentinel/validity check; PR comment says "X path can disable this" without documenting what guarantees the invariant holds on that path.
Real example (ATOM#1498): `CHECK_NEG_ONE_SENTINEL=False` disables -1 slot filter in paged prefill kernel; illegal access if any -1 slot appears without the check.
→ `⚠️ B4: [constexpr] disables [check] — document which caller invariant guarantees no [invalid value] on that path`

**B5 — API propagation incompleteness** 🔴/⚠️
When an API surface changes in dimension X, all downstream receivers (Y) must be updated. Unhandled propagation silently falls through to wrong behavior (Z).

| Sub-type | X (what changed) | Y (downstream not updated) | Z (failure) | Sev |
|----------|-----------------|---------------------------|-------------|-----|
| param-discard | new param in signature | function body | value accepted but never used | ⚠️/🔴 |
| param-removed | param removed from signature | all call sites (subclasses/bridges if public) | TypeError at call time | 🔴 |
| attr-missing | new `getattr` attribute key | all call paths without the attr | silent zero/None fallback | ⚠️/🔴 |
| dispatch-silent | multi-backend fallback | caller logging | backend switch with no diagnostic | ⚠️ |
| rename | public symbol renamed | all importers (subclasses/bridges if public) | AttributeError at import/call time | 🔴 |

Severity (param-discard): 🔴 if param controls output correctness (`expert_mask`, `q_scale`, `expert_mapping`); ⚠️ for performance knobs or optional features with working defaults.
**Public/base-class scope:** if the changed symbol is a base-class method (`deepseek_v2.py` is the base for DSv2/V3/V3-0324/Kimi) or a signature read by a plugin bridge, param-removed and rename break all subclasses and bridges, not just same-file call sites — also apply E2 (plugin bridge sync).
Exception: method override where base class forces the signature but subclass legitimately ignores the param — flag as 📝 (structural discard, not a bug). E.g., `needs_independent_noise` structurally `del`-ed inside `ATOM_USE_TORCH_SAMPLER` override → 📝.
FP self-check (rename / param-removed): before firing, confirm the old symbol is NOT preserved by a compatibility shim added in the same PR — a same-named wrapper, an alias, or a backward-compat re-export. A rename/removal behind such a shim is backward-compatible — do not fire. (cf. aiter#4227: `get_mla_metadata_v1` renamed but a same-named wrapper preserved the symbol → not a break.)
Real examples (param-discard): `expert_mask` accepted but `# return None` commented out → TP expert-parallel callers silently routed wrong; `needs_independent_noise` dropped in `prefill_forward` — first token uses wrong sampling mode (ATOM#860).
Real example (attr-missing): `getattr(self.args,'n_shared_experts',0)` silent zero fallback — shared expert slots silently dropped from mapping with no warning if attribute absent (ATOM#1548).
→ `🔴/⚠️ B5-[sub-type]: [what changed] — [downstream not updated] — [failure]`

**B6 — New dispatch value not handled by all paths, no warning** ⚠️/🔴
When a PR introduces a new routing value to a multi-way dispatch — a new dtype, a new arch string, a new layout flag, a new `getattr` attribute key — every reachable dispatch branch must either (a) handle it explicitly, (b) fall through to a documented safe default, or (c) assert/warn before the wrong branch is reached. Silent fallback to an incorrect behavior is a bug.
Severity: 🔴 if the wrong path produces incorrect output (wrong expert mapping, wrong kernel, wrong scale). ⚠️ if the wrong path is a safe-but-suboptimal default.
Exception: an upstream assert/raise/isinstance check that prevents the bad value from reaching the branch → not B6. A runtime assert that fires for the dangerous combo → not B6.
FP self-check: Is the uncovered branch actually reachable with the new value? Is there a caller contract preventing the bad combo?
Real examples: `getattr(self.args,'n_shared_experts',0)` silent zero fallback — shared expert slots silently dropped from mapping with no warning if attribute absent (ATOM#1548); `x_pad_to_multiple=256` kwarg added to fused AR+RMSNorm call before aiter PR merged — TypeError at dispatch for all callers (ATOM#841).
→ `🔴/⚠️ B6: [new value/attribute] reaches [branch] which assumes [old behavior] — [what wrong thing happens] — add assert or explicit handling`

---

### C — Hardcoded Arch / Dtype Assumptions
_"The constant is correct for gfx942/bf16; it silently breaks on gfx950 or fp8."_

**C1 — Dtype hardcoded without checking actual tensor** ⚠️
Fixed `bf16`, `fp8_e8m0`, or similar in an attention backend, norm, or scale path that handles multiple configs.
FP self-check first: search the unchanged lines of this file for the same hardcoded dtype — if it already appears pre-existing on the same path, this is not a new violation (do not fire as new). Fire only when the hardcode is newly introduced, or the path newly handles more than one dtype/config.
Real examples: ATOM#1423 valarLip: "not always bf16"; ATOM#1458 valarLip: "hard code to fp8_e8m0?"
→ `⚠️ C1: dtype hardcoded to [type] — should derive from actual tensor/config dtype`

**C2 — FP8 fnuz check uses arch name** ⚠️
`if "gfx942" in arch: treat_as_fnuz()` — wrong. Same arch can have both fn and fnuz tensors in flight (fnuz KV, fn Q).
Use `tensor.dtype == fp8_fnuz` to check IS fnuz. Gating CONVERSION by arch is OK.
Real example (aiter#4073): valarLip: "check _is_fnuz by tensor's DType instead of arch."
→ `⚠️ C2: fnuz check uses arch name — use tensor.dtype comparison`

**C3 — New GPU arch string or arch-specific constant hardcoded in dispatch** ⚠️
**FP self-check first (do this before deciding to fire):** Search the unchanged lines of this file for the same arch string or constant value (e.g., `'gfx942'`, `576`). If it already appears on an unchanged line → **do not fire** (pre-existing style). Only proceed if it is genuinely new to this file.
Trigger (only after self-check passes): a new `+` line introduces an arch string literal in a dispatch condition (`if arch == 'gfx942':`, `'gfx950' in arch_str`), or a magic constant tied to a specific arch/model config (`576` for MLA kv_lora_rank+qk_rope_head_dim), rather than deriving from config or a named constant.
Also exempt: strings in comments, docstrings, or directory names; values imported from a central registry; arch strings used as **capability guards inside a model-specific or kernel-specific function** (not in the centralized dispatch layer) — e.g., `gfx_name.startswith("gfx1201")` inside `activation.py`'s `_detect_gfx1201()` function determines backend availability; that check belongs at the detection layer and does not trigger C3.
Real examples: hardcoded `576` in `_bind_kv_cache_to_modules()` instead of `kv_lora_rank + qk_rope_head_dim` — breaks any MLA variant where the sum ≠ 576 (ATOM#860 → fire C3); `'gfx942'` already in `attention_mla.py` and new `+` line extends same pattern (→ skip, pre-existing style); `gcnArchName.startswith("gfx1201")` in a dedicated `_detect_gfx1201()` helper (ATOM#749 → skip, capability guard).
→ `⚠️ C3: new arch-specific [string/constant] hardcoded in dispatch — derive from config or named constant`

---

### D — Uninitialized / Boundary State
_"The buffer is used before it's properly set up."_

_Rule numbers are aligned with aiter's D-section. D3 (hipblaslt tuning config) is aiter-only and does not apply to ATOM._

**D1 — Atomic reduction on uninitialized buffer** 🔴
`atomic_fmax(*ptr, val)` = `*ptr = max(*ptr, val)`. If `*ptr` is uninitialized (from `::empty()` / `torch.empty()`), garbage dominates the max → corrupted amax → corrupted FP8 descale → silent wrong quantization.
Trigger: ATOM code passes a freshly-allocated `torch.empty()` tensor to an aiter kernel that uses atomic reductions internally (e.g., `fused_qk_norm_rope_cache_pts_quant_shuffle` v_amax buffers); or any new allocation on the `torch.empty` path near a kernel that calls `atomic_fmax`.
Severity: 🔴 for atomic accumulation (atomic_fmax, atomicAdd) — garbage propagates into every output element. ⚠️ for partial-sum buffers where a zero-weight coefficient mathematically cancels the contribution (e.g., online softmax with empty batch: `exp(-inf) × garbage = 0`); still flag because `0.0 × NaN = NaN` on IEEE hardware if the allocator returns dirty pages.
Real example (aiter#4015): yzhou103: "AiterTensor::empty does not zero-initialize... garbage in v_amax silently corrupts descale."
→ `🔴 D1: [buffer] passed to atomic kernel not zero-initialized — use torch.zeros not torch.empty`

**D1b — Python-side UnboundLocalError from conditional assignment** 🔴
A variable is assigned inside an `if/elif` branch but referenced unconditionally after the block. Python does not detect this statically — `UnboundLocalError` or `NameError` fires only at runtime when the skipped branch is exercised. Silent in test environments that never hit the uninitialized path.
Trigger: new `if/elif` gate assigns a variable (`result = ...`) on some branches; a later line references it without a pre-block default. Check: is there a `var = None` or `var = default_val` before the if-block?
Exception: if there is a definitive `else` branch that also assigns the variable, or if the variable is only ever used inside the branch that assigns it.
Real example (ATOM#860): `needs_independent_noise` returned from `prepare_model()` tuple but assigned only in one branch of `prefill_forward` — other branch paths raised `NameError` when the sampler tried to use it.
→ `🔴 D1b: [var] assigned only inside [branch] but referenced unconditionally — add [var = default] before the if-block`

**D2 — New default path without rollback env-var** ⚠️
New implementation replaces existing default before wide validation: is there an env var to revert?
→ `⚠️ D2: new default path needs rollback env-var for safe rollout`

**D4 — Invariant reversal without citation** 🔴
A documented safety invariant is reversed: old comment says "must X because Y" → new code removes X claiming "X not needed" but no spec/asm/test is cited to prove Y no longer holds.
Trigger: `torch.zeros → torch.empty` where old comment mentions "must" / "required" / "read back as zero"; assert deletion without explanation; `.contiguous()` removal; zero-init removal with contradicting justification.
Real example (aiter#4043): old: "trailing pad must read back as zero for the asm reader, so zero-initialise it here" → new: "trailing pad is never read by the asm reader, so no zero-init is needed" — two comments directly contradict; PR cites no spec.
→ `🔴 D4: [operation] reverses a documented safety invariant — cite the spec/asm/test proving new assumption is safe`

**D5 — Verbatim duplication across backbone files** ⚠️
The same fix is copy-pasted into 2+ Tier 1/2 backbone files with trivial name substitution (different variable names, identical algorithm and comments). AI code signature: changes look symmetric but each file's invariants may differ and were not independently verified.
Trigger: nearly identical `+` blocks appearing in two backbone files in the same PR diff; same formula / same comment structure / same magic constants, only variable names differ.
Real example (ATOM#1493): chunked indexer loop copy-pasted verbatim between `deepseek_v2.py` (num_rows, total_kv) and `deepseek_v4.py` (total_tokens, total_committed) — same `(budget_rows // 128) * 128` formula and same `bit_length() - 1` fallback, same block of comments.
→ `⚠️ D5: identical algorithm in [file_a] and [file_b] — was correctness verified independently in each context, or copy-pasted?`

**D6 — Fake / meta function dtype or shape mismatch** 🔴
When a `gen_fake` / `_fake` / `abstract_impl` function is added or modified, its return tensor dtypes and shapes must match the real op exactly. torch.compile uses the fake to infer output types; a wrong dtype compiles cleanly but fails at runtime with a dtype mismatch or silently produces wrong values.
Trigger (1): diff contains a `_fake` / `gen_fake` function alongside the real op; compare each return tensor's dtype and shape against the real op's actual output.
Trigger (2): real op's return dtype or arity changes in the diff but no corresponding `_fake` / `gen_fake` change appears — the existing fake is now stale.
Real example (aiter#4110): `fused_allreduce_rmsnorm_quant_fake` returned `torch.empty_like(res_inp)` (bf16) as first element, but real op returns fp8 — wrong dtype for torch.compile's dtype checks.
→ `🔴 [fake_fn] return [N] dtype is [X] but real op returns [Y] — torch.compile will assert or silently miscompute`

**D7 — New torch.compile op without fake function** 🔴
A new op registered via `torch.library.custom_op` or `@compile_ops` has no corresponding `_fake` / `gen_fake` / `abstract_impl`. torch.compile traces using fake tensors; without a fake the op is a black box → runtime crash or silent fallback to eager inside a compiled region.
Trigger: diff adds a new function decorated with `@compile_ops` or `torch.library.custom_op`; grep for a matching `_fake` / `gen_fake` function — if absent, flag.
→ `🔴 D7: [op_name] has no fake/abstract implementation — torch.compile will crash or silently fall back to eager`

**D8 — Kernel call missing contiguous check** ⚠️
Python code calls a C++ / aiter kernel but doesn't assert `.is_contiguous()` or call `.contiguous()` on inputs that may arrive strided (slice of larger tensor, `.T`, output of non-contiguous `view()`). Kernel reads from wrong addresses — completely silent wrong result.
Trigger: new call to an aiter kernel or C-extension; check that non-trivially-shaped inputs are either asserted contiguous or made contiguous before the call.
→ `⚠️ D8: [tensor] passed to [kernel] without contiguous check — add .contiguous() or assert .is_contiguous()`

**D9 — INT32 overflow in GPU pointer arithmetic** 🔴
Python wrapper or model code computes a buffer offset or index in `int32` (or `torch.int32`) when the product of dimensions can exceed 2^31 (~2 billion) at production scale.
Common patterns: `token_id * (num_heads * head_dim)` overflows at token_id > 16M with H=32, D=128; `total_tokens * head_dim` overflows for long-context batches.
Trigger: arithmetic involving `token_id`, `total_tokens`, `seq_start`, or `batch_offset` that produces a buffer address or index in int32 without an explicit widening cast to int64 before the multiply.
→ `🔴 D9: [expr] in int32 — widen [token_id / total_tokens / seq_start] to int64 before multiplying by [stride]`

---

### E — Cross-Repo Sync
_"The change is incomplete without a matching update in another repo or file."_

**E1 — New aiter symbol or kwarg without linked aiter PR** ⚠️
New `from aiter import X`, new kwargs on aiter calls: PR links an aiter PR?
New kwargs may require an aiter version not yet released.
Real example (ATOM#1494): `emit_bf16=True` kwarg added → needed aiter PR first.
→ `⚠️ E1: new aiter usage — corresponding aiter PR not mentioned`

**E2 — Plugin bridge not updated** ⚠️
PR changes KV layout, function signature, or data structure that `deepseek_v4_bridge.py` / `sglang_bridge.py` read directly.
Real example (ATOM#1423): paged-SWA layout changed; bridge still used old layout.
→ `⚠️ E2: [structure] changed — plugin bridge sync needed`

**E3 — New param with backward-compatible default is dead code** 📝
New aiter param added with default that preserves old behavior: fix only activates when consumer passes non-default. Who updates the consumer?
Real example (aiter#3773): `max_seqlen=-1` — fix never activated until ATOM passed actual value.
→ `📝 E3: new API param needs consumer-side update to activate — follow-up tracked?`

---

### F — Resource Duplication
_"The change pins the same data twice without freeing the original."_

**F1 — New weight variant alongside original** ⚠️
New `w13_weight_preshuffled` / `w_quantized` stored alongside `w13_weight`: both pinned simultaneously → double HBM.
For large MoE models (w13 = 7B+), OOM only at peak load — silent until prod.
Real example (ATOM#1469): valarLip: "this will make us pin double weight."
Check: is the original freed after the new variant is created?
→ `⚠️ F1: [new_attr] alongside [original] — doubles HBM; is original freed?`

---

### G — Multi-Stream Synchronization
_"Written on stream A, consumed on stream B — no sync between them."_

**G1 — Missing HIP/CUDA stream synchronization** 🔴
HIP/CUDA streams execute concurrently by default. A tensor produced on stream A and consumed by a kernel on stream B without an explicit sync between them causes the consumer to read garbage — no crash, no error, silent wrong output.
Trigger: diff introduces a non-default `torch.cuda.Stream`, passes an explicit `stream=` argument to a kernel, or prepares weights/KV buffers on a side stream at model-load time that are later consumed during forward pass on the compute stream. Check: is there `stream.synchronize()`, `stream.wait_stream(other)`, `hipEventRecord` + `hipStreamWaitEvent`, or `torch.cuda.current_stream().wait_stream(other)` between the last write on stream A and the first read on stream B?
→ `🔴 G1: [tensor] written on [stream A] consumed on [stream B] without sync — add stream.wait_stream() or hipStreamWaitEvent`

**G1b — Blocking queue.get() without timeout in production serving code** ⚠️
`queue.get()` without `timeout=` in a worker or service thread that depends on an external producer (decode loop, stream consumer, request handler). If the producer exits abnormally, the worker blocks forever — no crash, no log, hung process.
Trigger: `queue.get()` or `asyncio.Queue.get()` inside a `while True:` worker loop in production serving paths (entrypoints, engine loop, scheduler) without `timeout=` and without a corresponding `except queue.Empty` / `asyncio.TimeoutError` handler or a `done` flag.
Exception: test code, CLI tools, or one-shot scripts where a hang is detectable (CI timeout, interactive TTY).
Real example (ATOM#789): `loop.call_soon_threadsafe` routes incremental detokenizer state mutations to the event-loop thread — this is correct and is NOT a G1b violation.
→ `⚠️ G1b: [worker] blocks on queue.get() without timeout — add timeout= and handle Empty/TimeoutError to survive producer failure`

---

### Performance Evidence (always check)

**P1 — Perf PR without benchmark numbers** ⚠️
Trigger words: perf, optimize, fuse, faster, improve, +X%, replace kernel, OOM fix that changes algo.
Description must have numbers with units (ms, tokens/s, TFLOPS, %). Accuracy tables (gsm8k, MMLU) do NOT count. Screenshots ≠ numbers.
→ `⚠️ P1: perf claimed — no benchmark numbers with units`

**P2 — Benchmark covers only toy shapes** ⚠️
Numbers exist but only for M≤256, only 1 token, or one model.
Production: DSv4 (E=385/topk=7, TP=4/8), GPT-OSS 120B, Kimi-K2.5, GLM5; token range 1→16384 decode + prefill 1k/4k/32k.
Staleness guard: the production config list is a snapshot — verify current E/topk/TP and the model roster from `atom/config.py` or a recent benchmark before asserting what counts as "production".
→ `⚠️ P2: benchmark missing production shapes — [what's absent]`

**P3 — Perf claim not reproducible** ⚠️
Missing: test script/command, ROCm version, GPU model, TP/DP config, model checkpoint.
→ `⚠️ P3: perf claim missing reproduction info — [what's absent]`

**P4 — TP split shapes not covered** ⚠️
New attention / norm op tested only at full head count (TP=1 equivalent). At TP=4/8, `num_heads_q` / `num_heads_k` per device is divided by TP. A kernel passing at H=128 may OOB or produce wrong output at H=32 (TP=4) if shape math doesn't account for the split.
Trigger: new kernel or dispatch path that takes per-device head count; PR shows only one head count without TP=4/8 variant.
→ `⚠️ P4: test covers only TP=1 head count — verify at num_heads÷TP=4 (e.g., [128→32])`

**P5 — Benchmark hides a cost real users pay on every call or cold start** ⚠️
The perf claim is measured with the timing window drawn so a *recurring* production cost is excluded: a first-call JIT compile on a path that is NOT cached across calls, or a setup step that runs on the live stream inside the timed region on every cold start. If that cost is real and recurring, omitting it can turn a net regression into an apparent speedup.
Do NOT fire on a genuinely one-time, amortizable setup that production pays once at model init — excluding weight shuffle/preshuffle, model weight loading, or a first-call JIT whose result is cached forever from steady-state per-call latency is CORRECT methodology, not deception. `warmup_iters` before a steady-state loop is standard and by itself is not P5.
FP self-check (do this before firing): is the excluded cost paid **once per deployment** (amortizable → do NOT fire) or **again on every call / every cold start / inside the timed stream** (→ fire)? If you cannot show it recurs, do not fire.
Counter-example (does NOT trigger P5): aiter#4166 preshuffles the static weight once outside the timing loop and honestly reports a geomean 0.69x result — a correct steady-state benchmark, not a hidden cost. Charging that one-time shuffle against a single call to manufacture a "regression" is itself the false positive this rule must avoid.
→ `⚠️ P5: timing window excludes [cost] that recurs per call / per cold start — re-run including it, or confirm it is one-time amortizable`

---

### Housekeeping (quick scan)

| Check | Trigger | Flag |
|---|---|---|
| Temp script | `runperf-*.sh`, `test_local_*.py` in diff | `⚠️ HK1: [file] looks temporary — remove before merge` |
| Unrelated files | Files with no connection to PR purpose | `⚠️ HK2: [file] appears unrelated` |
| Bug fix without test | No regression test or repro script | `📝 HK3: bug fix without test — how to prevent regression?` |
| TODO/stub in new path | `# TODO`, `# FIXME`, `raise NotImplementedError`, lone `pass` on a `+` line inside a new branch | `⚠️ HK4: [location] — incomplete implementation in new code path, will silently not execute on default path` |
| Undocumented new env var | `os.environ.get("ATOM_...` or `os.environ.get("AITER_...` on a `+` line | `📝 HK5: new env var [NAME] — register in atom/envs.py AND document in README` |
| Test reference dtype promotion | New test reference impl uses Python float literal (`1.0 + weight`, `0.5 * x.float()`) or explicit upcast (`.to(torch.float32)`, `.double()`) promoting to fp32 while kernel runs in bf16/fp8 — comparison calibrated against wrong-precision baseline | `⚠️ HK6: reference [fn] promotes to fp32 — cast back to [kernel dtype] before comparison` |
| New third-party dependency | New package in `requirements*.txt`, `setup.py`, `pyproject.toml`; or new top-level `import [pkg]` not already a project dep. Exception: ROCm system packages (`amdsmi`, `hip`, `rccl`) are intentionally not on PyPI — flag only if there is no `try/except ImportError` guard AND no comment explaining the ROCm-only dependency | `📝 HK7: new dependency [pkg] — add to requirements, or add try/except ImportError with a comment for ROCm system packages` |

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

## Step 8 — Verdict

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
