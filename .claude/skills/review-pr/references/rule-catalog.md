# ATOM PR rule catalog

The team checklist behind `review-pr`. SKILL.md section 2 sends you here by PR
type; the maintainer's lens (SKILL.md section 3) takes precedence where the two
differ. Rule codes (A1, B3, ...) are internal labels — never put them in the
posted review.

Severity labels: **[blocker]** wrong output, crash, OOM, hang; **[should fix]**
real but bounded; **[note]** worth saying, not worth holding the PR for.

**Blocker gate — before calling anything a blocker, write down the concrete
input that triggers it**: the shape / scale / dtype / arch / value that makes it
fire (e.g. "at `token_id` > 16M with H=32, D=128 the int32 product exceeds
2^31"). No concrete trigger → it is "worth checking", not a blocker. This applies
to every rule below, including those that state no self-check of their own.

Contents
1. PR type → rules to check
2. Backbone file risk
3. Rules: A coverage · B silent bypass · C hardcoded arch/dtype · D uninitialized
   state · E cross-repo sync · F resource duplication · G multi-stream
4. Performance evidence (P)
5. Housekeeping (HK)
6. AI-generated code: structural checks
7. Free-form questions and the blind-spot pass

---

## 1. PR type → rules to check

| The diff… | Check |
|---|---|
| claims performance | P1 numbers, P5 setup cost, P2 production shapes, P3 reproducible |
| adds aiter op usage or changes an aiter call | E1 aiter dep, B5 propagation, B6 new kwarg on every branch, E3 dead param |
| changes an API signature (param added / removed / renamed, default or return type) | B5, E1 (aiter side), E2 plugin bridge |
| adds a constexpr / routing flag / attribute key | B6, C3 |
| integrates a model (`atom/models/*.py`) | A2 shared backbone, section 2 risk |
| changes dispatch | B1, B2, B6, A3 |
| touches cudagraph / capture | B3 |
| touches a plugin (vLLM / SGLang / RTP) interface | E2 |
| changes config / tuning | A2, A3 |
| fixes a bug | A1 sibling variant, HK3 regression test, root cause explained? |
| touches FP8 / quantization | C1, C2 |
| transforms weights / adds a weight attr | F1 |
| adds async / multi-stream / weight prep | G1, G1b |
| adds `if/elif` that assigns a variable | D1b |
| adds `@compile_ops` / `torch.library.custom_op`, or changes an op's return dtype/arity | D7, D6 |
| changes launcher / buffer-offset / index arithmetic on long-context or large-batch paths | D9 |
| adds a C-extension / aiter kernel call | D8 |
| removes or reverses a zero-init / assert / `.contiguous()` / documented invariant | D4 |

## 2. Backbone file risk

A file is **Tier 1** if it runs on every forward of every model
(`model_runner.py`, `engine_core.py`, `scheduler.py`, `config.py`) or is a base
class for more than two production families (`deepseek_v2.py`: DSv2 / V3 /
V3-0324 / Kimi). **Tier 2** implements an op shared across families, where a
bug silently corrupts all of them (`attention_mla.py`, `attentions/aiter_mla.py`,
`linear.py`, `moe.py`, `attention_mha.py`, `layernorm.py`). Plugin bridges are
Tier 3 by blast radius but highly visible. ATOM has no `import atom` chain, so
tiers come from "runs every request" or "base of many", not imports.

Snapshot of the most-changed backbone files (2025–2026; recount with
`git log --oneline -- <file> | wc -l` before relying on it):

| Tier | File | Common failure |
|---|---|---|
| 1 | `model_engine/model_runner.py` | OOM, cudagraph break, wrong batch assembly |
| 1 | `config.py` | wrong config silently changes kernel path |
| 1 | `models/deepseek_v2.py` | wrong MLA, OOM, accuracy drop for all DSv* |
| 2 | `model_ops/moe.py` | wrong expert routing, double weight pinning |
| 2 | `model_engine/scheduler.py` | stall, wrong decode/prefill split |
| 2 | `model_ops/attention_mla.py`, `attentions/aiter_mla.py` | wrong KV / kernel / dtype |
| 2 | `model_ops/linear.py` | wrong GEMM dispatch or quant |
| 2 | `model_ops/attention_mha.py`, `layernorm.py` | wrong attention output, wrong scale |
| 2 | `models/deepseek_v4.py` | wrong sparse MLA, SWA layout break |

When a Tier 1 file changes:
- List every changed function and grep its callers (`grep -rn 'def <name>'
  atom/`). A caller the PR does not mention is a finding to trace.
- For `deepseek_v2.py`: is the changed method overridden in a subclass? If not,
  every DSv2/V3/V3-0324/Kimi is affected even when the PR says "Kimi-only".
- Does anything new sit in the cudagraph-captured region (B3)?
- State what breaks if the change is wrong (crash / silent wrong value / OOM /
  graph break) and what would detect it.

When a Tier 2 file changes: which families use it, is one from each tested,
are production shapes covered (TP4/TP8, single-token decode, prefill ≥ 4K), and
are both the FP8 and BF16 paths exercised?

Verbatim duplication of one algorithm across backbone files (same formula and
comments, names swapped) is D5.

---

## 3. Rules

### A — Coverage gaps: "fixed one path; the same bug lives in a sibling"

**A1 Sibling not fixed** [should fix; blocker on Tier 1/2]. A fix to address
math, bounds or layout: scan the same file for `_opt`, `_prefill_opt`, `_decode`,
`_v2` variants. (aiter#3841: strided `q_nope` fixed in decode, not `_prefill_opt`.)

**A2 One model/GPU tested, shared path affects others** [should fix]. "[MI308]"
or "DSv4-only" on a backbone file still reaches gfx950 / Kimi / DSv3
(ATOM#1498). If benchmarks show one arch, ask about the other.

**A3 Activation broader than validated** [should fix]. A new dispatch enables a
kernel for family X but only subcase Y was tested (vLLM#16435).

### B — Silent bypass: "the code looks complete but some inputs take the wrong path"

**B1 Gated-off parameter** [blocker]. In a new `if/elif/else`, is every
parameter that a branch cannot honour asserted or forwarded? Triggers:
`block_table`, `alibi_slopes`, `window_size`, `logits_soft_cap`, `dropout_p`.
(aiter#3576: the False branch computed dense attention and ignored `block_table`.)

**B2 Phase by size proxy** [should fix]. `max_q`, `q.shape[0] == 1`,
`seq_len <= 1` deciding decode vs prefill misclassifies short / chunked
prefill; use the explicit phase flag (ATOM#1372).

**B3 Cudagraph capture** [blocker]. New Python branching, `.item()` / `.tolist()`,
or `torch.empty` / `torch.zeros` inside a captured region. Constant arithmetic,
casts of existing tensors and fixed-shape kernel calls are safe. Ask why capture
stays valid (ATOM#1321).

**B4 Triton constexpr disables a check** [should fix]. A `tl.constexpr` bool that
skips a bounds / sentinel check needs the caller invariant documented
(ATOM#1498: `CHECK_NEG_ONE_SENTINEL=False`).

**B5 API propagation incomplete** [blocker / should fix]:

| Sub-type | Changed | Not updated | Failure |
|---|---|---|---|
| param-discard | new param in signature | body | accepted, never used |
| param-removed | param removed | call sites, subclasses, bridges | TypeError |
| attr-missing | new `getattr` key | paths without it | silent zero / None |
| dispatch-silent | multi-backend fallback | logging | silent backend switch |
| rename | public symbol | importers, subclasses, bridges | AttributeError |

Blocker when the param controls correctness (`expert_mask`, `q_scale`,
`expert_mapping`). A base-class method or a signature a plugin bridge reads
breaks every subclass and bridge (also E2). Self-check: a same-named wrapper,
alias or re-export added in the PR keeps it compatible (aiter#4227). A subclass
that legitimately ignores a param the base forces is a note. (ATOM#860
`needs_independent_noise`; ATOM#1548 `getattr(..., 'n_shared_experts', 0)`.)

**B6 New dispatch value not handled everywhere** [blocker / should fix]. A new
dtype / arch / layout flag / attribute key must be handled, fall to a documented
safe default, or be asserted on every reachable branch. Not B6 if an upstream
check already rejects it. (ATOM#1548; ATOM#841 kwarg passed before the aiter PR
merged.)

### C — Hardcoded arch / dtype

Pre-existing hardcodes are reported too, under SKILL.md section 4 with a
disposition; "already on an unchanged line" decides how it is filed, not
whether it is reported.

**C1 Dtype hardcoded** [should fix]. `bf16`, `fp8_e8m0`, `float8_e4m3fn` fixed in
a path that serves several configs (ATOM#1423 "not always bf16", ATOM#1458).

**C2 fnuz decided by arch name** [should fix]. Check `tensor.dtype ==
fp8_fnuz`; gating the conversion by arch is fine (aiter#4073).

**C3 Arch string or arch constant in dispatch** [should fix]. A new
`'gfx942' in arch` in central dispatch, or a magic model constant (`576`
instead of `kv_lora_rank + qk_rope_head_dim`, ATOM#860). Not C3: comments,
values from a central registry, a capability guard inside a dedicated detection
helper (ATOM#749 `_detect_gfx1201()`).

### D — Uninitialized / boundary state

**D1 Atomic reduction on `torch.empty`** [blocker]. Garbage dominates
`atomic_fmax` / `atomicAdd` → corrupt amax / descale (aiter#4015). Partial sums
cancelled by a zero weight are still a should-fix: `0 * NaN = NaN`.

**D1b Variable assigned on some branches only** [blocker]. `UnboundLocalError`
when the skipped branch runs (ATOM#860).

**D2 New default path and its off switch** [note]. A new path should be on by
default and not grow an env var just to turn it off (maintainer, #2808). Ask for
a rollback switch only when the PR replaces a path users depend on and the
description states why it may need to be turned off; otherwise an env var added
"for safety" is itself the finding (SKILL.md 3.8).

**D4 Invariant reversed without citation** [blocker]. An old "must X because Y"
comment, assert, zero-init or `.contiguous()` removed with no spec / asm / test
proving Y no longer holds (aiter#4043).

**D5 Verbatim duplication across backbone files** [should fix]. Same formula and
comments in two files with names swapped: was each verified in its own context?
(ATOM#1493.)

**D6 Fake impl dtype/shape mismatch** [blocker]. A `_fake` / `gen_fake` whose
outputs differ from the real op, or a real op whose dtype/arity changed with a
stale fake (aiter#4110).

**D7 New compiled op without a fake** [blocker]. `@compile_ops` /
`torch.library.custom_op` with no fake → compile crash or silent eager fallback.

**D8 Kernel input not checked contiguous** [should fix]. A possibly strided input
(slice, `.T`, `view`) to a C++ / aiter kernel with no assert or `.contiguous()`.
But do not add copies aiter does not need — its quant and GEMM take strides
(SKILL.md 3.4).

**D9 int32 overflow in offsets** [blocker, with the gate above]. `token_id *
(H * D)`, `total_tokens * head_dim` without widening to int64.

### E — Cross-repo sync

**E1 New aiter symbol / kwarg with no aiter PR named** [should fix]. State the
aiter PR or minimum version; a module-level import of a new symbol breaks every
model on an older aiter (ATOM#1494).

**E2 Plugin bridge not updated** [should fix]. KV layout, signature or structure
a vLLM / SGLang / RTP bridge reads directly (ATOM#1423).

**E3 New param whose default keeps the old behaviour** [note]. Who passes the
non-default to activate it? (aiter#3773 `max_seqlen=-1`.)

### F — Resource duplication

**F1 New weight variant beside the original** [should fix]. Both pinned → double
HBM, OOM only at peak (ATOM#1469). Is the original freed?

### G — Multi-stream

**G1 Written on stream A, read on stream B, no sync** [blocker]. Look for
`wait_stream`, `hipStreamWaitEvent`, or `synchronize` between the last write and
the first read.

**G1b Blocking `queue.get()` with no timeout in serving code** [should fix]. A
worker loop that hangs forever when its producer dies. Not in tests or one-shot
tools, and not `loop.call_soon_threadsafe` handing state to the event-loop
thread, which is the correct pattern (ATOM#789).

---

## 4. Performance evidence (P)

**P1** Perf claimed (perf, optimize, fuse, faster, +X%) with no numbers in units
(ms, tok/s, TFLOPS). Accuracy tables and screenshots are not numbers.

**P2** Only toy shapes (M ≤ 256, one token, one model). Production means the
current roster and TP (verify from `.github/benchmark/models.json` rather than
memory), decode 1→16K tokens, prefill 1K/4K/32K.

**P3** Not reproducible: missing command, ROCm, GPU, TP/DP, checkpoint.

**P4** Only the TP=1 head count tested; per-device heads at TP4/8 can break shape
math.

**P5** The timing window hides a cost paid on every call or cold start. A
one-time, amortizable setup (weight preshuffle, model load, a JIT cached
forever) is correctly excluded — check it recurs before raising it (aiter#4166 is
a correct benchmark).

## 5. Housekeeping (HK)

| Check | Trigger |
|---|---|
| HK1 temp script | `runperf-*.sh`, `test_local_*.py`, throwaway benches in the diff |
| HK2 unrelated files | files with no link to the PR's purpose |
| HK3 bug fix without a regression test | no test or repro |
| HK4 TODO / stub in a new path | `# TODO`, `raise NotImplementedError`, lone `pass` in a new branch |
| HK5 env var outside envs.py | `os.environ.get("ATOM_...")` on a `+` line: register it in `atom/utils/envs.py` and add a row to `docs/environment_variables.md` |
| HK6 test reference promoted to fp32 | reference upcasts while the kernel runs bf16/fp8 |
| HK7 new third-party dependency | new package or top-level import not already a dependency; ROCm system packages need an `ImportError` guard and a comment |

---

## 6. AI-generated code: structural checks

A clean description is cheap to produce; when the diff changes code, run these.

1. **Hallucinated symbols.** Grep every symbol new to the diff (function, kwarg,
   enum, attribute, import) against its real definition in aiter / ATOM / torch.
   One you cannot find is a defect until shown real.
2. **Twin divergence.** Compare mirrored code (fwd/bwd, v2/v3, prefill/decode,
   per-arch) field by field: one side int64 the other int32, one masked the
   other not.
3. **Claims and numbers.** Does the code enforce what the description or a
   comment says? Trace the most impressive number to its output; a number you
   cannot trace is `[unverified]`.
4. **Safety theater.** Is each new guard reachable, can it fire, does an
   `except: pass` swallow a real error?
5. **Tests that cannot fail.** Reference twin of the kernel, loosened tolerance,
   self-comparison.
6. **Magic constants** with no derivation or tuning basis.

Warning signs: description says WHAT not WHY; perf numbers are clean integers or
screenshots only; tests only at token_num ≤ 256; template "Test Plan" left
blank; `sys.path` edits at module level; an AI-attribution footer. Three or more
signs, or any structural check firing, means read dispatch logic and coverage
closely.

---

## 7. Free-form questions and the blind-spot pass

- Does the algorithm fit the MI300X / MI355X memory hierarchy?
- MLA: does the paged vs dense KV distinction hold throughout?
- MoE: routing correct under every EP configuration? Is the denominator the right
  parallelism degree — EP shards experts, TP shards matrices?
- Quant fusion: scale order and dtype promotion.
- Size arithmetic: missing factors (`num_swa_blocks` without `* block_size`,
  ATOM#1423).
- `.copy_()` into a buffer where an `out=` exists (ATOM#1493).
- Triton tile / blocking changes: LDS stays within 64 KB per CU, or the compiler
  spills.

Last, before writing: **is there any correctness risk, resource hazard or edge
case in this diff that nothing above caught?** If yes, add it.
