# ATOM review rules

Rule bodies, injected into a review by `triage.py expand` according to the
families triage derived for the diff. They are NOT in SKILL.md: the median PR
needs a fraction of them, and shipping all of them into every review costs
attention that the derived subset is supposed to buy back.

Every rule must cite a real ATOM PR as evidence. The rules below were carried
over verbatim from the pre-pipeline SKILL.md and most do not yet carry one --
that is the work of P1, not a licence to add more without evidence.

---

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

---

### A — Coverage Gaps
_"Fixed one path; the same bug lives in a sibling."_

**A1 — Sibling function/kernel not fixed** ⚠️ (🔴 if Tier-1/2 backbone)
Fix changes address calc, bounds check, or data layout: scan same file for variants named `_opt`, `_prefill_opt`, `_decode`, `_v2`. The sibling is not always same-file: also scan plugin backend directories (`atom/plugin/vllm/`, `atom/plugin/sglang/`) for a copy of the same allocation/dispatch pattern.
Real example (aiter#3841): strided q_nope fix on decode kernel; `_prefill_opt` in same file had same bug.
Real example (ATOM#1091, `atom/model_ops/moe.py`, fixed `5663a664`): `expert_data` narrowed to actual shard size with remainder zeroed — a trailing partial rank (e.g. `tp=4, D=10`) no longer hits a `copy_()` shape mismatch; the same PR also removed a redundant `all_reduce` in `Mxfp4MoEMethod.apply` ("other `apply()` methods don't reduce internally... reduction now happens once") — a second, distinct defect (duplicate collective → silently wrong numeric result), single occurrence so far, worth checking for alongside the shard-remainder bug.
Real example (ATOM#2607, valarLip, `COMMENTED`): "Incomplete fix (same allocation pattern remains): `atom/models/glm5_next.py:748` — `Glm5NextIndexer.forward` (non-kpool branch) still calls `sparse_attn_indexer_impl` without `indexer_logits_ws`... `atom/plugin/vllm/attention/layer_sparse_mla.py:466` and the SGLang plugin... still allocate a `[num_decode_tokens, max_model_len]` fp32 plane per layer — the same unbounded pattern this PR chunks natively." Here the "sibling" is a cross-plugin-backend copy, not a same-file variant — confirms the backend-directory scan above.
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

**A4 — Distributed handshake or coordination invariant changes on one side only** ⚠️/🔴
A new per-connection config field (parallelism degree, block size, dtype, layout) is read on one side of a cross-process relationship — a P/D handshake, a KV-connector aggregator, a producer/consumer coordination predicate — without the matching update on the other side: the exchanged metadata, the merge/aggregation logic, or the assumption the other side's predicate relies on. The mismatch is never caught; a downstream index or control-flow decision silently goes wrong.
Trigger: diff adds/reads a new `self.<field>` on a connector/engine/producer side (`kv_transfer`, `disaggregation`, `mooncake`, or similar cross-process code) that feeds an index computation or a cross-process control predicate, without the corresponding field being added to the handshake message exchanged with the remote side, or to the aggregator that merges per-connector output.
Real example (ATOM#1994): yitingw1: "The handshake exchanges `tp_size`/`remote_pp_size` but never `hash_block_size`, so nothing catches a P/D pair with mismatched `decode_context_parallel_size`; since `num_computed_blocks` is computed on one side and used as a block-table index on the other, a mismatch here silently misaligns the slice." Author (Phi-C): "we can add this warning and fallback path here as you suggested." — caught in review.
Real example (ATOM#2000, yhl-amd, `COMMENTED`): "`MultiConnector.get_finished()` drops `connector_completions`. DSV4 reports SLOT/sidecar save success or failure through `KVConnectorOutput.connector_completions`, but the `MultiConnector` merge only collects recv/send/load/save fields. A sidecar completion entering a sub-connector therefore produces an empty `MultiConnector` output." Same shape, at the aggregator layer instead of the wire layer — also caught in review.
Real example (Revert ATOM#2315, `Revert "feat(kimi-k3): support P/D disaggregation (#2154)"`): "The producer now disables deferred output, but the DPA draft lockstep predicate still uses deferred output to exclude pipeline parallelism. Output-less producer prefill chunks skip draft collectives while idle DP peers enter them, hanging V4 DPA P/D requests." This one was **not** caught in review and needed a full revert — a reminder that this defect class is genuinely hard to catch from a single-file diff.
→ `⚠️/🔴 A4: [field/invariant] changed on [side] but not matched on [other side] — a mismatch silently [misaligns index / hangs / drops output]`

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
Real example (ATOM#2371, review comment from xytpai, found via the closed-PR channel —
the PR itself was closed as superseded by #2431, not closed for this reason): *"The
wrapper validates the index tensors' device, shape, and dtype, but the kernel reads them
using `ptr + token_id`, which assumes contiguous storage. A non-contiguous view such as
`buffer[::2]` passes validation but produces incorrect output."* xytpai reproduced it on
GPU with logically identical index values — making only `src_token` non-contiguous
produced 1,056 mismatched output bytes, with no exception raised. D8's first ATOM-native
citation; until this, the rule had evidence from elsewhere (vendored prose) but no ATOM
instance of its own.
→ `⚠️ D8: [tensor] passed to [kernel] without contiguous check — add .contiguous() or assert .is_contiguous()`

**D9 — INT32 overflow in GPU pointer arithmetic** 🔴 *(scanner-backed — read the scan output, not this prose)*

ATOM's best-evidenced rule (three independent production fixes below) is live: `fetch.sh`
runs `review-pr/scan_index_width.py` over the diff and writes its candidate list to
`$WORK/index_width_scan.txt` as part of Step 1 evidence, before the rule pass even starts.
The defect this rule names is an *absence* — no int64 widening before a stride multiply
that reaches pointer arithmetic — and detecting an absence on a diff hunk needs the index
variable's type traced through the expression, which a regex cannot do (an earlier
name-list trigger caught 0 of 3 known ATOM overflow fixes in a controlled run). The scanner
is an AST pass, ported verbatim from aiter's `scan_index_width.py`: a multiplication is a
candidate when it feeds pointer/index arithmetic, at least one operand derives from a
non-`constexpr` kernel parameter (runtime-supplied — the only kind that can grow past
2^31), and no operand is 64-bit widened (counting widening on an earlier line, carried in
through a local name). Only sites on diff-added lines are reported.

Python wrapper or model code computes a buffer offset or index in `int32` (or
`torch.int32`) when the product of dimensions can exceed 2^31 (~2 billion) at production
scale. Common patterns: `token_id * (num_heads * head_dim)` overflows at token_id > 16M
with H=32, D=128; `total_tokens * head_dim` overflows for long-context batches.

**The candidate list is already in context.** Work it: clear each candidate, and fire D9
only where you can name the production scale at which the product exceeds 2^31. If the
list is empty, say so rather than skipping the category silently. If the scan printed a
`NOT SCANNED` section, D9 cannot be cleared for those files — an empty candidate list that
excluded them is not evidence of absence.

Real example (ATOM#2565, fixed `3795338967f`): packed KV output offsets computed in int32
could exceed int32 for long cached batches — fixed by widening the stride multiply to
int64 before it reaches pointer arithmetic. Scanned clean post-fix (see below).
Real example (ATOM#2211, fixed `1580bf1af6a`): `address` wraps at `vocab*hidden >= 2^31`,
shipped NaN at gqa=16 before the int64 widening landed. Scanned clean post-fix.
Real example (ATOM#1460, fixed `7986573b10c`, `v4_kernels/inverse_rope.py`): same shape,
same fix. Scanned clean post-fix.
→ `🔴 D9: [expr] in int32 — widen [token_id / total_tokens / seq_start] to int64 before multiplying by [stride]`

**D10 — Sentinel value collides with valid data** 🔴
A numeric sentinel (`-1`, `0`, `None`-as-0) is reused to mean "no limit / unset / unavailable" in a field that elsewhere holds a real count, offset, or limit. The collision is silent: the receiving code cannot tell "legitimately zero" from "not set."
Trigger: a new/changed line sets or compares a sentinel constant (`-1`, `0`) against a field whose name suggests a count/limit/offset/index/deadline (`max_tokens`, `limit`, `stop_at_idx`, `deadline`, `timeout`, `seq_num`, `budget`, `tier`), especially where the same field is read elsewhere as real data; or a function's return narrows away an `Optional[int]` in favor of returning `0`/`-1` for "no answer."
Real example (ATOM#2012): yitingw1 reproduced the hang independently — `max_tokens=0` → `stop_at_idx=-1` → the `if new_tokens` guard drops the terminal chunk → client blocks forever on `token_queue.get()`; the fix needed a second sentinel (`-2`) because `-1` was already load-bearing on the same path for the P/D `injected_t0` case. Fixed `9a70c95`, then `58ed37d3`.
Real example (ATOM#2046, `atom/model_ops/fused_moe_triton.py`): JiaoliangYu: "`limit=0` means the opposite here than it did on the old path, and it makes the whole MoE layer output exactly zero... `0.0` is taken as a real bound... No error, no assert."
Real example (ATOM#2305, `atom/kv_transfer/offload/chunked_scheduler.py`): yhl-amd: "`_MPLookupClient.lookup()` returns `0`, not `None`, when its poll deadline expires... this branch memoizes an MP timeout as a real tier miss." Confirmed + fixed `da8c39ea3`.
Real example (ATOM#1316, `atom/distributed/kv_events.py`): yitingw1: sequence numbers wrap into the reserved `REPLAY_DONE` sentinel value. Fixed `918f7dcf`.
→ `🔴 D10: [field] reuses [sentinel] to mean "unset" but also holds real data on [path] — disambiguate with Optional or a distinct sentinel`

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

**E4 — A production kernel or dispatch belongs in aiter, not ATOM** ⚠️
Found via the closed-unmerged-PR channel (ROCm/ATOM PRs closed, not merged, with a
stated technical reason), not the inline-comment corpus the rest of this file draws
from — the strongest cluster in that channel, well above the 1-citation bar. The
shape repeats: a contributor lands a self-contained production kernel or an
`ATOM_*` dispatch switch in ATOM, and the PR is later closed because the same
specialization already lives, or is decided to belong, in aiter — not because the
kernel itself was wrong. In every citation below, the kernel had already cleared
local correctness/perf review; the boundary violation was caught only by a
maintainer or author noticing the aiter PR existed, never by anything in the diff.
Real examples: ATOM#2465 (`atom/model_ops/triton_k3_grouped_topk.py` — closed,
"superseded by the AITER-native implementation... dispatches through the public
biased_grouped_topk / topk_gating APIs, so ATOM needs no production kernel copy");
ATOM#1104 (`atom/model_ops/mamba_ops/causal_conv1d.py` — closed, "superseded by
aiter #3803... future maintenance... will follow the aiter implementation as the
source of truth"); ATOM#2510 (new `ATOM_MHC_GLUON_POST_PRE` env-var — closed,
"migrate the switch on AITER side entirely"); ATOM#1593 (author's own close: "real
fix should be in aiter side"). ATOM#987 is the same boundary one layer up (a CI
job, not a kernel): closed as "wrong repo... the intent is a downstream CI check
in ROCm/aiter."
Scoped to **new files only**: ATOM has 116 existing Triton kernel files, and a
trigger on every edit to any of them would fire on nearly every Triton PR in the
repo for no reason connected to this defect — editing an existing, already-ATOM-
resident kernel is not the thing these closures are about. The trigger is a
**new** file (not a new function in an existing file) that is both self-contained
(no call into `aiter.<fn>`/`from aiter import`) and implements a kernel primitive,
OR a new `ATOM_*` env-var that switches between ATOM-side kernel variants for a
choice aiter itself could make.
→ `⚠️ E4: new [file] implements a production kernel/dispatch with no aiter counterpart linked — confirm this doesn't duplicate (or belong in) aiter, per #2465/#1104/#2510/#1593`

---

### F — Resource Duplication
_"The change pins the same data twice without freeing the original."_

**F1 — New weight variant alongside original** ⚠️
New `w13_weight_preshuffled` / `w_quantized` stored alongside `w13_weight`: both pinned simultaneously → double HBM.
For large MoE models (w13 = 7B+), OOM only at peak load — silent until prod.
Real example (ATOM#1469): valarLip: "this will make us pin double weight."
Check: is the original freed after the new variant is created?
→ `⚠️ F1: [new_attr] alongside [original] — doubles HBM; is original freed?`

**F2 — Unbounded per-request/per-shape cache never evicted on the no-op path** ⚠️
A dict/set keyed by request id, sequence id, or tensor shape gets an entry on every call (or every success), but the matching cleanup only fires on a narrower event some calls never reach — short requests, error paths, or shapes that stop recurring leave the entry forever.
Trigger: a new `+` line adds to a dict/set (`.add(`, `.append(`, `.setdefault(`, `[key] =`) whose name suggests request/shape tracking (`awaiting`, `pending`, `in_flight`/`inflight`, `finished_*`, `*_cache`, `retained`, `deferred`), while the removal (`del`, `.discard(`, `.pop(`) is reachable only from a different, narrower branch than the one that adds the entry.
Real example (ATOM#2146): yhl-amd: "Do not retain finished IDs for requests that never issued a save — this adds every completed request to `_finished_awaiting_save`, but an ID is removed only if it later appears in `finished_saving`. Short requests with no aligned chunk... remain in this set for the lifetime of the worker."
Real example (ATOM#1316, `atom/distributed/kv_events.py`): yitingw1: replay ring buffer reuses `buffer_steps`, "adding a long-lived, count-based memory retention... retaining up to 10000 encoded payloads (including potentially large token_ids) for the lifetime of the publisher." Fixed with a dedicated, decoupled knob in `75792547`.
Real example (ATOM#1881, `atom/plugin/sglang/attention_backend/attention_gdn.py`): zovonoir: per-shape caches "can grow unbounded over time as `bs` varies" — same defect class.
→ `⚠️ F2: [tracking_structure] grows on every call but is only cleared on [narrower path] — bound it or evict unconditionally`

---

### G — Multi-Stream Synchronization
_"Written on stream A, consumed on stream B — no sync between them."_

**derive() trigger (both G1 and G1b).** Previously neither had one: confirmed by re-running
`fetch.sh` on ATOM#1369, which derived `rules=5/25` with G1 absent even though it is the
PR's own citation. The `stream-lifetime` family now fires on added lines that create or
hand off a stream, or make a cross-stream lifetime promise explicit — `cuda.Stream(`,
`.record_stream(`, `.wait_stream(`, `.set_stream(`, or `.to(..., non_blocking=True)` — and
surfaces both rule bodies together, since a diff in this territory is exactly where the
reviewer needs to check G1's sync-ordering question and G1b's record_stream-vs-wait_stream
criterion side by side. Demonstrated firing on ATOM#2654 (`f08cd7f70`, the D2H citation
below): `rules=7/27`, `[stream-lifetime] G1 G1b`. Still does not fire on #1369 itself (that
PR's own diff does not touch any of the five regex shapes above; its hazard is a bare
tensor handoff with no stream API call in the diff at all) — reported honestly rather than
re-tuned to force a match on the PR that motivated the rule.

**G1 — Missing HIP/CUDA stream synchronization** 🔴
HIP/CUDA streams execute concurrently by default. A tensor produced on stream A and consumed by a kernel on stream B without an explicit sync between them causes the consumer to read garbage — no crash, no error, silent wrong output.
Trigger: diff introduces a non-default `torch.cuda.Stream`, passes an explicit `stream=` argument to a kernel, or prepares weights/KV buffers on a side stream at model-load time that are later consumed during forward pass on the compute stream. Check: is there `stream.synchronize()`, `stream.wait_stream(other)`, `hipEventRecord` + `hipStreamWaitEvent`, or `torch.cuda.current_stream().wait_stream(other)` between the last write on stream A and the first read on stream B?
Real example (ATOM#1369, `atom/model_ops/moe.py:3515`): jpy794: "This fixes the use-after-free race for the tensor allocated in stream A and used in stream B. Without this fix the intermediate tensor could be reused by pytorch in the allocating stream before its real use by kernels in the other stream." (H2D/compute-side: allocated on one stream, consumed on another.)
Real example (ATOM#2654, `fix(mtp): keep side-stream D2H sources alive until the copy runs`): the same rule in the opposite direction — a D2H race. Commit body: "tokenIDProcessor copies the sampled ids... to the host on `async_copy_stream`. The sources are fresh default-stream allocations that the caller drops right after queuing the copy, so the caching allocator can hand their block to the next default-stream allocation before the side-stream copy has run... Under MTP the stale `num_bonus` becomes `num_accepted_tokens`, which the GDN conv update kernel adds to the conv-state offset unchecked: an illegal memory access." The same PR fixed the identical pattern a second and third time in the same commit (dspark ell D2H, output D2H) — three instances of the same mechanism before it was generalized.
A new `.record_stream(...)` call (a non-deterministic cross-stream handoff — correct only while the allocator's own polling is guaranteed to run before the tensor is freed) should be held to the same bar as a missing sync: prefer deterministic `wait_stream`/event-based sync whenever the free point is actually controllable, and expect the diff to say why `record_stream` was chosen instead. `atom/models/kimi_k3/model.py:681,689` already relies on `record_stream` for the TBO shared-expert output handoff between `alt` and `current` streams, and `atom/utils/tbo/ubatch_wrapper.py`'s `comm_stream`/`compute_stream` pair does the same — a new `record_stream` call elsewhere should explain why `wait_stream` could not be used. (pytorch `.agents/skills/cuda-streams/SKILL.md`, borrowed for this sharper criterion — no independent ATOM citation for the `record_stream`-vs-`wait_stream` choice specifically.)
→ `🔴 G1: [tensor] written on [stream A] consumed on [stream B] without sync — add stream.wait_stream() or hipStreamWaitEvent`

**G1b — Blocking queue.get() without timeout in production serving code** ⚠️
`queue.get()` without `timeout=` in a worker or service thread that depends on an external producer (decode loop, stream consumer, request handler). If the producer exits abnormally, the worker blocks forever — no crash, no log, hung process.
Trigger: `queue.get()` or `asyncio.Queue.get()` inside a `while True:` worker loop in production serving paths (entrypoints, engine loop, scheduler) without `timeout=` and without a corresponding `except queue.Empty` / `asyncio.TimeoutError` handler or a `done` flag.
Exception: test code, CLI tools, or one-shot scripts where a hang is detectable (CI timeout, interactive TTY).
Real example (ATOM#789): `loop.call_soon_threadsafe` routes incremental detokenizer state mutations to the event-loop thread — this is correct and is NOT a G1b violation.
→ `⚠️ G1b: [worker] blocks on queue.get() without timeout — add timeout= and handle Empty/TimeoutError to survive producer failure`

**G2 — Blocking CPU/GPU synchronization added to the decode loop** ⚠️
A new `.item()`, `.tolist()`, boolean-tensor indexing, or blocking `.to("cpu")` (no `non_blocking=True`) inside the model runner's forward path or a model's `forward()` forces the CPU to wait for the GPU queue to drain at that point, on every call. ATOM has no `VLLM_GPU_SYNC_CHECK`-equivalent env var (verified: no such flag in `atom/utils/envs.py`), so there is no automated gate for it; the runnable check is `torch.cuda.set_sync_debug_mode('error')` around the suspect region, which raises instead of silently stalling.
Trigger: a new `.item()`, `.tolist()`, or `.to("cpu")` without `non_blocking=True` added inside `atom/model_engine/model_runner.py` or a model's `forward()`.
Grounded in: `model_runner.py`'s `tokenIDProcessor` class (`send_to_cpu_async`/`recv_async_output`, ~line 206-298) already hand-manages `non_blocking=True` transfers for logprobs and rejected/bonus-token counts specifically to avoid forcing a sync — the hazard is real and already worked around ad hoc for existing code; a new unguarded `.item()`/`.tolist()` elsewhere in the same hot path defeats that work.
Source: vLLM `.agents/skills/pr-checklist/SKILL.md` §1.5 (sync-check); detection method from TensorRT-LLM `.claude/skills/perf-torch-sync-free/SKILL.md` (`torch.cuda.set_sync_debug_mode`). Borrowed, not ATOM-native — no ATOM PR citation yet for this specific rule.
→ `⚠️ G2: [call] forces a CPU/GPU sync in [function] — confirm with torch.cuda.set_sync_debug_mode('error'), or move it off the hot path`

---

### H — Batch Object Invariants
_"`ScheduledBatch` is a snapshot, not a scratchpad."_

**H1 — ScheduledBatch mutated in place after capture** 🔴
`atom/model_engine/scheduler.py` docstrings `ScheduledBatch` as an "Immutable snapshot of sequences selected for a single forward pass," and `model_runner.py` holds `self.prev_batch: ScheduledBatch | None` across steps so a later step can inspect what an earlier one actually ran. An in-place mutation after capture (`.append()`/`.extend()`/`+=`/`-=`/`.add_()`/`.fill_()`/slice-assignment on a `batch.*` or `prev_batch.*` field) corrupts what `prev_batch` thinks it holds instead of producing a fresh value — the same failure mode sglang's own `ScheduledBatch` equivalent hit before it adopted an out-of-place-mutation rule.
Trigger: a diff calls `.append(`, `.extend(`, `.add_(`, `.fill_(`, or slice/index-assigns a field of a `batch` or `prev_batch` object, or does `batch.<field> +=`/`-=`, instead of rebinding to a new value.
Source: sglang `.claude/rules/schedule-batch-out-of-place-mutation.md`, adapted to ATOM's `ScheduledBatch`. Borrowed, not ATOM-native — no ATOM PR citation yet.
→ `🔴 H1: [field] mutated in place on [batch/prev_batch] — rebind to a new value instead of mutating the captured snapshot`

**H2 — Batch-consuming entry points must treat ScheduledBatch as read-only** ⚠️
Pairs with H1: entry points that consume a `ScheduledBatch` (`prepare_model`, `get_token_locations`, and similar methods in `atom/model_engine/model_runner.py`) are expected to read from `batch` and return/thread a new value, not write back into the `batch` object they were handed — the same "init must stay pure" discipline sglang's `forward-batch-init-new-purity` rule encodes for its own forward-batch construction.
Trigger: a diff inside a batch-consuming method of `atom/model_engine/model_runner.py` (signature takes a `batch: ScheduledBatch` parameter) writes to a `batch.<field> = ...` field instead of returning a new value.
Source: sglang `.claude/rules/forward-batch-init-new-purity.md`, adapted to ATOM's `prepare_model`/`get_token_locations`. Borrowed, not ATOM-native — no ATOM PR citation yet.
→ `⚠️ H2: [method] writes back into batch.[field] instead of returning a new value — keep batch-consuming entry points read-only`

---

### T — Triton Kernel Review

Ported from aiter's own `.claude/skills/review-pr/rules.md`, graded individually rather
than flattened: aiter's own T1-T9 (plus its B2, which duplicates T1 under a second id) are
not all equally evidenced on ATOM, and this section says so per rule instead of porting
the whole family at the same confidence. T2 (mask with no `other=`) and T4 (AMD launch
knobs — `waves_per_eu`/`matrix_instr_nonkdim`/`kpack` — hardcoded ACROSS archs) are **not
ported**: T2 was never checked against ATOM (absence of a citation here is "not checked",
not "clean"), and T4 was checked and came back with zero hits for those three names in the
directories sampled for this assessment — "not found in sampled directories", not "absent
from ATOM". Neither claim should be read as more than it is.

**T1 — Triton load/store without a bounds mask** ⚠️
aiter numbers this defect twice (its own T1 and B2 are the same check); ATOM already has
an unrelated B2 ("decode/prefill phase classified by size proxy"), so only "T1" is added
here to avoid an id collision. This is a **regression guard on currently-correct code**,
not a response to a known ATOM defect: `atom/model_ops/fla_ops/chunk_delta_h.py`'s tile
loads already carry `mask=`/`other=` throughout, and the handful of unmasked `tl.load`/
`tl.store` calls in that file are safe by construction — scalar, single-index loads with
no tile dimension to run past. The rule exists so a future edit that turns one of those
scalar loads into a tiled one doesn't lose the mask silently.
Trigger: `tl.load(`/`tl.store(` on a `+` line with no `mask=` kwarg, or a `tl.load(` with
`mask=` but no `other=`.
→ `⚠️ T1: [call] has no bounds mask ([or: no other= fallback]) — confirm the index range cannot exceed the tensor, or add mask=/other=`

**T3 — Launch config (`num_warps`/`num_stages`) carried over unretuned** ⚠️
ATOM evidence: `atom/model_ops/v4_kernels/paged_decode.py`'s `_kernel_config()` (lines
105-133) picks `num_warps`/`num_stages` from an in-source heuristic ("without autotune",
per its own docstring) based on `block_h`, and every call site that launches off it
(lines 712, 787-788, 846-847) inherits that heuristic's choice rather than a per-shape
tuned value. That is exactly T3's shape: a launch knob set once by reasoning about one
shape family and carried forward, with no autotune sweep re-deriving it as new shapes
(block sizes, head counts) are added.
Trigger: `num_warps=`/`num_stages=`/`num_ctas=` set in added lines, inside a file that
also defines or calls a Triton kernel.
→ `⚠️ T3: num_warps/num_stages set from [heuristic/literal] — confirm this was re-tuned for the new shape, not carried over from the shape it was picked for`

**T8 — Launch knob hardcoded in kernel source instead of the config system** ⚠️
ATOM evidence, same file: `paged_decode.py:896` hardcodes `num_warps=4` directly at a
Triton kernel call site, bypassing `_kernel_config()`'s own heuristic entirely for that
one launch. A literal at a call site is invisible to whatever tuning process the config
system represents — it will not be swept, and a later retune of the heuristic silently
stops applying to this one call.
Trigger: `hardcoded_launch_knob()` — a `waves_per_eu=`/`matrix_instr_nonkdim=`/`kpack=`/
`num_stages=`/`num_warps=`/`num_ctas=` literal (not sourced from `config`/`cfg`/`tune`/
`kwargs`/an uppercase constant) on an added line outside `configs/`, `op_tests/`, `tests/`.
→ `⚠️ T8: [knob]=[value] hardcoded at [location] instead of read from config — move it to the tuned config, or justify why this call site is exempt`

**T5 — Accumulator precision loses tail-of-sum bits** ⚠️
Carried from aiter **without an ATOM citation** — labeled as such, same as aiter marks
its own uncited rules, rather than silently presented as proven. The trigger shape (a
`tl.dot(`/`allow_tf32`/`input_precision` call, or an `acc += `/`acc = tl.zeros` manual
accumulator) fires throughout `fla_ops` and `v4_kernels`, but no ATOM defect commit was
found demonstrating this specific failure (an fp16/bf16 accumulator silently losing the
tail of a long reduction sum) during this assessment.
Trigger: `tl.dot(`, `allow_tf32`, `input_precision`, or a manual `acc\w* +=`/`acc\w* =
tl.zeros` accumulator pattern on an added line.
→ `⚠️ T5: accumulator [name] — confirm dtype is fp32 (or confirm the tail-of-sum loss at this length/dtype is acceptable) — no ATOM instance found for this specific rule, carried from aiter`

**T6 — Grid / `program_id` axis disagreement** ⚠️
Carried from aiter **without an ATOM citation**, same caveat as T5: the trigger shape
(`grid = lambda` or `tl.program_id`) fires throughout `fla_ops`/`v4_kernels`, but no ATOM
defect commit was found for this specific failure (a launch grid's axis order disagreeing
with the `tl.program_id(axis)` the kernel reads it back with — including a swizzled grid
where the swizzle axis and the `program_id` axis it's applied to don't agree).
Trigger: `grid = lambda` or `tl.program_id` on an added line.
→ `⚠️ T6: program_id(axis=[n]) — confirm this matches the grid's own axis order (and the swizzle axis, if the grid is swizzled) — no ATOM instance found for this specific rule, carried from aiter`

---

### Performance Evidence

Not derived by the current pipeline -- P1-P5 are in `UNREACHABLE_BY_DESIGN`
(see `triage.py` and `MAPPING.md`): the P0 minimal deriver does not emit this
family, so nothing puts these on `$WORK/rules.txt` today. "Always check"
described SKILL.md's pre-pipeline prose checklist, not what the deriver
does; kept here as the rule bodies P1 regrows the family from.

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
Beyond TP: `atom/config.py` also declares `pp_size`, `data_parallel_size`, and `decode_context_parallel_size` (`dcp_size`) alongside `tp_size` and MoE `ep_size`. The same per-device shape-math hazard P4 checks for under a TP split applies equally under a PP stage boundary, a DP batch split, DCP, or an EP expert split — a new shape-dependent kernel/dispatch path tested only at one setting of any of these dimensions (not just TP) should trigger P4, not only the TP=1-vs-TP=N case. (TensorRT-LLM `.claude/skills/trtllm-code-contribution/SKILL.md`'s "Check Parallelism Dimensions" table, borrowed for this broader framing — no new ATOM citation beyond P4's existing evidence.)
→ `⚠️ P4: test covers only TP=1 head count — verify at num_heads÷TP=4 (e.g., [128→32]); also check PP/DP/DCP/EP splits if the new path is shape-dependent on those`

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


## Refutation

Step 7.6 attacks each finding before it is reported. What to attack, in the order that has
actually killed things in the sibling aiter review skill's history (ATOM has not yet run this
skill long enough to have its own refutation incidents on record — these are borrowed, not
fabricated as ATOM's own):

1. **The premise.** aiter's sibling skill once reported most rows of a tuned-config table as
   unreachable, marked `[verified]`, reasoning about a padding chain — but the real lookup
   function tried the exact value first, so the rows were reachable and the finding was false.
   Ten seconds in the lookup function would have ended it. Read the code the claim depends on,
   not the code the claim is about.
2. **The tree you read.** A local worktree is not the merge target. Anchor line numbers with
   `git show <base>:<path>`; `base_head.txt` has the sha.
3. **Reachability.** Can the trigger you named actually occur, or does something upstream
   already prevent it — a caller that never passes that value, a guard added since, an arch
   gate?
4. **Suspicion versus defect.** "This looks wrong" is not a finding. If you cannot name
   inputs that produce a wrong result, it is a question, and 📝 is where questions go.
5. **Severity.** A 🔴 asserts a wrong result or a crash. Without a reachable triggering case
   it is ⚠️ at most.

Write one line per attempt to `$WORK/refutations.txt`:
`RED|WARN|NOTE SURVIVED|KILLED -- what you opened and what it said`. Name a file, a symbol
or a command. "I checked" is rejected by the gate, because a refutation nobody can repeat is
not one.

**This is the weaker of the two tiers, and aiter's numbers say how much weaker.** Across 200
PRs in four waves on the aiter sibling skill, Step 7.6 killed 260–280 candidates per 50 PRs
before they reached a card -- and 28% of what survived it was still killed by an independent
reader told the findings were false until defended. Refuting your own finding catches a
premise you never tested. It does not catch a conclusion you are committed to, and
instructing yourself more firmly is not a substitute for a reader who was never committed.
ATOM has not measured this on its own review history yet; treat the numbers above as aiter's
until ATOM has its own.

That is why Step 7.7 is a step and not a suggestion. It needs no ability to spawn anything:
hand over the card, the diff and the merge-target path with none of your reasoning attached,
and record what comes back. When no such reader exists, say so on the card -- a review that
had one and a review that did not are different objects, and a reader of the card cannot
tell them apart unless it is written down.

**A caveat that belongs next to that 28%:** it is the rate at which an adversary instructed
to assume falsity judged a finding dead. That adversary has not itself been audited, so the
number is an upper bound on the error rate, not a measurement of it. It says nothing about
false *clearance* -- reporting nothing when something is wrong. That number has never been
measured here at all.
