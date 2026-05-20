---
name: dump-bisect-debug
description: Locate forward numerical bugs by dumping intermediate tensors from a target implementation and a known-good reference, then bisecting layer by layer. Also covers batch-invariance bisect (the same token at any batch position should produce a bitwise-identical output, per DeepSeek V4 paper §3.3). Use when "the output is wrong but I don't know where" — model produces gibberish, degenerates, or picks the wrong token, but code review reveals nothing.
version: 3.0.0
scope: ATOM (generalizable to any PyTorch forward debug)
last_updated: 2026-04-30  # add Phase 8 batch invariance + atom/utils/debug_helper infra
---

## ATOM built-in infrastructure (v3.0+ — read first)

**Use these instead of hand-rolling hooks / compare scripts.**

| Module | Purpose |
|--------|---------|
| `atom/utils/debug_helper/dump.py` | Env-gated forward / weight / sampler dump; multi-class hooks; multi-call counter. |
| `atom/utils/debug_helper/compare.py` | `cos_max` (double-precision — avoids the fp32 `cos > 1.0` trap) + `slot_split` + `pick_prefill_call` + CLI. |
| `atom/utils/debug_helper/ref_patch.py` | Monkey-patch context managers for instrumenting read-only reference implementations. |
| `atom/utils/envs.py` "Debug Dump" section | 9 env vars (`ATOM_FWD_DUMP_*`, `ATOM_WEIGHT_DUMP_*`, `ATOM_DEBUG_TOPK*`), all no-op by default. |

Public API: `from atom.utils.debug_helper import ...`. CLI: `python -m atom.utils.debug_helper.compare <subcommand>`.

Once these are wired up, debugging a new model never requires editing `model_runner.py`, writing a `cos_max` again, or re-inventing the warmup-vs-prefill heuristic.


# Dump-Bisect Debug Methodology

> Looking at logits or output text only tells you *that* it's wrong, not *where* it's wrong.
> This methodology compresses *where* down to O(log N) dump-and-compare iterations.

## Quick-start decision tree

```
Is the symptom reproducible at temperature=0.0 / single prompt?
├─ NO  → batch / sampling-stochastic
│   └─ Same prompt yields different tokens across runs OR across batch slots?
│       ├─ across runs    → you ALREADY BROKE determinism — fix sampling seed or env
│       └─ across slots   → likely NOT a bug; jump to Phase 8 (batch invariance)
└─ YES → deterministic forward bug, run the linear pipeline:
    ├─ Have a reference impl?  ──NO──► build one (Phase 0)
    ├─ Schema/shape mismatch?  ──YES─► fix loader (Phase 1, do NOT skip)
    ├─ Single layer dump cos<1? ───── Phase 3 → Phase 4 → Phase 5
    └─ Pinpoint sub-stage cos drop ── Phase 6 (standalone GPU isolation)
                                   └─ Phase 7 (fix + isolation revert)
```

## Phase-at-a-glance

| # | Goal | Time | Move on when |
|---|------|------|--------------|
| 0 | Establish reference | once per project | reference reproduces user-expected output end-to-end |
| 1 | Rule out weight loading | ≤ 1h, 1–2 GPU runs | all dumped params byte-equal or cos > 0.9999 |
| 2 | Define dump protocol | once per model | both sides agree on stage names + tensor contract |
| 3 | First single-layer comparison | ≤ 30min, 2 GPU runs | layer 0 cos table drawn |
| 4 | Layer-level bisect | 1–2h | first layer with cos < 0.99 (or rel > 10%) located |
| 5 | Intra-layer sub-stage bisect | 1–3h per bug | sub-stage with single-step cos drop > 0.001 located |
| 6 | Standalone GPU kernel isolation | 1–2h | path A reproduces ATOM dump cos > 0.9999; cross-experiment confirms quant vs GEMM |
| 7 | Fix + isolation revert | ~1h | e2e byte-equal vs ref AND each fix's necessity verified |
| 8 | Batch invariance bisect (parallel mode) | 2–3h | slot-vs-slot cos vs spec — if broken, classify as kernel-stack issue, NOT model bug |

Total per root cause: 4–6h. Multiple root causes stack. **5–10× faster than "stare at the code and guess"** with no missed diagnoses.

## When to use

**Trigger conditions (any one applies)**:
- Model output is gibberish, degenerates, or picks the wrong token, but code review finds nothing.
- Output is correct on some prompts and wrong on others ("corner case").
- Single prompt OK / batch fails; prefill OK / decode fails; short prompt OK / long prompt fails.
- A correct reference implementation exists (HF transformers, official inference repo, a previously-verified commit).

**Don't use when**:
- No reference ground truth (only "I think it's wrong") — build a reference first.
- Difference of just a few tokens — could be numerical noise; confirm it's a real bug first.
- Symptom is "non-deterministic across runs at temp=0.0" — that's broken sampling determinism (env / RNG / kernel), not a forward bug; this skill won't help.

## Core principles

1. **Read the model paper before assuming a bug.** Some divergence is *expected by design* of the model architecture, not a bug to fix. V4 paper §3.3 explicitly assumes batch-invariant kernels; running on a non-batch-invariant inference stack will flip edge-confidence tokens and that is **not** an ATOM bug. Always check the model's reproducibility / determinism claims first — if the runtime stack doesn't meet them, classify as a kernel-stack limitation and document, don't bisect.
2. **The reference must actually run end-to-end**, not "the code looks right" — references can have bugs too.
3. **Reference may be batch=1 only.** Many official `inference/model.py` files (V4, parts of DeepSeek family) hardcode `max_batch_size=1`. You cannot directly compare batch>1 against them. Either modify the ref to support batches OR design experiments that work within the bsz=1 constraint (e.g. multiple seeded runs).
4. **Share as much code as possible**: reference and target use the same tokenizer / kernel / inputs, so the only variable is the code under investigation.
5. **Rule out weight loading first**: confirm byte-equality before any forward bisect. Schema diff (names + shapes + dtypes) BEFORE numerical comparison — a missing param prefix masquerades as a "forward bug".
6. **Fix isolation revert is mandatory after multi-bug fixes.** When 2+ fixes land together, revert each one in turn to identify which are *critical* (output-changing) vs *fine-tuning* (cosmetic / perf-cost). This decides PR split granularity — critical fixes merge fast, fine-tuning fixes can wait.
7. **Cross-check against a second reference**: besides the canonical reference, look at how sglang / vLLM implemented and fixed the same model.
8. **Dump names must be semantic**: `intra_attn_norm_in` / `intra_ffn_out`, never `tensor_5`.
9. **Dump one layer / one prompt at a time**: avoids file explosion and cross-contamination.
10. **Align `input_ids` first**: tokenizer mismatch is the most common false-positive bug source.
11. **Same stage name ≠ numerically equivalent**: confirm both sides have applied the same number of ops at dump time. The most common trap: ATOM dumps pre-all-reduce, ref dumps post-all-reduce.

## Eight-phase flow

Phases 0–7 are the linear pipeline for ref-vs-target bisect. Phase 8 is a parallel mode for batch-invariance investigation; trigger it independently when the symptom is "single prompt OK, batch fails".

### Phase 0: Establish a reference

If no reference exists yet, build one. Priority:
1. **The official repo's `inference/generate.py`**: run with `torchrun` (e.g. `/data/DeepSeek-V4-Pro/inference/generate.py`).
2. **HF transformers**: `AutoModelForCausalLM.from_pretrained(...).generate()`.
3. **A known-good prior commit**: `git checkout <commit>` and run.

Reference requirements:
- Same weights (stream from safetensors directly; do not convert).
- Same tokenizer / chat template.
- **Same GPU kernel** when investigating numerical drift — use the same aiter / cuBLAS kernels in the reference, otherwise you can't distinguish "algorithm bug" from "kernel numerical drift".
- End-to-end verified: the reference output must match user expectation (e.g. `"1+2+3=?"` answers `"6"`).

**Output**: `ref_full_generate.py` or similar — capable of producing a ground-truth token sequence.

### Phase 1: Rule out weight loading (before any forward bisect)

**Why first**: if weights load wrong, every subsequent forward comparison will mis-attribute the cause to the forward path. 10 minutes of weight comparison saves a day of forward bisect.

Use `maybe_dump_weights_and_exit(self.model)` from `atom.utils.debug_helper`, already wired in `model_runner.py`. It dumps params + buffers per rank and `sys.exit(0)`:

```bash
ATOM_WEIGHT_DUMP_DIR=/path/to/dump \
ATOM_WEIGHT_DUMP_LAYERS=0,2 \
python -m atom.examples.simple_inference --model ... -tp 8
```

**Comparison checklist**:

| Check type | What to do |
|------------|-----------|
| Schema diff first | List ATOM-only / ref-only / shape mismatch / dtype diff **before any numerical comparison**. |
| FP8 weight | Compare byte-equality after `aiter.ops.shuffle.shuffle_weight(ref_w, layout=(16,16))`. |
| FP8 scale (e8m0) | Cast via `aiter.utility.fp4_utils.e8m0_to_f32(ref_s)` to fp32, then compare. |
| TP-replicated layer | Per-rank ATOM byte-equal vs ref. |
| TP-sharded layer | `torch.cat(ref_rank0..7, dim=tp_dim)` vs full ATOM (or per-rank ATOM vs ref slice + shuffle). |
| Norm weight (BF16 vs FP32) | Different dtype, equal value — cast then `cos`. |
| MoE expert weights | Volume is huge; skip initially — assume the expert loader is consistent with other weights. |

**Conclusion patterns**:
- ✓ All byte-equal / cos > 0.9999 → weight loading is OK; proceed to Phase 2.
- ✗ Any byte mismatch → fix the loader first (WeightsMapper / shuffle / TP shard / quant_type).

### Phase 2: Define the forward dump protocol

Both sides agree on the same **checkpoint names** and **tensor shape contract**.

**Minimum set** (dump per layer):

| Stage | Meaning | Use |
|-------|---------|-----|
| `embed.input_ids` | Input token ids | Confirm tokenization consistency |
| `embed.embed_out` | Embedding output | Confirm lookup consistency |
| `layer{L}.hidden_in` | Hidden entering the layer | Confirm previous layer's output |
| `layer{L}.attn_norm_out` | After attention norm | Isolate norm differences |
| `layer{L}.attn_out` | Attention output | Whole attention block |
| `layer{L}.ffn_norm_out` | After FFN norm | Isolate norm |
| `layer{L}.ffn_out` | FFN output | Whole FFN block |
| `layer{L}.hidden_out` | Hidden leaving the layer | Feeds the next layer's comparison |
| `embed.final_h` | Pre-`lm_head` hidden | Accumulated diff vs `lm_head` amplification |
| `embed.final_logits` | Final logits | Vocab-space difference |

**Use the built-in dump infrastructure** (do not write hooks by hand):

```bash
# ATOM side
ATOM_FWD_DUMP_DIR=/path/to/dump \
ATOM_FWD_DUMP_LAYERS=0 \
ATOM_FWD_DUMP_BLOCK_CLASS=Block \
python -m atom.examples.simple_inference --prompt "1+2+3=?" --max-tokens 1
```

For the **reference side** (often a read-only `/data/<model>/inference/model.py`):

```python
from atom.utils.debug_helper import patch_block_forward, patch_module_dump

with patch_block_forward(ref_Block, layer_attr="layer_id", side_prefix="ref"):
    ref_model.forward(...)
    # writes ref_layer{LL}_Block__{stage}_rank{R}.pt to ATOM_FWD_DUMP_DIR
```

For deeper sub-stages (RoPE, q_norm, sparse_attn output …), patch the relevant method directly with `patch_method` and insert named `dump(stage, tensor)` calls — copy the original `forward` body verbatim and insert the dumps in between to avoid losing side effects.

### Phase 3: First comparison (single layer, single prompt)

```bash
# Reference
ATOM_FWD_DUMP_DIR=$DIR ATOM_FWD_DUMP_LAYERS=0 \
  torchrun --nproc-per-node=8 ref_full_generate.py --prompt "1+2+3=?" --max-new-tokens 1

# Target (ATOM)
ATOM_FWD_DUMP_DIR=$DIR ATOM_FWD_DUMP_LAYERS=0 \
  python -m atom.examples.simple_inference --prompt "1+2+3=?" --max-tokens 1

# Compare
python -m atom.utils.debug_helper.compare ref-vs-target --dir $DIR
```

The CLI uses the project's standard `cos_max` (double precision). It prints a per-stage table with severity flags and asserts `input_ids` match before doing anything else.

**Severity thresholds** (look at both `cos` and `rel`; `rel` is more sensitive):

| cos | rel | Meaning | Action |
|-----|-----|---------|--------|
| `> 0.9999` | `< 1%` | Bit-equal range | ✓ OK |
| `0.99 ~ 0.9999` | `1 ~ 10%` | Numerical drift (kernel/dtype) | ? Flag — watch for accumulation |
| `0.9 ~ 0.99` | `10 ~ 30%` | Mild algorithmic drift / partial heads wrong | ✗ Bisect to sub-stage |
| `< 0.9` | `> 30%` | Real bug | ✗ Locate immediately |
| `≈ 0` or negative | `> 50%` | Total scramble / sign flip | ✗ Usually weight loading / shuffle bug |

**Important: when `cos` and `rel` disagree, trust `rel`**:
- When hidden values span large ranges (e.g. `max_abs = 1e5`), `cos` is dominated by a few outliers and may read 0.9998 even though `rel = 57%`.
- A 50%+ per-element error → after `lm_head` amplification, logits are completely wrong.

### Phase 4: Layer-level bisect — find the first cos drop, with layer-class awareness

The first layer with `cos < 0.99` or `rel > 10%` = the layer where the bug lives.

**Key observation: layer class.** When dumping multiple layers (0 / N / 2N / 3N) to look at the decay curve, **focus on which layer's cos suddenly drops** and correlate with that layer's *class*:

| Model | Layer-class examples | Investigation direction |
|-------|---------------------|-------------------------|
| DeepSeek-V4 | First N layers use hash routing; rest use sqrtsoftplus routing | Did the non-hash path miss a fix? |
| DeepSeek-V4 | `compress_ratio=4` (sparse) vs `=128` (window) | Inside the sparse path? |
| Qwen3-Next | Hybrid attention vs full attention | Attention-type branch? |
| MTP | Base layer vs MTP block | MTP path needs an independent fix? |

**V4 example**: layer 0/2 `hidden_out` cos = 1.0 ✓, but **layer 3 suddenly drops to cos = 0.98**. Layers 0/1/2 are hash routing (layer_id < n_hash_layers); layer 3+ takes `select_experts(sqrtsoftplus)`.
→ Compare the hash path vs the sqrtsoftplus path → discover the latter is missing `* routed_scaling_factor`.

**Accumulated drift vs algorithmic bug**:
- Each layer cos = 0.999 but compounded to layer 60 it's 0.94 → kernel drift.
- One layer suddenly drops from 0.999 to 0.5 → algorithmic bug.
- Some *type* of layer (e.g. layer 3, 5, 7, …) consistently has poor cos while others are fine → layer-class branch bug.

### Phase 5: Intra-layer sub-stage bisect → component-level root cause

Once the layer is located, dump finer checkpoints inside it. **Each arrow = one dump**:

```
Attention:
  x_in → wq_a → q_norm → qr → wq_b → q_pre_norm → q_post_norm → q_post_rope → ┐
                                                                              ├ → sparse_attn → o_pre_invrope → o_post_invrope → wo_a → wo_b
  x_in → wkv → kv_pre_norm → kv_post_norm → kv_post_rope → kv_after_quant → ┘

FFN (MoE):
  x_in → ffn_norm → norm_out → gate(routing) → topk → ┐
                                                       ├ → expert_out → all_reduce → + shared_out → combined_out
                                                       ┘
```

Each sub-stage's single-step cos should be `> 0.999`. If a sub-stage **drops by more than 0.001 in a single step**, that op is the culprit.

**Common MoE trap: dump stages out of sync**
- ATOM `expert_out` is pre-all-reduce (per-rank partial).
- Ref `expert_out` is post-all-reduce (full).
- The two will read cos = 0.36, looking like a bug — but **the dump stage isn't aligned**.
- Fix: move ATOM's dump after the all-reduce, OR move ref's dump before — pick one, keep both sides aligned.

### Phase 6: Standalone GPU kernel isolation (key innovation)

Is `cos = 0.998` from input quantization noise or a GEMM kernel difference? **Re-run the kernel offline using dumped data** to isolate:

```python
# standalone_<op>_test.py — runs on a single GPU
qr = atom_dump["layer0.attn.qr"].cuda()                  # input
atom_w = atom_weight_dump["layers.0.attn.wq_b.weight"]   # ATOM-loaded weight (shuffled)
ref_w = ref_weight_dump["layers.0.attn.wq_b.weight"]     # ref-loaded weight (unshuffled)

# Path A: ATOM kernel = ATOM input quant + ATOM GEMM
qr_q_A, qr_s_A = atom_quant_func(qr)
out_A = atom_gemm(qr_q_A, atom_w, qr_s_A, atom_w_scale)

# Path B: REF kernel = ref input quant + ref GEMM
qr_q_B, qr_s_B = ref_quant_func(qr)
out_B = ref_gemm(qr_q_B, ref_w, qr_s_B, ref_w_scale)

# Sanity: A must perfectly reproduce the ATOM dump
assert cos_max(out_A, atom_dump["layer0.attn.q_pre_rope_pre_norm"])[0] > 0.9999

# Cross experiments isolate quant vs GEMM:
# C1: ref quant + ATOM GEMM   → measure quant impact
# C2: ATOM quant + ref GEMM   → measure GEMM impact
out_C1 = atom_gemm(qr_q_B, atom_w, qr_s_B, atom_w_scale)  # may have layout incompat
out_C2 = ref_gemm(qr_q_A, ref_w, qr_s_A, ref_w_scale)
print(cos_max(out_C1, ref_dump))
print(cos_max(out_C2, ref_dump))
```

**Validating a fix candidate**:
- Write a PyTorch reference quant function (e.g. `my_ue8m0_quant`).
- Confirm its FP8 / scale outputs are byte-equal vs the ref kernel (100%).
- Feed into the ref GEMM, confirm cos vs ref dump = 1.0.
- This step **doesn't need the model** — takes seconds to confirm the fix direction.

### Phase 7: Fix implementation + isolation revert

1. **Implement the fix env-gated** when its correctness is uncertain or it carries a perf cost: e.g. `V4_USE_REF_QUANT=1` toggles a PoC fix without changing default behavior. For high-confidence fixes with no perf cost, land them as the new default.
2. **Re-run the forward dump and verify cos**: confirm the fix is real and the sub-stage cos actually improved.
3. **Run e2e and look at token output**: from rambling → coherent output → byte-equal vs ref.
4. **Multiple bugs may need fixing**: one fix is often not enough (e.g. fixing V4 attention still leaves FFN wrong) → continue bisecting the next one.
5. **Fix isolation revert** ★ Once all fixes pass, **revert each one in turn** to identify the critical ones (this is core principle #6):
   - V4 case: reverting `linear.py` (Bug 8) showed Bug 9+10 alone were enough to output `"6"`.
   - Bug 8 was just fine-tuning (byte-equal vs ref) and had a perf cost.
   - This step decides the PR split granularity (critical fixes merge immediately; fine-tuning fixes can wait).

### Phase 8: Batch-invariance bisect (DeepSeek V4 paper §3.3)

**Trigger conditions**:
- Single prompt is correct; batch=N produces a wrong answer in some slot.
- The same prompt run multiple times yields different results (determinism is broken).
- The model paper / design assumes "bitwise reproducibility across pre-training, post-training, inference" (V4, Qwen3+).

**Core design** (V4 paper §3.3):

> "Batch invariance ensures that the output of any given token remains
> bitwise identical, regardless of its position within a batch."

The V4 team built three kinds of batch-invariant kernel for this:
- Attention: dual-kernel (single-SM full-seq + multi-SM single-seq, bitwise-identical).
- MatMul: replace cuBLAS with DeepGEMM, abandon split-K.
- MoE backward: token-order pre-processing + per-rank buffer isolation.

Running on a non-batch-invariant inference stack (aiter / cuBLAS / default vLLM) → slot-to-slot cos < 1.0 → edge-confidence prompts flip tokens. **This is not a bug; it's a violation of the model's architectural assumption.**

**Experiment design** (4 identical prompts + temp = 0.0):

```python
EXPERIMENTS = [
    ("E0_single",   [P]),                              # baseline
    ("E1_4xsame",   [P, P, P, P]),                     # all same → tests batch invariance
    ("E2_pos0",     [P, OTHER, OTHER, OTHER]),
    ("E3_pos1",     [OTHER, P, OTHER, OTHER]),
    ("E4_pos2",     [OTHER, OTHER, P, OTHER]),
    ("E5_pos3",     [OTHER, OTHER, OTHER, P]),
]
# Single model load; sequentially run all batches and look at slot-vs-slot output.
```

**Reading the results**:
- E1's four slots all agree → batch invariance is OK.
- E1 has divergent slots → batch invariance is broken (a kernel's reduction order depends on batch position).
- E2–E5 results vary with P's position → non-commutative path.

**Dump + bisect recipe**:

```bash
# 1. Run with multi-class hooks + multi-call (Compressor / Indexer live in a per-seq loop)
ATOM_FWD_DUMP_DIR=$DIR \
ATOM_FWD_DUMP_LAYERS="0,1,2,3" \
ATOM_FWD_DUMP_BLOCK_CLASS="Block,DeepseekV4Attention,MoE,Compressor,Indexer" \
ATOM_FWD_DUMP_ONE_SHOT=0 \
python debug_e1_4xsame.py

# 2. Compare per-slot
python -m atom.utils.debug_helper.compare slot-invariance --dir $DIR --n-slots 4
# Output: per (layer, stage) cos(slot 0, slot N) — find the first stage where it drops below 1.0.
```

**Critical trap: warmup vs prefill vs decode**

A forward hook with `one_shot=True` (the default) captures the **warmup forward** (`max_num_batched_tokens`, typically a power-of-2 dummy input), **not the real prefill**.

Set `ATOM_FWD_DUMP_ONE_SHOT=0` so each call writes its own file (`layer{LL}_{Cls}_rank{R}_call{NNN}.pt`), then use `pick_prefill_call()` to pick the real prefill:
- `n_tok % n_slots == 0` and `n_tok > n_slots` (excludes 1-tok-per-seq decode).
- Not a power-of-2 (excludes warmup dummies).
- Lowest `call_idx` among matches (earliest match).

**Per-seq dispatch loop trap (specific to V4 Attention)**:
- `Compressor` / `Indexer` are called inside a `for seq in range(num_seqs)` loop.
- A single prefill triggers N module calls, one per slot.
- `one_shot=True` only captures seq 0 — slot drift is invisible.
- **Required**: `ATOM_FWD_DUMP_ONE_SHOT=0` + use `call_idx` to disambiguate.

**Root-cause classification**:

| Stage cos drop | Candidates |
|----------------|-----------|
| Layer 0 attention `cos < 1.0` | `sparse_attn` split-KV / per-slot KV-cache memory layout. |
| Layer 0 MoE drops more than attn | Fused expert kernel split-K / silu+clamp boundary amplifies 1e-4 → 1e-3. |
| Layer 1+ attn cos explodes (0.97) | Hash routing / sqrtsoftplus cascades small upstream drift. |
| Whole model cos ≈ 0.999 yet final logits diverge | Edge-confidence token + sampling noise. |

**Example** (Bug 11, V4 batch=4 prompt P3 answers `"15"`):
- Single prompt: 100% answers `"6"`. Batch=4: 40% flip to `"15"`.
- E1 4×P3 + temp=0.0: slot 0/1/3 → `"6"`, slot 2 → `"1+2+"` (different).
- Layer 0 attn cos 0.99988 (near identical) → MoE 0.998 (10× amplification) → layer 1 attn 0.97 (cascaded).
- Full write-up: `/app/logs_claude/deepseek_v4/notes/21_bug11_isolation.md`.

**Fix direction**:
- Cannot be fixed at the ATOM layer — needs aiter to ship batch-invariant kernels (DeepGEMM-style + dual-kernel attention).
- Short term: document as a known limitation; only the single-prompt and high-confidence batch paths are guaranteed correct.

## Common root-cause categories (by frequency)

1. **Weight loading bug**: shape correct, content wrong (cos near 0 or negative).
   - WeightsMapper mismatch.
   - Wrong TP shard direction (ColumnParallel vs RowParallel vs Replicated).
   - Wrong FP8 shuffle (`quant_type`).
2. **Quant scale path differences**: each GEMM cos ≈ 0.998; compounded over 60 layers, final logits diverge.
   - HIP kernel doesn't support ue8m0 round but the reference does → scale value disagreement.
   - PyTorch `ceil` vs aiter round-to-even, 1-ulp difference.
3. **Routing scale missing a multiply**: a *type* of layer consistently has lower cos, others are fine.
   - One of hash path / non-hash path is missing `* routed_scaling_factor`.
4. **Layout-triple mismatch**: weight prep / input quant scale layout / GEMM kernel must all agree.
   - Changing one but not the other two → cos drops sharply.
5. **Cumulative kernel drift**: each layer cos 0.999, by layer 30 it's 0.94.
   - aiter `mhc_post` / different GEMM kernels with diverging numerical precision.
6. **Activation / algorithmic difference**: cos suddenly drops.
   - Swiglu vs Silu implementation mismatch.
   - Missing clamp / scale factor.
7. **TP reduce missing or duplicated**: cos = 1/tp_size off.
   - `reduce_results=True/False` mismatch.
   - `shared_expert` reduced twice.
8. **KV cache contamination**: first token correct, later tokens drift.
   - Warmup not reset.
   - `start_pos` misaligned.
9. **Batch=N vs batch=1 inconsistency**: single OK, batch fails (or vice versa).
   - `cu_seqlens` / padding / `block_table` has a bug specific to the multi-seq path.
   - After fixing the single prompt, always re-run a batch test.

## When to stop / accept divergence

Not every cos drop is a bug worth fixing. Stop bisecting and document as a known limitation when:

| Signal | Likely diagnosis | Action |
|--------|-----------------|--------|
| Slot-vs-slot cos < 1.0 with identical inputs | Non-batch-invariant kernels (V4 paper §3.3); model spec assumes invariance, runtime stack doesn't provide it | Document; do **not** try to "fix" the model code — fix is at the kernel/aiter layer |
| Per-layer cos = 0.999 stable across many runs, never drops further | Floating-point non-associativity baseline | Accept; verify final task accuracy (lm_eval / GSM8K) instead |
| Reference itself disagrees with user expectation | Bad reference | Build a better reference (Phase 0) before continuing |
| Different models, same kernel: same drift magnitude | Kernel-level noise floor | Accept; this is the fp8/fp4 budget |
| Symptom only at temp > 0 stochastic, never at temp=0.0 | Sampling RNG / borderline confidence | Verify with task accuracy; not a forward bug |

**Rule of thumb**: a divergence is worth chasing if (a) it changes greedy decode output, (b) it tanks task accuracy by > model-card noise, OR (c) the model paper explicitly calls out reproducibility/invariance as a design property.

## Anti-patterns (don't do this)

- ❌ **Only look at output tokens, never dump intermediate tensors**: you'll never find the bug.
- ❌ **Skip Phase 1 weight check, jump straight to forward bisect**: a day later you'll discover it was the loader.
- ❌ **Dump a pile of un-named dicts**: three days later even you can't read them.
- ❌ **Don't verify the reference itself is correct**: a day of "matching the reference" later, you discover the reference was wrong too.
- ❌ **Change two things at once and re-dump**: bisect breaks down.
- ❌ **Dump multiple prompts / multiple layers in one shot**: 100MB+ files, slow loads, tangled comparisons.
- ❌ **Look at `cos` only, not `rel`**: cos = 0.997 looks fine, but `rel = 11%` will compound across 60 layers.
- ❌ **Compare stages without checking the op count on each side**: ATOM pre-all-reduce vs ref post-all-reduce, cos = 0.36 misjudgment.
- ❌ **Stop after fixing one prompt**: must run batch + multiple prompts; there may be independent bugs.
- ❌ **Fix with a hardcoded change instead of env-gated**: production runs of other models will explode.
- ❌ **Look at a single reference implementation**: sglang / vLLM also implement the same model — use a second reference to confirm the fix direction.
- ❌ **Use `F.cosine_similarity` on bf16 / fp32**: cos can return > 1.0 (rounding artifact) — use `cos_max()` (double precision) instead.
- ❌ **Use `one_shot=True` for batch-invariance bisect**: you capture the warmup forward, not the real prefill; modules inside per-seq dispatch loops only get seq 0 captured.
- ❌ **Mistake batch invariance for an ATOM bug**: model papers explicitly assume "bitwise reproducibility" (V4, Qwen3+) — running on a non-batch-invariant inference stack will flip edge-confidence tokens; that's expected, not a single-point bug.

## Companion skills

- Pair with `/debug-guide` to handle server / GPU residual issues.
- Pair with `/capture-trace` to grab a GPU kernel trace for performance analysis.
- Pre-commit cleanup: env-gated dump hooks can stay (disabled by default); hardcoded debug prints must be removed.

## Case studies

| Bug | Key location step | Time |
|-----|-------------------|------|
| `weights_mapping` prefix conflict | Phase 1 weight comparison cos = 0.3 → grep substring matches | ~2h |
| `wo_a` FP8 shuffle | Layer 0 `attn_out` cos = -0.002 → check `quant_type` → set to No | ~1h |
| Swiglu activation 9× loss | Layer 0 `ffn_out` magnitude off by 9× → bisect activation → fix with silu+clamp | ~3h |
| aiter `mhc_post` cumulative drift | Single prompt: first 30 tokens OK, then drifts → disable `mhc_post` to confirm | ~4h |
| **V4 input quant missing ue8m0** | Phase 1 weight OK → Phase 5 `wq_b` GEMM cos = 0.998 → Phase 6 standalone test isolates quant as the cause (FIX-ref-GEMM cos = 1.0) | ~3h |
| **V4 `act_quant_inplace` ceil vs round-to-even** | `kv_after_quant` cos = 0.998 → diff against `ref_kernels.py` → `ceil` vs `f32_to_e8m0` 1-ulp difference | ~1h |
| **V4 `select_experts` missing `routed_scaling_factor`** | Phase 4 layer 0/2 cos = 1.0 but **layer 3 jumps to cos = 0.98** → check layer-class difference (hash vs sqrtsoftplus) → diff ATOM `select_experts` vs ref `Gate.forward` → discover the missing `*= route_scale` at the end | ~2h |
| **Fix isolation revert** | After three fixes pass e2e, revert `linear.py` → Bug 9+10 alone produce `"6"` → decide PR split (Bug 8 deferred) | ~30min |
| **Bug 11 batch invariance** | Phase 8 E1 4×P3 + temp=0.0 → slot 2 differs → CLI `slot-invariance` shows layer 0 attn cos 0.99988 → MoE 0.998 → layer 1 attn 0.97 → match V4 paper §3.3 → not an ATOM bug | ~3h |

## Key utility library

**Use the ATOM built-ins** (import from `atom.utils.debug_helper`):

```python
from atom.utils.debug_helper import (
    # numerical primitives
    cos_max,            # returns (cos, max_abs, rel) — DOUBLE precision, avoids fp32 cos > 1.0 bug
    flag_for,           # ✓ / ? / ✗ / ✗✗ severity
    byte_equal_pct,     # bytewise overlap fraction
    # batch invariance
    slot_split,         # flat tensor → per-slot list
    compare_slots,      # one-shot slot 0 vs others
    pick_prefill_call,  # warmup-vs-prefill-vs-decode heuristic
    # schema
    schema_diff,        # only_a / only_b / common
    # dump hooks
    install_block_forward_hooks, maybe_dump_weights_and_exit, maybe_log_topk,
    # ref monkey-patching
    patch_method, patch_block_forward, patch_module_dump,
    # thresholds
    COS_BIT_EQUAL, COS_NUM_DRIFT, COS_ALGO_DIFF, COS_BUG,
)
```

**CLI**:
```bash
python -m atom.utils.debug_helper.compare slot-invariance --dir DIR --n-slots 4
python -m atom.utils.debug_helper.compare ref-vs-target  --dir DIR
python -m atom.utils.debug_helper.compare schema --a A.pt --b B.pt
```

**Project-specific helpers** (aiter / TP gather):

```python
from aiter.utility.fp4_utils import e8m0_to_f32, f32_to_e8m0  # FP8 e8m0 ↔ FP32
from aiter.ops.shuffle import shuffle_weight                  # FP8 weight CK layout

# TP gather
ref_full = torch.cat([ref_rank[r]["weight"] for r in range(8)], dim=tp_dim)

# TP-replicated check
for r in range(1, 8):
    assert (ref_rank0["weight"].view(uint8) == ref_rank_r["weight"].view(uint8)).all()
```

## File location convention

```
/app/logs_claude/{project}/dumps/
  ├── weights/                  # atom_rank{R}_layer{L}.pt + ref_rank{R}_layer{L}.pt
  ├── fwd/                      # atom_rank{R}_fwd.pt + ref_rank{R}_fwd.pt (single-layer)
  ├── fwd_multi/                # multi-layer dumps (0,15,30,45,60)
  └── fwd_v4ref/                # one dir per fix variant
/app/logs_claude/{project}/standalone_<op>_test.py  # GPU isolation experiments
/app/logs_claude/{project}/notes/NN_*.md             # bisect notes, numbered sequentially
```

## Documentation discipline

After each phase, write a `notes/NN_*.md` containing:
1. Current hypothesis (what the bug is).
2. Experiment design (how to verify it).
3. Data (cos table / command output / key log).
4. Conclusion (hypothesis confirmed / refuted / rewritten).
5. Next step.

**Cost of skipping notes**: three days later you can't tell which phase you're in; handing over to someone else means redoing the work.
