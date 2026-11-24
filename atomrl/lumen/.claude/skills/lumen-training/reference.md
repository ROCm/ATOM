# Lumen Training Reference

## Temporary Bug Note File

Use the repo-local file `.claude/tmp-training-bugs.md` relative to the `Lumen` repo root to track possible bugs found during testing.

Rules:

- read the whole file at the start of every new debug session
- use it to avoid repeating dead ends and to reuse prior evidence
- do not treat it as proof that overrides fresh reference diffs
- append or update findings after each meaningful test, repro, or validation step

Treat any fresh return to the same debugging problem as a new debug session:

- a new chat or agent session
- a new day or work block
- returning after unrelated work
- starting a new round of debug after prior tests finished

Meaningful test or experiment means a step that changes confidence in a hypothesis, for example:

- a new minimal repro
- a new written reference diff
- a backend toggle check
- a layerwise forward or backward comparison
- a kernel unit test
- a targeted integration validation

Do not log every identical rerun. Do log negative results that rule a suspicion out.

Each entry should record:

- date or session marker
- symptom
- possible bug or suspicion
- evidence collected so far
- next check
- status: open, ruled out, or resolved

## If No Trusted Reference Exists

Do not tune first. Freeze a comparison target before bring-up or debugging:

- choose the closest trusted implementation, commit, or recipe
- define the success signal that will count as parity
- record the exact command, checkpoint state, and backend selection

## Pre-Training Comparison Checklist

Freeze one trusted reference before starting bring-up or debugging:

- branch or commit
- checkpoint or cold-start state
- exact command line
- environment variables and backend flags
- first signal that defines success or failure

Compare effective runtime values, not just config files:

- record the compared values in one diff note or table
- do not claim alignment from memory

### Model and parallelism

- model family and architecture
- hidden size, heads, layers, sequence length, vocab, rope settings
- tensor, pipeline, context, sequence, and data parallel settings
- microbatch, global batch, gradient accumulation, and token count semantics

### Optimizer and schedule

- optimizer type and param groups
- base LR, min LR, warmup, decay schedule
- weight decay, betas, eps, grad clip
- loss scaling or FP8 scaling behavior

### Precision and kernel path

- bf16, fp16, fp8, or mixed-precision mode
- FP8 recipe, amax history behavior, scaling type, and gather settings
- enabled backends, fallback flags, and AITER toggles

### Data and dataset

- tokenizer and preprocessing
- dataset version, filtering, packing, truncation, and masking
- data order, sampler, shuffle seed, shard mapping, and blend weights
- train, val, and test splits

### Resume state

- checkpoint step
- optimizer and scheduler restore state
- RNG state
- dataloader resume position

If any of the above differs from the reference, fix that diff before claiming the runs are aligned.

## FP8 Quantized GEMM Dispatch

Each scaling type routes through different backends. When debugging FP8 training, verify which backend was actually selected.

| scaling_type | Backends (priority order) | Tuning |
|---|---|---|
| delayed/dynamic | hipBLASLt -> CK (`gemm_a8w8_CK`) -> Triton | CK reads `a8w8_tuned_gemm.csv` |
| per_token | Triton only (`gemm_a8w8_per_token_scale`) | No tuned config |
| blockwise | CK (`gemm_a8w8_blockscale`) -> Triton | CK reads `a8w8_blockscale_tuned_gemm.csv` |
| mxfp8 | Triton only (`gemm_mxfp8`) | No tuned config |

## FP8 Backward Scale Misalignment

Per-tensor scalar scales (delayed/dynamic) survive weight transposition. Per-token, blockwise, and mxfp8 block scales become misaligned after `weight_fp8.t()`. Backward with full FP8 quantization is restricted to `["delayed", "dynamic", "none"]` scaling types.

When debugging wrong FP8 gradients:

1. Check the scaling type — if per-token, blockwise, or mxfp8, full FP8 backward is not supported
2. Verify backward is using BF16 fallback for unsupported scaling types
3. If delayed/dynamic, check amax history and scale computation

## BF16 GEMM Constraints

- **dtype matching**: `tuned_gemm` / `hipb_mm` / `F.linear` require both inputs to have the same dtype. After dequantizing FP8 weights, always cast back to target dtype.
- **Triton `gemm_a16w16`**: BLOCK_SIZE_K >= 128 required on gfx942.
- **ASM kernels**: Require K % 64 == 0 and N % 64 == 0.
- **TN layout**: All BF16/FP8 GEMM kernels compute Y = A @ W^T. Weight `w` is always (N, K).

## Debug Entry Checklist

Before running deeper debug:

1. Capture the exact failing behavior:
   - step number
   - loss or metric mismatch
   - NaN or Inf location
   - first suspicious layer or op if known
2. Reduce to the smallest repro that still fails:
   - same batch or microbatch
   - same seed
   - same checkpoint when possible
   - same backend path
3. Remove noise:
   - disable unrelated experiments
   - avoid changing multiple variables at once

If you are unsure whether the run is "bad enough" to stop, stop and diff against the reference anyway.

## Layerwise Forward and Backward Comparison

Use this only after relevant parameters, config, and dataset are aligned.

1. Freeze inputs:
   - same checkpoint
   - same tokenized batch
   - same masks and position ids
   - same seed and stochastic settings when possible
   - same dtype and backend selection
2. Compare forward layer by layer:
   - embeddings
   - each transformer block
   - final norm and logits
3. Record the first layer where activations drift outside tolerance.
4. Compare backward at and around the first failing layer:
   - output grad
   - parameter grads
   - grad norms
   - FP8 scales or amax metadata if relevant
5. Debug the first divergence, not every downstream difference.

Use dtype-appropriate tolerances. Prefer the same tolerance policy already used in nearby `lumen-test` coverage:

- `torch.testing.assert_close` for strict non-quantized or BF16 paths
- `compute_snr` or `check_close` for quantized paths
- reuse existing thresholds from nearby Lumen tests when possible, and write down the tolerance you used

Do not run layerwise comparison on different batches, different checkpoints, or different backend selections. That creates false leads.

## AITER Validation Ladder

Treat AITER as implicated only when the first failing op, backend toggle, or localized divergence points to an AITER-backed path.

If a backend toggle or logs already isolate the failing path to AITER before layerwise comparison, you may enter the AITER ladder immediately after the minimal repro. Otherwise, localize the first divergence with layerwise forward and backward comparison first.

Lumen uses `try_backends()` in `lumen/ops/dispatch.py` to fall through backends on failure. It catches `RuntimeError`, `NotImplementedError`, `TypeError`, `ValueError`, Triton `CompilationError`, and `OutOfResources`. Always call `torch.cuda.synchronize()` after GPU kernels to surface asynchronous errors before fallback.

AITER is pinned in the supported Lumen image. Import its public kernel API
directly; use `try_backends()` only for supported implementation fallbacks,
not for dependency-availability probes.

If AITER is implicated:

1. Identify the exact op and kernel path:
   - op name
   - shapes
   - dtypes
   - layout or stride assumptions
   - which backend `try_backends()` actually selected (check dispatch order for the scaling type)
2. Write or run a kernel-level unit test against a trusted implementation:
   - forward correctness
   - backward correctness if applicable
   - tolerances appropriate to dtype
3. If the kernel test fails, fix the kernel or integration and rerun the unit test first.
4. If the kernel test passes, run a targeted integration validation on the smallest model path that still exercises the op.
5. Only then run a short end-to-end confirmation.

Never use end-to-end validation as the first proof of AITER correctness.

## Normalization Backend Notes

- **Backward**: Prefer Triton when `requires_grad=True` (CK fwd doesn't save intermediates for bwd).
- **Fused norm+quant**: Available for all scaling types in both LayerNorm and RMSNorm.
- **ASM norm**: `layernorm2d_with_add_asm` available for LayerNorm only (no RMSNorm ASM).

## What Not To Do

- Do not change already-aligned parameters to hide a mismatch.
- Do not declare parity from config names alone.
- Do not declare alignment from memory without a written diff artifact.
- Do not skip dataset comparison because the model config matches.
- Do not jump from suspicion to end-to-end AITER testing without a kernel check.
- Do not run broad hyperparameter sweeps before the first real divergence is localized.
- Do not use per-token, blockwise, or mxfp8 scaling with full FP8 backward — scale misalignment after transposition makes results incorrect.
- Do not assume the same backend is used for different FP8 scaling types — each has its own dispatch chain.

## After Correctness

Once correctness is re-established and the run matches the reference, use `lumen-benchmark` only for performance tuning and overlap analysis.
