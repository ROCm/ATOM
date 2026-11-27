---
name: atomrl-training
description: >-
  Debugging and reviewing AtomRL RL training runs. Use when an RL run
  diverges, reward curves plateau or collapse, MoE routing becomes
  unstable, FP8 precision causes train-inference mismatch, or KL
  divergence spikes unexpectedly.
---

# AtomRL Training Guide

## Core Principle

When an RL run behaves unexpectedly, compare against a known-good BF16 baseline before changing any hyperparameters. Use `.claude/tmp-rl-bugs.md` in the AtomRL repo root as working memory. Do not solve stability problems by retuning parameters that already match the baseline.

## Use When

- Reward curve plateaus, collapses, or shows unexpected behavior
- KL divergence spikes (especially with FP8 or MoE)
- MoE router distributions diverge between train and inference
- FP8 rollout produces different behavior than BF16 rollout
- Training loss shows NaN or Inf
- Weight sync appears to corrupt model outputs
- Expert load balance degrades during training

Treat any fresh return to the same debugging problem as a new debug session.

## Default Workflow

1. Read `.claude/tmp-rl-bugs.md` at the start of every new debug session
2. Stop the run once behavior is clearly wrong; save logs and the last good checkpoint
3. Compare current config against the baseline recipe (YAML diff, not memory)
4. Check reward / KL / entropy curves for the first anomalous step
5. For FP8 issues: compare BF16 vs FP8 rollout log-probs; check if rollout correction (TIS/MIS) is enabled
6. For MoE issues: check if R3 is enabled; compare router distribution entropy between train and inference
7. For weight sync issues: compare model outputs before and after weight transfer on the same input
8. After each meaningful test, update `.claude/tmp-rl-bugs.md` with findings

## Hard Rules

- Stop the run when reward collapses. Debug from the last good checkpoint.
- Start every new debug session by reading `.claude/tmp-rl-bugs.md`
- Never disable R3 "to see if it helps" without first recording the routing distributions
- Never change clip ratio, KL coefficient, or learning rate if they already match the baseline
- FP8 precision issues are not hyperparameter problems; isolate the quantization path first
- Compare effective runtime values, not just config files
- Do not continue a clearly bad run "for more signal"

## FP8-Specific Debug

| Symptom | Check |
|---------|-------|
| Reward diverges from BF16 baseline | Compare per-token log-prob distributions (FP8 vs BF16) |
| Loss spikes after weight sync | Verify QKV scale recalibration runs every RL step |
| Gradual quality drift | Check if `rollout_correction.enabled: true` and try `num_first_layers_in_bf16: 1` |
| NaN in rollout | Check FP8 weight quantizer for overflow; verify blockwise scaling ranges |

Escalation order for FP8:
1. Compare BF16 vs FP8 rollout output on same input, same weights
2. If different: check weight quantizer round-trip error per layer
3. If quantizer is fine: check KV-cache scale recalibration
4. If KV-cache is fine: check attention output (FP8 attention vs BF16)
5. Enable rollout correction (TIS) and verify it reduces the gap

## R3-Specific Debug

| Symptom | Check |
|---------|-------|
| MoE training collapse (reward -> 0) | Verify R3 is enabled; check `moe.r3.enabled: true` |
| KL divergence spikes with MoE | Compare router logits between ATOM rollout and Lumen training |
| Expert load imbalance grows | Check if R3 replay preserves expert selection distribution |
| R3 enabled but still unstable | Verify recorded logits are non-NaN, correct shape, correct layer count |

Escalation order for R3:
1. Verify router logits are recorded correctly (shape, range, no NaN)
2. Diff recorded logits vs replayed logits (must be bit-identical)
3. Check expert load balance before and after R3 (should be similar)
4. If load balance differs: check if EP layout differs between ATOM and Megatron
5. If still unstable: check if replay mode should be `hard_assignment` instead of `distribution`

## Weight Sync Debug

| Symptom | Check |
|---------|-------|
| Rollout ignores training updates | Verify `RLTrainer._sync_weights_ipc()` runs after actor update, or `AsyncRLTrainer._trigger_param_sync()` advances the version |
| Model outputs differ after sync | Check sender/receiver completion and weight coverage; compare actor and rollout parameter checksums |
| OOM during sync | Check IPC bucket size, rollout sleep mode, and whether safetensors fallback creates an extra full copy |
| MoE experts scrambled after sync | Check `FusedMoEWeightRouter` coverage and EP layout resharding |

## Rationalizations to Reject

| Excuse | Reality |
|--------|---------|
| "Just tweak the clip ratio" | If it matches the baseline, the bug is elsewhere |
| "Disable R3, it's probably the problem" | R3 fixes instability; disabling it hides the root cause |
| "The FP8 gap is acceptable" | Quantify it. If `>5%` reward difference at convergence, investigate |
| "Keep running for more signal" | Sunk cost. Stop the bad run and debug from a snapshot |
| "The configs are basically the same" | "Basically" is not enough. Diff effective values |

## Red Flags

- Starting a debug session without reading `.claude/tmp-rl-bugs.md`
- Changing aligned hyperparameters to chase a symptom
- Running with R3 disabled on MoE models without documenting why
- Claiming FP8 parity from memory instead of a written comparison
- Ignoring weight sync step when debugging output quality
- Comparing runs with different seeds, datasets, or tokenizers

## Dashboard Organization

After a training run, the analysis dashboard must be saved to the `dashboards/` folder under a subfolder that identifies the run.

**Naming convention:** `{model}-{quantization}-{run_mode}`

| Component | Examples |
|-----------|----------|
| `{model}` | `1a` (Qwen3-1.7B), `8b` (Qwen3-8B), `32b`, etc. |
| `{quantization}` | `bf16`, `fp8`, `fp8-kv` |
| `{run_mode}` | `baseline`, `async`, `sync`, `moe`, `r3` |

**Examples:**
- `dashboards/1a-bf16-baseline/` — 1A model, BF16, baseline sync training
- `dashboards/1a-bf16-async/` — 1A model, BF16, fully async training
- `dashboards/8b-fp8-async/` — 8B model, FP8, async training
- `dashboards/32b-fp8-moe-r3/` — 32B MoE model, FP8, with R3

A single-file dashboard (e.g. `1a-bf16-async.html`) at the `dashboards/` root is acceptable for lightweight summaries; use a subfolder when additional artifacts (images, JSON data, comparison files) accompany the dashboard.

### Live Monitoring Cadence

Once an RL training run is confirmed running (first few steps complete without crash), monitor and update the dashboard **every 30 minutes**:

1. Read the latest training logs (tail the log file or check WandB)
2. Check for red flags: reward collapse, NaN spike, KL blowup, stuck generation
3. Update the dashboard HTML with fresh data points (all charts and summary cards)
4. If any red flag is detected, alert immediately — do not wait for the next 30-minute check
5. Record observations in `.claude/tmp-rl-bugs.md` if anything is anomalous

The 30-minute cycle continues until the run finishes or is stopped. If the run is healthy and uneventful for 3+ consecutive checks, the interval can relax to every 1 hour.

### Dashboard Metrics Reference

Every dashboard should include a metrics reference table explaining how each displayed metric is computed. Use the table below as the canonical source.

| Metric | Formula / Computation | Source | Granularity |
|--------|----------------------|--------|-------------|
| **Training Loss** | Clipped surrogate: `-E[min(r*A, clip(r,1-e,1+e)*A)]` + optional `kl_coeff * kl`. `r = exp(logp - logp_old)`. DAPO uses asymmetric clip `[1-e_low, 1+e_high]`. Masked mean over response tokens. | `loss_functions.py`, `grpo.py`, `dapo.py` | Mean over mini-batches per step |
| **Reward Mean** | `mean(rewards)` over all rollout sequences in the batch | `rl_trainer.py` | Per step |
| **Reward Accuracy** | Fraction of sequences with `reward > 0` | `rl_trainer.py` | Per step |
| **Step Time** | Wall clock from step start to end (includes rollout + train + sync) | `rl_trainer.py` | Per step |
| **Gen Time** | Wall clock for rollout phase only | `rl_trainer.py` | Per step |
| **Gen Throughput** | `generated_tokens / gen_time` | `rl_trainer.py` | Per step |
| **Grad Norm** | `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` — global L2 norm | `rl_trainer.py` (sync), `async_trainer.py` | Per step (logged in async only) |
| **NaN Grads** | Count of parameters whose `.grad` contains any NaN; NaN elements zeroed before clip | `async_trainer.py` | Per step (async only, parsed from logs) |
| **Active Fraction** | `mean(row_mask)` — fraction of rollout rows kept after filtering groups with `std <= 1e-6` (DAPO dynamic sampling) | `dapo.py` | Per step (log string only) |
| **KL Divergence** | `masked_mean(ref_logp - logp)` — Monte Carlo estimate; only when `kl_coeff > 0` and `ref_log_probs` present | `loss_functions.py` | Mean over mini-batches per step |
| **Mean Response Length** | Mean token count from `response_mask` | `rl_trainer.py` | Per step |
| **Max Sequence Length** | Max padded sequence length in the batch | `rl_trainer.py` | Per step |

## References

- `.claude/tmp-rl-bugs.md` in the AtomRL repo root — open bug candidates and evidence
- [reference.md](reference.md) — pre-training diff checklist, FP8 debug checklist, R3 debug checklist, weight sync debug checklist, reward debug checklist
- `configs/grpo_dense_bf16.yaml` — BF16 baseline recipe (the primary reference for parity testing)

## Pairing

Pair with `atomrl-coding` when code changes are needed.
Pair with `atomrl-debug` when the bug originates in Lumen, ATOM, or AITER rather than AtomRL itself.
