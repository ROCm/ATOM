# Rollout Correction

## Attribution

This implementation is derived from the **verl** project
([verl-project/verl](https://github.com/verl-project/verl)), specifically the
rollout correction framework designed by
[Yingru Li](https://richardli.xyz/) and Jiacai Liu.

### References

```bibtex
@online{liu-li-2025-rl-collapse,
  title   = {When Speed Kills Stability: Demystifying {RL} Collapse
             from the Training-Inference Mismatch},
  author  = {Liu, Jiacai and Li, Yingru and Fu, Yuqian and Wang, Jiawei
             and Liu, Qian and Shen, Yu},
  year    = {2025},
  month   = sep,
  url     = {https://richardli.xyz/rl-collapse}
}

@article{li2025trust,
  title   = {Trust Region Masking for Long-Horizon LLM Reinforcement Learning},
  author  = {Li, Yingru and Liu, Jiacai and Xu, Jiawei and Tong, Yuxuan
             and Li, Ziniu and Liu, Qian and Wang, Baoxiang},
  journal = {arXiv preprint arXiv:2512.23075},
  year    = {2025}
}
```

Blog series:
- [Part 1: Why Mismatch Breaks LLM-RL](https://richardli.xyz/rl-collapse-1)
- [Part 2: The Gradient Estimator Trials](https://richardli.xyz/rl-collapse-2)
- [Part 3: When Math Meets Reality — Toxic Tails and Length Traps](https://richardli.xyz/rl-collapse-3)
- Paper: https://arxiv.org/abs/2512.23075

### verl source files

| verl file | AtomRL counterpart |
|-----------|---------------------|
| `verl/trainer/ppo/rollout_corr_helper.py` | [`atomrl/algorithms/rollout_correction.py`](../atomrl/algorithms/rollout_correction.py) |
| `verl/trainer/ppo/core_algos.py` (bypass_mode, reinforce losses) | [`atomrl/algorithms/policy_losses.py`](../atomrl/algorithms/policy_losses.py) |
| `verl/trainer/ppo/ray_trainer.py` (training loop integration) | [`atomrl/trainer/rl_trainer.py`](../atomrl/trainer/rl_trainer.py) |
| `verl/trainer/config/algorithm.py` (RolloutCorrectionConfig) | [`atomrl/core/config.py`](../atomrl/core/config.py) |
| `verl/docs/algo/rollout_corr.md` | This document |
| `verl/docs/algo/rollout_corr_math.md` | (see verl repo for full mathematical derivations) |

---

## Overview

Rollout correction handles **off-policy distribution shifts** between the
rollout (data-collection) policy and the training policy. Common sources:

1. **Precision mismatch** — FP8/BF16/FP32 differences between rollout and
   training backends
2. **Temporal lag** — stale checkpoints from async workers
3. **Replay buffers** — training on historical trajectories
4. **Backend differences** — vLLM vs FSDP vs Megatron producing different
   logits for identical weights

Without correction, these shifts cause biased gradient estimates that can lead
to training instability and policy collapse.

The framework provides two orthogonal mechanisms:

- **Importance Sampling (IS) weights** — continuous reweighting for gradient
  correction (variance reduction)
- **Rejection Sampling (RS)** — binary filtering to exclude outlier
  tokens/sequences from training

---

## Files

| File | Role |
|------|------|
| [`atomrl/algorithms/rollout_correction.py`](../atomrl/algorithms/rollout_correction.py) | Primary implementation: IS weights, 11 RS criteria, off-policy diagnostics, bypass mode, factory presets |
| [`atomrl/quantization/rollout_correction.py`](../atomrl/quantization/rollout_correction.py) | Backward-compatible re-exports (thin redirect to `algorithms/`) |
| [`atomrl/algorithms/policy_losses.py`](../atomrl/algorithms/policy_losses.py) | `reinforce` and `bypass_mode` registered loss functions |
| [`atomrl/core/config.py`](../atomrl/core/config.py) | `RolloutCorrectionConfig` dataclass |
| [`atomrl/trainer/rl_trainer.py`](../atomrl/trainer/rl_trainer.py) | Training loop integration (torchrun + Ray paths) |

---

## Operating Modes

### Decoupled Mode (3 policies) — `bypass_mode: false`

Three distinct policies:
- **π_rollout** — behavior policy (data collection, e.g. vLLM)
- **π_old** — proximal policy (computed via `actor.compute_log_prob()` at
  start of each training epoch)
- **π_θ** — current policy (being updated)

IS weights correct for Drift 1 (π_rollout → π_old). PPO clipping handles
Drift 2 (π_old → π_θ). This achieves batch-size invariance (Hilton et al.,
2021).

### Bypass Mode (2 policies) — `bypass_mode: true`

Sets π_old = π_rollout, skipping the expensive `compute_log_prob()` forward
pass. Two loss types:

| `loss_type` | Behavior |
|-------------|----------|
| `ppo_clip` (default) | PPO clipped objective; ratio = π_θ/π_rollout handles IS implicitly |
| `reinforce` | Pure policy gradient with explicit IS weights; no PPO clipping |

---

## Configuration

All parameters live under `quantization.rollout_correction` in YAML:

```yaml
quantization:
  rollout_correction:
    # --- Legacy TIS/MIS (corrects advantages directly) ---
    enabled: false          # enable legacy TIS/MIS on advantages
    method: tis             # "tis" or "mis"
    clip: 1.5               # symmetric clip bound for legacy TIS

    # --- IS weights (verl rollout_corr) ---
    rollout_is: ""          # "token", "sequence", or "" (disabled)
    rollout_is_threshold: "2.0"   # float or "lower_upper" for IcePop
    rollout_is_batch_normalize: false

    # --- Rejection sampling ---
    rollout_rs: ""          # comma-separated criteria (see below)
    rollout_rs_threshold: ""  # comma-separated thresholds

    # --- Bypass mode ---
    bypass_mode: false
    loss_type: ppo_clip     # "ppo_clip" or "reinforce"
```

**Prerequisite**: rollout engine must set `calculate_log_probs: true` so that
`rollout_log_probs` are available in the batch.

### `rollout_is` — IS weight aggregation

| Value | Behavior | Typical threshold |
|-------|----------|-------------------|
| `"token"` | Per-token IS weights, independently truncated | 1.5 – 5.0 |
| `"sequence"` | Sequence-level weight (product of token ratios), broadcast | 2.0 – 10.0 |
| `""` (empty) | Disabled | — |

### `rollout_is_threshold` — IS truncation

- **Single float** (e.g. `"2.0"`): Truncated IS (TIS) — `.clamp(max=threshold)`
- **`"lower_upper"`** (e.g. `"0.5_5.0"`): IcePop — zero weights outside
  `[lower, upper]` without modifying response_mask

### `rollout_rs` — Rejection sampling criteria

11 supported criteria, combinable via comma (logical AND):

| Criterion | Level | Statistic | Threshold format |
|-----------|-------|-----------|------------------|
| `token_k1` | token | −log(ρ) | `"lower_upper"` |
| `token_k2` | token | 0.5·(log ρ)² | upper bound |
| `token_k3` | token | exp(log ρ) − 1 − log ρ | upper bound |
| `seq_sum_k1` | sequence (sum) | Σ(−log ρ) | `"lower_upper"` |
| `seq_sum_k2` | sequence (sum) | Σ 0.5·(log ρ)² | upper bound |
| `seq_sum_k3` | sequence (sum) | Σ(exp(log ρ) − 1 − log ρ) | upper bound |
| `seq_mean_k1` | sequence (mean) | geometric mean ratio | `"lower_upper"` |
| `seq_mean_k2` | sequence (mean) | mean 0.5·(log ρ)² | upper bound |
| `seq_mean_k3` | sequence (mean) | mean K3 divergence | upper bound |
| `seq_max_k2` | sequence (max) | max 0.5·(log ρ)² | upper bound |
| `seq_max_k3` | sequence (max) | max K3 divergence | upper bound |

Where ρ_t = π_old(a_t|s_t) / π_rollout(a_t|s_t).

**K1** uses the log-ratio directly; threshold is a ratio bound (e.g. `"0.999_1.001"`).
**K2** is a symmetric quadratic penalty; good for detecting general mismatch.
**K3** equals KL(π_rollout ‖ π_old) in expectation; always ≥ 0 per token;
more stable than K1 for small divergences.

### Geometric-mean RS (`seq_mean_k1`)

Solves the **length trap**: standard sequence-level IS ratios are
multiplicative products that explode with sequence length (e.g. 1.1¹⁰⁰ ≈
13,780), causing long CoT sequences to be systematically rejected. The
geometric mean normalizes by sequence length (1.1¹⁰⁰/¹⁰⁰ = 1.1), making
the trust region length-invariant.

---

## Presets

Factory functions in `atomrl.algorithms.rollout_correction`:

### Decoupled mode

| Preset | IS | RS |
|--------|----|----|
| `preset_decoupled_token_is(threshold=2.0)` | token TIS | — |
| `preset_decoupled_seq_is(threshold=2.0)` | sequence TIS | — |
| `preset_decoupled_token_icepop(lower=0.5, upper=5.0)` | token IcePop | — |
| `preset_decoupled_seq_is_rs(is_threshold=2.0, rs_threshold="0.5_2.0")` | sequence TIS | seq_sum_k1 |
| `preset_decoupled_geo_rs(threshold="0.999_1.001")` | — | seq_mean_k1 (Geo-RS) |
| `preset_decoupled_geo_rs_token_tis(is_threshold=2.0, rs_threshold="0.999_1.001")` | token TIS | seq_mean_k1 |
| `preset_decoupled_geo_rs_seq_tis(...)` | sequence TIS | seq_mean_k1 |
| `preset_decoupled_k3_rs(threshold=0.005)` | — | seq_mean_k3 |
| `preset_decoupled_k3_rs_token_tis(is_threshold=2.0, rs_threshold=0.005)` | token TIS | seq_mean_k3 |
| `preset_decoupled_k3_rs_seq_tis(...)` | sequence TIS | seq_mean_k3 |

### Bypass mode

| Preset | Loss | IS | RS |
|--------|------|----|----|
| `preset_bypass_ppo_clip()` | PPO-clip | (ratio) | — |
| `preset_bypass_ppo_clip_geo_rs(rs_threshold="0.999_1.001")` | PPO-clip | (ratio) | seq_mean_k1 |
| `preset_bypass_ppo_clip_k3_rs(rs_threshold=0.005)` | PPO-clip | (ratio) | seq_mean_k3 |
| `preset_bypass_pg_is(threshold=2.0)` | REINFORCE | sequence TIS | — |
| `preset_bypass_pg_geo_rs(rs_threshold="0.999_1.001")` | REINFORCE | — | seq_mean_k1 |
| `preset_bypass_pg_geo_rs_token_tis(...)` | REINFORCE | token TIS | seq_mean_k1 |
| `preset_bypass_pg_geo_rs_seq_tis(...)` | REINFORCE | sequence TIS | seq_mean_k1 |
| `preset_disabled()` | — | — | — |

---

## YAML Examples

### Token-level IS only (current default)

```yaml
quantization:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: "2.0"
```

### Geo-RS + Token-TIS (recommended for long sequences)

```yaml
quantization:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: "2.0"
    rollout_rs: seq_mean_k1
    rollout_rs_threshold: "0.999_1.001"
```

### K3-RS + Token-TIS

```yaml
quantization:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: "2.0"
    rollout_rs: seq_mean_k3
    rollout_rs_threshold: "0.005"
```

### Bypass mode (skip old_log_prob forward pass)

```yaml
quantization:
  rollout_correction:
    bypass_mode: true
    loss_type: ppo_clip
    rollout_rs: seq_mean_k1
    rollout_rs_threshold: "0.999_1.001"
```

### Multiple RS criteria (comma-separated, logical AND)

```yaml
quantization:
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: "2.0"
    rollout_rs: "token_k1,seq_mean_k3"
    rollout_rs_threshold: "0.5_2.0,0.005"
```

---

## Metrics

All metrics are prefixed with `rollout_corr/` in W&B / logs.

### IS weight metrics

| Metric | Description |
|--------|-------------|
| `rollout_is_mean` | Mean IS weight (should be ~1.0) |
| `rollout_is_std` | Std of IS weights |
| `rollout_is_max` / `rollout_is_min` | Extreme IS weights |
| `rollout_is_eff_sample_size` | Effective sample size (0–1 fraction) |
| `rollout_is_ratio_fraction_high` | Fraction exceeding upper threshold |
| `rollout_is_ratio_fraction_low` | Fraction below lower threshold |
| `rollout_is_seq_mean` / `seq_std` / `seq_max` | Sequence-level weight stats |
| `rollout_is_batch_norm_factor` | Normalization factor (if batch_normalize=true) |

### Rejection sampling metrics

| Metric | Description |
|--------|-------------|
| `rollout_rs_masked_fraction` | Fraction of tokens rejected (combined) |
| `rollout_rs_seq_masked_fraction` | Fraction of sequences with any rejection |
| `rollout_rs_<option>_masked_fraction` | Per-criterion token rejection rate |
| `rollout_rs_<option>_seq_masked_fraction` | Per-criterion sequence rejection rate |

### Off-policy diagnostics

| Metric | Description |
|--------|-------------|
| `kl` | KL(π_rollout ‖ π_training) — direct estimator |
| `k3_kl` | K3 estimator (always ≥ 0, more stable for small KL) |
| `training_ppl` / `rollout_ppl` | Policy perplexities |
| `ppl_ratio` | training_ppl / rollout_ppl |
| `log_ppl_diff` / `log_ppl_abs_diff` | Log-perplexity differences |
| `chi2_token` | Token-level χ² divergence: E[ρ²] − 1 |
| `chi2_seq` | Sequence-level χ² divergence: E[(∏ρ_t)²] − 1 |

### Health checks

| Condition | Action |
|-----------|--------|
| `rollout_is_mean` far from 1.0 (< 0.5 or > 2.0) | Significant off-policy gap; verify `calculate_log_probs=true` |
| `rollout_is_eff_sample_size` < 0.3 | High weight concentration; tighten threshold |
| `rollout_rs_masked_fraction` > 0.5 | Too many rejections; loosen RS threshold |
| `chi2_token` > 1.0 | Severe distribution shift; investigate cause |

---

## Choosing a Method

### By off-policy severity

| Severity | Scenario | Recommended |
|----------|----------|-------------|
| Negligible | Same checkpoint, BF16 everywhere | Token-TIS only (default) |
| Moderate | FP8 rollout + BF16 training | Token-TIS or Geo-RS + Token-TIS |
| Severe | Async workers, replay buffers | Seq-TIS or K3-RS + Token-TIS |

### By sequence length

| Length | Concern | Recommended |
|--------|---------|-------------|
| Short (< 2K tokens) | Standard chat | Seq-TIS |
| Long (> 8K tokens, CoT) | Length trap | Geo-RS (`seq_mean_k1`) or K3-RS (`seq_mean_k3`) |

### Bypass vs Decoupled

| Mode | Pros | Cons |
|------|------|------|
| Decoupled | Batch-size invariance; separate drift correction | Extra forward pass |
| Bypass | Faster (skip old_log_prob) | No batch-size invariance |

---

## Background: The Three-Policy Framework

Standard PPO assumes π_old (the reference for clipping) is also the behavior
policy that collected the data. In LLM-RL this assumption breaks because
rollout and training use different precision/backends.

**Decoupled PPO** (Hilton et al., 2021) separates these roles:

```
L = −E_{(s,a)∼μ}[ w_t · min(r_t·A_t, clip(r_t)·A_t) ]
```

where:
- w_t = π_old / π_rollout — IS weight correcting for behavior policy
- r_t = π_θ / π_old — PPO ratio controlling update magnitude

This separation was identified as critical for training stability in
[Liu, Li et al. (2025)](https://richardli.xyz/rl-collapse), which showed that
ignoring the rollout policy leads to biased gradients and RL collapse.
