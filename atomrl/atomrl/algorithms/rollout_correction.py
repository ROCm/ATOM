# Copyright 2025 The AtomRL Authors.
# Derived from verl (verl-project/verl):
#   verl/trainer/ppo/rollout_corr_helper.py (IS weights, rejection sampling, metrics)
#   verl/trainer/ppo/core_algos.py (bypass_mode loss integration)
#   verl/trainer/ppo/ray_trainer.py (training loop integration)
#
# Original rollout correction framework by Yingru Li (https://richardli.xyz/)
# and Jiacai Liu, as described in:
#   - Liu, Li et al. "When Speed Kills Stability: Demystifying RL Collapse
#     from the Training-Inference Mismatch" (https://richardli.xyz/rl-collapse)
#   - Li et al. "Trust Region Masking for Long-Horizon LLM Reinforcement
#     Learning" (arXiv:2512.23075)
#
# Licensed under the Apache License, Version 2.0.
"""Rollout correction: IS weights, rejection sampling, and off-policy diagnostics.

Ported from verl/trainer/ppo/rollout_corr_helper.py with adaptations for
AtomRL's DataProto and config system. Handles general off-policy problems:
precision mismatch (FP8 vs BF16), temporal lag (async workers), replay buffers.

Three operating modes:
  - Decoupled (3 policies): pi_rollout, pi_old, pi_theta — IS corrects drift 1
  - Bypass PPO-clip (2 policies): pi_old = pi_rollout, ratio handles IS
  - Bypass REINFORCE (2 policies): explicit IS weights, no PPO clipping

See docs/rollout_corr.md for usage guide and verl references.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor

from atomrl.core.config import AtomRLConfig, QuantizationConfig, RolloutCorrectionConfig
from atomrl.core.protocol import DataProto

logger = logging.getLogger(__name__)

SAFETY_BOUND = 20.0

SUPPORTED_ROLLOUT_RS_OPTIONS: set[str] = {
    "token_k1",
    "token_k2",
    "token_k3",
    "seq_sum_k1",
    "seq_sum_k2",
    "seq_sum_k3",
    "seq_mean_k1",
    "seq_mean_k2",
    "seq_mean_k3",
    "seq_max_k2",
    "seq_max_k3",
}
TOKEN_LEVEL_ROLLOUT_RS_OPTIONS: set[str] = {"token_k1", "token_k2", "token_k3"}


# ---------------------------------------------------------------------------
# Masked helpers (standalone, no verl_F dependency)
# ---------------------------------------------------------------------------


def _masked_sum(values: Tensor, mask: Tensor, axis: int | None = None) -> Tensor:
    valid = torch.where(mask.bool(), values, torch.zeros_like(values))
    return (valid * mask).sum(dim=axis)


def _masked_mean(values: Tensor, mask: Tensor, axis: int | None = None) -> Tensor:
    return _masked_sum(values, mask, axis) / (mask.sum(dim=axis).clamp(min=1e-8))


# ---------------------------------------------------------------------------
# Threshold parsers
# ---------------------------------------------------------------------------


def _parse_rollout_is_threshold(
    threshold_spec: str | float,
) -> tuple[float, float | None]:
    """Parse IS threshold: single float for TIS, 'lower_upper' for IcePop."""
    if isinstance(threshold_spec, bool):
        raise TypeError("rollout_is_threshold must be a float or string, not bool.")
    if isinstance(threshold_spec, (int, float)):
        upper = float(threshold_spec)
        lower = None
    elif isinstance(threshold_spec, str):
        spec = threshold_spec.strip()
        if not spec:
            raise ValueError("rollout_is_threshold must not be empty.")
        if "_" in spec:
            lo, hi = spec.split("_", 1)
            lower, upper = float(lo), float(hi)
        else:
            upper = float(spec)
            lower = None
    else:
        raise TypeError(
            f"rollout_is_threshold must be float or str, got {type(threshold_spec)}"
        )
    if upper <= 0:
        raise ValueError(f"rollout_is_threshold upper must be positive, got {upper}")
    if lower is not None:
        if lower <= 0:
            raise ValueError(
                f"rollout_is_threshold lower must be positive, got {lower}"
            )
        if lower > upper:
            raise ValueError("rollout_is_threshold lower must be <= upper.")
    return upper, lower


def _parse_rollout_rs_thresholds(
    options: list[str],
    threshold_spec: str | float | None,
) -> dict[str, dict[str, float | None]]:
    """Parse per-option RS thresholds."""
    if threshold_spec is None:
        raise ValueError(
            "rollout_rs_threshold must be provided for rejection sampling."
        )
    if isinstance(threshold_spec, (int, float)):
        raw_specs: list[str] = [str(threshold_spec)]
    elif isinstance(threshold_spec, str):
        raw_specs = [p.strip() for p in threshold_spec.split(",") if p.strip()]
    else:
        raise TypeError(
            f"rollout_rs_threshold must be str or numeric, got {type(threshold_spec)}"
        )
    if not raw_specs:
        raise ValueError("rollout_rs_threshold must contain at least one value.")
    if len(raw_specs) not in (1, len(options)):
        raise ValueError(
            f"rollout_rs_threshold expects 1 or {len(options)} thresholds, got {len(raw_specs)}."
        )
    if len(raw_specs) == 1 and len(options) > 1:
        raw_specs = raw_specs * len(options)

    thresholds: dict[str, dict[str, float | None]] = {}
    for option, spec in zip(options, raw_specs):
        if option.endswith("k1"):
            if "_" in spec:
                lo_s, hi_s = spec.split("_", 1)
            else:
                hi_s = spec
                lo_s = str(1.0 / float(hi_s))
            lower_v, upper_v = float(lo_s), float(hi_s)
            if lower_v <= 0 or upper_v <= 0:
                raise ValueError(
                    f"Thresholds for '{option}' must be positive, got {spec}."
                )
            thresholds[option] = {"lower": lower_v, "upper": upper_v}
        else:
            if "_" in spec:
                raise ValueError(
                    f"RS threshold for '{option}' must be a single upper bound, got '{spec}'."
                )
            upper_v = float(spec)
            if upper_v <= 0:
                raise ValueError(
                    f"Threshold for '{option}' must be positive, got {spec}."
                )
            thresholds[option] = {"lower": None, "upper": upper_v}
    return thresholds


# ---------------------------------------------------------------------------
# Rejection sampling (11 criteria)
# ---------------------------------------------------------------------------


def compute_rollout_rejection_mask(
    log_ratio: Tensor,
    response_mask: Tensor,
    rollout_rs: str = "token_k1",
    rollout_rs_threshold: str | float | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute hard trust region mask using divergence estimators.

    Returns (modified_response_mask, metrics).
    """
    if rollout_rs is None or not isinstance(rollout_rs, str):
        raise ValueError("rollout_rs must be a non-empty string.")
    if rollout_rs_threshold is None:
        raise ValueError("rollout_rs_threshold must be provided.")
    if log_ratio.shape[0] == 0:
        return response_mask, {}

    option_modes = [o.strip() for o in rollout_rs.split(",") if o.strip()]
    if not option_modes:
        raise ValueError("rollout_rs must contain at least one option.")

    normalized: list[str] = []
    seen: set[str] = set()
    for o in option_modes:
        if o not in SUPPORTED_ROLLOUT_RS_OPTIONS:
            raise ValueError(
                f"Invalid rollout_rs option: {o}. Must be one of {sorted(SUPPORTED_ROLLOUT_RS_OPTIONS)}."
            )
        if o not in seen:
            normalized.append(o)
            seen.add(o)

    threshold_specs = _parse_rollout_rs_thresholds(normalized, rollout_rs_threshold)

    log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    token_k1 = -log_ratio_safe
    token_k2 = 0.5 * log_ratio_safe**2
    token_k3 = torch.exp(log_ratio_safe) - 1.0 - log_ratio_safe

    response_mask_bool = response_mask.bool()
    combined_mask = torch.ones_like(response_mask, dtype=log_ratio.dtype)
    metrics: dict[str, float] = {}

    def _seq_sum(v: Tensor) -> Tensor:
        return _masked_sum(v, response_mask, axis=-1)

    def _seq_mean(v: Tensor) -> Tensor:
        return _masked_mean(v, response_mask, axis=-1)

    def _seq_max(v: Tensor) -> Tensor:
        neg_inf = torch.tensor(float("-inf"), device=v.device, dtype=v.dtype)
        masked_v = v.masked_fill(~response_mask_bool, neg_inf)
        mx = masked_v.max(dim=-1).values
        return torch.where(mx == neg_inf, torch.zeros_like(mx), mx)

    for opt in normalized:
        th = threshold_specs[opt]
        is_k1 = opt.endswith("k1")
        upper_v = th["upper"]
        lower_v = th["lower"]

        lower_log: float | None = None
        upper_log: float | None = None
        if is_k1:
            lower_log = math.log(lower_v)  # type: ignore[arg-type]
            upper_log = math.log(upper_v)  # type: ignore[arg-type]

        token_keep_bool: Tensor

        if opt == "token_k1":
            token_keep_bool = (token_k1 >= lower_log) & (token_k1 <= upper_log)  # type: ignore[arg-type]
        elif opt == "token_k2":
            token_keep_bool = token_k2 <= upper_v
        elif opt == "token_k3":
            token_keep_bool = token_k3 <= upper_v
        elif opt.startswith("seq_sum"):
            kx = (
                token_k1
                if opt.endswith("k1")
                else (token_k2 if opt.endswith("k2") else token_k3)
            )
            seq_stat = _seq_sum(kx)
            if is_k1:
                seq_keep = (seq_stat >= lower_log) & (seq_stat <= upper_log)  # type: ignore[arg-type]
            else:
                seq_keep = seq_stat <= upper_v
            token_keep_bool = seq_keep.unsqueeze(-1).expand_as(response_mask_bool)
        elif opt.startswith("seq_mean"):
            kx = (
                token_k1
                if opt.endswith("k1")
                else (token_k2 if opt.endswith("k2") else token_k3)
            )
            seq_stat = _seq_mean(kx)
            if is_k1:
                seq_keep = (seq_stat >= lower_log) & (seq_stat <= upper_log)  # type: ignore[arg-type]
            else:
                seq_keep = seq_stat <= upper_v
            token_keep_bool = seq_keep.unsqueeze(-1).expand_as(response_mask_bool)
        elif opt.startswith("seq_max"):
            kx = token_k2 if opt.endswith("k2") else token_k3
            seq_stat = _seq_max(kx)
            seq_keep = seq_stat <= upper_v
            token_keep_bool = seq_keep.unsqueeze(-1).expand_as(response_mask_bool)
        else:
            raise ValueError(f"Unsupported rollout_rs option: {opt}")

        token_keep_mask = token_keep_bool.to(dtype=log_ratio.dtype)
        combined_mask = combined_mask * token_keep_mask

        token_mf = _masked_mean(1.0 - token_keep_mask, response_mask).item()
        seq_valid = response_mask.sum(dim=-1) > 0
        seq_keep_tensor = (~((~token_keep_bool) & response_mask_bool)).all(dim=-1)
        sv_f = seq_valid.float()
        if sv_f.sum() > 0:
            seq_mf = (
                ((1.0 - seq_keep_tensor.float()) * sv_f).sum() / sv_f.sum()
            ).item()
        else:
            seq_mf = 0.0
        metrics[f"rollout_rs_{opt}_masked_fraction"] = token_mf
        metrics[f"rollout_rs_{opt}_seq_masked_fraction"] = seq_mf

    metrics["rollout_rs_masked_fraction"] = _masked_mean(
        1.0 - combined_mask, response_mask
    ).item()
    final_keep = (combined_mask > 0.5) & response_mask_bool
    seq_has_masked = (~final_keep & response_mask_bool).any(dim=-1)
    metrics["rollout_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()

    modified = (response_mask * combined_mask).to(dtype=response_mask.dtype)
    return modified, metrics


# ---------------------------------------------------------------------------
# IS weight computation
# ---------------------------------------------------------------------------


def _seq_log_ratio(
    log_ratio: Tensor,
    response_mask: Tensor,
    reduction: str = "sum",
) -> Tensor:
    """Reduce a sequence's per-token log-ratios to one value per sequence.

    ``sum`` is the full sequence likelihood ratio; ``mean`` divides by the
    response length, giving the per-token geometric mean. They differ by
    orders of magnitude on long responses, so this is a deliberate choice
    rather than a normalisation detail -- see ``rollout_is_seq_reduction``.
    """
    lr_sum = _masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
    if reduction == "mean":
        seq_len = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        return lr_sum / seq_len.to(dtype=lr_sum.dtype)
    return lr_sum


def rollout_is_weights(
    log_ratio: Tensor,
    response_mask: Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: str | float = 2.0,
    rollout_is_seq_reduction: str = "sum",
) -> tuple[Tensor, Tensor]:
    """Truncated importance-sampling weights ``exp(log_ratio)`` and the raw ones
    before truncation, both masked; no metrics, so cheap enough to call per row."""
    valid_levels = {"token", "sequence"}
    if rollout_is not in valid_levels:
        raise ValueError(
            f"Invalid rollout_is: {rollout_is}. Must be one of {valid_levels}."
        )

    threshold_upper, threshold_lower = _parse_rollout_is_threshold(rollout_is_threshold)

    if rollout_is == "token":
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        raw_weights = torch.exp(log_ratio_safe)
    else:  # sequence
        lr_sum = _seq_log_ratio(log_ratio, response_mask, rollout_is_seq_reduction)
        lr_sum_safe = torch.clamp(lr_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        raw_weights = torch.exp(lr_sum_safe).expand_as(log_ratio)

    raw_weights = raw_weights * response_mask

    if threshold_lower is None:
        is_weights = raw_weights.clamp(max=threshold_upper)
    else:
        kept = (raw_weights >= threshold_lower) & (raw_weights <= threshold_upper)
        is_weights = torch.where(kept, raw_weights, torch.zeros_like(raw_weights))
    return is_weights, raw_weights


def compute_rollout_correction_weights(
    log_ratio: Tensor,
    response_mask: Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: str | float = 2.0,
    rollout_is_batch_normalize: bool = False,
    rollout_is_seq_reduction: str = "sum",
) -> tuple[Tensor, dict[str, float]]:
    """Compute truncated importance sampling weights.

    Returns (is_weights [B, T], metrics).
    """
    threshold_upper, threshold_lower = _parse_rollout_is_threshold(rollout_is_threshold)
    is_weights, raw_weights = rollout_is_weights(
        log_ratio,
        response_mask,
        rollout_is,
        rollout_is_threshold,
        rollout_is_seq_reduction,
    )

    # Metrics
    metrics: dict[str, float] = {}
    mask_bool = response_mask.bool()
    metrics["rollout_is_mean"] = _masked_mean(is_weights, response_mask).item()

    if rollout_is == "token":
        above = raw_weights > threshold_upper
        th_lower_eff = (
            threshold_lower if threshold_lower is not None else 1.0 / threshold_upper
        )
        below = raw_weights < th_lower_eff
        metrics["rollout_is_ratio_fraction_high"] = _masked_mean(
            above.float(), response_mask
        ).item()
        metrics["rollout_is_ratio_fraction_low"] = _masked_mean(
            below.float(), response_mask
        ).item()
        metrics["rollout_is_max"] = (
            is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        )
        metrics["rollout_is_min"] = (
            is_weights.masked_fill(~mask_bool, float("inf")).min().item()
        )
    else:
        lr_sum_raw = _seq_log_ratio(log_ratio, response_mask, rollout_is_seq_reduction)
        log_th_upper = torch.log(torch.tensor(threshold_upper, device=log_ratio.device))
        th_lower_eff = (
            threshold_lower if threshold_lower is not None else 1.0 / threshold_upper
        )
        log_th_lower = torch.log(torch.tensor(th_lower_eff, device=log_ratio.device))
        metrics["rollout_is_max"] = torch.exp(
            torch.clamp(lr_sum_raw.max(), max=SAFETY_BOUND)
        ).item()
        metrics["rollout_is_min"] = torch.exp(lr_sum_raw.min()).item()
        metrics["rollout_is_ratio_fraction_high"] = (
            (lr_sum_raw > log_th_upper).float().mean().item()
        )
        metrics["rollout_is_ratio_fraction_low"] = (
            (lr_sum_raw < log_th_lower).float().mean().item()
        )

    # Std
    mask_count = response_mask.sum()
    if mask_count > 1:
        w_clamped = is_weights.clamp(min=0.0, max=threshold_upper)
        m_c = _masked_mean(w_clamped, response_mask)
        var = _masked_mean(w_clamped.square(), response_mask) - m_c.square()
        metrics["rollout_is_std"] = torch.sqrt(torch.clamp(var, min=0.0)).item()
    else:
        metrics["rollout_is_std"] = 0.0

    # ESS
    w_ess = is_weights.clamp(min=0.0, max=threshold_upper)
    m_ess = _masked_mean(w_ess, response_mask)
    w_norm = w_ess / (m_ess + 1e-8)
    metrics["rollout_is_eff_sample_size"] = (
        1.0 / _masked_mean(w_norm.square(), response_mask).item()
    )

    # Sequence-level metrics
    if is_weights.dim() > 1:
        seq_mean_w = _masked_mean(is_weights, response_mask, axis=-1)
        metrics["rollout_is_seq_mean"] = seq_mean_w.mean().item()
        metrics["rollout_is_seq_std"] = (
            seq_mean_w.std().item() if seq_mean_w.numel() > 1 else 0.0
        )
        metrics["rollout_is_seq_max"] = seq_mean_w.max().item()
        metrics["rollout_is_seq_min"] = seq_mean_w.min().item()
        metrics["rollout_is_seq_max_deviation"] = (seq_mean_w - 1.0).abs().max().item()
        raw_seq_mean = _masked_mean(raw_weights, response_mask, axis=-1)
        metrics["rollout_is_seq_fraction_high"] = (
            (raw_seq_mean > threshold_upper).float().mean().item()
        )
        th_lower_eff2 = (
            threshold_lower if threshold_lower is not None else 1.0 / threshold_upper
        )
        metrics["rollout_is_seq_fraction_low"] = (
            (raw_seq_mean < th_lower_eff2).float().mean().item()
        )

    if threshold_lower is not None:
        oob = (raw_weights < threshold_lower) | (raw_weights > threshold_upper)
        metrics["rollout_is_oob_ratio"] = _masked_mean(
            oob.float(), response_mask
        ).item()

    is_weights = is_weights.detach()

    # Batch normalization
    if rollout_is_batch_normalize:
        response_mask.to(dtype=is_weights.dtype)
        if rollout_is == "token":
            w_mean = _masked_mean(is_weights, response_mask)
        else:
            seq_w = _masked_mean(is_weights, response_mask, axis=-1)
            seq_mask = (response_mask.sum(dim=-1) > 0).to(dtype=is_weights.dtype)
            w_mean = (seq_w * seq_mask).sum() / seq_mask.sum().clamp(min=1e-8)
        if w_mean > 1e-8:
            is_weights = is_weights / w_mean
            metrics["rollout_is_batch_norm_factor"] = w_mean.item()
        else:
            metrics["rollout_is_batch_norm_factor"] = 1.0

    return is_weights, metrics


# ---------------------------------------------------------------------------
# Off-policy diagnostic metrics
# ---------------------------------------------------------------------------


def compute_offpolicy_metrics(
    old_log_prob: Tensor,
    rollout_log_prob: Tensor | None,
    response_mask: Tensor,
) -> dict[str, Any]:
    """Compute off-policy diagnostics: KL, PPL, chi-squared divergence."""
    metrics: dict[str, Any] = {}

    mean_lp_train = _masked_mean(old_log_prob, response_mask, axis=-1)
    training_ppl = torch.exp(-mean_lp_train).mean()
    metrics["training_ppl"] = training_ppl.detach().item()
    metrics["training_log_ppl"] = (-mean_lp_train).mean().detach().item()

    if rollout_log_prob is not None:
        metrics["kl"] = (
            _masked_mean(rollout_log_prob - old_log_prob, response_mask).detach().item()
        )

        log_ratio = old_log_prob - rollout_log_prob
        k3 = torch.exp(log_ratio) - log_ratio - 1
        metrics["k3_kl"] = _masked_mean(k3, response_mask).detach().item()

        mean_lp_rollout = _masked_mean(rollout_log_prob, response_mask, axis=-1)
        metrics["rollout_ppl"] = torch.exp(-mean_lp_rollout).mean().detach().item()
        metrics["rollout_log_ppl"] = (-mean_lp_rollout).mean().detach().item()

        log_ppl_diff = mean_lp_rollout - mean_lp_train
        metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
        metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
        metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item()
        metrics["ppl_ratio"] = torch.exp(log_ppl_diff).mean().detach().item()

        lr_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho = torch.exp(lr_safe)
        chi2_tok = _masked_mean(rho.square(), response_mask) - 1.0
        metrics["chi2_token"] = chi2_tok.detach().item()

        lr_sum = _masked_sum(log_ratio, response_mask, axis=-1)
        lr_sum_safe = torch.clamp(lr_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_sq_seq = torch.exp(2.0 * lr_sum_safe)
        chi2_seq = rho_sq_seq.mean() - 1.0
        metrics["chi2_seq"] = chi2_seq.detach().item()

    return metrics


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

# legacy ``rollout_correction/<key>`` -> current unprefixed key.
_LEGACY_METRIC_ALIASES = {
    "is_weight_mean": "rollout_is_mean",
    "is_weight_max": "rollout_is_max",
    "kl": "kl",
    "k3_kl": "k3_kl",
}


def compute_rollout_correction_and_rejection_mask(
    old_log_prob: Tensor,
    rollout_log_prob: Tensor,
    response_mask: Tensor,
    rollout_is: str | None = None,
    rollout_is_threshold: str | float | None = 2.0,
    rollout_is_batch_normalize: bool = False,
    rollout_rs: str | None = None,
    rollout_rs_threshold: str | float | None = None,
    rollout_is_seq_reduction: str = "sum",
) -> tuple[Tensor | None, Tensor, dict[str, float]]:
    """Unified interface: IS weights + rejection mask + off-policy metrics.

    Returns (is_weights_or_None, modified_response_mask, prefixed_metrics).
    """
    log_ratio = old_log_prob - rollout_log_prob
    metrics: dict[str, float] = {}

    is_weights: Tensor | None = None
    if rollout_is is not None and rollout_is_threshold is not None:
        is_weights, is_m = compute_rollout_correction_weights(
            log_ratio,
            response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_is_threshold,
            rollout_is_batch_normalize=rollout_is_batch_normalize,
            rollout_is_seq_reduction=rollout_is_seq_reduction,
        )
        metrics.update(is_m)

    modified_mask = response_mask.clone()
    if rollout_rs is not None:
        if rollout_rs_threshold is None:
            raise ValueError(
                "rollout_rs_threshold must be provided when rollout_rs is enabled."
            )
        modified_mask, rs_m = compute_rollout_rejection_mask(
            log_ratio,
            response_mask,
            rollout_rs=rollout_rs,
            rollout_rs_threshold=rollout_rs_threshold,
        )
        metrics.update(rs_m)

    op_m = compute_offpolicy_metrics(old_log_prob, rollout_log_prob, response_mask)
    metrics.update(op_m)

    prefixed: dict[str, float] = {}
    for k, v in metrics.items():
        val = v.item() if isinstance(v, Tensor) else v
        prefixed[f"rollout_corr/{k}"] = val

    # The four keys the pre-move implementation published, kept so that
    # existing dashboards and alerts do not go blank on the new prefix.
    for legacy, new in _LEGACY_METRIC_ALIASES.items():
        if new in metrics:
            val = metrics[new]
            prefixed[f"rollout_correction/{legacy}"] = (
                val.item() if isinstance(val, Tensor) else val
            )

    return is_weights, modified_mask, prefixed


# ---------------------------------------------------------------------------
# Batch convenience wrappers
# ---------------------------------------------------------------------------


def _resolve_rollout_correction_config(config: Any) -> RolloutCorrectionConfig:
    if isinstance(config, RolloutCorrectionConfig):
        return config
    if isinstance(config, QuantizationConfig):
        return config.rollout_correction
    if isinstance(config, AtomRLConfig):
        return config.quantization.rollout_correction
    raise TypeError(f"Expected RolloutCorrectionConfig, got {type(config)!r}")


def _clean_batch_logprobs(
    batch: DataProto,
) -> tuple[Tensor, Tensor, Tensor, str]:
    """Line up the three tensors the IS ratio is built from, and de-alias zeros.

    Entry i of each scores token i+1, but the widths need not agree: the mask
    and the rollout log-probs are built pre-shifted (``[:, 1:]``, width S-1)
    while ``old_log_probs`` may be padded out to the full width S depending on
    the training backend. Same frame, so a common width lines them up.
    """
    old_lp = batch.tensors["old_log_probs"]
    rollout_lp = batch.tensors.get(
        "rollout_log_probs", batch.tensors.get("fp8_logprobs")
    )
    mask_key = "response_mask" if "response_mask" in batch.tensors else "attention_mask"
    mask = batch.tensors.get(mask_key)
    if rollout_lp is None or mask is None:
        raise KeyError(
            "rollout correction needs rollout_log_probs (or fp8_logprobs) and "
            "response_mask (or attention_mask) in the batch"
        )

    width = min(old_lp.shape[-1], rollout_lp.shape[-1], mask.shape[-1])
    old_w, roll_w, mask_w = (
        old_lp[..., :width],
        rollout_lp[..., :width],
        mask[..., :width],
    )

    # ``rollout_log_probs`` is zero-initialised and then filled per reported
    # token, so an exact 0.0 inside the response mask is ambiguous: either the
    # engine never reported that position, or the token was so certain that
    # log p underflowed to 0. Reading an unreported position literally means
    # probability 1, the largest value possible, which saturates the IS ratio;
    # substituting the trainer's own value makes the ratio 1. For a genuinely
    # saturated token ``old_log_probs`` is ~0 there too, so the substitution
    # costs nothing. The logged mean tells the two apart.
    unreported = (roll_w == 0.0) & (mask_w > 0)
    n_unreported = int(unreported.sum())
    if n_unreported:
        subst_mean = float(old_w[unreported].mean())
        roll_w = torch.where(unreported, old_w, roll_w)
        patched = rollout_lp.clone()
        patched[..., :width] = roll_w
        batch.tensors["rollout_log_probs"] = patched
        logger.info(
            "rollout_log_probs: %d of %d response positions are exactly 0.0; "
            "substituted old_log_probs (mean %.4g there -- near 0 means the "
            "token really was certain, large means a reporting gap)",
            n_unreported,
            int(mask_w.sum()),
            subst_mean,
        )

    return old_w, roll_w, mask_w, mask_key


def compute_rollout_correction_and_add_to_batch(
    batch: DataProto,
    config: Any,
) -> tuple[DataProto, dict[str, float]]:
    """Compute IS weights + rejection and update batch in-place.

    Reads old_log_probs, rollout_log_probs, response_mask from batch.tensors.
    Updates response_mask (always) and rollout_is_weights (if IS enabled).
    """
    rcfg = _resolve_rollout_correction_config(config)
    r_is = rcfg.rollout_is or None
    r_is_th = rcfg.rollout_is_threshold or None
    r_rs = rcfg.rollout_rs or None
    r_rs_th = rcfg.rollout_rs_threshold or None

    old_w, roll_w, mask_w, mask_key = _clean_batch_logprobs(batch)

    is_weights, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_w,
        rollout_log_prob=roll_w,
        response_mask=mask_w,
        rollout_is=r_is,
        rollout_is_threshold=r_is_th,
        rollout_is_batch_normalize=rcfg.rollout_is_batch_normalize,
        rollout_rs=r_rs,
        rollout_rs_threshold=r_rs_th,
        rollout_is_seq_reduction=getattr(rcfg, "rollout_is_seq_reduction", "sum"),
    )

    # The width alignment above may have narrowed the mask; splice the result
    # back rather than reshaping the batch, since the columns past the common
    # width were never compared and other consumers expect the full width.
    original_mask = batch.tensors[mask_key]
    if modified_mask.shape[-1] == original_mask.shape[-1]:
        batch.tensors[mask_key] = modified_mask
    else:
        spliced = original_mask.clone()
        spliced[..., : modified_mask.shape[-1]] = modified_mask
        batch.tensors[mask_key] = spliced
    if is_weights is not None:
        batch.tensors["rollout_is_weights"] = is_weights

    return batch, metrics


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: Tensor,
    rollout_log_prob: Tensor,
    response_mask: Tensor,
) -> dict[str, float]:
    """Compute off-policy metrics during training (current vs rollout policy)."""
    op = compute_offpolicy_metrics(log_prob, rollout_log_prob, response_mask)
    return {
        f"rollout_corr/{k}": (v.item() if isinstance(v, Tensor) else v)
        for k, v in op.items()
    }


# ---------------------------------------------------------------------------
# Bypass mode
# ---------------------------------------------------------------------------


def apply_bypass_mode(batch: DataProto, config: Any) -> None:
    """Set pi_old = pi_rollout (skip old_log_prob computation)."""
    if "rollout_log_probs" not in batch.tensors:
        raise ValueError(
            "bypass_mode=True requires rollout_log_probs in batch. "
            "Ensure calculate_log_probs=true in rollout config."
        )
    batch.tensors["old_log_probs"] = batch.tensors["rollout_log_probs"]


# ---------------------------------------------------------------------------
# Legacy API (backward compat with atomrl/quantization/rollout_correction.py)
# ---------------------------------------------------------------------------


def token_level_tis(
    bf16_logprobs: Tensor,
    fp8_logprobs: Tensor,
    advantages: Tensor,
    clip: float = 1.5,
) -> Tensor:
    """Truncated importance sampling correction on advantages (legacy)."""
    log_ratio = torch.clamp(bf16_logprobs - fp8_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    return torch.clamp(ratio, 1.0 / clip, clip) * advantages


def token_level_mis(
    bf16_logprobs: Tensor,
    fp8_logprobs: Tensor,
    advantages: Tensor,
) -> Tensor:
    """Self-normalizing MIS correction on advantages (legacy)."""
    log_ratio = torch.clamp(bf16_logprobs - fp8_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    return (ratio / ratio.mean().clamp(min=1e-8)) * advantages


def _pick_fp8_logprobs(batch: DataProto) -> Tensor:
    for key in ("fp8_logprobs", "fp8_log_probs"):
        if key in batch.tensors:
            return batch[key]
    raise KeyError("DataProto must contain 'fp8_logprobs' or 'fp8_log_probs'.")


def _pick_bf16_logprobs(batch: DataProto) -> Tensor:
    for key in ("bf16_logprobs", "old_log_probs", "ref_log_probs"):
        if key in batch.tensors:
            return batch[key]
    raise KeyError(
        "DataProto must contain 'bf16_logprobs', 'old_log_probs', or 'ref_log_probs'."
    )


def apply_rollout_correction(batch: DataProto, config: Any) -> DataProto:
    """Apply legacy TIS/MIS correction to advantages (backward compat)."""
    rcfg = _resolve_rollout_correction_config(config)
    if not rcfg.enabled:
        return batch

    bf16_lp = _pick_bf16_logprobs(batch)
    fp8_lp = _pick_fp8_logprobs(batch)
    if "advantages" not in batch.tensors:
        raise KeyError("DataProto must contain 'advantages' for rollout correction.")
    advantages = batch["advantages"]

    method = rcfg.method.lower().strip()
    if method == "tis":
        corrected = token_level_tis(bf16_lp, fp8_lp, advantages, clip=rcfg.clip)
    elif method in {"mis", "multiplicative"}:
        corrected = token_level_mis(bf16_lp, fp8_lp, advantages)
    else:
        raise ValueError(f"Unknown rollout correction method: {rcfg.method!r}")

    out = DataProto(tensors=dict(batch.tensors), meta=dict(batch.meta))
    out["advantages"] = corrected
    out.meta["rollout_correction"] = {"method": method, "clip": rcfg.clip}
    logger.debug("apply_rollout_correction: method=%s", method)
    return out


def compute_rollout_is_weights(
    old_log_probs: Tensor,
    rollout_log_probs: Tensor,
    response_mask: Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float | str = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Compute IS weights (legacy API, delegates to new implementation)."""
    log_ratio = old_log_probs - rollout_log_probs
    is_weights, metrics = compute_rollout_correction_weights(
        log_ratio,
        response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
    )
    prefixed = {f"rollout_correction/{k}": v for k, v in metrics.items()}
    return is_weights * response_mask, prefixed


def apply_rejection_sampling(
    response_mask: Tensor,
    is_weights: Tensor,
    threshold: float = 0.0,
) -> Tensor:
    """Zero out response_mask where IS weight exceeds threshold (legacy)."""
    if threshold <= 0.0:
        return response_mask
    modified = response_mask.clone()
    modified[is_weights > threshold] = 0
    return modified


# ---------------------------------------------------------------------------
# Factory presets on RolloutCorrectionConfig
# ---------------------------------------------------------------------------


def preset_decoupled_token_is(threshold: float = 2.0) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_is="token", rollout_is_threshold=str(threshold)
    )


def preset_decoupled_seq_is(threshold: float = 2.0) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_is="sequence", rollout_is_threshold=str(threshold)
    )


def preset_decoupled_token_icepop(
    lower: float = 0.5, upper: float = 5.0
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_is="token", rollout_is_threshold=f"{lower}_{upper}"
    )


def preset_decoupled_seq_is_rs(
    is_threshold: float = 2.0,
    rs_threshold: str = "0.5_2.0",
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_is="sequence",
        rollout_is_threshold=str(is_threshold),
        rollout_rs="seq_sum_k1",
        rollout_rs_threshold=rs_threshold,
    )


def preset_decoupled_geo_rs(threshold: str = "0.999_1.001") -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_rs="seq_mean_k1", rollout_rs_threshold=threshold
    )


def preset_decoupled_geo_rs_seq_tis(
    is_threshold: float = 2.0,
    rs_threshold: str = "0.999_1.001",
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_is="sequence",
        rollout_is_threshold=str(is_threshold),
        rollout_rs="seq_mean_k1",
        rollout_rs_threshold=rs_threshold,
    )


def preset_decoupled_geo_rs_token_tis(
    is_threshold: float = 2.0,
    rs_threshold: str = "0.999_1.001",
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_is="token",
        rollout_is_threshold=str(is_threshold),
        rollout_rs="seq_mean_k1",
        rollout_rs_threshold=rs_threshold,
    )


def preset_decoupled_k3_rs(threshold: float = 0.005) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_rs="seq_mean_k3", rollout_rs_threshold=str(threshold)
    )


def preset_decoupled_k3_rs_seq_tis(
    is_threshold: float = 2.0,
    rs_threshold: float = 0.005,
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_is="sequence",
        rollout_is_threshold=str(is_threshold),
        rollout_rs="seq_mean_k3",
        rollout_rs_threshold=str(rs_threshold),
    )


def preset_decoupled_k3_rs_token_tis(
    is_threshold: float = 2.0,
    rs_threshold: float = 0.005,
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        rollout_is="token",
        rollout_is_threshold=str(is_threshold),
        rollout_rs="seq_mean_k3",
        rollout_rs_threshold=str(rs_threshold),
    )


def preset_bypass_ppo_clip() -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(bypass_mode=True, loss_type="ppo_clip")


def preset_bypass_ppo_clip_geo_rs(
    rs_threshold: str = "0.999_1.001",
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        bypass_mode=True,
        loss_type="ppo_clip",
        rollout_rs="seq_mean_k1",
        rollout_rs_threshold=rs_threshold,
    )


def preset_bypass_ppo_clip_k3_rs(
    rs_threshold: float = 0.005,
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        bypass_mode=True,
        loss_type="ppo_clip",
        rollout_rs="seq_mean_k3",
        rollout_rs_threshold=str(rs_threshold),
    )


def preset_bypass_pg_is(threshold: float = 2.0) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        bypass_mode=True,
        loss_type="reinforce",
        rollout_is="sequence",
        rollout_is_threshold=str(threshold),
    )


def preset_bypass_pg_geo_rs(
    rs_threshold: str = "0.999_1.001",
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        bypass_mode=True,
        loss_type="reinforce",
        rollout_rs="seq_mean_k1",
        rollout_rs_threshold=rs_threshold,
    )


def preset_bypass_pg_geo_rs_seq_tis(
    is_threshold: float = 2.0,
    rs_threshold: str = "0.999_1.001",
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        bypass_mode=True,
        loss_type="reinforce",
        rollout_is="sequence",
        rollout_is_threshold=str(is_threshold),
        rollout_rs="seq_mean_k1",
        rollout_rs_threshold=rs_threshold,
    )


def preset_bypass_pg_geo_rs_token_tis(
    is_threshold: float = 2.0,
    rs_threshold: str = "0.999_1.001",
) -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig(
        bypass_mode=True,
        loss_type="reinforce",
        rollout_is="token",
        rollout_is_threshold=str(is_threshold),
        rollout_rs="seq_mean_k1",
        rollout_rs_threshold=rs_threshold,
    )


def preset_disabled() -> RolloutCorrectionConfig:
    return RolloutCorrectionConfig()
