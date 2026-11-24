# Copyright 2025 The AtomRL Authors.
# Derived from verl (verl-project/verl). See atomrl/algorithms/rollout_correction.py
# for full attribution and docs/rollout_corr.md for references.
"""Backward-compatible re-exports.

Primary implementation moved to atomrl.algorithms.rollout_correction.
"""

from atomrl.algorithms.rollout_correction import (  # noqa: F401
    SAFETY_BOUND,
    SUPPORTED_ROLLOUT_RS_OPTIONS,
    TOKEN_LEVEL_ROLLOUT_RS_OPTIONS,
    apply_bypass_mode,
    apply_rejection_sampling,
    apply_rollout_correction,
    compute_offpolicy_metrics,
    compute_rollout_corr_metrics_from_logprobs,
    compute_rollout_correction_and_add_to_batch,
    compute_rollout_correction_and_rejection_mask,
    compute_rollout_correction_weights,
    compute_rollout_is_weights,
    compute_rollout_rejection_mask,
    token_level_mis,
    token_level_tis,
)
