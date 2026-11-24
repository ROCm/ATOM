# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Backward-compatible re-exports.

Re-exports the Apache-2.0-derived implementation in
``atomrl.algorithms.rollout_correction`` without copying its implementation.
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
