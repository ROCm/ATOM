# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Expert-parallel vs tensor-parallel-only parity (placeholder for multi-GPU CI)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.multigpu, pytest.mark.moe]


def test_ep_output_matches_tp_only() -> None:
    """Placeholder: once EP resharding lands, compare logits against TP-only baseline."""
    pytest.skip("Requires multi-GPU MoE expert-parallel harness (not wired in CI yet).")
