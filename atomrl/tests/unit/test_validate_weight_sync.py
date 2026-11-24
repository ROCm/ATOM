# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import pytest
import torch
from examples.GRPO.dsv4 import validate_weight_sync

compare_snapshots = validate_weight_sync.compare_snapshots


def test_compare_snapshots_reports_token_and_logprob_differences() -> None:
    before = {
        "token_ids": [[1, 2, 3], [4, 5]],
        "logprobs": [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
    }
    after = {
        "token_ids": [[1, 2, 3], [4, 6]],
        "logprobs": [[-0.1, -0.25, -0.3], [-0.4, -0.7]],
    }

    summary = compare_snapshots(before, after)

    assert summary["exact_token_matches"] == 1
    assert summary["total"] == 2
    assert summary["first_token_mismatch"] == 1
    assert summary["matching_logprob_mae"] == pytest.approx(0.05 / 3)


def test_compare_cross_engine_logprobs_uses_only_response_tokens() -> None:
    assert hasattr(validate_weight_sync, "compare_cross_engine_logprobs")
    rollout = torch.tensor([[-1.0, -2.0, -3.0]])
    megatron = torch.tensor([[-1.5, -1.0, -5.0]])
    response_mask = torch.tensor([[True, False, True]])

    summary = validate_weight_sync.compare_cross_engine_logprobs(
        rollout,
        megatron,
        response_mask,
    )

    assert summary["token_count"] == 2
    assert summary["rollout_minus_megatron_mean"] == pytest.approx(1.25)
    assert summary["logprob_mae"] == pytest.approx(1.25)
    assert summary["rollout_nll"] == pytest.approx(2.0)
    assert summary["megatron_nll"] == pytest.approx(3.25)


def test_skip_sync_flag_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hasattr(validate_weight_sync, "_skip_sync_requested")
    monkeypatch.delenv("ATOMRL_SYNC_VALIDATE_SKIP_SYNC", raising=False)
    assert validate_weight_sync._skip_sync_requested() is False
    monkeypatch.setenv("ATOMRL_SYNC_VALIDATE_SKIP_SYNC", "1")
    assert validate_weight_sync._skip_sync_requested() is True
