# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The reported DAPO loss must not depend on how the batch is sharded or packed.

``agg_loss`` already divides by the GLOBAL ``batch_num_tokens`` and scales by
``dp_size``, so the sum over accumulation units is the globally-normalized
objective -- the quantity that was differentiated. Both engines used to divide
that sum a second time, by a count that is a property of the *configuration*
rather than of the model:

* the Megatron path divided by ``n_rows``, the per-rank row count
  (``global_batch / DP``);
* the FSDP path divided by the micro-batch count, set by
  ``max_token_len_per_gpu``.

The consequence was a reported loss that moved by ~17x between two runs that were
optimizing the same objective to within 2% (measured on Qwen3-8B, DP=2 vs DP=8).
These tests pin the invariant directly, at the reduction level, so neither path
can reintroduce a divisor.
"""

import re

import pytest
import torch
from atomrl.algorithms.loss_functions import agg_loss, reduce_reported_loss

# --- the reduction the engines report -----------------------------------------
#
# These are the tests that fail on the unfixed tree: the defect was in the
# reporting, not in agg_loss, so an agg_loss-only test passes either way.


def test_reported_loss_is_the_sum_not_the_mean():
    """Averaging here is what made the metric depend on the unit count."""
    assert reduce_reported_loss([0.1, 0.2, 0.3]) == pytest.approx(0.6)


def test_reported_loss_does_not_depend_on_how_units_are_grouped():
    """Same total objective split 2, 3 or 6 ways must report the same number.

    Directly mirrors the two production divisors: rows-per-rank (global_batch/DP)
    on the Megatron path and micro-batch count (max_token_len_per_gpu) on the
    FSDP path. Under the old mean, these three report 0.3, 0.2 and 0.1.
    """
    total = 0.6
    for units in ([0.3, 0.3], [0.2, 0.2, 0.2], [0.1] * 6):
        assert reduce_reported_loss(units) == pytest.approx(total)


def test_a_bare_scalar_passes_through():
    """The Megatron path hands in an already-summed float, not a list."""
    assert reduce_reported_loss(0.42) == pytest.approx(0.42)


def test_both_engines_route_through_the_shared_reduction():
    """Pin that no call site re-implements the reduction.

    The bug existed because each engine had its own inline reduction and they
    disagreed. If a future edit inlines one again, this fails.

    The worker and Megatron engine share the same reduction helper.
    """
    import inspect

    from atomrl.engine.training import megatron_base_engine
    from atomrl.workers import actor_worker

    for mod, fn in (
        (megatron_base_engine, "engine_update_policy"),
        (actor_worker, "update_policy"),
    ):
        cls = next(
            o
            for _, o in inspect.getmembers(mod, inspect.isclass)
            if hasattr(o, fn) and o.__module__ == mod.__name__
        )
        src = inspect.getsource(getattr(cls, fn))
        assert "reduce_reported_loss" in src, f"{mod.__name__}.{fn} bypasses it"
        # SFT is the one branch that legitimately averages (see the test below);
        # drop it, and nothing that remains may divide by a unit count.
        rl = re.sub(
            r'\n +if .*task_type.*"sft":.*?\n +else:\n', "\n", src, flags=re.DOTALL
        )
        assert "len(lv)" not in rl, f"{mod.__name__}.{fn} still divides by a unit count"
        assert (
            "max(1, n_rows)" not in rl
        ), f"{mod.__name__}.{fn} still divides by n_rows"


def test_sft_reports_the_mean_because_its_units_are_already_means():
    """The sum is right for RL and wrong for SFT, so the branch must stay.

    ``agg_loss`` on the RL path divides by the GLOBAL ``batch_num_tokens``, so
    units sum. ``sft_loss`` computes ``batch_num_tokens`` from the micro's own
    ``loss_mask``, so each unit is already a mean over its own tokens and a sum
    would scale by the unit count -- the very defect this file pins, mirrored.
    Unreachable today (nothing sets ``task_type``), which is exactly why it
    needs a test rather than a reader noticing.
    """
    import inspect

    from atomrl.workers import actor_worker

    for mod in (actor_worker,):
        cls = next(
            o
            for _, o in inspect.getmembers(mod, inspect.isclass)
            if hasattr(o, "update_policy") and o.__module__ == mod.__name__
        )
        src = inspect.getsource(cls.update_policy)
        assert 'task_type") == "sft"' in src, f"{mod.__name__} sums the SFT loss"
        assert "sum(lv) / len(lv)" in src, f"{mod.__name__} does not mean the SFT loss"


def _loss_matrix(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed per-token loss and an all-valid mask."""
    torch.manual_seed(0)
    return torch.rand(rows, cols), torch.ones(rows, cols)


def _sum_over_shards(loss_mat, loss_mask, dp_size: int, total_tokens: int) -> float:
    """Reduce as the engines do: each DP rank aggregates its own slice, then sum.

    This mirrors the production contract -- every rank calls ``agg_loss`` with the
    GLOBAL token count and the DP width, and the reported metric is the sum of
    what the units produced.
    """
    rows_per_rank = loss_mat.shape[0] // dp_size
    total = 0.0
    for r in range(dp_size):
        sl = slice(r * rows_per_rank, (r + 1) * rows_per_rank)
        total += float(
            agg_loss(
                loss_mat[sl],
                loss_mask[sl],
                "token-mean",
                dp_size=dp_size,
                batch_num_tokens=total_tokens,
            )
        )
    return total / dp_size  # metric is averaged across ranks, not summed


def test_token_mean_is_invariant_to_dp_width():
    """DP=1/2/4/8 over the same data must report the same loss.

    This is the Megatron half of the bug: dividing by ``n_rows`` made the value
    scale with ``global_batch / DP``.
    """
    loss_mat, loss_mask = _loss_matrix(8, 16)
    total_tokens = int(loss_mask.sum().item())
    values = [
        _sum_over_shards(loss_mat, loss_mask, dp, total_tokens) for dp in (1, 2, 4, 8)
    ]
    for dp, v in zip((2, 4, 8), values[1:]):
        assert abs(v - values[0]) < 1e-5, (
            f"DP={dp} reports {v:.6f}, DP=1 reports {values[0]:.6f} -- "
            "token-mean must not depend on the DP width"
        )


def test_token_mean_is_invariant_to_micro_batch_packing():
    """Splitting one rank's rows into more micro-batches must not change the sum.

    This is the FSDP half: averaging over micro-batches made the value scale with
    ``max_token_len_per_gpu``.
    """
    loss_mat, loss_mask = _loss_matrix(8, 16)
    total_tokens = int(loss_mask.sum().item())

    def summed(chunks: int) -> float:
        per = loss_mat.shape[0] // chunks
        return sum(
            float(
                agg_loss(
                    loss_mat[i * per : (i + 1) * per],
                    loss_mask[i * per : (i + 1) * per],
                    "token-mean",
                    dp_size=1,
                    batch_num_tokens=total_tokens,
                )
            )
            for i in range(chunks)
        )

    whole = summed(1)
    for chunks in (2, 4, 8):
        assert abs(summed(chunks) - whole) < 1e-5, (
            f"{chunks} micro-batches report {summed(chunks):.6f}, one reports "
            f"{whole:.6f} -- packing must not change the reported loss"
        )


def test_token_mean_equals_the_global_mean_per_token_loss():
    """The invariant value is the plain masked mean -- an interpretable number.

    Anchors what the metric *is*, so a future change that keeps the two tests
    above passing (any constant factor does) still has to keep the scale honest.
    """
    loss_mat, loss_mask = _loss_matrix(8, 16)
    total_tokens = int(loss_mask.sum().item())
    got = _sum_over_shards(loss_mat, loss_mask, 4, total_tokens)
    want = float((loss_mat * loss_mask).sum() / loss_mask.sum())
    assert abs(got - want) < 1e-5, f"{got:.6f} != masked mean {want:.6f}"


def test_ragged_masks_still_normalize_globally():
    """Uneven valid-token counts per row must not bias the result.

    Real batches are ragged; if normalization were per-row or per-shard, a rank
    holding short sequences would contribute disproportionately.
    """
    torch.manual_seed(1)
    loss_mat = torch.rand(8, 16)
    loss_mask = torch.zeros(8, 16)
    for i in range(8):
        loss_mask[i, : 4 + 2 * (i % 5)] = 1.0  # 4..12 valid tokens, varying
    total_tokens = int(loss_mask.sum().item())

    want = float((loss_mat * loss_mask).sum() / loss_mask.sum())
    for dp in (1, 2, 4):
        got = _sum_over_shards(loss_mat, loss_mask, dp, total_tokens)
        assert (
            abs(got - want) < 1e-5
        ), f"DP={dp} on a ragged batch reports {got:.6f}, expected {want:.6f}"
