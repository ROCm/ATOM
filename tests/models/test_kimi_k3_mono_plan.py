# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The K3 mono decode's plain-Python decisions: the TP shard, the AttnRes block
bookkeeping, and which steps the kernels are allowed to serve.

These are the decisions that, taken wrongly, make a kernel read the right bytes
at the wrong offset -- so each is pinned here rather than only inside a kernel
that needs a GPU to run.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono.config import (
    ATTN_RES_BLOCK,
    HEAD_DIM,
    HIDDEN,
    KDA_HEADS,
    MAX_ROWS,
    SUPPORTED_TP,
    Dims,
    layer_block,
    num_blocks,
)
from atom.mono.plan.shard import shard_refusal

LAYERS = 93  # Kimi-K3


def test_tp8_shard_matches_the_kernel_constants():
    """The one place the widths are derived, against the ones K1 bakes."""
    kda = pytest.importorskip("atom.models.kimi_k3.mono.kernels.attention.kda")
    d = Dims(8)
    assert (d.heads, d.proj, d.qkv) == (kda.NH, kda.PROJ, kda.QKV)
    assert (d.fused, d.nproj) == (kda.FUSED, kda.NPROJ)
    assert kda.HIDDEN == HIDDEN and kda.HD == HEAD_DIM
    # the fused hidden projection is [q | k | v | g | beta | f_a], which is the
    # order KimiKDAAttention.process_weights_after_loading builds
    assert kda.G2_0 == 3 * d.proj
    assert kda.BETA_0 == 4 * d.proj
    assert kda.FA_0 == kda.BETA_0 + d.heads
    assert kda.FA_0 + HEAD_DIM == d.fused


def test_tp8_moe_widths_match_the_kernel_constants():
    moe = pytest.importorskip("atom.models.kimi_k3.mono.kernels.stages.moe")
    d = Dims(8)
    assert (d.inter, d.sh_inter, d.up_n) == (moe.RI, moe.SI, moe.UP_N)


def test_supported_tp_is_what_dims_accepts():
    for tp in SUPPORTED_TP:
        assert shard_refusal(Dims, tp) is None


@pytest.mark.parametrize("tp", [5, 7, 9, 96 + 1])
def test_dims_refuses_a_tp_that_does_not_divide_the_widths(tp):
    why = shard_refusal(Dims, tp)
    assert why is not None and str(tp) in why


def test_attn_res_block_bookkeeping():
    """A block closes every ATTN_RES_BLOCK layers; layer 12 j writes block j and
    layer k mixes ceil(k / 12) blocks before it."""
    assert layer_block(0) == (0, 0)
    assert layer_block(1) == (1, -1)
    assert layer_block(ATTN_RES_BLOCK) == (1, 1)
    assert layer_block(ATTN_RES_BLOCK + 1) == (2, -1)
    assert layer_block(LAYERS - 1) == (8, -1)
    assert num_blocks(LAYERS) == 8
    # every write lands in a distinct slot inside the buffer
    writes = [w for _, w in map(layer_block, range(LAYERS)) if w >= 0]
    assert writes == sorted(set(writes))
    assert max(writes) < num_blocks(LAYERS)


def _metadata(**over):
    """A KDA metadata of the one shape the kernels serve."""
    base = {
        "spec_sequence_masks": object(),
        "num_prefills": 0,
        "num_decodes": 0,
        "num_spec_decodes": 2,
        "num_actual_tokens": 8,
        "spec_state_indices_tensor": torch.zeros(2, 4, dtype=torch.int32),
        "num_accepted_tokens": torch.ones(2, dtype=torch.int32),
        "replayssm": False,
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_step_rows_accepts_a_uniform_spec_verify():
    from atom.models.kimi_k3.mono.plan import step_rows

    assert step_rows(_metadata(), 8) == 4


@pytest.mark.parametrize(
    "over",
    [
        {"spec_sequence_masks": None},  # not a spec step
        {"num_prefills": 1},  # a prefill in the batch
        {"num_decodes": 1},  # a plain decode mixed in
        {"num_spec_decodes": 0},
        {"replayssm": True},  # only column 0 of the slot table is live
        {"num_accepted_tokens": None},
        {"num_actual_tokens": 7},  # disagrees with the rows
        {"spec_state_indices_tensor": torch.zeros(2, 4, dtype=torch.int64)},
    ],
)
def test_step_rows_refuses_every_other_shape(over):
    from atom.models.kimi_k3.mono.plan import step_rows

    assert step_rows(_metadata(**over), 8) is None


def test_step_rows_refuses_a_step_wider_than_the_buffers():
    from atom.models.kimi_k3.mono.plan import step_rows

    rows = MAX_ROWS * 2
    wide = _metadata(
        num_spec_decodes=rows // 4,
        num_actual_tokens=rows,
        spec_state_indices_tensor=torch.zeros(rows // 4, 4, dtype=torch.int32),
        num_accepted_tokens=torch.ones(rows // 4, dtype=torch.int32),
    )
    assert step_rows(wide, rows) is None


def test_step_rows_refuses_a_ragged_batch():
    from atom.models.kimi_k3.mono.plan import step_rows

    # three requests of 4 tokens would be 12 rows, not the 8 the step reports
    assert step_rows(_metadata(num_spec_decodes=3), 8) is None


def _spec_metadata(requests: int, qlen: int):
    """A uniform spec verify of ``requests`` requests of ``qlen`` tokens."""
    return _metadata(
        num_spec_decodes=requests,
        num_actual_tokens=requests * qlen,
        spec_state_indices_tensor=torch.zeros(requests, qlen, dtype=torch.int32),
        num_accepted_tokens=torch.ones(requests, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("cap", "requests", "qlen", "served"),
    [
        (8, 1, 8, True),  # bs=1 x MTP7
        (8, 2, 8, False),  # bs=2 x MTP7 declines under a cap of 8...
        (16, 1, 8, True),
        (16, 2, 8, True),  # ...and is served at 16 (the shipped cap), the bs=2 graph
        (16, 1, 16, True),
        (16, 3, 8, False),  # 24 rows: past either cap
        (16, 2, 9, False),  # 18 rows
    ],
)
def test_the_row_cap_decides_which_uniform_steps_are_served(
    monkeypatch, cap, requests, qlen, served
):
    from atom.models.kimi_k3.mono import plan

    monkeypatch.setattr(plan, "MAX_ROWS", cap)
    rows = requests * qlen
    got, why = plan.step_shape(_spec_metadata(requests, qlen), rows)
    assert (got == qlen) is served
    if not served:
        assert got is None and f"{rows} rows > the {cap}-row buffers" in why


@pytest.mark.parametrize(
    ("rows", "qlen"),
    [(1, 1), (8, 8), (8, 2), (4, 4), (6, 3), (16, 8), (16, 16), (16, 1), (12, 4)],
)
def test_width_key_round_trips(rows, qlen):
    from atom.models.kimi_k3.mono.plan import _width, width_key

    assert _width(width_key(rows, qlen)) == (rows, qlen)


def test_width_keys_stay_distinct_up_to_the_widest_cap():
    from atom.models.kimi_k3.mono.config import MAX_QLEN, MAX_ROWS
    from atom.models.kimi_k3.mono.plan import width_key

    rows = MAX_ROWS
    keys = {width_key(r, q) for r in range(1, rows + 1) for q in range(1, MAX_QLEN + 1)}
    assert len(keys) == rows * MAX_QLEN


def test_kda_head_count_is_the_model_s():
    assert KDA_HEADS % max(SUPPORTED_TP) == 0


# --------------------------------------------------- the block's thread budget
# The kernels build for at most 16 rows (``layer.build``'s assert). A stage that
# covers its work in a single pass over the block is only correct while that
# work fits in ``THREADS`` threads -- past that it silently computes the first
# ``THREADS`` items and leaves the rest of the step uncomputed, which is what
# made a 16-row step answer rows 8-15 with garbage. These are the single-pass
# stages and what each one has to cover.
BUILD_ROWS = (8, 16)


def _single_pass_items(s: int) -> dict[str, int]:
    kda = pytest.importorskip("atom.models.kimi_k3.mono.kernels.attention.kda")
    back = pytest.importorskip("atom.models.kimi_k3.mono.kernels.attention.back")
    moe = pytest.importorskip("atom.models.kimi_k3.mono.kernels.stages.moe")
    ns = ATTN_RES_BLOCK + 1  # the most sources an AttnRes seam ever mixes
    return {
        "kda.stage_mix mix": s * kda.QC,
        "kda.stage_mix weights": s * (ns + 1),
        "kda.stage_mix msq fan-in": 8 * s,
        "kda.gemv_finish": s * kda.ROWS,
        "back.stage_o": s * (back.ROWS // 2),
        "moe.load_route route": s * 2 * moe.TOPK,
        "moe.load_route flags": s * moe.TOPK,
        "moe.stage_ug epilogue": s * 32,
        "moe.early_out": s * moe.ROWS,
        "moe.stage_down fold": s * (moe.ROWS // 2),
        "moe.stage_sdown": s * (moe.ROWS // 2),
        "moe.stage_up": s * (moe.ROWS // 2),
    }


@pytest.mark.parametrize("rows", BUILD_ROWS)
def test_single_pass_stages_fit_the_block(rows):
    """Raising the row cap past these is not a constant change: each name here
    has to become a strided pass first."""
    from atom.mono.plan.execution import THREADS

    over = {n: v for n, v in _single_pass_items(rows).items() if v > THREADS}
    assert not over, f"{rows} rows: single-pass stages past {THREADS} threads: {over}"


@pytest.mark.parametrize("rows", BUILD_ROWS)
def test_the_per_128_column_stages_must_stride(rows):
    """``back.stage_reduce``, ``moe.stage_lnorm`` and ``moe.stage_reduce`` each
    own 128 columns and give a thread to every (token, column pair), so they
    have ``s * 64`` items -- over the block from 9 rows on. They are strided for
    that reason; this pins the arithmetic so the loops are not 'simplified' back
    into a single pass."""
    from atom.mono.plan.execution import THREADS

    kda = pytest.importorskip("atom.models.kimi_k3.mono.kernels.attention.kda")
    items = rows * (kda.COLS // 2)
    assert items == rows * 64
    assert (items > THREADS) == (rows > THREADS // 64)


def test_bind_moe_does_not_refuse_a_dual_stream_moe():
    """A dual-stream MoE must still bind: mono never calls the module.

    `_use_dual_stream` only selects a branch inside `KimiSparseMoeBlock.forward`,
    and a mono layer computes the latent MoE in K2 instead of calling it. The
    alt stream cannot contend either -- `dual_stream_moe_forward` joins before
    returning. Refusing on it disarmed mono in the default environment (shared
    experts present, alt stream set, threshold 1024), which silently scored the
    production path in three separate GSM8K runs, so this is pinned here.
    """
    from atom.models.kimi_k3.mono.weights import bind_moe

    moe = SimpleNamespace(
        use_latent_moe=True,
        shared_experts=object(),
        routed_expert_norm=object(),
        _use_dual_stream=True,
    )
    layer = SimpleNamespace(block_sparse_moe=moe)

    # This fake carries no gate or expert tensors, so binding still fails --
    # the point is only that it gets PAST the dual-stream check to fail on
    # something else. Catching Exception keeps the assertion about the one
    # thing under test rather than about which later check happens to fire.
    with pytest.raises(Exception) as refusal:
        bind_moe(layer, 0, Dims(tp=8), rank=0)
    assert "dual-stream" not in str(refusal.value)
