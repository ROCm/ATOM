# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono width whose fused K2 spills, served unfused instead.

No GPU, no kernel build and no AITER, FlyDSL or Triton, so the non-GPU CI runs
all of it: the lean choice is a pure function over a build record, the unfused
keys are ``replace`` on a key, and the selection mapping is torch on the CPU.
The keys are stand-ins with the real keys' fields, because the real ones live
in FlyDSL kernel modules; those go through the same transforms where the
kernels import, in ``test_deepseek_v41_mono_build_key`` (a binary of their
own) and ``test_deepseek_v41_mono_contract`` (their hand-off contract). That
they are scratch-free is a compile probe, not a test.
"""

import dataclasses

import pytest
import torch

from atom.models.deepseek_v41.mono.builds import (
    RowsBuild,
    lean_width,
    unfused_keys,
    unfused_post_key,
    write_selection_rows,
)


@dataclasses.dataclass(frozen=True)
class PostKey:
    """``kernels.layer_post.LayerPostBuild``'s fields this file reads."""

    tokens: int
    index_bound_max: int
    index_fp4: bool
    index_mfma: bool
    tp: int
    tok_par: bool
    kv_fp8: bool


@dataclasses.dataclass(frozen=True)
class PreKey:
    """``kernels.attn_pre.AttnPreBuild``'s fields this file reads."""

    tokens: int
    fold: bool
    feeds_index: bool
    index_fp4: bool
    tp: int
    tok_par: bool
    kv_fp8: bool


# the 48-row TP4 step DSpark's verify runs, on the FP4 MFMA index layout: the
# build the compile probe measures at 256 VGPRs and 8 B of scratch a lane
FUSED = PostKey(
    tokens=48,
    index_bound_max=1 << 20,
    index_fp4=True,
    index_mfma=True,
    tp=4,
    tok_par=True,
    kv_fp8=True,
)
# that step's K1 builds: layer 0 and a folding layer, with and without the
# indexer query, all token parallel because the width is
K1_KEYS = [
    PreKey(
        tokens=48,
        fold=fold,
        feeds_index=feeds,
        index_fp4=True,
        tp=4,
        tok_par=True,
        kv_fp8=True,
    )
    for fold in (False, True)
    for feeds in (False, True)
]


# ── the unfused build keys ────────────────────────────────────────────────


def test_the_unfused_k2_has_no_index_stage_and_no_token_parallel():
    key = unfused_post_key(FUSED)
    assert key.index_bound_max == 0
    assert key.tok_par is False


def test_the_unfused_k2_changes_nothing_else():
    """Only the two fields: the width, the experts, the KV dtype and the index
    layout are the step's, not the variant's."""
    key = unfused_post_key(FUSED)
    back = dataclasses.replace(
        key, index_bound_max=FUSED.index_bound_max, tok_par=FUSED.tok_par
    )
    assert back == FUSED


def test_every_k1_of_an_unfused_width_loses_token_parallel():
    """A token-parallel K1 writes its own rank's rows alone, which is whose
    selection that rank's K2 indexer would have made; without the indexer in
    K2 every rank runs every row."""
    keys, post_key = unfused_keys(K1_KEYS, FUSED)
    assert post_key is not FUSED and all(key.tok_par is False for key in keys)
    # and nothing else about K1: it still projects the indexer query
    assert [dataclasses.replace(k, tok_par=True) for k in keys] == K1_KEYS
    assert [k.feeds_index for k in keys] == [k.feeds_index for k in K1_KEYS]


def test_an_unfused_build_reports_no_token_parallel():
    """``RowsBuild.tok_par`` is the build's answer: a 48-row step would run
    token parallel, its unfused build does not."""
    assert RowsBuild([], {}, post_key=FUSED).tok_par is True
    assert RowsBuild([], {}, post_key=unfused_post_key(FUSED)).tok_par is False
    assert RowsBuild([], {}).tok_par is False


# ── the lean choice ───────────────────────────────────────────────────────


def _rebuilder():
    """A rebuild that records whether it ran, returning the unfused build."""
    calls = []

    def rebuild():
        calls.append(True)
        return RowsBuild(
            [], {}, layer_post=object(), post_key=unfused_post_key(FUSED), unfused=True
        )

    return rebuild, calls


def _fused(**over):
    """A fused build record with distinguishable launcher objects."""
    return RowsBuild(
        keys=[object()],
        launchers={"a": object(), "b": object()},
        layer_post=object(),
        post_key=dataclasses.replace(FUSED, **over),
    )


def _split(**over):
    """Check mode's build: K2a and K2b launched apart."""
    return RowsBuild(
        keys=[object()],
        launchers={"a": object()},
        attn_post=object(),
        moe=object(),
        post_key=dataclasses.replace(FUSED, **over),
    )


def test_a_k2_spill_is_rebuilt_unfused():
    built = _fused()
    rebuild, calls = _rebuilder()
    lean = lean_width(built, [built.layer_post], rebuild)
    assert calls == [True]
    assert lean.unfused and lean.post_key.index_bound_max == 0
    assert lean.tok_par is False


def test_a_k2a_spill_in_check_mode_is_rebuilt_unfused():
    built = _split()
    rebuild, calls = _rebuilder()
    assert lean_width(built, [built.attn_post], rebuild).unfused
    assert calls == [True]


def test_a_k1_spill_has_nothing_leaner():
    """K1 never spills in the probe, and the unfused rebuild is not a K1
    change: a K1 spill keeps today's refusal."""
    built = _fused()
    rebuild, calls = _rebuilder()
    assert lean_width(built, [built.launchers["a"]], rebuild) is None
    assert calls == []


def test_a_k2b_spill_has_nothing_leaner():
    """``LayerPostBuild.halves`` gives K2b no index field and no ``tok_par``:
    the unfused key is the same MoE build, so rebuilding would not help."""
    built = _split()
    rebuild, calls = _rebuilder()
    assert lean_width(built, [built.moe], rebuild) is None
    assert calls == []


def test_an_unfused_build_that_still_spills_has_nothing_leaner():
    built = dataclasses.replace(
        _fused(), post_key=unfused_post_key(FUSED), unfused=True
    )
    rebuild, calls = _rebuilder()
    assert lean_width(built, [built.layer_post], rebuild) is None
    assert calls == []


def test_a_build_without_an_index_stage_has_nothing_leaner():
    """The DSpark draft's K2 (``index_bound_max`` 0) is already the shape the
    rebuild would produce."""
    built = _fused(index_bound_max=0)
    rebuild, calls = _rebuilder()
    assert lean_width(built, [built.layer_post], rebuild) is None
    assert calls == []


def test_a_build_with_no_recorded_key_has_nothing_leaner():
    built = RowsBuild([], {}, layer_post=object())
    rebuild, calls = _rebuilder()
    assert lean_width(built, [built.layer_post], rebuild) is None
    assert calls == []


def test_nothing_spilled_is_not_a_rebuild():
    """``WidthBuilds`` asks ``lean`` only about a spill; asked about none, the
    answer is still that there is nothing to swap."""
    built = _fused()
    rebuild, calls = _rebuilder()
    assert lean_width(built, [], rebuild) is None
    assert calls == []


# ── the selection a published step hands K2 ───────────────────────────────


def _reference_rows(selected, batch_ids, table, rows_per_page, page_rows, main_off):
    """``index_score._selected_rows`` and ``_final``, one element at a time:
    each valid column's pool row, written at its rank among the valid ones."""
    out = [[0] * len(row) for row in selected]
    for t, row in enumerate(selected):
        if batch_ids[t] < 0:
            continue
        at = 0
        for col in row:
            if col < 0:
                continue
            page = table[batch_ids[t]][col // rows_per_page]
            out[t][at] = page * page_rows + main_off + col % rows_per_page
            at += 1
    return out


def test_the_selection_mapping_against_a_hand_worked_case():
    # 4 rows a page, 16 pool rows a page, this KV owner 100 rows in
    table = torch.tensor([[2, 5, 7, 0], [3, 1, 4, 0]], dtype=torch.int32)
    batch_ids = torch.tensor([0, 1, -1], dtype=torch.int32)
    selected = torch.tensor(
        [[0, 5, -1, 3], [-1, 6, 1, -1], [0, 1, 2, 3]], dtype=torch.int32
    )
    out = torch.full((3, 4), -7, dtype=torch.int32)
    write_selection_rows(out, selected, batch_ids, table, 4, 16, 100)
    assert out.tolist() == [
        # cols 0, 5, 3 -> pages 2, 5, 2 -> 32+100+0, 80+100+1, 32+100+3
        [132, 181, 135, 0],
        # cols 6, 1 -> pages 1, 3 -> 16+100+2, 48+100+1
        [118, 149, 0, 0],
        # a pad row selects nothing
        [0, 0, 0, 0],
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_selection_mapping_is_the_kernels_arithmetic(seed):
    torch.manual_seed(seed)
    rows, k, pages, rows_per_page, page_rows, main_off = 6, 8, 5, 4, 32, 64
    table = torch.randint(0, pages, (3, 4), dtype=torch.int32)
    batch_ids = torch.tensor([0, 1, 2, -1, 1, -1], dtype=torch.int32)
    columns = rows_per_page * table.shape[1]
    selected = torch.randint(-2, columns, (rows, k), dtype=torch.int32).clamp_min(-1)
    out = torch.zeros(rows, k, dtype=torch.int32)
    write_selection_rows(
        out, selected, batch_ids, table, rows_per_page, page_rows, main_off
    )
    assert out.tolist() == _reference_rows(
        selected.tolist(),
        batch_ids.tolist(),
        table.tolist(),
        rows_per_page,
        page_rows,
        main_off,
    )


def test_the_valid_entries_are_packed_first_and_the_rest_are_empty():
    """The attention reads the first ``index_plan.selected(bound)`` entries and
    counts nothing itself, so a hole in the packing is a key read from pool
    row 0."""
    table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    batch_ids = torch.zeros(1, dtype=torch.int32)
    selected = torch.tensor([[-1, 4, -1, -1, 9]], dtype=torch.int32)
    out = torch.full((1, 5), -1, dtype=torch.int32)
    write_selection_rows(out, selected, batch_ids, table, 4, 8, 0)
    # col 4 -> page 2 -> 16 + 0, col 9 -> page 3 -> 24 + 1
    assert out.tolist() == [[16, 25, 0, 0, 0]]


def test_a_short_output_takes_the_steps_rows():
    """``irow`` is the widest step's, narrowed to this one's (``_use_rows``);
    ``batch_ids`` and ``selected`` may be longer."""
    table = torch.tensor([[1], [2]], dtype=torch.int32)
    batch_ids = torch.tensor([0, 1, 0], dtype=torch.int32)
    selected = torch.tensor([[0], [0], [0]], dtype=torch.int32)
    out = torch.zeros(2, 1, dtype=torch.int32)
    write_selection_rows(out, selected, batch_ids, table, 4, 8, 0)
    assert out.tolist() == [[8], [16]]


def test_a_mismatched_output_width_is_refused():
    table = torch.tensor([[0]], dtype=torch.int32)
    with pytest.raises(AssertionError):
        write_selection_rows(
            torch.zeros(1, 3, dtype=torch.int32),
            torch.zeros(1, 4, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            table,
            4,
            8,
            0,
        )
