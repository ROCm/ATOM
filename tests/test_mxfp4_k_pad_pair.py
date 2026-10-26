# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K padding of a gate_up -> act -> down MLP pair, inside the parallel linears.

The preshuffled MXFP4 GEMMs cannot take a K that is not a multiple of 256. An
MLP's down_proj K is its per-rank intermediate (Kimi-K3's dense MLP: 33792 / TP8
= 4224), so the pair pads it together: MergedColumnParallelLinear(
output_pad_align=) gives each per-rank output shard zero rows, RowParallelLinear(
input_pad_align=) zero columns. The checkpoint still holds the real sizes.

Every load path has to write each rank's real shard at the front of its padded
slot -- including a gate_up merged on disk, and the module-level weight_loader
an online weight sync calls -- and the padded slots have to be inert.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "atom.model_ops.linear imports aiter, which resolves the chip "
        "architecture through rocminfo",
        allow_module_level=True,
    )

from aiter import QuantType, dtypes

from atom.config import QuantizationConfig
from atom.model_ops import linear as linear_mod
from atom.model_ops.linear import (
    MXFP4_PRESHUFFLE_K_ALIGN,
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
    mxfp4_k_pad_align,
)
from atom.quant_spec import LayerQuantConfig

TP = 8
INTER = 33792  # Kimi-K3 dense MLP
REAL, PADDED = INTER // TP, 4352  # 4224 -> 4352 per rank
HIDDEN = 256  # small: K of gate_up, N of down


def _qc(quant_type, quant_dtype):
    qc = QuantizationConfig()
    qc.global_spec = LayerQuantConfig(quant_type=quant_type, quant_dtype=quant_dtype)
    return qc


MXFP4 = (QuantType.per_1x32, dtypes.fp4x2)


@pytest.fixture
def tp_rank(monkeypatch, request):
    rank = getattr(request, "param", 0)
    group = SimpleNamespace(world_size=TP, rank_in_group=rank)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: group)
    return rank


def _pair(quant=MXFP4):
    qc = _qc(*quant)
    align = mxfp4_k_pad_align(qc, "mlp.down_proj")
    gate_up = MergedColumnParallelLinear(
        HIDDEN,
        [INTER, INTER],
        quant_config=qc,
        prefix="mlp.gate_up_proj",
        output_pad_align=align,
    )
    down = RowParallelLinear(
        INTER,
        HIDDEN,
        quant_config=qc,
        reduce_results=False,
        prefix="mlp.down_proj",
        input_pad_align=align,
    )
    return gate_up, down


def _pattern(rows, cols):
    """Distinct per (row, col) modulo 251, so a misplaced block shows up."""
    r = torch.arange(rows).unsqueeze(1)
    c = torch.arange(cols).unsqueeze(0)
    return ((r * 7 + c * 3) % 251).to(torch.uint8)


def _u8(p):
    return p.data.view(torch.uint8).cpu()


# ── the decision ──────────────────────────────────────────────────────────


def test_align_is_requested_only_for_mxfp4():
    assert mxfp4_k_pad_align(_qc(*MXFP4), "x") == MXFP4_PRESHUFFLE_K_ALIGN
    assert mxfp4_k_pad_align(_qc(QuantType.No, torch.bfloat16), "x") is None
    assert mxfp4_k_pad_align(_qc(QuantType.per_Token, dtypes.fp8), "x") is None
    assert mxfp4_k_pad_align(None, "x") is None


def test_align_matches_quant_type_by_value():
    """A QuantType from another aiter import (plugin mode) has the same value
    but does not compare equal; the decision must still pad."""

    class Foreign:
        value = QuantType.per_1x32.value
        name = "per_1x32"

        def __eq__(self, other):
            return self is other

        __hash__ = object.__hash__

    assert mxfp4_k_pad_align(_qc(Foreign(), dtypes.fp4x2), "x") == 256


# ── shapes ────────────────────────────────────────────────────────────────


def test_mxfp4_pair_is_padded(tp_rank):
    gate_up, down = _pair()
    assert gate_up.weight.shape == (2 * PADDED, HIDDEN // 2)
    assert down.weight.shape == (HIDDEN, PADDED // 2)
    assert gate_up._pad_output_rows == [(REAL, PADDED), (PADDED + REAL, 2 * PADDED)]
    assert down._pad_input == (REAL, PADDED)


def test_unquantized_pair_is_untouched(tp_rank):
    gate_up, down = _pair((QuantType.No, torch.bfloat16))
    assert gate_up.weight.shape == (2 * REAL, HIDDEN)
    assert down.weight.shape == (HIDDEN, REAL)
    assert getattr(gate_up, "_pad_output_rows", None) is None
    assert down._pad_input is None


# ── every load path lands the real shard at the front of the padded slot ──


def _expect_gate_up(gate, up, rank, layer_w):
    w = _u8(layer_w)
    assert torch.equal(w[:REAL], gate.chunk(TP, 0)[rank]), "gate rows misplaced"
    assert torch.equal(w[PADDED : PADDED + REAL], up.chunk(TP, 0)[rank]), "up rows"


@pytest.mark.parametrize("tp_rank", [0, 3, 7], indirect=True)
def test_gate_up_separate_shards(tp_rank):
    """Checkpoint stores gate_proj and up_proj; the dispatcher (and an online
    weight sync, which calls module.weight_loader directly) passes shard ids."""
    gate_up, _ = _pair()
    gate, up = _pattern(INTER, HIDDEN // 2), _pattern(INTER, HIDDEN // 2).flip(0)
    gate_up.weight_loader(gate_up.weight, gate, 0)
    gate_up.weight_loader(gate_up.weight, up, 1)
    _expect_gate_up(gate, up, tp_rank, gate_up.weight)


@pytest.mark.parametrize("tp_rank", [0, 3, 7], indirect=True)
def test_gate_up_merged_on_disk(tp_rank):
    """Checkpoint stores gate_up_proj merged: the loader gets loaded_shard_id
    None and must split at the REAL sizes, not the padded ones."""
    gate_up, _ = _pair()
    gate, up = _pattern(INTER, HIDDEN // 2), _pattern(INTER, HIDDEN // 2).flip(0)
    gate_up.weight_loader(gate_up.weight, torch.cat([gate, up]), None)
    _expect_gate_up(gate, up, tp_rank, gate_up.weight)


@pytest.mark.parametrize("tp_rank", [0, 7], indirect=True)
def test_gate_up_tuple_shard_ids(tp_rank):
    gate_up, _ = _pair()
    gate, up = _pattern(INTER, HIDDEN // 2), _pattern(INTER, HIDDEN // 2).flip(0)
    gate_up.weight_loader(gate_up.weight, torch.cat([gate, up]), (0, 1))
    _expect_gate_up(gate, up, tp_rank, gate_up.weight)


@pytest.mark.parametrize("tp_rank", [0, 3, 7], indirect=True)
def test_gate_up_scales(tp_rank):
    gate_up, _ = _pair()
    gs, us = _pattern(INTER, HIDDEN // 32), _pattern(INTER, HIDDEN // 32).flip(0)
    gate_up.weight_loader(gate_up.weight_scale, gs, 0)
    gate_up.weight_loader(gate_up.weight_scale, us, 1)
    _expect_gate_up(gs, us, tp_rank, gate_up.weight_scale)


@pytest.mark.parametrize("tp_rank", [0, 3, 7], indirect=True)
def test_down_weight_and_scale(tp_rank):
    """K runs along dim 1: fp4x2 bytes (K/2) and e8m0 groups (K/32) alike."""
    _, down = _pair()
    w = _pattern(HIDDEN, INTER // 2)
    s = _pattern(HIDDEN, INTER // 32)
    down.weight_loader(down.weight, w)
    down.weight_loader(down.weight_scale, s)
    assert torch.equal(_u8(down.weight)[:, : REAL // 2], w.chunk(TP, 1)[tp_rank])
    assert torch.equal(_u8(down.weight_scale)[:, : REAL // 32], s.chunk(TP, 1)[tp_rank])


# ── the padded slots are inert, and the pair passes the load-time check ───


def _process(monkeypatch, layer):
    monkeypatch.setattr(linear_mod, "shuffle_weights", lambda *a, **kw: None)
    monkeypatch.setattr(
        linear_mod.fp4_utils, "e8m0_shuffle", lambda s: s, raising=False
    )
    monkeypatch.setattr(linear_mod, "use_fp4_non_shuffle_triton_gemm", lambda: False)
    LinearBase.process_weights_after_loading(layer)


def test_padded_slots_are_zero_weights_and_unit_scales(monkeypatch, tp_rank):
    gate_up, down = _pair()
    for layer in (gate_up, down):
        layer.weight.data.view(torch.uint8).fill_(
            0xAB
        )  # what an empty alloc might hold
        layer.weight_scale.data.view(torch.uint8).fill_(0xCD)
        _process(monkeypatch, layer)

    gw, gs = _u8(gate_up.weight), _u8(gate_up.weight_scale)
    for start, end in gate_up._pad_output_rows:
        assert (gw[start:end] == 0).all() and (gs[start:end] == 127).all()
    assert (gw[:REAL] == 0xAB).all(), "real rows must not be touched"

    dw, ds = _u8(down.weight), _u8(down.weight_scale)
    assert (dw[:, REAL // 2 :] == 0).all() and (ds[:, REAL // 32 :] == 127).all()
    assert (dw[:, : REAL // 2] == 0xAB).all(), "real columns must not be touched"


def test_an_unpadded_mxfp4_down_is_refused(monkeypatch, tp_rank):
    """Without the pair's padding, the backstop refuses K=4224 at load."""
    qc = _qc(*MXFP4)
    down = RowParallelLinear(INTER, HIDDEN, quant_config=qc, prefix="mlp.down_proj")
    with pytest.raises(RuntimeError, match=r"mlp\.down_proj.*K=4224"):
        _process(monkeypatch, down)
