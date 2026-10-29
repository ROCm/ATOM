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
from atom.model_loader.online_quant_streaming import OnlineQuantStreamer
from atom.model_ops import linear as linear_mod
from atom.model_ops.linear import (
    MXFP4_N32K4_K_ALIGN,
    MXFP4_PRESHUFFLE_K_ALIGN,
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
    _fill_pad_tails,
    mxfp4_k_pad_align,
    pad_slot_numel,
)
from atom.quant_spec import NVFP4_DTYPE, LayerQuantConfig
from atom.quantization.quark.utils import dequantize_nvfp4

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


# ── a source requantized online to MXFP4 is padded in its own scheme ──────


def _online_to_mxfp4(source):
    """A checkpoint in ``source``, requantized online to MXFP4."""
    target = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2)
    return SimpleNamespace(
        online_quant=True,
        blockscale_e8m0_scale=False,
        get_layer_quant_config=lambda *_a, use_online_quant=False, **_k: (
            target if use_online_quant else source
        ),
    )


NVFP4_SRC = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=NVFP4_DTYPE)
FP8_BLOCK_SRC = LayerQuantConfig(
    quant_type=QuantType.per_1x128,
    quant_dtype=dtypes.fp8,
    weight_block_size=[128, 128],
)


@pytest.fixture
def online_build(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )


def _online_pair(source, inter=INTER):
    qc = _online_to_mxfp4(source)
    align = mxfp4_k_pad_align(qc, "mlp.down_proj")
    assert align == MXFP4_PRESHUFFLE_K_ALIGN
    gate_up = MergedColumnParallelLinear(
        HIDDEN,
        [inter, inter],
        quant_config=qc,
        prefix="mlp.gate_up_proj",
        output_pad_align=align,
    )
    down = RowParallelLinear(
        inter,
        HIDDEN,
        quant_config=qc,
        reduce_results=False,
        prefix="mlp.down_proj",
        input_pad_align=align,
    )
    return gate_up, down


def _poison(layer):
    layer.weight.data.view(torch.uint8).fill_(0xAB)
    layer.weight_scale.data.view(torch.uint8).fill_(0x7F)  # NaN as e4m3fn
    _fill_pad_tails(layer)


def test_nvfp4_source_pad_scales_are_one_not_nan(tp_rank, online_build):
    """NVFP4 scales are e4m3fn, where the e8m0 "1.0" byte 0x7F is NaN; the
    online requant dequantizes them, so a NaN there would reach the output."""
    gate_up, down = _online_pair(NVFP4_SRC)
    assert gate_up.weight_scale.dtype == torch.float8_e4m3fn
    for layer in (gate_up, down):
        _poison(layer)

    for start, end in gate_up._pad_output_rows:
        assert (gate_up.weight_scale.data[start:end].float() == 1.0).all()
        tail = dequantize_nvfp4(
            gate_up.weight.data[start:end],
            gate_up.weight_scale.data[start:end],
            torch.tensor(1.0),
        )
        assert (tail == 0).all()
    s_start = REAL // 16  # one e4m3fn scale per 16 of K
    assert (down.weight_scale.data[:, s_start:].float() == 1.0).all()
    tail = dequantize_nvfp4(
        down.weight.data[:, REAL // 2 :],
        down.weight_scale.data[:, s_start:],
        torch.tensor(1.0),
    )
    assert (tail == 0).all()


@pytest.mark.parametrize("tp_rank", [0, 3, 7], indirect=True)
def test_fp8_block_source_is_padded_by_whole_blocks(tp_rank, online_build):
    """Kimi-K3's 4224 per rank is 33 blocks of 128: the pad is one more block
    of zero weights with scale 1.0, and each rank's real blocks load in front."""
    gate_up, down = _online_pair(FP8_BLOCK_SRC)
    assert gate_up.weight_scale.shape == (2 * PADDED // 128, HIDDEN // 128)
    assert down.weight_scale.shape == (HIDDEN // 128, PADDED // 128)

    blocks = INTER // 128
    gs = torch.arange(blocks * 2, dtype=torch.float32).view(blocks, 2) + 2
    us = gs + 1000
    gate_up.weight_loader(gate_up.weight_scale, gs, 0)
    gate_up.weight_loader(gate_up.weight_scale, us, 1)
    ds = torch.arange(2 * blocks, dtype=torch.float32).view(2, blocks) + 2
    down.weight_loader(down.weight_scale, ds)
    for layer in (gate_up, down):
        layer.weight.data.view(torch.uint8).fill_(0xAB)
        _fill_pad_tails(layer)

    real_b, pad_b = REAL // 128, PADDED // 128  # 33, 34
    g = gate_up.weight_scale.data.cpu()
    assert torch.equal(g[:real_b], gs.chunk(TP, 0)[tp_rank])
    assert torch.equal(g[pad_b : pad_b + real_b], us.chunk(TP, 0)[tp_rank])
    assert (g[real_b:pad_b] == 1.0).all() and (g[pad_b + real_b :] == 1.0).all()
    d = down.weight_scale.data.cpu()
    assert torch.equal(d[:, :real_b], ds.chunk(TP, 1)[tp_rank])
    assert (d[:, real_b:] == 1.0).all()
    assert (_u8(gate_up.weight)[REAL:PADDED] == 0).all()
    assert (_u8(down.weight)[:, REAL:] == 0).all()


@pytest.mark.parametrize("tp_rank", [0], indirect=True)
def test_fp8_block_source_refused_when_blocks_straddle_the_pad(tp_rank, online_build):
    """Qwen3-Next's 512-wide shared expert at TP8 is 64 per rank: a 128-row
    scale block would cover real and padded rows, so it can't be padded."""
    with pytest.raises(NotImplementedError, match=r"per_1x128.*straddle"):
        _online_pair(FP8_BLOCK_SRC, inter=512)


# ── online-quant streaming does not wait for the padded slots ─────────────


@pytest.mark.parametrize("tp_rank", [0, 7], indirect=True)
def test_streaming_expects_only_the_real_shards(monkeypatch, tp_rank):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "1")
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING", "1")
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING_THREADS", "0")
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    gate_up, down = _online_pair(NVFP4_SRC)
    for layer in (gate_up, down):
        assert layer._stream_online_quant and layer.weight.is_meta
        streamer = OnlineQuantStreamer.maybe_create(layer, load_dummy=None)
        padded = sum(
            pad_slot_numel(layer, p) for p in (layer.weight, layer.weight_scale)
        )
        assert padded > 0
        full = sum(p.numel() for p in layer.parameters(recurse=False))
        assert layer._stream_expected_numel == full - padded
        del streamer

    # gate_up: the real shards (and global scales) alone complete the module.
    monkeypatch.setattr(
        linear_mod,
        "quant_weight_online",
        lambda w, *_a, **_k: (
            torch.zeros(w.shape[0], w.shape[1] // 2, dtype=torch.uint8).view(
                dtypes.fp4x2
            ),
            torch.full((w.shape[0], w.shape[1] // 32), 127, dtype=torch.uint8).view(
                dtypes.fp8_e8m0
            ),
        ),
    )
    gate_up, _ = _online_pair(NVFP4_SRC)
    # Built padded for the preshuffled GEMM; skip only the shuffle at finalize.
    monkeypatch.setattr(linear_mod, "use_fp4_non_shuffle_triton_gemm", lambda: True)
    streamer = OnlineQuantStreamer.maybe_create(gate_up, load_dummy=None)
    for shard_id in (0, 1):
        for param, value in (
            (gate_up.weight, torch.full((INTER, HIDDEN // 2), 0x22, dtype=torch.uint8)),
            (
                gate_up.weight_scale,
                torch.full((INTER, HIDDEN // 16), 2.0, dtype=torch.float8_e4m3fn),
            ),
            (gate_up.weight_scale_2, torch.tensor(1.0)),
        ):
            streamer.run(param.weight_loader, (param, value, shard_id))
    assert id(gate_up) in streamer.done_module_ids


# ── the alignment follows the GEMM the layer will run on ──────────────────


def test_align_follows_the_backend(monkeypatch):
    qc = _qc(*MXFP4)
    assert mxfp4_k_pad_align(qc, "x") == MXFP4_PRESHUFFLE_K_ALIGN
    # gfx1250's 32x4-scale kernels step K by 128: Kimi-K3's 4224 needs nothing.
    monkeypatch.setattr(linear_mod, "_mxfp4_n32k4_backend", lambda: True)
    assert mxfp4_k_pad_align(qc, "x") == MXFP4_N32K4_K_ALIGN == 128
    # The non-shuffle Triton GEMM reads the raw layout: no tail, no padding.
    monkeypatch.setattr(linear_mod, "use_fp4_non_shuffle_triton_gemm", lambda: True)
    assert mxfp4_k_pad_align(qc, "x") is None


def test_no_padding_when_the_backend_takes_the_real_k(monkeypatch, tp_rank):
    monkeypatch.setattr(linear_mod, "_mxfp4_n32k4_backend", lambda: True)
    gate_up, down = _pair()
    assert gate_up.weight.shape == (2 * REAL, HIDDEN // 2)
    assert down.weight.shape == (HIDDEN, REAL // 2)
    assert down._pad_input is None


# ── W4A8: the native group32 path pads, so its slots are filled too ───────


def test_w4a8_native_pad_slots_are_filled(monkeypatch, tp_rank):
    """fp4 weights + FP8 activations take the native group32 branch of
    process_weights_after_loading, which returns early; the fill runs first."""
    qc = QuantizationConfig()
    qc.global_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
        activation_dtype=torch.float8_e4m3fn,
        weight_block_size=(1, 32),
        is_dynamic=True,
    )
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
    assert gate_up.native_a8_group_rows is not None
    for layer in (gate_up, down):
        layer.weight.data.view(torch.uint8).fill_(0xAB)
        layer.weight_scale.data.view(torch.uint8).fill_(0xFF)  # NaN as e8m0
        _process(monkeypatch, layer)

    gw, gs = _u8(gate_up.weight), _u8(gate_up.weight_scale)
    for start, end in gate_up._pad_output_rows:
        assert (gw[start:end] == 0).all() and (gs[start:end] == 127).all()
    dw, ds = _u8(down.weight), _u8(down.weight_scale)
    assert (dw[:, REAL // 2 :] == 0).all() and (ds[:, REAL // 32 :] == 127).all()


# ── TP=1: a per-token (N, 1) scale has no K to pad ────────────────────────


def test_tp1_per_token_scale_loads_into_a_padded_down(monkeypatch, online_build):
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: group)
    src = LayerQuantConfig(quant_type=QuantType.per_Token, quant_dtype=dtypes.fp8)
    qc = _online_to_mxfp4(src)
    down = RowParallelLinear(
        REAL,
        HIDDEN,
        quant_config=qc,
        reduce_results=False,
        prefix="mlp.down_proj",
        input_pad_align=mxfp4_k_pad_align(qc, "mlp.down_proj"),
    )
    assert down._pad_input == (REAL, PADDED)
    assert down.weight_scale.shape == (HIDDEN, 1)
    scale = torch.arange(HIDDEN, dtype=torch.float32).view(HIDDEN, 1) + 1
    down.weight_loader(down.weight_scale, scale)
    assert torch.equal(down.weight_scale.data.cpu(), scale)
    w = _pattern(HIDDEN, REAL)
    down.weight_loader(down.weight, w.view(dtypes.fp8))
    assert torch.equal(_u8(down.weight)[:, :REAL], w)
