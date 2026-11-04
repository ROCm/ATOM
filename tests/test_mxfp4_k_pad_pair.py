# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K padding of a gate_up -> act -> down MLP pair, inside the parallel linears.

The preshuffled MXFP4 GEMMs cannot take a K that is not a multiple of 256. An
MLP's down_proj K is its per-rank intermediate (Kimi-K3's dense MLP: 33792 / TP8
= 4224), so the pair pads it together, like MoE's intermediate_pad:
MergedColumnParallelLinear(output_pad_align=) gives each per-rank output shard
zero rows, RowParallelLinear(input_pad_align=) zero columns. Only a pair that is
MXFP4 or bf16 in the checkpoint is padded, zeroed at allocation; a bf16 pair may
be quantized online, while online requant from a quantized source is refused.

Every load path has to write each rank's real shard at the front of its padded
slot -- including a gate_up merged on disk, and the module-level weight_loader
an online weight sync calls -- and the padded slots have to stay zero.
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
    MXFP4_N32K4_K_ALIGN,
    MXFP4_PRESHUFFLE_K_ALIGN,
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
    mxfp4_k_pad_align,
)
from atom.quant_spec import NVFP4_DTYPE, LayerQuantConfig

TP = 8
INTER = 33792  # Kimi-K3 dense MLP
REAL, PADDED = INTER // TP, 4352  # 4224 -> 4352 per rank
HIDDEN = 256  # small: K of gate_up, N of down


def _qc(quant_type, quant_dtype):
    qc = QuantizationConfig()
    qc.global_spec = LayerQuantConfig(quant_type=quant_type, quant_dtype=quant_dtype)
    return qc


MXFP4 = (QuantType.per_1x32, dtypes.fp4x2)


@pytest.fixture(autouse=True)
def _default_backends(mxfp4_default_backends):
    """The alignment depends on the backend; pin the default ASM one."""


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
    assert gate_up._ckpt_output_sizes == [INTER, INTER]
    assert down._k_padded


def test_unquantized_pair_is_untouched(tp_rank):
    gate_up, down = _pair((QuantType.No, torch.bfloat16))
    assert gate_up.weight.shape == (2 * REAL, HIDDEN)
    assert down.weight.shape == (HIDDEN, REAL)
    assert gate_up.output_sizes == gate_up._ckpt_output_sizes
    assert not down._k_padded


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


@pytest.mark.parametrize("tp_rank", [0, 3, 7], indirect=True)
def test_padded_slots_are_zeroed_at_allocation_and_stay_zero(monkeypatch, tp_rank):
    """Zeroed when built (byte 0 is finite in e8m0, as in MoE); loading the
    real shards and process_weights_after_loading leave the pad zero."""
    gate_up, down = _pair()
    for layer in (gate_up, down):
        assert (_u8(layer.weight) == 0).all() and (_u8(layer.weight_scale) == 0).all()
    w = _pattern(INTER, HIDDEN // 2) | 1  # nonzero real bytes
    sc = _pattern(INTER, HIDDEN // 32) | 1
    for shard_id in (0, 1):
        gate_up.weight_loader(gate_up.weight, w, shard_id)
        gate_up.weight_loader(gate_up.weight_scale, sc, shard_id)
    down.weight_loader(down.weight, _pattern(HIDDEN, INTER // 2) | 1)
    down.weight_loader(down.weight_scale, _pattern(HIDDEN, INTER // 32) | 1)
    for layer in (gate_up, down):
        _process(monkeypatch, layer)

    gw, gs = _u8(gate_up.weight), _u8(gate_up.weight_scale)
    for real, pad in (
        (slice(0, REAL), slice(REAL, PADDED)),
        (slice(PADDED, PADDED + REAL), slice(PADDED + REAL, 2 * PADDED)),
    ):
        assert (gw[pad] == 0).all() and (gs[pad] == 0).all()
        assert (gw[real] != 0).all() and (gs[real] != 0).all()
    dw, ds = _u8(down.weight), _u8(down.weight_scale)
    assert (dw[:, REAL // 2 :] == 0).all() and (ds[:, REAL // 32 :] == 0).all()
    assert (dw[:, : REAL // 2] != 0).all() and (ds[:, : REAL // 32] != 0).all()


def test_an_unpadded_mxfp4_down_is_refused(monkeypatch, tp_rank):
    """Without the pair's padding, the backstop refuses K=4224 at load."""
    qc = _qc(*MXFP4)
    down = RowParallelLinear(INTER, HIDDEN, quant_config=qc, prefix="mlp.down_proj")
    with pytest.raises(RuntimeError, match=r"mlp\.down_proj.*K=4224"):
        _process(monkeypatch, down)


# ── online requant: bf16 sources pad, quantized sources are refused ─────


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
FP8_PER_TOKEN_SRC = LayerQuantConfig(
    quant_type=QuantType.per_Token, quant_dtype=dtypes.fp8
)
BF16_SRC = LayerQuantConfig(quant_type=QuantType.No, quant_dtype=torch.bfloat16)


@pytest.fixture
def online_build(monkeypatch):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    "source",
    [NVFP4_SRC, FP8_BLOCK_SRC, FP8_PER_TOKEN_SRC],
    ids=["nvfp4", "fp8_block", "fp8_per_token"],
)
def test_online_requant_into_a_padded_layer_is_refused(tp_rank, online_build, source):
    """A quantized source would need its scales padded before requant, which
    MoE never does either; an unaligned pair from one is refused, loudly."""
    qc = _online_to_mxfp4(source)
    align = mxfp4_k_pad_align(qc, "mlp.down_proj")
    assert align == MXFP4_PRESHUFFLE_K_ALIGN
    with pytest.raises(NotImplementedError, match="exclude_layer"):
        RowParallelLinear(
            INTER,
            HIDDEN,
            quant_config=qc,
            prefix="mlp.down_proj",
            input_pad_align=align,
        )
    with pytest.raises(NotImplementedError, match="exclude_layer"):
        MergedColumnParallelLinear(
            HIDDEN,
            [INTER, INTER],
            quant_config=qc,
            prefix="mlp.gate_up_proj",
            output_pad_align=align,
        )


def test_an_aligned_online_requant_pair_is_built_unpadded(tp_rank, online_build):
    """The refusal is only for a layer padding would change."""
    qc = _online_to_mxfp4(BF16_SRC)
    align = mxfp4_k_pad_align(qc, "mlp.down_proj")
    aligned = PADDED * TP  # per rank already a multiple of 256
    down = RowParallelLinear(
        aligned,
        HIDDEN,
        quant_config=qc,
        prefix="mlp.down_proj",
        input_pad_align=align,
    )
    assert not down._k_padded


def test_a_static_mixed_scheme_pair_is_refused_without_exclude_advice(tp_rank):
    """A static FP8 gate_up beside an MXFP4 down has no online config to
    exclude it from; the message says what the pair needs instead."""
    with pytest.raises(NotImplementedError, match="both halves") as err:
        MergedColumnParallelLinear(
            HIDDEN,
            [INTER, INTER],
            quant_config=_qc(QuantType.per_Token, dtypes.fp8),
            prefix="mlp.gate_up_proj",
            output_pad_align=MXFP4_PRESHUFFLE_K_ALIGN,
        )
    assert "exclude_layer" not in str(err.value)


def test_a_bf16_pair_requantized_online_is_padded(tp_rank, online_build):
    """The common online case: a bf16 checkpoint quantized online to MXFP4.
    bf16 has no scales to pad; its zero rows quantize to zero."""
    qc = _online_to_mxfp4(BF16_SRC)
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
    assert gate_up.weight.shape == (2 * PADDED, HIDDEN)
    assert down.weight.shape == (HIDDEN, PADDED)
    for layer in (gate_up, down):
        assert (_u8(layer.weight) == 0).all()


def test_an_unquantized_gate_up_half_is_padded_with_zeros(tp_rank):
    """A bf16 gate_up feeding an MXFP4 down: zero rows are trivially exact."""
    gate_up = MergedColumnParallelLinear(
        HIDDEN,
        [INTER, INTER],
        quant_config=_qc(QuantType.No, torch.bfloat16),
        prefix="mlp.gate_up_proj",
        output_pad_align=MXFP4_PRESHUFFLE_K_ALIGN,
    )
    assert gate_up.weight.shape == (2 * PADDED, HIDDEN)
    assert (_u8(gate_up.weight) == 0).all()


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
    assert not down._k_padded


# ── W4A8: the native group32 path is padded too, zeroed at allocation ─────


def test_w4a8_native_pad_slots_stay_zero(monkeypatch, tp_rank):
    """fp4 weights + FP8 activations take the native group32 branch of
    process_weights_after_loading, which returns early; the pad was zeroed when
    the layer was built, so nothing there needs to run."""
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
        _process(monkeypatch, layer)
    gw, gs = _u8(gate_up.weight), _u8(gate_up.weight_scale)
    for start in (REAL, PADDED + REAL):
        assert (gw[start : start + PADDED - REAL] == 0).all()
        assert (gs[start : start + PADDED - REAL] == 0).all()
    dw, ds = _u8(down.weight), _u8(down.weight_scale)
    assert (dw[:, REAL // 2 :] == 0).all() and (ds[:, REAL // 32 :] == 0).all()


# ── a padded param never takes the fused-on-disk direct copy ──────────────


@pytest.mark.parametrize("rank", [0, 1])
def test_merged_on_disk_whose_shape_matches_the_padded_param(monkeypatch, rank):
    """intermediate 256 at TP2: 128 per rank pads to 256, so the padded param
    (2 x 256 rows) has the unsharded [gate | up] checkpoint's shape. It must
    still be split and sharded, not copied whole."""
    group = SimpleNamespace(world_size=2, rank_in_group=rank)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: group)
    qc = _qc(*MXFP4)
    inter = 256
    gate_up = MergedColumnParallelLinear(
        HIDDEN,
        [inter, inter],
        quant_config=qc,
        prefix="mlp.gate_up_proj",
        output_pad_align=mxfp4_k_pad_align(qc, "mlp.down_proj"),
    )
    assert gate_up.weight.shape == (2 * 256, HIDDEN // 2)
    gate, up = _pattern(inter, HIDDEN // 2), _pattern(inter, HIDDEN // 2).flip(0)
    fused = torch.cat([gate, up])
    assert fused.shape == gate_up.weight.shape  # the coincidence
    gate_up.weight_loader(gate_up.weight, fused, None)
    w = _u8(gate_up.weight)
    assert torch.equal(w[:128], gate.chunk(2, 0)[rank])
    assert torch.equal(w[256:384], up.chunk(2, 0)[rank])


# ── one resolver for the runtime scheme ───────────────────────────────────


def test_effective_layer_quant_resolves_the_online_target():
    """The one resolver the pad decision and Kimi-K3's fused act quant read."""
    from atom.quant_spec import effective_layer_quant

    for source in (NVFP4_SRC, FP8_BLOCK_SRC):
        qc = _online_to_mxfp4(source)
        qt, qd = effective_layer_quant(qc, "mlp.down_proj")
        assert (qt.value, qd) == (QuantType.per_1x32.value, dtypes.fp4x2)


def test_weight_scale_row_group_compares_by_value():
    """A per_1x128 QuantType from another aiter import still groups by 128."""

    class Foreign:
        value = QuantType.per_1x128.value

        def __eq__(self, other):
            return self is other

        __hash__ = object.__hash__

    double = SimpleNamespace(native_a8_group_rows=None, quant_type=Foreign())
    assert LinearBase.weight_scale_row_group.fget(double) == 128


# ── the padded pair runs forward like the unpadded math ───────────────────


def test_padded_pair_forward_matches_the_real_math(monkeypatch, tp_rank):
    """gate_up -> silu(gate) * up -> down through the padded MXFP4 pair at
    TP8, against the same quantized weights in float at the real K. Catches a
    wiring slip (one align without the other, a misplaced shard) that the
    hand-padded GEMM test in test_mxfp4_k_alignment.py cannot."""
    quant = linear_mod.get_hip_quant(QuantType.per_1x32)
    torch.manual_seed(0)
    w_gu = torch.randn(2 * INTER, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05
    w_d = torch.randn(HIDDEN, INTER, dtype=torch.bfloat16, device="cuda") * 0.05
    gu_q, gu_s = quant(w_gu, quant_dtype=dtypes.fp4x2, shuffle=False)
    d_q, d_s = quant(w_d, quant_dtype=dtypes.fp4x2, shuffle=False)

    with torch.device("cuda"):
        gate_up, down = _pair()
    for shard_id in (0, 1):
        rows = slice(shard_id * INTER, (shard_id + 1) * INTER)
        gate_up.weight_loader(gate_up.weight, gu_q.view(torch.uint8)[rows], shard_id)
        gate_up.weight_loader(
            gate_up.weight_scale, gu_s.view(torch.uint8)[rows], shard_id
        )
    down.weight_loader(down.weight, d_q.view(torch.uint8))
    down.weight_loader(down.weight_scale, d_s.view(torch.uint8))
    for layer in (gate_up, down):
        layer.process_weights_after_loading()

    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device="cuda")
    h = gate_up(x)
    gate, up = h.float().chunk(2, -1)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    y = down(act).float()

    # Reference: this rank's real shards, dequantized, in float.
    def deq(q, s):
        p = q.cpu()
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        v = torch.stack([lut[(p & 0xF).long()], lut[(p >> 4).long()]], -1)
        v = v.reshape(p.shape[0], -1)
        scale = torch.pow(2.0, s.cpu().float() - 127).repeat_interleave(32, 1)
        return (v * scale[:, : v.shape[1]]).cuda()

    r = slice(tp_rank * REAL, (tp_rank + 1) * REAL)
    g_ref = deq(gu_q.view(torch.uint8)[:INTER][r], gu_s.view(torch.uint8)[:INTER][r])
    u_ref = deq(gu_q.view(torch.uint8)[INTER:][r], gu_s.view(torch.uint8)[INTER:][r])
    d_ref = deq(
        d_q.view(torch.uint8)[:, r.start // 2 : r.stop // 2],
        d_s.view(torch.uint8)[:, r.start // 32 : r.stop // 32],
    )

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0)

    # Each stage against the float math on its own input, so each carries one
    # fp4 activation-quant step (~0.99); a misplaced shard would be ~0.
    assert gate.shape[-1] == PADDED
    for name, got, ref in (
        ("gate", gate[:, :REAL], x.float() @ g_ref.T),
        ("up", up[:, :REAL], x.float() @ u_ref.T),
    ):
        assert cos(got, ref) > 0.98, f"rank {tp_rank} {name}: {cos(got, ref):.4f}"
    # The padded tail is exactly zero, so down's padded K contributes nothing.
    assert (gate[:, REAL:] == 0).all() and (up[:, REAL:] == 0).all()
    assert (act[:, REAL:] == 0).all()
    y_ref = act[:, :REAL].float() @ d_ref.T
    assert torch.isfinite(y).all()
    assert cos(y, y_ref) > 0.98, f"rank {tp_rank} down: {cos(y, y_ref):.4f}"


def test_bf16_pair_requantized_online_runs_forward_like_the_real_math(
    monkeypatch, tp_rank, online_build
):
    """bf16 checkpoint -> online MXFP4 through the padded pair at TP8: load the
    real bf16 shards, requantize in process_weights_after_loading, run
    gate_up -> silu(gate) * up -> down; each stage against bf16 float math."""
    qc = _online_to_mxfp4(BF16_SRC)
    align = mxfp4_k_pad_align(qc, "mlp.down_proj")
    torch.manual_seed(0)
    w_gu = torch.randn(2 * INTER, HIDDEN, dtype=torch.bfloat16) * 0.05
    w_d = torch.randn(HIDDEN, INTER, dtype=torch.bfloat16) * 0.05
    with torch.device("cuda"):
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
    for shard_id in (0, 1):
        rows = slice(shard_id * INTER, (shard_id + 1) * INTER)
        gate_up.weight_loader(gate_up.weight, w_gu[rows], shard_id)
    down.weight_loader(down.weight, w_d)
    for layer in (gate_up, down):
        layer.process_weights_after_loading()
    assert gate_up.weight.dtype == dtypes.fp4x2 and down.weight.dtype == dtypes.fp4x2

    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device="cuda")
    gate, up = gate_up(x).float().chunk(2, -1)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    y = down(act).float()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0)

    r = slice(tp_rank * REAL, (tp_rank + 1) * REAL)
    g_ref = w_gu[:INTER][r].float().cuda()
    u_ref = w_gu[INTER:][r].float().cuda()
    d_ref = w_d[:, r].float().cuda()
    assert gate.shape[-1] == PADDED
    for name, got, ref in (
        ("gate", gate[:, :REAL], x.float() @ g_ref.T),
        ("up", up[:, :REAL], x.float() @ u_ref.T),
    ):
        assert cos(got, ref) > 0.98, f"rank {tp_rank} {name}: {cos(got, ref):.4f}"
    assert (gate[:, REAL:] == 0).all() and (act[:, REAL:] == 0).all()
    y_ref = act[:, :REAL].float() @ d_ref.T
    assert torch.isfinite(y).all()
    assert cos(y, y_ref) > 0.98, f"rank {tp_rank} down: {cos(y, y_ref):.4f}"
