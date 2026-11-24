# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""K padding of a group32 FP8 gate_up -> act -> down MLP pair on gfx950.

AITER's gfx950 group32 FP8 GEMM (bmm_a8w8_mxscale_gfx950) steps K in 64s and
zero-fills no 32-wide tail (ROCm/aiter#7967). DeepSeek-V4.1-Flash's shared
expert at TP8 has a per-rank intermediate of 2304 / 8 = 288, so the pair pads it
to 320 the way an MXFP4 pair pads (test_mxfp4_k_pad_pair.py):
MergedColumnParallelLinear(output_pad_align=) gives each per-rank output shard
zero rows, RowParallelLinear(input_pad_align=) zero columns, zero scale codes
under them. gate_up's padded rows give silu(0) * 0 = 0, so down's padded K
multiplies exact zeros: no activation pad on a call, one weight. Only a native
32x32 weight the load holds preshuffled is padded: a row-major one
(ATOM_GROUP32_WEIGHT_PRESHUFFLE=0) runs K = 288 as it is.
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
from aiter.jit.utils import chip_info
from aiter.ops.shuffle import shuffle_weight

from atom.config import QuantizationConfig
from atom.model_ops import linear as linear_mod
from atom.model_ops.linear import (
    GROUP32_FP8_K_ALIGN,
    MergedColumnParallelLinear,
    RowParallelLinear,
    fp8_group32_k_pad_align,
    weight_is_stored_preshuffled,
)
from atom.quant_spec import LayerQuantConfig

TP = 8
INTER = 2304  # DeepSeek-V4.1-Flash's shared expert
REAL, PADDED = INTER // TP, 320  # 288 -> 320 per rank
HIDDEN = 256  # small: K of gate_up, N of down


def _qc(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp8, block=(32, 32)):
    qc = QuantizationConfig()
    qc.global_spec = LayerQuantConfig(
        quant_type=quant_type, quant_dtype=quant_dtype, weight_block_size=block
    )
    return qc


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx950")
    monkeypatch.setenv("ATOM_GROUP32_WEIGHT_PRESHUFFLE", "1")


@pytest.fixture
def tp_rank(monkeypatch, request):
    rank = getattr(request, "param", 0)
    group = SimpleNamespace(world_size=TP, rank_in_group=rank)
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: group)
    return rank


def _pair(qc=None):
    qc = qc or _qc()
    align = fp8_group32_k_pad_align(qc, "shared_experts.w2")
    gate_up = MergedColumnParallelLinear(
        HIDDEN, [INTER, INTER], quant_config=qc, prefix="shared_experts.gate_up_proj",
        output_pad_align=align,
    )  # fmt: skip
    down = RowParallelLinear(
        INTER, HIDDEN, quant_config=qc, reduce_results=False, prefix="shared_experts.w2",
        input_pad_align=align,
    )  # fmt: skip
    return gate_up, down


def _u8(t):
    return t.data.view(torch.uint8).cpu()


def _shuffled(rows):
    """Row-major FP8 codes as the layer holds them: aiter's (16, 16) preshuffle."""
    w = rows.contiguous().view(torch.float8_e4m3fn)
    return shuffle_weight(w, layout=(16, 16)).view(torch.uint8).reshape(rows.shape)


def test_the_align_is_gfx950_group32_fp8_only(monkeypatch, gfx950):
    assert fp8_group32_k_pad_align(_qc(), "x") == GROUP32_FP8_K_ALIGN == 64
    assert fp8_group32_k_pad_align(_qc(QuantType.per_1x32, dtypes.fp4x2), "x") is None
    assert fp8_group32_k_pad_align(_qc(QuantType.per_1x128, dtypes.fp8), "x") is None
    assert fp8_group32_k_pad_align(_qc(QuantType.No, torch.bfloat16), "x") is None
    assert fp8_group32_k_pad_align(None, "x") is None
    # gfx1250's group32 GEMM runs a ragged K (a 32x32 w_scale): nothing to pad
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx1250")
    assert fp8_group32_k_pad_align(_qc(), "x") is None


def test_the_mono_dims_pad_as_the_loader_does():
    """V4.1 mono's ``SHARED_PAD`` (``kernels.dims``, free of AITER) is the loader's K step."""
    from atom.models.deepseek_v41.mono.kernels.dims import SHARED_PAD

    assert SHARED_PAD == GROUP32_FP8_K_ALIGN


def test_the_pair_is_padded_at_tp8_only(gfx950, tp_rank):
    gate_up, down = _pair()
    assert gate_up.weight.shape == (2 * PADDED, HIDDEN)
    assert gate_up.weight_scale.shape == (2 * PADDED // 32, HIDDEN // 32)
    assert down.weight.shape == (HIDDEN, PADDED)
    assert down.weight_scale.shape == (HIDDEN // 32, PADDED // 32)
    # a share the GEMM steps (TP4's 576) is not padded
    linear_mod.get_tp_group().world_size = 4
    gate_up, down = _pair()
    assert gate_up.weight.shape == (2 * 576, HIDDEN) and down.weight.shape == (
        HIDDEN,
        576,
    )


@pytest.mark.parametrize("block", [(32, 32), (1, 32)])
@pytest.mark.parametrize("preshuffle", ["1", "0"])
def test_the_pad_follows_the_loads_preshuffle_decision(
    monkeypatch, gfx950, tp_rank, preshuffle, block
):
    """The pair is padded exactly where the load holds a native 32x32 down weight preshuffled
    (``weight_is_stored_preshuffled``, ATOM_GROUP32_WEIGHT_PRESHUFFLE on): the weight the preshuffled group32 GEMM
    reads. A row-major weight keeps its K = 288; so does a native 1x32 one (its MXFP8 backend's K rule is not the
    32x32 GEMM's step)."""
    monkeypatch.setenv("ATOM_GROUP32_WEIGHT_PRESHUFFLE", preshuffle)
    qc = _qc(block=block)
    gate_up, down = _pair(qc)
    assert down.native_a8_group_rows == block[0]
    shuffled = weight_is_stored_preshuffled(
        down.quant_type, down.params_dtype, native_group_rows=down.native_a8_group_rows
    )
    assert shuffled == (preshuffle == "1")
    pad = shuffled and block == (32, 32)
    assert fp8_group32_k_pad_align(qc, "shared_experts.w2") == (
        GROUP32_FP8_K_ALIGN if pad else None
    )
    k = PADDED if pad else REAL
    assert gate_up.weight.shape == (2 * k, HIDDEN) and down.weight.shape == (HIDDEN, k)


def _load(gate_up, down, gu, gu_s, d, d_s):
    for shard_id in (0, 1):
        rows = slice(shard_id * INTER, (shard_id + 1) * INTER)
        gate_up.weight_loader(gate_up.weight, gu[rows], shard_id)
        gate_up.weight_loader(
            gate_up.weight_scale, gu_s[shard_id * INTER // 32 : (shard_id + 1) * INTER // 32],
            shard_id,
        )  # fmt: skip
    down.weight_loader(down.weight, d)
    down.weight_loader(down.weight_scale, d_s)


def _checkpoint(seed):
    g = torch.Generator().manual_seed(seed)
    # finite E4M3 codes of either sign (|code| <= 0x77: never the NaN 0x7f)
    codes = lambda *s: torch.randint(1, 120, s, dtype=torch.uint8, generator=g) | (
        torch.randint(0, 2, s, dtype=torch.uint8, generator=g) << 7
    )
    scales = lambda *s: torch.randint(115, 130, s, dtype=torch.uint8, generator=g)
    return (codes(2 * INTER, HIDDEN), scales(2 * INTER // 32, HIDDEN // 32),
            codes(HIDDEN, INTER), scales(HIDDEN // 32, INTER // 32))  # fmt: skip


def _expect(gate_up, down, rank, gu, gu_s, d, d_s):
    """This rank's shards at the front of each padded slot, zero codes and scale codes behind them, preshuffled as
    a whole."""
    r = slice(rank * REAL, (rank + 1) * REAL)
    rs = slice(rank * REAL // 32, (rank + 1) * REAL // 32)
    gw = torch.zeros(2 * PADDED, HIDDEN, dtype=torch.uint8)
    gs = torch.zeros(2 * PADDED // 32, HIDDEN // 32, dtype=torch.uint8)
    for half in (0, 1):
        w0, s0 = half * PADDED, half * PADDED // 32
        gw[w0 : w0 + REAL] = gu[half * INTER :][r]
        gs[s0 : s0 + REAL // 32] = gu_s[half * INTER // 32 :][rs]
    dw = torch.zeros(HIDDEN, PADDED, dtype=torch.uint8)
    ds = torch.zeros(HIDDEN // 32, PADDED // 32, dtype=torch.uint8)
    dw[:, :REAL], ds[:, : REAL // 32] = d[:, r], d_s[:, rs]
    assert torch.equal(_u8(gate_up.weight), _shuffled(gw))
    assert torch.equal(_u8(gate_up.weight_scale), gs)
    assert torch.equal(_u8(down.weight), _shuffled(dw))
    assert torch.equal(_u8(down.weight_scale), ds)


@pytest.mark.parametrize("tp_rank", [0, 3, 7], indirect=True)
def test_load_and_a_weight_sync_keep_the_tail_zero(gfx950, tp_rank):
    """Loaded and preshuffled, then a weight sync: its module-level loads write the new shards over the preshuffled
    weight (whose bytes in the pad's place belong to other tiles: the loads zero the pad again) and the updater
    re-shuffles it. Each time: the real shards at the front of each slot, zero behind them.
    """
    from atom.rollout.weight_updater import WeightUpdaterMixin

    with torch.device("cuda"):
        gate_up, down = _pair()
    ckpt = _checkpoint(0)
    _load(gate_up, down, *(t.cuda() for t in ckpt))
    for layer in (gate_up, down):
        layer.process_weights_after_loading()
        assert layer.weight.is_shuffled
    _expect(gate_up, down, tp_rank, *ckpt)

    # a sync: the module-level loaders write the new rows, then the updater restores the preshuffle
    new = _checkpoint(1)
    updater = object.__new__(WeightUpdaterMixin)
    _load(gate_up, down, *(t.cuda() for t in new))
    for layer in (gate_up, down):
        updater._post_process_fp8_weight(layer, layer.weight)
    _expect(gate_up, down, tp_rank, *new)


def test_an_all_zero_group_quantizes_to_zero_codes_under_a_finite_scale():
    """The act's padded columns reach down's activation quant as an all-zero 32-group: zero codes, a finite E8M0."""
    from atom.model_ops.blockscale import (
        quantize_fp8,
    )  # (native_quant_linear's activation quant)

    x = torch.randn(5, PADDED, dtype=torch.bfloat16, device="cuda")
    x[:, REAL:] = 0
    q, s = quantize_fp8(x)
    assert (q.view(torch.uint8)[:, REAL:] == 0).all()
    tail = s.view(torch.uint8)[:, REAL // 32 :]
    assert (tail != 0xFF).all() and torch.isfinite(s.float()).all()


@pytest.mark.parametrize("preshuffle", ["1", "0"])
@pytest.mark.parametrize("tp_rank", [0, 7], indirect=True)
def test_the_pair_runs_the_real_math(monkeypatch, gfx950, tp_rank, preshuffle):
    """gate_up -> silu(gate) * up -> down through the pair on gfx950's group32 GEMM at TP8, against the same FP8
    weights at the real K = 288 in float: preshuffled, the GEMM steps K = 320 and the padded columns contribute
    nothing; row-major (ATOM_GROUP32_WEIGHT_PRESHUFFLE=0), the pair is not padded and runs K = 288.
    """
    monkeypatch.setenv("ATOM_GROUP32_WEIGHT_PRESHUFFLE", preshuffle)
    k = PADDED if preshuffle == "1" else REAL
    torch.manual_seed(0)
    with torch.device("cuda"):
        gate_up, down = _pair()
    gu, gu_s, d, d_s = _checkpoint(2)
    _load(gate_up, down, gu.cuda(), gu_s.cuda(), d.cuda(), d_s.cuda())
    for layer in (gate_up, down):
        layer.process_weights_after_loading()
        assert getattr(layer.weight, "is_shuffled", False) == (preshuffle == "1")

    x = torch.randn(6, HIDDEN, dtype=torch.bfloat16, device="cuda")
    h = gate_up(x).float()
    gate, up = h.chunk(2, -1)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    y = down(act).float()

    def deq(codes, scales):
        v = codes.view(torch.float8_e4m3fn).float()
        s = torch.pow(2.0, scales.float() - 127)
        return v * s.repeat_interleave(32, 0).repeat_interleave(32, 1)

    r = slice(tp_rank * REAL, (tp_rank + 1) * REAL)
    rs = slice(tp_rank * REAL // 32, (tp_rank + 1) * REAL // 32)
    g_ref = deq(gu[:INTER][r], gu_s[: INTER // 32][rs]).cuda()
    u_ref = deq(gu[INTER:][r], gu_s[INTER // 32 :][rs]).cuda()
    d_ref = deq(d[:, r], d_s[:, rs]).cuda()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0)

    assert gate.shape[-1] == k
    assert (gate[:, REAL:] == 0).all() and (up[:, REAL:] == 0).all()
    assert (act[:, REAL:] == 0).all()
    assert cos(gate[:, :REAL], x.float() @ g_ref.T) > 0.99
    assert cos(up[:, :REAL], x.float() @ u_ref.T) > 0.99
    y_ref = act[:, :REAL].float() @ d_ref.T
    assert torch.isfinite(y).all() and cos(y, y_ref) > 0.99
