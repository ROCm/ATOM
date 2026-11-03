# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 Quark MXFP4 export: the MXFP8 shared expert fused into the MXFP4
routed experts (ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4), one 129-expert / top-5 MoE.
The shared expert is staged in MXFP8 while loading and re-quantized into its
MXFP4 slot afterwards."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

aiter = pytest.importorskip("aiter")
QuantType = aiter.QuantType
dtypes = aiter.dtypes

moe_mod = importlib.import_module("atom.model_ops.moe")
quant_spec = importlib.import_module("atom.quant_spec")
staging_mod = importlib.import_module(
    "atom.model_ops.fused_moe.mxfp8_shared_into_mxfp4"
)
FusedMoE = moe_mod.FusedMoE
LayerQuantConfig = quant_spec.LayerQuantConfig

H, I_LOCAL, ROUTED, SHARED = 64, 32, 2, 1
PAD = 16  # the MXFP4 target may pad the intermediate size
FP8 = torch.float8_e4m3fn


def _spec(dtype):
    return LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtype)


def _param(*shape, dtype=torch.float32):
    return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)


def _fused_mxfp4_layer():
    """An MXFP4 FusedMoE with one fused shared slot, buffers as the target
    lays them out (w13 halves padded), its shared expert staged in MXFP8."""
    layer = object.__new__(FusedMoE)
    nn.Module.__init__(layer)
    layer.has_bias = False
    layer.prefix = layer.layer_name = "model.layers.3.block_sparse_moe.experts"
    layer.layer_quant_config = _spec(dtypes.fp4x2)
    layer.moe_parallel_config = SimpleNamespace(
        tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, dp_rank=0, use_ep=False
    )
    layer.expert_map = None
    layer.expert_layout = SimpleNamespace(
        routed_physical_per_rank=ROUTED,
        num_fused_shared_experts=SHARED,
        shared_is_routed=False,
    )
    layer.local_num_experts = ROUTED + SHARED
    e, half = ROUTED + SHARED, I_LOCAL + PAD
    # Stand-ins for the packed fp4x2 / e8m0 buffers: the fake quantizer below
    # "packs" by halving columns, and these record what lands where.
    layer.w13_weight = _param(e, 2 * half, H // 2)
    layer.w13_weight_scale = _param(e, 2 * half, H // 32)
    layer.w2_weight = _param(e, H, I_LOCAL // 2)
    layer.w2_weight_scale = _param(e, H, I_LOCAL // 32)
    layer.shared_mxfp8_into_mxfp4 = staging_mod.Mxfp8SharedIntoMxfp4(
        layer, None, H, I_LOCAL
    )
    return layer


def _shared_checkpoint():
    """MXFP8 shared expert as the checkpoint stores it (full, unsharded)."""
    torch.manual_seed(0)
    w = {
        "w1": (torch.randn(I_LOCAL, H) * 4).to(FP8),
        "w3": (torch.randn(I_LOCAL, H) * 4).to(FP8),
        "w2": (torch.randn(H, I_LOCAL) * 4).to(FP8),
    }
    s = {
        k: torch.randint(100, 140, (v.shape[0], v.shape[1] // 32), dtype=torch.uint8)
        for k, v in w.items()
    }
    return w, s


def _load_shared(layer, w, s, skip=()):
    for shard in ("w1", "w3", "w2"):
        wp = layer.w2_weight if shard == "w2" else layer.w13_weight
        sp = layer.w2_weight_scale if shard == "w2" else layer.w13_weight_scale
        if ("weight", shard) not in skip:
            layer.weight_loader(wp, w[shard], "shared.weight", shard, ROUTED)
        if ("scale", shard) not in skip:
            layer.weight_loader(sp, s[shard], "shared.weight_scale", shard, ROUTED)


@pytest.mark.parametrize(
    ("env", "routed", "fused"),
    [
        ("0", dtypes.fp4x2, False),  # default: shared expert stays an MXFP8 MLP
        ("1", dtypes.fp4x2, True),
        ("1", dtypes.fp8, False),  # only the MXFP8 -> MXFP4 pairing is opted in
    ],
)
def test_mxfp4_fusion_is_opt_in(monkeypatch, env, routed, fused):
    monkeypatch.setenv("ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4", env)
    assert (
        bool(quant_spec.fuses_mxfp8_shared_into_mxfp4(_spec(dtypes.fp8), _spec(routed)))
        is fused
    )


def test_shared_slot_loads_into_mxfp8_staging_not_mxfp4_buffers():
    layer = _fused_mxfp4_layer()
    w, s = _shared_checkpoint()
    _load_shared(layer, w, s)

    staged = layer.shared_mxfp8_into_mxfp4.staged
    assert torch.equal(
        staged["w13_weight"][0, :I_LOCAL].view(torch.uint8), w["w1"].view(torch.uint8)
    )
    assert torch.equal(
        staged["w13_weight"][0, I_LOCAL:].view(torch.uint8), w["w3"].view(torch.uint8)
    )
    assert torch.equal(
        staged["w2_weight"][0].view(torch.uint8), w["w2"].view(torch.uint8)
    )
    assert torch.equal(staged["w13_weight_scale"][0, I_LOCAL:], s["w3"])
    assert torch.equal(staged["w2_weight_scale"][0], s["w2"])
    # The MXFP4 buffers never saw the shared expert.
    assert not layer.w13_weight.data.any() and not layer.w2_weight.data.any()


def test_missing_shared_tensor_is_reported():
    layer = _fused_mxfp4_layer()
    w, s = _shared_checkpoint()
    _load_shared(layer, w, s, skip={("scale", "w3")})
    with pytest.raises(RuntimeError, match="w13_weight_scale"):
        layer.shared_mxfp8_into_mxfp4.fuse(layer)


def test_shared_expert_requantized_into_the_mxfp4_slot(monkeypatch):
    """Dequantized MXFP8 goes through the MXFP4 quantizer, gate/up land in the
    padded halves, the routed slots are untouched and the staging is freed."""
    layer = _fused_mxfp4_layer()
    w, s = _shared_checkpoint()
    _load_shared(layer, w, s)
    staged = {k: v.clone() for k, v in layer.shared_mxfp8_into_mxfp4.staged.items()}

    seen = []

    def fake_quant(x, online_quant_type, online_quant_dtype):
        assert online_quant_type == QuantType.per_1x32
        assert online_quant_dtype == dtypes.fp4x2
        seen.append(x)
        return x[:, : x.shape[1] // 2].float(), torch.full(
            (x.shape[0], x.shape[1] // 32), 7.0
        )

    monkeypatch.setattr(staging_mod, "quant_weight_online", fake_quant)
    layer.shared_mxfp8_into_mxfp4.fuse(layer)

    assert layer.shared_mxfp8_into_mxfp4.staged is None
    deq = staging_mod.dequant_mxfp8
    w1 = deq(staged["w13_weight"][0, :I_LOCAL], staged["w13_weight_scale"][0, :I_LOCAL])
    w3 = deq(staged["w13_weight"][0, I_LOCAL:], staged["w13_weight_scale"][0, I_LOCAL:])
    assert len(seen) == 3 and all(x.dtype == torch.bfloat16 for x in seen)
    assert torch.equal(seen[0], w1.to(torch.bfloat16))
    assert torch.equal(seen[1], w3.to(torch.bfloat16))
    half = I_LOCAL + PAD
    slot = layer.w13_weight.data[ROUTED]
    assert torch.equal(slot[:I_LOCAL], seen[0][:, : H // 2].float())
    assert torch.equal(slot[half : half + I_LOCAL], seen[1][:, : H // 2].float())
    assert not slot[I_LOCAL:half].any()  # padding untouched
    assert (layer.w13_weight_scale.data[ROUTED, half : half + I_LOCAL] == 7).all()
    assert (layer.w2_weight_scale.data[ROUTED] == 7).all()
    assert not layer.w13_weight.data[:ROUTED].any()  # routed slots untouched


def test_reloading_the_fused_shared_expert_is_refused(monkeypatch):
    """The staging is freed once the slot is MXFP4, so a later load of the
    shared expert must not fall through into the MXFP4 buffers."""
    layer = _fused_mxfp4_layer()
    w, s = _shared_checkpoint()
    _load_shared(layer, w, s)
    monkeypatch.setattr(
        staging_mod,
        "quant_weight_online",
        lambda x, **_: (x[:, : x.shape[1] // 2].float(), torch.ones(x.shape[0], 1)),
    )
    layer.shared_mxfp8_into_mxfp4.fuse(layer)
    before = layer.w13_weight.data.clone()

    with pytest.raises(NotImplementedError, match="reloading"):
        _load_shared(layer, w, s)
    assert torch.equal(layer.w13_weight.data, before)


def test_refuses_eplb_routed_shared_expert():
    """EPLB under all2all EP routes the shared expert in the routed slots."""
    layer = _fused_mxfp4_layer()
    layer.expert_layout.shared_is_routed = True
    with pytest.raises(NotImplementedError, match="EPLB"):
        staging_mod.Mxfp8SharedIntoMxfp4(layer, None, H, I_LOCAL)


def test_refuses_online_requant_of_the_routed_experts(monkeypatch):
    """The shared slot is written as MXFP4, so the routed experts must stay it."""
    layer = _fused_mxfp4_layer()
    monkeypatch.setattr(staging_mod, "will_online_requant", lambda *_: True)
    with pytest.raises(NotImplementedError, match="online_quant_config"):
        staging_mod.Mxfp8SharedIntoMxfp4(layer, None, H, I_LOCAL)
