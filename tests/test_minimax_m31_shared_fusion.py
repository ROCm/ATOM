# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 Quark MXFP4 export: the MXFP8 shared expert fused into the MXFP4
routed experts (ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4), one 129-expert / top-5 MoE.
The shared expert is staged in MXFP8 while loading and re-quantized into its
MXFP4 slot afterwards."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from torch import nn

with atom_without_aiter() as load:
    moe_mod = load("atom.model_ops.moe")
    quant_spec = load("atom.quant_spec")
    topk_mod = load("atom.model_ops.topK")
    staging_mod = load("atom.model_ops.fused_moe.shared_expert_staging")
    QuantType = moe_mod.QuantType
    dtypes = moe_mod.dtypes
    d_dtypes = {name: getattr(dtypes, name) for name in ("fp8", "fp4x2")}
FusedMoE = moe_mod.FusedMoE
LayerQuantConfig = quant_spec.LayerQuantConfig

H, I_LOCAL, ROUTED, SHARED = 64, 32, 2, 1
PAD = 16  # the MXFP4 target may pad the intermediate size
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def quant_handles(monkeypatch):
    # quant_spec defers these imports until use, after the import context has
    # restored the CPU host's modules. Patch only for this test and restore
    # its original lazy handles so other files still import their own stack.
    monkeypatch.setattr(quant_spec, "QuantType", QuantType)
    monkeypatch.setattr(quant_spec, "d_dtypes", d_dtypes)


def _spec(dtype):
    return LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtype)


def _param(*shape, dtype=torch.float32):
    return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)


def _fused_mxfp4_layer(ptpc=False, tp_size=1, tp_rank=0, block_rows=1):
    """An MXFP4 FusedMoE with one fused shared slot, buffers as the target
    lays them out (w13 halves padded), its shared expert staged in MXFP8
    (per-channel FP8 with ``ptpc``)."""
    layer = object.__new__(FusedMoE)
    nn.Module.__init__(layer)
    layer.quant_method = None
    layer._stream_online_quant = False
    layer.has_bias = False
    layer.prefix = layer.layer_name = "model.layers.3.block_sparse_moe.experts"
    layer.layer_quant_config = _spec(dtypes.fp4x2)
    layer.moe_parallel_config = SimpleNamespace(
        tp_size=tp_size, tp_rank=tp_rank, ep_size=1, ep_rank=0, dp_rank=0, use_ep=False
    )
    layer.expert_map = None
    layer.expert_layout = SimpleNamespace(
        routed_physical_per_rank=ROUTED,
        num_fused_shared_experts=SHARED,
        shared_is_routed=False,
        uses_all2all_fusion=False,
    )
    layer.local_num_experts = ROUTED + SHARED
    e, half = ROUTED + SHARED, I_LOCAL + PAD
    # Stand-ins for the packed fp4x2 / e8m0 buffers: the fake quantizer below
    # "packs" by halving columns, and these record what lands where.
    layer.w13_weight = _param(e, 2 * half, H // 2)
    layer.w13_weight_scale = _param(e, 2 * half, H // 32)
    layer.w2_weight = _param(e, H, I_LOCAL // 2)
    layer.w2_weight_scale = _param(e, H, I_LOCAL // 32)
    layer.shared_expert_staging = staging_mod.SharedExpertStaging(
        layer,
        None,
        H,
        I_LOCAL,
        ptpc=ptpc,
        shared_expert_spec=LayerQuantConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=dtypes.fp8,
            weight_block_size=(block_rows, 32),
        ),
    )
    return layer


def _shared_checkpoint(tp_size=1):
    """MXFP8 shared expert as the checkpoint stores it (full, unsharded)."""
    torch.manual_seed(0)
    inter = I_LOCAL * tp_size
    w = {
        "w1": (torch.randn(inter, H) * 4).to(FP8),
        "w3": (torch.randn(inter, H) * 4).to(FP8),
        "w2": (torch.randn(H, inter) * 4).to(FP8),
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

    staged = layer.shared_expert_staging.staged
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


@pytest.mark.parametrize(("tp_size", "tp_rank"), [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_32x32_block_scales_stage_as_their_1x32_rows(tp_size, tp_rank):
    """DeepSeek-V4.1's FP8 shared expert has one E8M0 scale per 32x32 block."""
    w, _ = _shared_checkpoint(tp_size)
    blocks = {
        k: torch.randint(
            100, 140, (v.shape[0] // 32, v.shape[1] // 32), dtype=torch.uint8
        )
        for k, v in w.items()
    }
    rows = {k: v.repeat_interleave(32, dim=0) for k, v in blocks.items()}
    staged = []
    for scales, block_rows in ((blocks, 32), (rows, 1)):
        layer = _fused_mxfp4_layer(
            tp_size=tp_size, tp_rank=tp_rank, block_rows=block_rows
        )
        _load_shared(layer, w, scales)
        staged.append(layer.shared_expert_staging.staged)
    for name, got in staged[0].items():
        assert torch.equal(got.view(torch.uint8), staged[1][name].view(torch.uint8))


@pytest.mark.parametrize(
    "prefix", ["layers.3.ffn.experts", "mtp.0.ffn.experts", "other.draft.experts"]
)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("same_dtype", [False, True])
def test_model_owns_shared_fusion_policy(monkeypatch, prefix, fused, same_dtype):
    monkeypatch.setenv("ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4", "1")
    specs = {
        "shared_experts": _spec(dtypes.fp8),
        "experts": _spec(dtypes.fp8 if same_dtype else dtypes.fp4x2),
    }
    qc = SimpleNamespace(
        fuse_shared_experts=fused,
        get_layer_quant_config=lambda p, check_children=False: specs[
            p.rsplit(".", 1)[-1]
        ],
    )
    monkeypatch.setattr(
        topk_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            quant_config=qc,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            moe_ep_flatten_tp_across_dp=False,
        ),
    )
    shared = prefix.removesuffix("experts") + "shared_experts"
    assert (
        topk_mod.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
            qc, shared_expert_prefix=shared, routed_expert_prefix=prefix
        )
        is fused
    )


def test_missing_shared_tensor_is_reported():
    layer = _fused_mxfp4_layer()
    w, s = _shared_checkpoint()
    _load_shared(layer, w, s, skip={("scale", "w3")})
    with pytest.raises(RuntimeError, match="w13_weight_scale"):
        layer.shared_expert_staging.fuse(layer)


def test_shared_expert_requantized_into_the_mxfp4_slot(monkeypatch):
    """Dequantized MXFP8 goes through the MXFP4 quantizer, gate/up land in the
    padded halves, the routed slots are untouched and the staging is freed."""
    layer = _fused_mxfp4_layer()
    w, s = _shared_checkpoint()
    _load_shared(layer, w, s)
    staged = {k: v.clone() for k, v in layer.shared_expert_staging.staged.items()}

    seen = []

    def fake_quant(x, online_quant_type, online_quant_dtype):
        assert online_quant_type == QuantType.per_1x32
        assert online_quant_dtype == dtypes.fp4x2
        seen.append(x)
        return x[:, : x.shape[1] // 2].float(), torch.full(
            (x.shape[0], x.shape[1] // 32), 7.0
        )

    monkeypatch.setattr(staging_mod, "quant_weight_online", fake_quant)
    layer.shared_expert_staging.fuse(layer)

    assert layer.shared_expert_staging.staged is None
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
    layer.shared_expert_staging.fuse(layer)
    before = layer.w13_weight.data.clone()

    with pytest.raises(NotImplementedError, match="reloading"):
        _load_shared(layer, w, s)
    assert torch.equal(layer.w13_weight.data, before)


def test_refuses_eplb_routed_shared_expert():
    """EPLB under all2all EP routes the shared expert in the routed slots."""
    layer = _fused_mxfp4_layer()
    layer.expert_layout.shared_is_routed = True
    with pytest.raises(NotImplementedError, match="EPLB"):
        staging_mod.SharedExpertStaging(layer, None, H, I_LOCAL)


def test_refuses_online_requant_of_the_routed_experts(monkeypatch):
    """The shared slot is written as MXFP4, so the routed experts must stay it."""
    layer = _fused_mxfp4_layer()
    monkeypatch.setattr(staging_mod, "will_online_requant", lambda *_: True)
    with pytest.raises(NotImplementedError, match="online_quant_config"):
        staging_mod.SharedExpertStaging(layer, None, H, I_LOCAL)
