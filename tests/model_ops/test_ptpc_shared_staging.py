# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 FP4PTPC: the per-channel FP8 shared expert staged while loading
and copied into its fused slot once online quant made the routed experts
per-channel FP8."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

aiter = pytest.importorskip("aiter")

moe_mod = importlib.import_module("atom.model_ops.moe")
quant_spec = importlib.import_module("atom.quant_spec")
staging_mod = importlib.import_module("atom.model_ops.fused_moe.ptpc_shared_staging")
FusedMoE = moe_mod.FusedMoE

H, I_LOCAL, ROUTED, SHARED = 64, 32, 2, 1
FP8 = torch.float8_e4m3fn


def _param(*shape, dtype=torch.float32):
    return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)


def _layer(tp_size=1, tp_rank=0):
    """A FusedMoE with FP4 routed experts (packed stand-ins) and one fused
    shared slot."""
    layer = object.__new__(FusedMoE)
    nn.Module.__init__(layer)
    layer.quant_method = None
    layer._stream_online_quant = False
    layer.has_bias = False
    layer.prefix = layer.layer_name = "model.layers.3.ffn.experts"
    layer.layer_quant_config = quant_spec.LayerQuantConfig(
        quant_type=aiter.QuantType.per_1x32, quant_dtype=aiter.dtypes.fp4x2
    )
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
    e = ROUTED + SHARED
    layer.w13_weight = _param(e, 2 * I_LOCAL, H // 2)
    layer.w13_weight_scale = _param(e, 2 * I_LOCAL, H // 32)
    layer.w2_weight = _param(e, H, I_LOCAL // 2)
    layer.w2_weight_scale = _param(e, H, I_LOCAL // 32)
    layer.ptpc_shared_staging = staging_mod.PtpcSharedStaging(layer, H, I_LOCAL)
    return layer


def _load_shared(layer, w, s):
    for shard in ("w1", "w3", "w2"):
        wp = layer.w2_weight if shard == "w2" else layer.w13_weight
        sp = layer.w2_weight_scale if shard == "w2" else layer.w13_weight_scale
        layer.weight_loader(wp, w[shard], "shared.weight", shard, ROUTED)
        layer.weight_loader(sp, s[shard], "shared.weight_scale", shard, ROUTED)


@pytest.mark.parametrize("fnuz", [False, True])
@pytest.mark.parametrize(("tp_size", "tp_rank"), [(1, 0), (2, 1)])
def test_shared_expert_is_copied_into_its_slot(fnuz, tp_size, tp_rank):
    """This rank's shard of the shared expert goes into its slot as it is (FNUZ
    bytes and doubled scales where the routed experts are FNUZ); a w2 row keeps
    its whole scale."""
    layer = _layer(tp_size=tp_size, tp_rank=tp_rank)
    torch.manual_seed(0)
    inter = I_LOCAL * tp_size
    w = {k: (torch.randn(*s) * 4).to(FP8) for k, s in
         (("w1", (inter, H)), ("w3", (inter, H)), ("w2", (H, inter)))}  # fmt: skip
    s = {k: torch.rand(v.shape[0], 1) + 0.5 for k, v in w.items()}
    _load_shared(layer, w, s)
    # What `_online_quant` leaves: unpadded per-channel FP8 buffers.
    e = ROUTED + SHARED
    f8 = torch.float8_e4m3fnuz if fnuz else FP8
    layer.w13_weight = nn.Parameter(_param(e, 2 * I_LOCAL, H).fill_(9).to(f8), False)
    layer.w13_weight_scale = _param(e, 2 * I_LOCAL).fill_(9)
    layer.w2_weight = nn.Parameter(_param(e, H, I_LOCAL).fill_(9).to(f8), False)
    layer.w2_weight_scale = _param(e, H).fill_(9)
    layer.ptpc_shared_staging.fuse(layer)

    k = 2.0 if fnuz else 1.0
    rows = slice(tp_rank * I_LOCAL, (tp_rank + 1) * I_LOCAL)
    w13, w13_s = layer.w13_weight.data[ROUTED], layer.w13_weight_scale.data[ROUTED]
    for shard, off in (("w1", 0), ("w3", I_LOCAL)):
        got = w13[off : off + I_LOCAL].float() * w13_s[off : off + I_LOCAL, None]
        assert torch.equal(got, w[shard][rows].float() * s[shard][rows])
        assert torch.equal(w13_s[off : off + I_LOCAL], k * s[shard][rows].flatten())
    w2, w2_s = layer.w2_weight.data[ROUTED], layer.w2_weight_scale.data[ROUTED]
    assert torch.equal(w2.float() * w2_s[:, None], w["w2"][:, rows].float() * s["w2"])
    assert torch.equal(w2_s, k * s["w2"].flatten())
    assert (layer.w13_weight.data[:ROUTED].float() == 9).all()  # routed untouched


@pytest.mark.parametrize("field", ["uses_all2all_fusion", "shared_is_routed"])
def test_refuses_all2all(field):
    """The FP8 online method has no all2all shared-expert dispatch."""
    layer = _layer()
    setattr(layer.expert_layout, field, True)
    with pytest.raises(NotImplementedError, match="all2all"):
        staging_mod.PtpcSharedStaging(layer, H, I_LOCAL)


def test_dummy_load_leaves_the_slot_alone(monkeypatch):
    layer = _layer()
    monkeypatch.setattr(
        staging_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(load_dummy="zero"),
    )
    layer.ptpc_shared_staging.fuse(layer)
    assert layer.ptpc_shared_staging.staged is None
