# SPDX-License-Identifier: MIT
"""FusedMoE's online re-quantization where the platform FP8 is E4M3FNUZ: an
FP8 source's OCP E4M3 bytes are decoded as E4M3 before the dequant.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

aiter = pytest.importorskip("aiter")
QuantType, dtypes = aiter.QuantType, aiter.dtypes

moe_mod = importlib.import_module("atom.model_ops.moe")
LayerQuantConfig = importlib.import_module("atom.quant_spec").LayerQuantConfig
dequant_moe_weight_online = importlib.import_module(
    "atom.quantization.quark.utils"
).dequant_moe_weight_online

FNUZ = torch.float8_e4m3fnuz


def test_moe_decodes_its_checkpoint_bytes_before_requantizing(monkeypatch):
    """FusedMoE hands the conversion the values, not E4M3FNUZ reads of them."""
    experts, inter, hidden = 2, 128, 256
    source = LayerQuantConfig(quant_type=QuantType.per_Token, quant_dtype=FNUZ)
    target = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2)

    class TargetMethod:
        def create_weights(self, layer, **_kwargs):
            pass

    monkeypatch.setattr(moe_mod, "_make_mxfp4_moe_method", lambda *_a: TargetMethod())
    torch.manual_seed(0)
    ref = {
        "w13": torch.randn(experts, 2 * inter, hidden) * 0.05,
        "w2": torch.randn(experts, hidden, inter) * 0.05,
    }
    params, want = {}, {}
    for name, value in ref.items():
        scale = value.abs().amax(dim=-1, keepdim=True) / 448
        q = (value / scale).to(torch.float8_e4m3fn)
        q.view(torch.uint8)[0, 0, 0] = 0x80
        want[name] = (q.float() * scale).reshape(-1, q.shape[-1])
        # The loader's byte copy into the E4M3FNUZ parameter.
        stored = torch.empty_like(q, dtype=FNUZ)
        stored.view(torch.uint8).copy_(q.view(torch.uint8))
        params[f"{name}_weight"] = nn.Parameter(stored, requires_grad=False)
        params[f"{name}_weight_scale"] = nn.Parameter(scale, requires_grad=False)
    handed = {}
    layer = SimpleNamespace(
        online_quant=True,
        quant_config=SimpleNamespace(get_layer_quant_config=lambda *_a, **_k: target),
        layer_name="model.layers.3.mlp.experts",
        layer_quant_config=source,
        source_is_nvfp4=False,
        params_dtype=FNUZ,
        # The source Fp8MoEMethod: it would fix the bytes up after loading.
        quant_method=SimpleNamespace(need_normalize_e4m3fn_to_e4m3fnuz=True),
        moe_config=SimpleNamespace(),
        moe_quant_params={"params_dtype": FNUZ},
        _stream_online_quant=False,
        local_num_experts=experts,
        intermediate_size_per_partition=inter,
        use_ep=False,
        tp_size=1,
        tp_rank=0,
        _online_quant_row_local_batched=lambda **kwargs: handed.update(kwargs),
        **params,
    )

    moe_mod.FusedMoE._online_quant(layer)

    for name in ("w13", "w2"):
        got = dequant_moe_weight_online(
            handed[f"old_{name}_data"],
            handed[f"old_{name}_scale"],
            handed["source_quant_type"],
            handed["source_quant_dtype"],
        )
        assert torch.equal(got.float(), want[name].to(got.dtype).float())
