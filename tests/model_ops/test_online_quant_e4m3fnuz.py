# SPDX-License-Identifier: MIT
"""Online re-quantization where the platform FP8 is E4M3FNUZ: the source's OCP
E4M3 bytes are decoded as E4M3, and the re-quantized weight gets no FNUZ fixup.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

aiter = pytest.importorskip("aiter")
QuantType, dtypes = aiter.QuantType, aiter.dtypes

linear_mod = importlib.import_module("atom.model_ops.linear")
moe_mod = importlib.import_module("atom.model_ops.moe")
LayerQuantConfig = importlib.import_module("atom.quant_spec").LayerQuantConfig
dequant_moe_weight_online = importlib.import_module(
    "atom.quantization.quark.utils"
).dequant_moe_weight_online

FNUZ = torch.float8_e4m3fnuz
N, K = 256, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _per_row_quant(weight, online_quant_type, online_quant_dtype, scale_dtype=None):
    """quant_weight_online's per-token FP8 result, in plain torch."""
    assert online_quant_type == QuantType.per_Token
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True) / torch.finfo(online_quant_dtype).max
    return (w / scale).to(online_quant_dtype), scale


def _checkpoint(quant_type, blocks):
    """The bytes and scale an OCP checkpoint stores, and the values they mean.

    The first byte is 0x80: zero in E4M3, NaN when read as E4M3FNUZ.
    """
    torch.manual_seed(0)
    ref = torch.randn(N, K, device=DEVICE) * 0.05
    if quant_type == QuantType.No:
        weight = ref.bfloat16()
        return weight, None, weight.float()
    amax = ref.abs().view(N // blocks[0], blocks[0], K // blocks[1], blocks[1])
    scale = amax.amax(dim=(1, 3)) / 448
    expanded = scale.repeat_interleave(blocks[0], 0).repeat_interleave(blocks[1], 1)
    weight = (ref / expanded).to(torch.float8_e4m3fn)
    weight.view(torch.uint8)[0, 0] = 0x80
    return weight, scale, weight.float() * expanded


def _close_per_row(got, want):
    """Within one E4M3 rounding step of each row's largest value."""
    assert not got.isnan().any()
    bound = want.abs().amax(dim=1, keepdim=True) / 16
    assert torch.all((got - want).abs() <= bound)


@pytest.mark.parametrize(
    "quant_type,blocks",
    [
        (QuantType.No, None),
        (QuantType.per_Tensor, (N, K)),
        pytest.param(
            QuantType.per_1x128,
            (128, 128),
            marks=pytest.mark.skipif(DEVICE == "cpu", reason="Triton dequant"),
        ),
    ],
    ids=["bf16", "fp8-per-tensor", "fp8-128x128"],
)
def test_linear_reaches_ptpc_with_the_checkpoint_values(
    monkeypatch, quant_type, blocks
):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    monkeypatch.setattr(
        linear_mod,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(linear_mod, "quant_weight_online", _per_row_quant)
    # The GEMM layout is beside the point here; keep the weight row-major.
    monkeypatch.setattr(linear_mod, "shuffle_weights", lambda *_a, **_k: None)
    source = LayerQuantConfig(
        quant_type=quant_type,
        quant_dtype=torch.bfloat16 if quant_type == QuantType.No else FNUZ,
    )
    target = LayerQuantConfig(quant_type=QuantType.per_Token, quant_dtype=FNUZ)
    quant_config = SimpleNamespace(
        online_quant=True,
        blockscale_e8m0_scale=False,
        get_layer_quant_config=lambda *_a, use_online_quant=False, **_k: (
            target if use_online_quant else source
        ),
    )
    with torch.device(DEVICE):
        layer = linear_mod.ReplicatedLinear(K, N, quant_config=quant_config)
    weight, scale, want = _checkpoint(quant_type, blocks)
    layer.weight_loader(layer.weight, weight)
    if scale is not None:
        layer.weight_loader(layer.weight_scale, scale.reshape(layer.weight_scale.shape))

    layer.process_weights_after_loading()

    assert layer.quant_type == QuantType.per_Token
    assert layer.weight.dtype == FNUZ
    got = layer.weight.float() * layer.weight_scale.float()
    _close_per_row(got, want)


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
