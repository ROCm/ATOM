# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU dispatch and row-slice checks around the real KDA load and forward paths."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from aiter_stub import atom_without_aiter
from torch import nn


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            kimi=load("atom.models.kimi_k3.model"),
            ops=load("atom.model_ops.kimi_k3"),
            glm=load("atom.models.glm5_next"),
            chip=load("aiter.jit.utils.chip_info"),
        )


@pytest.mark.parametrize(
    "override,arch,accepted",
    [
        ({}, "gfx950", True),
        ({}, "gfx942", False),
        ({}, "gfx1250", False),
        ({"tp_size": 4}, "gfx950", False),
        ({"hidden_size": 4096}, "gfx950", False),
        ({"local_proj_size": 3072}, "gfx950", False),
        ({"num_local_heads": 24}, "gfx950", False),
        ({"head_dim": 64}, "gfx950", False),
        ({"dtype": torch.float16}, "gfx950", False),
        ({"dtype": torch.float32}, "gfx950", False),
        ({"shape": (6285, 7168)}, "gfx950", False),
        ({"shape": (6284, 4096)}, "gfx950", False),
        ({"glm": True}, "gfx950", False),
        ({"online_type": "No"}, "gfx950", True),
        ({"online_type": "per_Tensor"}, "gfx950", False),
        ({"online_type": "per_Token"}, "gfx950", False),
        ({"online_type": "per_1x128"}, "gfx950", False),
        ({"online_type": "per_1x32"}, "gfx950", False),
    ],
)
def test_only_measured_native_geometry_is_supported(
    modules, monkeypatch, override, arch, accepted
):
    cls = (
        modules.glm.Glm5NextKDAAttention
        if override.get("glm")
        else modules.kimi.KimiKDAAttention
    )
    attn = object.__new__(cls)
    nn.Module.__init__(attn)
    for name, value in {
        "tp_size": 8,
        "hidden_size": 7168,
        "local_proj_size": 1536,
        "num_local_heads": 12,
        "head_dim": 128,
    }.items():
        setattr(attn, name, override.get(name, value))
    # Parent KDA fuses BF16 before the child can apply its online target.
    quant_config = None
    prefix = "layers.0.self_attn.in_proj"
    if "online_type" in override:

        def layer_config(name, use_online_quant=False):
            assert name == prefix
            return SimpleNamespace(
                quant_type=(
                    getattr(modules.kimi.QuantType, override["online_type"])
                    if use_online_quant
                    else modules.kimi.QuantType.No
                ),
                quant_dtype=(
                    modules.kimi.dtypes.fp8 if use_online_quant else torch.bfloat16
                ),
            )

        quant_config = SimpleNamespace(
            online_quant=True, get_layer_quant_config=layer_config
        )
    # Metadata only: no production-sized weights or device allocation.
    attn.in_proj = SimpleNamespace(
        quant_config=quant_config,
        prefix=prefix,
        weight=SimpleNamespace(
            dtype=override.get("dtype", torch.bfloat16),
            shape=override.get("shape", (6284, 7168)),
        ),
    )
    monkeypatch.setattr(modules.chip, "get_gfx_runtime", lambda: arch)
    # Plugin construction temporarily rebinds this module symbol.
    monkeypatch.setattr(
        modules.kimi, "KimiKDAAttention", modules.glm.Glm5NextKDAAttention
    )
    assert attn._supports_in_proj_split() is accepted


HIDDEN, HEADS, HEAD_DIM = 16, 2, 4
LP = HEADS * HEAD_DIM


class _Projection(nn.Module):
    def __init__(self, in_features, out_features, quant_type):
        super().__init__()
        self.weight = nn.Parameter(
            (torch.randn(out_features, in_features) / 8).bfloat16(),
            requires_grad=False,
        )
        self.quant_type = quant_type
        self.calls = []

    def forward(self, x, x_scale=None):
        self.calls.append((x, x_scale))
        return F.linear(x, self.weight)


def _attention(modules, monkeypatch, tokens):
    kimi = modules.kimi
    torch.manual_seed(0)

    def linear(inputs, outputs, **kwargs):
        outputs = sum(outputs) if isinstance(outputs, list) else outputs
        return _Projection(inputs, outputs, kimi.QuantType.No)

    for name in (
        "MergedColumnParallelLinear",
        "ColumnParallelLinear",
        "ReplicatedLinear",
        "RowParallelLinear",
    ):
        monkeypatch.setattr(kimi, name, linear)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_rank", lambda: 0)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        linear_num_key_heads=HEADS,
        linear_key_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1e-6,
    )
    attn = kimi.KimiKDAAttention(
        SimpleNamespace(
            hf_config=config,
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
        None,
        prefix="layers.0.self_attn",
    )
    assert attn._split_in_proj is False
    expected = torch.cat(
        [attn.in_proj.weight, attn.b_proj.weight, attn.f_a_proj.weight]
    )
    qualifications = []

    def qualify():
        # The real load hook must finish fusing before it caches eligibility.
        torch.testing.assert_close(attn.in_proj.weight, expected)
        qualifications.append(True)
        return True

    monkeypatch.setattr(attn, "_supports_in_proj_split", qualify)
    attn.process_weights_after_loading()
    attn.process_weights_after_loading()
    assert qualifications == [True] and attn._split_in_proj
    assert attn.b_proj.weight.numel() == attn.f_a_proj.weight.numel() == 0
    monkeypatch.setattr(kimi, "causal_conv1d_fn", lambda x, *a, **k: x.T.split(LP, -1))
    monkeypatch.setattr(
        modules.ops, "gather_kda_initial_state", lambda s, idx, _: s[idx].clone()
    )
    attn._run_kda = lambda q, k, v, g, b, s, *_: (
        (q + k + v + g * b.unsqueeze(-1)).contiguous(),
        s,
    )
    monkeypatch.setattr(
        attn.o_norm, "forward", lambda out, gate, is_prefill: out * gate
    )
    monkeypatch.setattr(attn.o_proj, "forward", lambda x: x)
    meta = SimpleNamespace(
        num_actual_tokens=tokens,
        num_prefills=1,
        has_initial_state=torch.tensor([False]),
        non_spec_state_indices_tensor=torch.tensor([0]),
        non_spec_state_indices_in_tensor=None,
        non_spec_query_start_loc=torch.tensor([0, tokens]),
    )
    cache = SimpleNamespace(
        k_cache=torch.zeros(1, 3 * LP, 3),
        v_cache=torch.zeros(1, HEADS, HEAD_DIM, HEAD_DIM),
    )
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(kda_metadata=meta),
        kv_cache_data={"layer_0": cache},
    )
    monkeypatch.setattr(kimi, "get_forward_context", lambda: ctx)
    gemms = []

    def mm(x, weight):
        gemms.append(weight)
        return F.linear(x, weight)

    monkeypatch.setattr(kimi, "tgemm", SimpleNamespace(mm=mm))
    return attn, meta, gemms, qualifications


@pytest.mark.parametrize("tokens,padded", [(1, 1), (8191, 8192), (8192, 8192)])
@torch.inference_mode()
def test_real_prefill_matches_fused_projection(modules, monkeypatch, tokens, padded):
    attn, _, gemms, qualifications = _attention(modules, monkeypatch, tokens)
    x = torch.randn(padded, HIDDEN).bfloat16()
    got = attn._forward_impl(x)
    weights = list(gemms)
    assert len(attn.in_proj.calls) == (tokens < 8192)
    assert len(weights) == (2 if tokens >= 8192 else 0)
    if weights:
        assert [tuple(w.shape) for w in weights] == [
            (4 * LP, HIDDEN),
            (HEADS + HEAD_DIM, HIDDEN),
        ]
        assert all(
            w.untyped_storage().data_ptr() == attn.in_proj.weight.data_ptr()
            for w in weights
        )
        assert [w.storage_offset() for w in weights] == [0, 4 * LP * HIDDEN]
    # f_a must have its own packed rows even for the one-row fallback.
    f_a = attn.f_b_proj.calls[-1][0]
    assert f_a.stride() == (HEAD_DIM, 1) and f_a.storage_offset() == 0
    attn._split_in_proj = False
    expected = attn._forward_impl(x)
    torch.testing.assert_close(got, expected)
    fused = F.linear(x[:tokens], attn.in_proj.weight)
    q, k, v, out_gate, beta, f_a = fused.split([LP, LP, LP, LP, HEADS, HEAD_DIM], -1)
    gate = F.linear(f_a.contiguous(), attn.f_b_proj.weight)
    beta = beta.unsqueeze(-1).expand(-1, -1, HEAD_DIM).reshape(tokens, LP)
    reference = (q + k + v + gate * beta) * out_gate
    torch.testing.assert_close(got, reference)
    assert got.shape == (tokens, LP)
    assert qualifications == [True]  # No qualification/probing during forward.


@pytest.mark.parametrize(
    "fallback", ["disabled", "decode", "scale", "dtype", "strided"]
)
def test_unsupported_inputs_call_the_original_projection(
    modules, monkeypatch, fallback
):
    attn, meta, gemms, _ = _attention(modules, monkeypatch, 8192)
    x = torch.zeros(8193, HIDDEN, dtype=torch.bfloat16)
    scale = None
    if fallback == "disabled":
        attn._split_in_proj = False
    elif fallback == "decode":
        meta.num_prefills = 0
    elif fallback == "scale":
        scale = torch.ones(8193, 1)
    elif fallback == "dtype":
        x = x.float()
    elif fallback == "strided":
        x = torch.zeros(8193, HIDDEN * 2, dtype=torch.bfloat16)[:, ::2]
    calls = []

    class ProjectionReached(Exception):
        pass

    def fused(inp, x_scale=None):
        calls.append((inp, x_scale))
        raise ProjectionReached

    monkeypatch.setattr(attn.in_proj, "forward", fused)
    with pytest.raises(ProjectionReached):
        attn._forward_impl(x, scale)
    assert len(calls) == 1 and gemms == []
    assert torch.equal(calls[0][0], x[:8192])
    assert calls[0][0].stride() == x.stride()
    if scale is not None:
        assert torch.equal(calls[0][1], scale[:8192])
    else:
        assert calls[0][1] is None
