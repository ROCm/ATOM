# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU regression coverage of the production K-up and decode dispatch bodies.

GPU GEMM/RoPE arithmetic is stubbed here; bit-exact GPU validation is separate.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def projection(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mla_test_envs", root / "atom/utils/envs.py"
    )
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    for name in (
        "ATOM_MLA_ABSORB_BF16",
        "ATOM_MLA_ABSORB_BF16_FUSE_QOUT",
        "ATOM_MLA_KV_SHUFFLED",
    ):
        monkeypatch.setenv(name, "1")
    path = root / "atom/model_ops/attention_mla.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLAAttention"
    )
    method_names = {"_q_proj_and_k_up_proj", "forward_impl"}
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names
    ]
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "_mla_absorb_bf16_enabled",
            "_mla_absorb_bf16_fuse_qout_enabled",
        }
    ]
    assert {n.name for n in methods} == method_names
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    static_gate = next(
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_absorb_bf16_fuse_qout"
            for t in n.targets
        )
    )
    configure = ast.FunctionDef(
        name="_configure_feature",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[static_gate],
        decorator_list=[],
    )
    methods.append(configure)
    for method in methods:
        method.decorator_list = []
    namespace = {
        "torch": torch,
        "envs": envs,
        "is_rocm_aiter_fp4bmm_enabled": lambda: False,
        "qrep_for_step": lambda *args: False,
        "pcp_is_enabled": lambda: False,
        "_MLA_Q_OUT_PADDED_DIM": 16,
    }
    exec(  # noqa: S102 - checked-in production bodies, following the V-up CPU test
        compile(
            ast.fix_missing_locations(
                ast.Module(body=helpers + methods, type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    layer = type(
        "Projection",
        (),
        {name: namespace[name] for name in method_names | {"_configure_feature"}},
    )()
    layer.num_heads, layer.num_kv_heads = 3, 1
    layer.qk_nope_head_dim, layer.qk_rope_head_dim, layer.kv_lora_rank = 4, 2, 8
    layer.qk_head_dim = 6
    layer.qrep_enabled = False
    layer.dcp_prefill_qrep = False
    layer.W_K = (torch.arange(3 * 8 * 4) % 3).reshape(3, 8, 4).to(torch.bfloat16)
    layer.W_K_scale = None
    layer.q_proj = lambda x, scale=None: x
    layer.layer_num = 0
    layer.kv_shuffled = envs.ATOM_MLA_KV_SHUFFLED
    layer._q_nope_prestored_supported = True
    layer._configure_feature()
    layer.use_seg_mla = layer._fused_q_head_pad = False
    layer.dcp_world_size, layer.padded_num_heads = 1, 4
    layer.is_sparse_mla = False
    layer._k_scale, layer._q_scale = torch.tensor(0.5), torch.tensor(2.0)
    layer.rotary_emb = SimpleNamespace(
        cos_cache=None, sin_cache=None, is_neox_style=True
    )
    layer._shuffled_kv_view = lambda cache: cache
    layer._seg_kv_cache_view = lambda cache: cache
    for name in (
        "_forward_decode",
        "_dcp_decode",
        "_forward_prefill_mla",
        "_dcp_sparse_prefill",
    ):
        setattr(layer, name, lambda q, *args, **kwargs: q)
    seen = {}

    def bmm(q, weights, *, YQ):
        seen["YQ"] = YQ
        YQ.copy_(
            torch.bmm(q.float(), weights.float().transpose(1, 2)).to(torch.bfloat16)
        )
        return YQ

    namespace["_aiter_triton_bf16_bmm"] = bmm
    return layer, namespace, seen


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_bf16_k_up_writes_strided_q_nope_slice_without_touching_rope(projection, dtype):
    layer, _, seen = projection
    tokens = 5
    x = (torch.arange(tokens * 3 * 6) % 7).reshape(tokens, -1).to(torch.bfloat16)
    # Nonzero storage offset plus a RoPE tail expose incorrect contiguous stores.
    backing = torch.full((tokens, 3, 12), 3.0, dtype=dtype)
    out = backing[..., 1:9]
    result, rope = layer._q_proj_and_k_up_proj(x, q_nope_out=out)
    assert result is out
    yq = seen["YQ"]
    assert yq.shape == (3, tokens, 8)
    assert yq.stride() == (12, 36, 1)
    assert yq.data_ptr() == out.data_ptr()
    expected = torch.einsum(
        "tnp,nlp->tnl", x.reshape(tokens, 3, 6)[..., :4].float(), layer.W_K.float()
    )
    assert torch.equal(result.float(), expected.to(torch.bfloat16).to(dtype).float())
    assert torch.all(backing[..., 0].float() == 3)
    assert torch.all(backing[..., 9:].float() == 3)
    assert torch.equal(rope, x.reshape(tokens, 3, 6)[..., 4:])


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_default_bf16_k_up_allocates_contiguous_output(projection, noncontiguous):
    layer, _, seen = projection
    tokens = 5
    x = (torch.arange(tokens * 36) % 7).reshape(tokens, 36).to(torch.bfloat16)
    x = x[:, ::2] if noncontiguous else x[:, :18].contiguous()
    result, rope = layer._q_proj_and_k_up_proj(x)
    assert result.shape == (tokens, 3, 8)
    assert result.is_contiguous() and result.dtype == torch.bfloat16
    assert result.stride() == (24, 8, 1)
    assert seen["YQ"].shape == (3, tokens, 8)
    assert seen["YQ"].stride() == (8, 24, 1)
    assert seen["YQ"].data_ptr() == result.data_ptr()
    expected = torch.einsum(
        "tnp,nlp->tnl", x.reshape(tokens, 3, 6)[..., :4].float(), layer.W_K.float()
    )
    assert torch.equal(result, expected.to(torch.bfloat16))
    assert torch.equal(rope, x.reshape(tokens, 3, 6)[..., 4:])


@pytest.mark.parametrize(
    "case",
    [
        "on",
        "off",
        "small",
        "unsupported",
        "scaled_weight",
        "unshuffled",
        "fp4",
        "seg",
        "head_pad",
        "dcp",
        "prefill",
        "slots",
        "empty_cache",
    ],
)
def test_forward_selects_prestored_only_for_supported_decode(
    projection, monkeypatch, case
):
    layer, namespace, seen = projection
    tokens = 63 if case == "small" else 64
    if case == "off":
        monkeypatch.setenv("ATOM_MLA_ABSORB_BF16_FUSE_QOUT", "0")
    if case == "unsupported":
        # aiter's q_nope_prestored_supported() said no (gfx942/950).
        layer._q_nope_prestored_supported = False
    layer.kv_shuffled = case not in ("unshuffled", "seg", "head_pad", "dcp", "prefill")
    layer.use_seg_mla = case == "seg"
    layer._fused_q_head_pad = case == "head_pad"
    layer.dcp_world_size = 2 if case == "dcp" else 1
    if case == "scaled_weight":
        layer.W_K_scale = torch.ones(3)
    namespace["is_rocm_aiter_fp4bmm_enabled"] = lambda: case == "fp4"
    layer._configure_feature()
    # Production initialization reads the flag once; forward must not query it.
    namespace["_mla_absorb_bf16_fuse_qout_enabled"] = lambda: pytest.fail(
        "flag read during forward"
    )

    # Observe the real dispatch independently from projection arithmetic.
    def project(x, **kwargs):
        out = kwargs.get("q_nope_out")
        seen["projection_kwargs"] = kwargs
        seen["destination"] = out
        if out is None:
            out = torch.ones(tokens, 3, 8, dtype=torch.bfloat16)
        else:
            out.fill_(1)
        return out, torch.ones(tokens, 3, 2, dtype=torch.bfloat16)

    layer._q_proj_and_k_up_proj = project

    def rope(*args, **kwargs):
        seen["rope_kwargs"] = kwargs

    namespace.update(
        {
            name: rope
            for name in (
                "triton_fused_qk_rope_cat_and_cache_mla",
                "fused_qk_rope_concat_and_cache_mla",
                "fused_qk_rope_concat_and_cache_mla_seg",
            )
        }
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(tokens + (case == "slots")),
        dtype_q=torch.bfloat16,
        run_sparse=True,
    )
    context = SimpleNamespace(is_prefill=case == "prefill", is_dummy_run=False)
    layer.is_sparse_mla = case == "prefill"
    cache = torch.empty(0) if case == "empty_cache" else torch.empty(tokens, 1, 10)
    namespace["get_forward_context"] = lambda: SimpleNamespace(
        attn_metadata=metadata,
        context=context,
        kv_cache_data={"layer_0": SimpleNamespace(k_cache=cache)},
    )
    x = torch.zeros(tokens, 18, dtype=torch.bfloat16)
    result = layer.forward_impl(x, torch.zeros(tokens, 8), torch.zeros(tokens, 1, 2))
    if case == "on":
        assert seen["destination"] is not None
        assert seen["destination"].stride() == (30, 10, 1)
        assert result.data_ptr() == seen["destination"].data_ptr()
        assert seen["rope_kwargs"]["q_nope_prestored"] is True
    else:
        assert seen["destination"] is None
        assert seen.get("rope_kwargs", {}).get("q_nope_prestored") is not True
        assert "q_nope_out" not in seen["projection_kwargs"]


def test_registered_absorb_flags_defaults_and_overrides(projection, monkeypatch):
    _, namespace, _ = projection
    monkeypatch.delenv("ATOM_MLA_ABSORB_BF16")
    monkeypatch.delenv("ATOM_MLA_ABSORB_BF16_FUSE_QOUT")
    assert namespace["_mla_absorb_bf16_enabled"]() is False
    assert namespace["_mla_absorb_bf16_fuse_qout_enabled"]() is False
    monkeypatch.setenv("ATOM_MLA_ABSORB_BF16_FUSE_QOUT", "1")
    assert namespace["_mla_absorb_bf16_fuse_qout_enabled"]() is True
    monkeypatch.setenv("ATOM_MLA_ABSORB_BF16", "1")
    monkeypatch.setenv("ATOM_MLA_ABSORB_BF16_FUSE_QOUT", "0")
    assert namespace["_mla_absorb_bf16_enabled"]() is True
    assert namespace["_mla_absorb_bf16_fuse_qout_enabled"]() is False
