# SPDX-License-Identifier: MIT
"""Exercise the GPU gate's real RoPE and writer dispatch on CPU.

GPU primitives and tracing are replaced; the fixture constructor,
RotaryEmbedding forward and PagedAttentionImpl writer selection execute
their real code.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("tokens", [8192, 8193, 16384])
def test_target_prefix_writer_invokes_real_rotary_above_threshold(
    monkeypatch, heads, tokens
):
    calls = []

    def rope_gpu(q, k, cos, sin, pos, style, **kw):
        assert style == 0 and kw["reuse_freqs_front_part"]
        for x in (q, k):
            # Independent CPU substitute for the in-place GPU primitive.
            x1, x2 = x.float().chunk(2, -1)
            c, s = cos[pos.flatten()].view(1, tokens, 1, 64), sin[pos.flatten()].view(
                1, tokens, 1, 64
            )
            x.copy_(torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1).to(x.dtype))
        calls.append("rotary")

    def cache_gpu(*args, **kw):
        assert kw["asm_layout"]
        calls.append("cache")

    aiter = SimpleNamespace(
        dtypes=SimpleNamespace(fp32=torch.float32),
        rope_cached_positions_2c_fwd_inplace=rope_gpu,
        reshape_and_cache_with_pertoken_quant=cache_gpu,
    )
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    # Tracing is outside this CPU writer test. Importing its runtime module
    # pulls in atom.utils, model loading and distributed dependencies that
    # are deliberately absent from the torch/pytest-only CI environment.
    # Keep the real RoPE constructor and forward body unchanged.
    monkeypatch.setitem(
        sys.modules,
        "atom.utils.decorators",
        SimpleNamespace(mark_trace=lambda **kwargs: lambda fn: fn),
    )
    rotary_module = load(
        ROOT / "atom/model_ops/rotary_embedding.py", "real_fixture_rotary"
    )
    monkeypatch.setitem(sys.modules, "atom.model_ops.rotary_embedding", rotary_module)
    gate = load(
        ROOT / ".github/scripts/atomesh/check_m3_gluon_adapter.py", "rope_fixture_gate"
    )
    rope = gate.make_rotary(128, tokens + 1, "cpu")
    assert isinstance(rope, rotary_module.RotaryEmbedding)

    class Gemma:
        pass

    class Dual:
        def __init__(self, *args, **kw):
            pass

        def __call__(self, q, k):
            return q.clone(), k.clone()

    def fused(q, k, v, pos, **kw):
        calls.append("fused")
        assert kw["rotary_emb"] is rope
        return q.clone(), k.clone()

    for name, module in {
        "layernorm": SimpleNamespace(GemmaRMSNorm=Gemma, DualRMSNorm=Dual),
        "triton_mrope": SimpleNamespace(try_mrope_qk_fused=lambda *a: None),
        "triton_fused_qkv_norm_rope_cache": SimpleNamespace(
            triton_fused_norm_rope_cache=fused
        ),
    }.items():
        monkeypatch.setitem(sys.modules, f"atom.model_ops.{name}", module)
    source = ROOT / "atom/model_ops/attention_mha.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PagedAttentionImpl"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {"rope_cache", "_gemma_norm_rope_then_cache", "_use_asm_cache_layout"}
    ]
    for n in cls.body:
        n.decorator_list = []
    code = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], cls],
        type_ignores=[],
    )
    ns = {
        "torch": torch,
        "nn": torch.nn,
        "aiter": aiter,
        "envs": SimpleNamespace(
            ATOM_FORCE_ATTN_TRITON=True,
            ATOM_USE_UNIFIED_ATTN=False,
            ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS=8192,
        ),
    }
    exec(  # noqa: S102
        compile(ast.fix_missing_locations(code), str(source), "exec"), ns
    )
    layer = ns["PagedAttentionImpl"]()
    for name, value in {
        "layer_num": 0,
        "sliding_window": -1,
        "head_dim": 128,
        "num_heads": heads,
        "num_kv_heads": 1,
        "rotary_emb": rope,
        "q_norm": Gemma(),
        "k_norm": Gemma(),
        "kv_cache_dtype": "fp8",
        "use_flash_layout": False,
    }.items():
        setattr(layer, name, value)
    kv = torch.empty(1, 1, 8, 128, 16, dtype=torch.float8_e4m3fn)
    scales = torch.ones(1, 1, 128)
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(slot_mapping=None),
        kv_cache_data={
            "layer_0": SimpleNamespace(
                k_cache=kv, v_cache=kv, k_scale=scales, v_scale=scales
            )
        },
    )
    torch.manual_seed(42)
    packed = torch.randn(tokens, (heads + 2) * 128, dtype=torch.bfloat16)
    q, k, v = packed.split((heads * 128, 128, 128), -1)
    pos = torch.arange(1, tokens + 1)
    out = layer.rope_cache(q, k, v, packed, pos, ctx)
    assert calls == (["rotary", "cache"] if tokens > 8192 else ["fused"])
    assert out[0].shape == (tokens, heads, 128)
    if tokens > 8192:
        # Check a nonzero position; construction and the callable must use
        # identical cos/sin values, including at the last prefix token.
        expected = rotary_module.apply_rotary_emb(
            q[-1].view(heads, 128),
            rope.cos_cache[tokens].reshape(64),
            rope.sin_cache[tokens].reshape(64),
        )
        torch.testing.assert_close(out[0][-1], expected)
