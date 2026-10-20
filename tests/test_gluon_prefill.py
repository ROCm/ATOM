# SPDX-License-Identifier: MIT
"""Verify the C32 adapter preserves native cache and request metadata."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from atom.model_ops.gluon_prefill import _scale_cache_view, run_gluon_prefill


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("scale_mode", ["token", "scalar0", "scalar1", "mixed"])
def test_native_pages_and_padded_metadata_are_passed_without_gather(draft, scale_mode):
    heads = 16 if draft else 1
    q = torch.randn(5, 16, 128, dtype=torch.bfloat16)
    k = torch.empty(8, heads, 8, 128, 16, dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    scales = torch.rand(8, heads, 256)[..., ::2]
    ks, vs = scales, scales
    if scale_mode != "token":
        ks = torch.tensor(0.375)
        vs = scales if scale_mode == "mixed" else torch.tensor(0.625)
        if scale_mode == "scalar1":
            ks, vs = ks.view(1), vs.view(1)
    meta = SimpleNamespace(
        has_cached=True,
        block_tables=torch.tensor([[4, 1], [6, 2], [-1, -1]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 5, 999], dtype=torch.int32),
        context_lens=torch.tensor([129, 255, 999], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_k=255,
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_draft=draft, scheduled_bs=2),
        attn_metadata=meta,
    )
    seen = []

    def attention(query, kc, vc, out, **kw):
        assert query is q and kc is k and vc is v
        assert kw["backend"] == "gluon"
        assert kw["k_descale"] is None and kw["v_descale"] is None
        for name, source in (("k_scale_cache", ks), ("v_scale_cache", vs)):
            view = kw[name]
            assert view.data_ptr() == source.data_ptr()
            assert view.untyped_storage().nbytes() == source.untyped_storage().nbytes()
            if source.numel() == 1:
                assert view.shape == (8, heads, 128)
                assert view.stride() == (0, 0, 0)
                assert view[7, heads - 1, 127] == source.reshape(())
            else:
                assert view is source
        assert kw["cu_seqlens_q"].tolist() == [0, 1, 5]
        assert kw["seqused_k"].tolist() == [129, 255]
        assert kw["block_table"].tolist() == [[4, 1], [6, 2]]
        assert kw["block_table"].data_ptr() == meta.block_tables.data_ptr()
        out.copy_(query)
        seen.append(kw)

    with patch("atom.model_ops.gluon_prefill.torch.empty", wraps=torch.empty) as alloc:
        out = run_gluon_prefill(attention, q, k, v, ks, vs, ctx, softmax_scale=0.125)
    assert len(seen) == 1 and alloc.call_count == 1
    assert alloc.call_args.args[0] == q.shape
    torch.testing.assert_close(out, q)
    assert meta.cu_seqlens_q[-1] == 999


def test_unsupported_layout_is_rejected_before_launch():
    with pytest.raises(ValueError, match="native M3"):
        run_gluon_prefill(
            None,
            torch.empty(1, 16, 128, dtype=torch.bfloat16),
            torch.empty(1, 1, 8, 128, 16, dtype=torch.float8_e4m3fn),
            torch.empty(1, 1, 128, 128, dtype=torch.float8_e4m3fn),
            None,
            None,
            None,
            softmax_scale=0.125,
        )


@pytest.mark.parametrize(
    "scale",
    [
        None,
        torch.ones(2),
        torch.ones(8, 16, 127),
        torch.ones((), dtype=torch.bfloat16),
        torch.ones((), device="meta"),
    ],
)
def test_invalid_scales_are_rejected(scale):
    with pytest.raises(ValueError, match="Gluon C32 requires"):
        _scale_cache_view(scale, (8, 16, 128), torch.device("cpu"))


def test_scalar_view_storage_does_not_grow_with_real_draft_pool():
    scalar = torch.tensor(0.375)
    view = _scale_cache_view(scalar, (53602, 16, 128), scalar.device)
    assert view.stride() == (0, 0, 0)
    assert view.untyped_storage().nbytes() == 4
    assert view.data_ptr() == scalar.data_ptr()


def test_real_draft_rope_cache_selects_scalar_instead_of_unused_pool_scales():
    # Execute the actual Python method; only its GPU writer is stubbed on CPU.
    # The GPU gate calls the complete PagedAttentionImpl methods and real writer.
    path = Path(__file__).resolve().parents[1] / "atom/model_ops/attention_mha.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PagedAttentionImpl"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "rope_cache"
    )
    fn.decorator_list = []
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    scalar = torch.tensor(0.375)
    pool_scales = torch.full((8, 16, 128), float("nan"))
    kc = torch.empty(8, 16, 8, 128, 16, dtype=torch.float8_e4m3fn)
    vc = torch.empty_like(kc)
    q = torch.randn(4, 16, 128, dtype=torch.bfloat16)
    meta = SimpleNamespace(
        has_cached=True,
        block_tables=torch.tensor([[4, 1, 3]]),
        cu_seqlens_q=torch.tensor([0, 4]),
        context_lens=torch.tensor([257]),
        max_seqlen_q=4,
        max_seqlen_k=257,
        slot_mapping=None,
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_draft=True, scheduled_bs=1),
        attn_metadata=meta,
        kv_cache_data={
            "layer_60": SimpleNamespace(
                k_cache=kc, v_cache=vc, k_scale=pool_scales, v_scale=pool_scales
            )
        },
    )
    layer = SimpleNamespace(
        layer_num=60,
        sliding_window=-1,
        head_dim=128,
        rotary_emb=SimpleNamespace(cos_cache=None, sin_cache=None, is_neox_style=True),
        q_norm=None,
        k_norm=None,
        kv_scale=scalar,
        kv_cache_dtype="fp8",
        use_flash_layout=False,
    )
    calls = []

    def writer(q, k, v, kc, vc, slots, positions, cos, sin, ks, vs, *args, **kwargs):
        assert ks is scalar and vs is scalar
        calls.append("writer")
        return q, k, kc, vc

    ns = {
        "torch": torch,
        "envs": SimpleNamespace(
            ATOM_FORCE_ATTN_TRITON=True, ATOM_USE_UNIFIED_ATTN=False
        ),
        "fused_qk_rope_reshape_and_cache": writer,
    }
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)  # noqa: S102
    result = ns["rope_cache"](layer, q, q, q, None, None, ctx)
    assert not layer.per_token_quant
    assert result[-2] is scalar and result[-1] is scalar
    assert pool_scales.isnan().all()

    def attention(query, k, v, out, **kwargs):
        for name in ("k_scale_cache", "v_scale_cache"):
            view = kwargs[name]
            assert view.data_ptr() == scalar.data_ptr()
            assert view.stride() == (0, 0, 0)
        calls.append("attention")
        out.copy_(query)

    out = run_gluon_prefill(
        attention, q, kc, vc, *result[-2:], ctx, softmax_scale=0.125
    )
    torch.testing.assert_close(out, q)
    assert calls == ["writer", "attention"]
