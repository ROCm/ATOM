# SPDX-License-Identifier: MIT
"""CPU checks of SP MegaMoE tiling, native fallback and fused-block ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_comm_fused_moe import atom_modules  # noqa: F401
from test_sp_moe_dispatch import _layer
from test_sp_moe_dispatch import sp_modules as _sp_modules

from atom.config import use_custom_atom_config
from atom.model_ops import sp_mega_moe as fusion
from atom.model_ops.sp_mega_moe import SPMegaMoE, local_token_tile


@pytest.fixture(autouse=True)
def mock_mega_buffer_ops(monkeypatch):
    from atom.model_ops import sp_mega_moe

    # Replace only the GPU copy boundary for CPU orchestration tests. The real
    # copies and launch counts are covered on GPU.
    def copy(source, source_scale, out, out_scale):
        out.view(torch.uint8).copy_(source.view(torch.uint8))
        if source_scale is not None:
            out_scale.view(-1).copy_(source_scale)

    monkeypatch.setattr(sp_mega_moe, "copy_tail", copy)


@pytest.fixture
def sp_modules(request):
    return _sp_modules.__wrapped__(request)


@pytest.mark.parametrize("rows,use_mega", [(16, False), (63, False), (64, True)])
def test_small_batches_keep_the_native_moe(monkeypatch, sp_modules, rows, use_mega):
    from importlib import import_module

    moe = sp_modules.moe
    layer = _layer(moe)
    layer.w13_weight = torch.empty(1, dtype=moe.dtypes.fp4x2)
    resolver = Mock(return_value=moe.dtypes.fp4x2 if use_mega else torch.bfloat16)
    monkeypatch.setattr(
        import_module("aiter.fused_moe"),
        "resolve_activation_dtype",
        resolver,
        raising=False,
    )
    runtime = SimpleNamespace(cfg=SimpleNamespace(world_size=4), forward=Mock())
    prefix = SimpleNamespace(eps=1e-6, forward=Mock())
    backend = SPMegaMoE(runtime, prefix)
    assert backend.supports(layer, rows) is use_mega
    assert resolver.call_args.kwargs["M"] == rows * 4
    if use_mega:
        return

    def norm(x, residual, weight, eps, fp8):
        return x + residual, torch.ones(x.shape[0], 1), x + residual

    monkeypatch.setattr(fusion, "_norm_input", norm)
    layer.forward_impl = Mock(side_effect=lambda x, logits: x * 2)
    attention = torch.ones(rows, 8, dtype=torch.bfloat16)
    gate = torch.zeros(128, 8, dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=torch.bfloat16)
    normalized, _, residual = backend.forward_block(
        layer, attention, attention * 3, weight, gate, None, weight, 1e-6, False
    )
    x, logits = layer.forward_impl.call_args.args
    torch.testing.assert_close(x, attention * 4)
    assert logits.shape == (rows, 128)
    torch.testing.assert_close(normalized, attention * 12)
    torch.testing.assert_close(residual, attention * 12)
    prefix.forward.assert_not_called()
    runtime.forward.assert_not_called()


@pytest.mark.parametrize(
    ("budget", "world_size", "tile"),
    [(2048, 4, 512), (4098, 4, 1025), (65536, 4, 8192), (16385, 8, 2049)],
)
def test_tile_covers_the_largest_local_shard(budget, world_size, tile):
    assert local_token_tile(budget, world_size) == tile


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rows", [0, 16, 65])
def test_full_op_preserves_rank_rows_residual_and_output_ownership(
    monkeypatch, sp_modules, fp8, rows
):
    from aiter import dtypes

    # Real custom-op implementation and backend orchestration; only the device
    # boundaries are replaced. Shared scratch is overwritten on every launch.
    x = (torch.arange(rows) % 8).bfloat16()[:, None].expand(rows, 8).contiguous()
    residual = torch.ones_like(x)
    expected_residual = (x + residual) * 2 + (x + residual)
    scratch = torch.empty((4 * 32, 8), dtype=torch.bfloat16)
    tail_calls, plain_calls = [], []

    def norm(x, residual, weight, eps, fp8):
        r = x + residual
        return r.to(dtypes.fp8 if fp8 else r.dtype), torch.ones(r.shape[0], 1), r

    monkeypatch.setattr(fusion, "_norm_input", norm)

    def route(layer, hidden, router):
        return (
            torch.ones(hidden.shape[0], 5),
            hidden[:, :1].int().expand(-1, 5).contiguous(),
        )

    def prefix_forward(x, r, w, *, out, res_out, router):
        _gate, _bias, ids, weights = router
        out.copy_(x + r)
        res_out.copy_(out)
        tw, ti = route(None, out, None)
        ids.copy_(ti)
        weights.copy_(tw)

    def launch(hidden, weights, ids, *, tail=None, bf16=False):
        torch.testing.assert_close(ids[:, :1].bfloat16(), hidden[:, :1])
        y = hidden * 2
        if tail is None:
            plain_calls.append(hidden.shape[0])
            return y
        tail_calls.append(hidden.shape[0])
        res_in, res_out, _nw = tail
        res_out.copy_(y + res_in)
        m = hidden.shape[0]
        # Non-owning rows contain poison; selecting global row 0 would fail.
        scratch.fill_(-42)
        scratch[2 * m : 3 * m].copy_(res_out)
        q = scratch.to(dtypes.fp8).view(torch.uint8)
        result = (y, q, torch.ones(m * 4))
        return (*result, scratch) if bf16 else result

    backend = fusion.SPMegaMoE(
        SimpleNamespace(
            cfg=SimpleNamespace(rank=2, world_size=4, tail_eps=1e-6),
            max_local_tokens=32,
            set_weights=Mock(),
            forward=launch,
            tail_ok=lambda m: m >= 16,
        ),
        SimpleNamespace(
            mmax=64,
            eps=1e-6,
            routes=lambda m: m % 16 == 0,
            forward=prefix_forward,
        ),
    )
    backend.supports = lambda layer, rows: True
    backend._route = route
    layer = SimpleNamespace(
        _sp_mega_moe=backend,
        w13_weight=object(),
        w13_weight_scale=object(),
        w2_weight=object(),
        w2_weight_scale=object(),
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"e": layer})
    )
    weight = torch.ones(8, dtype=torch.bfloat16)
    gate = torch.zeros(128, 8, dtype=torch.bfloat16)
    bias = torch.zeros(128)
    before_x, before_res = x.clone(), residual.clone()
    with use_custom_atom_config(config):
        first = fusion.minimax_m3_sp_mega_block(
            x, residual, weight, gate, bias, weight, "e", 1e-6, fp8
        )
        second = fusion.minimax_m3_sp_mega_block(
            x, residual, weight, gate, bias, weight, "e", 1e-6, fp8
        )
    second[0].zero_()
    second[2].zero_()
    expected_normalized = expected_residual.to(dtypes.fp8) if fp8 else expected_residual
    torch.testing.assert_close(first[0].float(), expected_normalized.float())
    torch.testing.assert_close(first[2], expected_residual)
    torch.testing.assert_close(x, before_x)
    torch.testing.assert_close(residual, before_res)
    assert first[1].shape == (rows, 1)
    assert first[0].dtype == (dtypes.fp8 if fp8 else torch.bfloat16)
    if rows == 65:
        assert plain_calls == [1, 1]
        assert tail_calls == [32, 32, 32, 32]
