# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 DSpark stacked context K/V (`dspark_ctx_kv`): the stacking and the
writer's reach, then on gfx950 every pool byte against the layer-by-layer write."""

import statistics
import types

import pytest
import torch
from conftest import gpu_stack_import

pytest.importorskip("triton")

from atom.model_ops.minimax_m3.dspark_ctx_kv import (
    CTX_KV_ROW_PAD,
    _element_offsets,
    ctx_kv_gemm,
    stack_kv_rows,
)

LAYERS, HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 5, 64, 4, 128, 6144
Q_SIZE, KV_SIZE = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
BLOCKS = 256  # page-128 blocks: room for 32768 random slots
# Ms reaching every kernel of the qkv shape's tuned rows up to 32768, ragged ones too.
MS = [1, 8, 16, 31, 64, 100, 129, 333, 1000, 2048, 2049, 6001, 8191, 8192, 12288, 16384]
MS += [24576, 32768]  # the TP1 rows' 8-wave 1x4 kernel serves only these
MODES = ["randn", "outlier", "tiny", "normed"]
SEEDS = [101, 202, 303]


def test_layers_in_one_allocation_are_reached_from_layer_0():
    pool = torch.zeros(5, 4, 8)
    views = [pool[i] for i in range(5)]
    assert _element_offsets(views, views[0]).tolist() == [32 * i for i in range(5)]


def test_writer_refuses_caches_it_cannot_reach_from_layer_0():
    separate = [torch.zeros(4, 8) for _ in range(3)]
    with pytest.raises(ValueError, match="layer 0's allocation"):
        _element_offsets(separate, separate[0])
    pool = torch.zeros(3, 4, 8)
    with pytest.raises(ValueError, match="at or above it"):
        _element_offsets([pool[2], pool[1], pool[0]], pool[2])


def test_stacked_rows_are_each_layers_kv_rows_then_padding():
    q, kv, k = 96, 64, 64
    weights = [torch.randn(q + kv, k).to(torch.float8_e4m3fn) for _ in range(LAYERS)]
    scales = [
        torch.randint(100, 140, (q + kv, k // 32), dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        for _ in weights
    ]
    w, s = stack_kv_rows(weights, scales, slice(q, q + kv))
    assert w.shape == (CTX_KV_ROW_PAD, k) and s.shape == (CTX_KV_ROW_PAD, k // 32)
    for i in range(LAYERS):
        rows = slice(i * kv, (i + 1) * kv)
        assert torch.equal(w[rows].view(torch.uint8), weights[i][q:].view(torch.uint8))
        assert torch.equal(s[rows].view(torch.uint8), scales[i][q:].view(torch.uint8))
    assert not w[LAYERS * kv :].view(torch.uint8).any()
    assert (s[LAYERS * kv :].view(torch.uint8) == 127).all()


gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="needs gfx950",
)


class _NativeMxfp8Qkv(torch.nn.Module):
    """`LinearBase.forward`'s native MXFP8 1x32 branch on a preshuffled weight."""

    native_a8_group_rows = 1

    def __init__(self, weight, scale):
        super().__init__()
        self.weight = weight
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

    def forward(self, x):
        from atom.model_ops.blockscale import native_quant_linear

        return native_quant_linear(
            x,
            self.weight,
            self.weight_scale,
            weight_group_rows=1,
            weight_preshuffled=True,
        )


def _pool(separate=False):
    """Each layer's page-128 ``(k, v, k_scale, v_scale)`` in the draft pool's layout,
    random bytes; ``separate``: one pool per layer."""
    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool

    pools = []
    for _ in range(LAYERS if separate else 1):
        pool = MhaKvPool(
            layers=1 if separate else LAYERS,
            block_size=128,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            kv_dtype=torch.float8_e4m3fn,
        )
        pool.allocate(BLOCKS, "cuda")
        pools.append(pool)
    caches = []
    for i in range(LAYERS):
        pool, row = (pools[i], 0) if separate else (pools[0], i)
        caches.append((*pool.kv_views(row), *pool.scale_views(row)))
    for t in _flat(caches):
        t.view(torch.uint8).random_(0, 256)
    return pools, caches


def _flat(caches):
    return [t.view(torch.uint8) for layer_caches in caches for t in layer_caches]


def _shell(dspark, caches):
    """A `MiniMaxM3DSpark` of random native MXFP8 layers bound to ``caches``."""
    from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale
    from aiter.rotary_embedding import get_rope

    from atom.model_ops.utils import shuffle_weights
    from atom.models.dspark_draft import KVCacheBinding

    g = torch.Generator(device="cuda").manual_seed(0)
    rows = Q_SIZE + 2 * KV_SIZE
    rope = get_rope(
        HEAD_DIM, rotary_dim=64, max_position=1 << 20, base=1e7, rope_scaling=None
    ).cuda()

    def norm():
        w = torch.randn(HEAD_DIM, device="cuda", generator=g) * 0.1
        return types.SimpleNamespace(weight=w.bfloat16(), variance_epsilon=1e-6)

    layers = []
    for i, layer_caches in enumerate(caches):
        attn = object.__new__(dspark.M3DSparkAttention)
        torch.nn.Module.__init__(attn)
        w = torch.randn(rows, HIDDEN, device="cuda", generator=g) * 32
        weight = torch.nn.Parameter(w.to(torch.float8_e4m3fn), requires_grad=False)
        shuffle_weights(weight)
        scale = torch.randint(
            112, 124, (rows, HIDDEN // 32), device="cuda", generator=g
        ).to(torch.uint8)
        scale = shuffle_mxfp8_bpreshuffle_scale(scale.view(torch.float8_e8m0fnu))
        attn.qkv_proj = _NativeMxfp8Qkv(weight, scale)
        attn.q_norm, attn.k_norm, attn.rotary_emb = norm(), norm(), rope
        attn.num_heads, attn.num_kv_heads, attn.head_dim = HEADS, KV_HEADS, HEAD_DIM
        attn.q_size, attn.kv_size, attn.kv_cache_dtype = Q_SIZE, KV_SIZE, "fp8"
        attn.kv = KVCacheBinding(i, page_tokens=16)
        kv = attn.kv
        kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale = layer_caches
        layer = object.__new__(dspark.M3DSparkDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.self_attn = attn
        layers.append(layer)
    model = object.__new__(dspark.MiniMaxM3DSpark)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList(layers)
    model._ctx_kv = None
    return model


def _inputs(m, seed, mode, slots):
    """A context of ``m`` rows, its positions and its page-16 slots (-1 pads)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HIDDEN, device="cuda", generator=g)
    if mode == "outlier":  # outlier channels and all-zero rows
        x[:, torch.randint(0, HIDDEN, (24,), device="cuda", generator=g)] *= 80.0
        x[torch.rand(m, device="cuda", generator=g) < 0.01] = 0.0
    elif mode == "tiny":
        x *= 1e-3
    elif mode == "normed":  # what hidden_norm hands on: RMS-normed, wide channels
        x *= torch.exp(torch.randn(HIDDEN, device="cuda", generator=g))
        x *= torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    if slots == "contig":  # consecutive positions on whole page-16 runs
        pages = torch.randperm(BLOCKS * 8, device="cuda", generator=g)[: -(-m // 16)]
        slot = (pages[:, None] * 16 + torch.arange(16, device="cuda")).flatten()[:m]
        start = int(torch.randint(0, 900_000, (1,), device="cuda", generator=g))
        pos = torch.arange(start, start + m, device="cuda")
    else:
        slot = torch.randperm(BLOCKS * 128, device="cuda", generator=g)[:m]
        pos = torch.randint(0, 1 << 20, (m,), device="cuda", generator=g)
    if m > 8:
        slot[torch.rand(m, device="cuda", generator=g) < 0.03] = -1
        slot[-1] = -1
    return x.bfloat16(), pos, slot


def _layer_by_layer_then_stacked(dspark, model, caches, x, pos, slot):
    """Pool bytes after the layer-by-layer write and after the stacked one, from the
    same starting bytes."""
    start = [t.clone() for t in _flat(caches)]
    dspark.DSparkDraftModel.write_context_rows(model, x, pos, slot)
    want = [t.clone() for t in _flat(caches)]
    for t, s in zip(_flat(caches), start):
        t.copy_(s)
    model.write_context_rows(x, pos, slot)
    return want, _flat(caches)


@pytest.fixture(scope="module")
def arena():
    dspark = gpu_stack_import("atom.models.minimax_m3.dspark")
    pools, caches = _pool()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
        yield dspark, _shell(dspark, caches), caches, pools


@gfx950
@pytest.mark.parametrize("slots", ["contig", "random"])
@pytest.mark.parametrize("m", MS)
def test_stacked_write_is_the_layer_by_layer_write_bit_for_bit(arena, m, slots):
    dspark, model, caches, _ = arena
    for seed in SEEDS:
        for mode in MODES:
            x, pos, slot = _inputs(m, seed, mode, slots)
            want, got = _layer_by_layer_then_stacked(
                dspark, model, caches, x, pos, slot
            )
            assert all(map(torch.equal, got, want)), (mode, seed)
            weight, scale, qkv_rows, _ = model._ctx_kv[1]
            kv = ctx_kv_gemm(x, weight, scale, qkv_rows)
            assert kv is not None, f"no tuned qkv row at M={m}: nothing stacked"
            for i, layer in enumerate(model.layers):
                ref = layer.self_attn.qkv_proj(x)[:, Q_SIZE:]
                cols = kv[:, i * 2 * KV_SIZE : (i + 1) * 2 * KV_SIZE]
                assert torch.equal(cols.view(torch.int16), ref.view(torch.int16))


@gfx950
def test_caches_outside_layer_0s_allocation_are_written_layer_by_layer(monkeypatch):
    dspark = gpu_stack_import("atom.models.minimax_m3.dspark")
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    _pools, caches = _pool(separate=True)
    model = _shell(dspark, caches)
    want, got = _layer_by_layer_then_stacked(
        dspark, model, caches, *_inputs(2048, 7, "randn", "random")
    )
    assert model._ctx_kv[1] is None
    assert all(map(torch.equal, got, want))


@gfx950
def test_tp_above_1_writes_layer_by_layer(arena, monkeypatch):
    dspark, model, _, _ = arena
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(model, "_ctx_kv", None)
    model.write_context_rows(*_inputs(64, 7, "randn", "random"))
    assert model._ctx_kv[1] is None


if __name__ == "__main__":
    # us per write, layer by layer vs stacked: medians of 9 interleaved rounds of 20.
    dspark = gpu_stack_import("atom.models.minimax_m3.dspark")
    dspark.get_tensor_model_parallel_world_size = lambda: 1
    _pools, caches = _pool()
    model = _shell(dspark, caches)
    for m in (8, 64, 256, 1024, 8192):
        x, pos, slot = _inputs(m, 1, "normed", "contig")
        arms = {
            "layer by layer": dspark.DSparkDraftModel.write_context_rows,
            "stacked": dspark.MiniMaxM3DSpark.write_context_rows,
        }
        times = {name: [] for name in arms}
        for r in range(10):
            for name in arms if r % 2 else reversed(list(arms)):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(20):
                    arms[name](model, x, pos, slot)
                end.record()
                torch.cuda.synchronize()
                if r:  # round 0 warms up
                    times[name].append(start.elapsed_time(end) * 1e3 / 20)
        a, b = (statistics.median(t) for t in times.values())
        print(f"M={m}: layer by layer {a:.1f} us, stacked {b:.1f} us, {a / b:.2f}x")
