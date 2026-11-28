# SPDX-License-Identifier: MIT

"""The tiled prefill causal conv1d (`causal_conv1d_fn`'s prefill path) against
the 8-token row kernel it replaces. The conv-state pool must match bit for bit
everywhere; the outputs at Kimi-K3's shape and layouts (width 4, no bias), and
within 1 bf16 ulp elsewhere, where the row kernel's backend may fuse a tap.

The cases are the ones a tile can get wrong and a long sequence never shows:
sequences shorter than the conv state (shift-left update), sequences ending
inside a tile or exactly on its edge, forks (read slot != write slot), pad
slots, a bias, and widths below 4.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.mamba_ops import causal_conv1d as cc

# (k_dim_size = v_dim_size, projection row width): Kimi-K3 per rank under the
# split in_proj (qkvg rows) and the fused one, then a small layout.
GEOMS = {"k3_split": (1536, 6144), "k3_fused": (1536, 6284), "small": (512, 1600)}
SLOTS = 12
COLS = 7  # pool token axis: 3 + num_spec


def _inputs(lens, width, seed, bias, geom="k3_split"):
    kd, row = GEOMS[geom]
    dim = 3 * kd
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lens)
    fused = torch.randn(T, row, generator=g, device="cuda").bfloat16()
    x = fused[:, :dim].transpose(0, 1)  # channel-last view, projection row stride
    w = torch.randn(dim, width, generator=g, device="cuda") * 0.3
    b = torch.randn(dim, generator=g, device="cuda") if bias else None
    pool = torch.randn(SLOTS, COLS, dim, generator=g, device="cuda").bfloat16()
    cu = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    return x, w, b, pool, cu


def _close(r, t, exact):
    if exact:
        return torch.equal(r, t)
    # 1 bf16 ulp of the output, or 1e-5 where the fp32 sum cancels below that.
    r, t = r.float(), t.float()
    _, e = torch.frexp(torch.maximum(r.abs(), t.abs()))
    ulp = torch.ldexp(torch.ones_like(r), e - 8).clamp(min=1e-5)
    return bool(((r - t).abs() <= ulp).all())


def _run(x, w, b, pool, cu, idx, idx_in, init, tiled, tile=()):
    pool = pool.clone()
    kd = x.shape[0] // 3
    md = SimpleNamespace()
    md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr = (
        cc.compute_causal_conv1d_metadata(cu, x.shape[1])
    )
    args = {
        "activation": "silu",
        "conv_states": pool.transpose(-1, -2),
        "has_initial_state": init,
        "cache_indices": idx,
        "cache_indices_in": idx_in,
        "query_start_loc": cu,
        "k_dim_size": kd,
        "v_dim_size": kd,
        "metadata": md,
    }
    if tiled:
        assert cc._tiled_conv1d_applies(
            x, w, args["conv_states"], kd, kd, idx, init
        ), "case would not take the tiled kernel"
        out = cc._causal_conv1d_fwd_tiled(
            x,
            w,
            b,
            args["conv_states"],
            cu,
            kd,
            kd,
            idx,
            idx_in,
            init,
            "silu",
            cc.PAD_SLOT_ID,
            md,
            *tile,
        )
    elif tiled is None:
        out = cc.causal_conv1d_fn(x, w, b, token_tiles=True, **args)
    else:
        # The reference: `causal_conv1d_fn` on the row kernel (no opt-in).
        out = cc.causal_conv1d_fn(x, w, b, **args)
    return out, pool


CASES = [
    # lens, has_initial_state, fork
    ([1], [True], False),
    ([2], [True], False),
    ([2], [False], False),
    ([3], [True], False),
    ([1, 2, 3, 5], [True, True, False, True], False),
    ([64], [True], False),
    ([65], [False], False),
    ([63, 1, 129, 2], [True, False, True, True], True),
    ([4096, 7, 2048, 1000], [True, False, False, True], True),
]


SHAPES = [("k3_split", 4, False), ("k3_fused", 4, False)] + [
    ("small", width, bias) for width in (4, 3, 2) for bias in (False, True)
]


@pytest.mark.parametrize("geom,width,bias", SHAPES)
@pytest.mark.parametrize("lens,init,fork", CASES)
# (block_m, block_n, num_warps, channel_fast); () is the module default
@pytest.mark.parametrize(
    "tile",
    [
        (),
        (16, 512, 4, False),
        (32, 256, 4, True),
        (16, 128, 2, False),
        (64, 128, 8, True),
    ],
)
def test_tiled_matches_row_kernel(lens, init, fork, geom, width, bias, tile):
    x, w, b, pool, cu = _inputs(lens, width, len(lens) * 7 + width, bias, geom)
    n = len(lens)
    idx = torch.tensor([1 + 2 * i for i in range(n)], dtype=torch.int32, device="cuda")
    idx_in = idx + 1 if fork else idx
    init_t = torch.tensor(init, dtype=torch.bool, device="cuda")
    (qr, kr, vr), pool_r = _run(x, w, b, pool, cu, idx, idx_in, init_t, tiled=False)
    (qt, kt, vt), pool_t = _run(x, w, b, pool, cu, idx, idx_in, init_t, True, tile)
    exact = geom != "small" and width == 4 and not bias
    assert all(_close(r, t, exact) for r, t in ((qr, qt), (kr, kt), (vr, vt)))
    assert torch.equal(pool_r, pool_t)


def test_pad_slot_rows_are_skipped():
    x, w, b, pool, cu = _inputs([40, 9, 70], 4, seed=3, bias=False)
    idx = torch.tensor([2, cc.PAD_SLOT_ID, 5], dtype=torch.int32, device="cuda")
    init = torch.tensor([True, True, False], device="cuda")
    (qr, kr, vr), pool_r = _run(x, w, b, pool, cu, idx, idx, init, tiled=False)
    (qt, kt, vt), pool_t = _run(x, w, b, pool, cu, idx, idx, init, tiled=True)
    live = torch.cat([torch.arange(0, 40), torch.arange(49, 119)]).cuda()
    for r, t in ((qr, qt), (kr, kt), (vr, vt)):
        assert torch.equal(r[live], t[live])
    assert torch.equal(pool_r, pool_t)


def test_prefill_takes_the_tiled_kernel(monkeypatch):
    # Width 3: width 4 takes the sliding-window kernel (test_causal_conv1d_slide).
    x, w, b, pool, cu = _inputs([300, 5], 3, seed=11, bias=False)
    idx = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    init = torch.tensor([True, False], device="cuda")
    ref, pool_r = _run(x, w, b, pool, cu, idx, idx, init, tiled=False)
    calls = []
    real = cc._causal_conv1d_fwd_tiled
    monkeypatch.setattr(
        cc, "_causal_conv1d_fwd_tiled", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    got, pool_t = _run(x, w, b, pool, cu, idx, idx, init, tiled=None)
    assert calls, "causal_conv1d_fn kept the row kernel"
    assert all(_close(r, t, exact=False) for r, t in zip(ref, got))
    assert torch.equal(pool_r, pool_t)
