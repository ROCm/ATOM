# SPDX-License-Identifier: MIT

"""The sliding-window prefill causal conv1d (`causal_conv1d_fn`'s prefill path
at width 4) against the tiled kernel it replaces there. Both round every product (fp fusion
off) and sum the taps in the same order, so the outputs and the conv-state pool
must match bit for bit at every layout, with or without a bias.

Cases as in test_causal_conv1d_tiled.py: sequences shorter than the conv state,
sequences ending inside a tile or on its edge, forks (read slot != write slot),
pad slots, a bias.
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


def _inputs(lens, seed, bias, geom="k3_split", width=4):
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


def _metadata(cu, num_tokens):
    md = SimpleNamespace()
    md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr = (
        cc.compute_causal_conv1d_metadata(cu, num_tokens)
    )
    return md


def _run(fwd, x, w, b, pool, cu, idx, idx_in, init, **tile):
    pool = pool.clone()
    kd = x.shape[0] // 3
    out = fwd(
        x,
        w,
        b,
        pool.transpose(-1, -2),
        cu,
        kd,
        kd,
        idx,
        idx_in,
        init,
        "silu",
        cc.PAD_SLOT_ID,
        _metadata(cu, x.shape[1]),
        **tile,
    )
    return out, pool


CASES = [
    # lens, has_initial_state, fork
    ([1], [True], False),
    ([2], [True], False),
    ([2], [False], False),
    ([3], [True], False),
    ([1, 2, 3, 5], [True, True, False, True], False),
    ([16], [True], False),
    ([17], [False], False),
    ([64], [True], False),
    ([65], [False], False),
    ([63, 1, 129, 2], [True, False, True, True], True),
    ([4096, 7, 2048, 1000], [True, False, False, True], True),
]


@pytest.mark.parametrize(
    "geom,bias", [("k3_split", False), ("k3_fused", False), ("small", True)]
)
@pytest.mark.parametrize("lens,init,fork", CASES)
@pytest.mark.parametrize(
    "tile",
    [
        {},
        {"block_n": 128, "num_warps": 1, "num_stages": 1},
        {"block_n": 512, "num_warps": 4, "num_stages": 2},
    ],
)
def test_slide_matches_tiled(lens, init, fork, geom, bias, tile):
    x, w, b, pool, cu = _inputs(lens, len(lens) * 7, bias, geom)
    n = len(lens)
    idx = torch.tensor([1 + 2 * i for i in range(n)], dtype=torch.int32, device="cuda")
    idx_in = idx + 1 if fork else idx
    init_t = torch.tensor(init, dtype=torch.bool, device="cuda")
    args = (x, w, b, pool, cu, idx, idx_in, init_t)
    ref, pool_r = _run(cc._causal_conv1d_fwd_tiled, *args)
    got, pool_t = _run(cc._causal_conv1d_fwd_slide, *args, **tile)
    assert all(torch.equal(r, t) for r, t in zip(ref, got))
    assert torch.equal(pool_r, pool_t)


def test_pad_slot_rows_are_skipped():
    x, w, b, pool, cu = _inputs([40, 9, 70], seed=3, bias=False)
    idx = torch.tensor([2, cc.PAD_SLOT_ID, 5], dtype=torch.int32, device="cuda")
    init = torch.tensor([True, True, False], device="cuda")
    args = (x, w, b, pool, cu, idx, idx, init)
    ref, pool_r = _run(cc._causal_conv1d_fwd_tiled, *args)
    got, pool_t = _run(cc._causal_conv1d_fwd_slide, *args)
    live = torch.cat([torch.arange(0, 40), torch.arange(49, 119)]).cuda()
    for r, t in zip(ref, got):
        assert torch.equal(r[live], t[live])
    assert torch.equal(pool_r, pool_t)


@pytest.mark.parametrize("width,slides", [(4, True), (3, False)])
def test_dispatch_by_width(monkeypatch, width, slides):
    x, w, b, pool, cu = _inputs([300, 5], seed=11, bias=False, width=width)
    idx = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    init = torch.tensor([True, False], device="cuda")
    kd = x.shape[0] // 3

    def fn():
        p = pool.clone()
        out = cc.causal_conv1d_fn(
            x,
            w,
            b,
            activation="silu",
            conv_states=p.transpose(-1, -2),
            has_initial_state=init,
            cache_indices=idx,
            query_start_loc=cu,
            k_dim_size=kd,
            v_dim_size=kd,
            metadata=_metadata(cu, x.shape[1]),
        )
        return out, p

    with pytest.MonkeyPatch.context() as patch:
        # The reference: the tiled kernel at every width.
        patch.setattr(cc, "_causal_conv1d_fwd_slide", cc._causal_conv1d_fwd_tiled)
        ref, pool_r = fn()
    calls = []
    real = cc._causal_conv1d_fwd_slide
    monkeypatch.setattr(
        cc, "_causal_conv1d_fwd_slide", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    got, pool_t = fn()
    assert bool(calls) == slides
    assert all(torch.equal(r, t) for r, t in zip(ref, got))
    assert torch.equal(pool_r, pool_t)
