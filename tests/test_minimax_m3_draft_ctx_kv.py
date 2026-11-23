# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 draft's one-GEMM context K/V (`MiniMaxM3DSpark._context_kv_weight`):
the K/V rows of several preshuffled MXFP8 qkv weights, row-sliced on 16-row
tiles and concatenated, give those columns of each full GEMM -- up to the
accumulation order of whichever tuned kernel each N picks (a rare 1-ulp bf16
difference)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _mx8_weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(120, 130, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    return w, s.view(torch.float8_e8m0fnu)


@pytest.mark.parametrize("rows", [8, 56, 448])
def test_concat_kv_rows_gemm(rows):
    from aiter.ops.shuffle import shuffle_weight

    from atom.model_ops.blockscale import native_quant_linear, quantize_fp8

    N, K, Q, KV = 2304, 6144, 2048, 128
    x = (torch.randn(rows, K, device="cuda") * 2).to(torch.bfloat16)
    xq, xs = quantize_fp8(x)
    full, parts_w, parts_s = [], [], []
    for layer in range(5):
        w, s = _mx8_weight(N, K, layer)
        w_sh = shuffle_weight(w, layout=(16, 16))
        full.append(
            native_quant_linear(xq, w_sh, s, x_scale=xs, weight_group_rows=1, weight_preshuffled=True)
        )
        parts_w.append(w_sh.narrow(0, Q, 2 * KV))
        parts_s.append(s.narrow(0, Q, 2 * KV))
    kv = native_quant_linear(
        xq, torch.cat(parts_w), torch.cat(parts_s), x_scale=xs, weight_group_rows=1,
        weight_preshuffled=True,
    )  # fmt: skip
    for layer in range(5):
        got = kv[:, layer * 2 * KV : (layer + 1) * 2 * KV].float()
        want = full[layer][:, Q:].float()
        ulp = want.abs().clamp(min=2**-126) * 2**-7  # one bf16 ulp, rounded up
        assert bool(((got - want).abs() <= ulp).all())
        assert (got != want).float().mean().item() < 0.005


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rows", [8, 64, 448])
def test_ctx_rope_cache_one_launch(rows, fp8):
    """`ctx_rope_cache_write` (every layer in one launch) writes the caches the
    per-layer `triton_fused_norm_rope_cache` calls write, bit for bit."""
    from types import SimpleNamespace

    from atom.model_ops.triton_fused_qkv_norm_rope_cache import triton_fused_norm_rope_cache
    from atom.models.minimax_m3.dspark import ctx_rope_cache_write

    torch.manual_seed(rows + fp8)
    L, H, D, R, P16, MAXPOS = 5, 1, 128, 64, 64, 8192
    dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
    X = 16 // torch.empty(0, dtype=dt).element_size()
    kv = (torch.randn(rows, L * 2 * H * D, device="cuda") * 2).to(torch.bfloat16)
    positions = torch.randint(0, MAXPOS, (rows,), device="cuda", dtype=torch.int64)
    slots = torch.randperm(P16 * 16, device="cuda")[:rows].to(torch.int64)
    slots[::7] = -1
    inv = 1.0 / (10000 ** (torch.arange(0, R, 2, device="cuda").float() / R))
    ang = torch.arange(MAXPOS, device="cuda").float()[:, None] * inv[None, :]
    rot = SimpleNamespace(
        rotary_dim=R,
        cos_cache=ang.cos().to(torch.bfloat16)[:, None, None, :],
        sin_cache=ang.sin().to(torch.bfloat16)[:, None, None, :],
    )
    norms = [
        SimpleNamespace(weight=(torch.randn(D, device="cuda") * 0.3).to(torch.bfloat16), variance_epsilon=1e-6)
        for _ in range(L)
    ]

    def pools():
        k = torch.zeros(L, P16, H, D // X, 16, X, device="cuda", dtype=dt)
        v = torch.zeros(L, P16, H, 16 // X, D, X, device="cuda", dtype=dt)
        ks = torch.zeros(L, P16, H, 16, device="cuda") if fp8 else None
        vs = torch.zeros(L, P16, H, 16, device="cuda") if fp8 else None
        return k, v, ks, vs

    def layer(p, i):
        k, v, ks, vs = p
        return k[i], v[i], None if ks is None else ks[i], None if vs is None else vs[i]

    ref, got = pools(), pools()
    for i in range(L):
        kk = kv[:, 2 * i * H * D : (2 * i + 1) * H * D]
        vv = kv[:, (2 * i + 1) * H * D : (2 * i + 2) * H * D]
        triton_fused_norm_rope_cache(
            kk, kk, vv, positions, q_norm=norms[i], k_norm=norms[i], rotary_emb=rot,
            num_heads=0, num_kv_heads=H, head_dim=D, k_cache=ref[0][i], v_cache=ref[1][i],
            k_scale=layer(ref, i)[2], v_scale=layer(ref, i)[3], slot_mapping=slots,
            kv_cache_dtype="fp8" if fp8 else "auto",
        )  # fmt: skip
    kw = torch.stack([n.weight for n in norms])
    assert ctx_rope_cache_write(
        kv, positions, slots, kw, rot.cos_cache.squeeze(-2).squeeze(-2),
        rot.sin_cache.squeeze(-2).squeeze(-2), [layer(got, i) for i in range(L)], H, R, 1e-6,
        "fp8" if fp8 else "auto",
    )  # fmt: skip
    def bits(t):
        return t.view(torch.uint8) if t.element_size() == 1 else t

    for a, b in zip(ref, got):
        if a is not None:
            assert torch.equal(bits(a), bits(b))
    # layers not evenly spaced (one pool, out of order): refused, nothing written
    other = pools()
    shuffled = [layer(other, i) for i in (0, 1, 3, 2, 4)]
    assert not ctx_rope_cache_write(
        kv, positions, slots, kw, rot.cos_cache.squeeze(-2).squeeze(-2),
        rot.sin_cache.squeeze(-2).squeeze(-2), shuffled, H, R, 1e-6, "fp8" if fp8 else "auto",
    )  # fmt: skip
    assert not bits(other[0]).any()
