# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`causal_conv1d_fn`'s width-4 rows kernel against the reference kernel.

A channel-last `x` (the strided in_proj output, as the GDN layer passes it)
takes `_causal_conv1d_fwd_rows_kernel`; the same values copied channel-first
take `_causal_conv1d_fwd_kernel`, with `l2norm_fwd` after it for the fused
q / k norm. Both must give the same q, k, v and conv-state pool, bit for bit.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_fn

DEV = "cuda"
HEAD = 128


def _run(
    x, weight, bias, pool, seqlens, slots_in, slots_out, has_init, k_dim, v_dim, norm
):
    cu = torch.tensor([0] + seqlens, device=DEV).cumsum(0).to(torch.int32)
    return causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states=pool,
        query_start_loc=cu,
        k_dim_size=k_dim,
        v_dim_size=v_dim,
        cache_indices=slots_out,
        cache_indices_in=slots_in,
        has_initial_state=has_init,
        activation="silu",
        qk_l2norm_head_dim=HEAD if norm else None,
    )


@pytest.mark.parametrize("k_heads,v_heads", [(16, 32), (4, 8), (1, 2)])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("fork", [False, True])
def test_rows_kernel_matches_reference(k_heads, v_heads, norm, fork):
    k_dim, v_dim = k_heads * HEAD, v_heads * HEAD
    dim = 2 * k_dim + v_dim
    # single tokens, sequences shorter than the conv state, chunk boundaries
    seqlens = [1, 2, 3, 4, 9, 70, 129, 517, 3, 1000]
    tokens = sum(seqlens)
    n = len(seqlens)
    g = torch.Generator(device="cpu").manual_seed(k_heads * 4 + 2 * norm + fork)
    # q|k|v|z rows, as the in_proj output the layer slices
    proj = torch.randn(tokens, dim + v_dim, generator=g).to(torch.bfloat16).to(DEV)
    weight = (torch.randn(dim, 4, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    bias = (torch.randn(dim, generator=g) * 0.1).to(torch.bfloat16).to(DEV)
    alloc = torch.randn(64, 3, dim, generator=g).to(torch.bfloat16).to(DEV)
    perm = torch.randperm(64, generator=g).to(torch.int32).to(DEV)
    slots_out = perm[:n]
    slots_in = perm[n : 2 * n] if fork else slots_out
    has_init = (torch.rand(n, generator=g) < 0.7).to(DEV)

    x_rows = proj[:, :dim].T
    assert x_rows.stride(0) == 1
    x_ref = x_rows.contiguous()
    assert x_ref.stride(0) != 1

    out = {}
    pools = {}
    for name, x in (("rows", x_rows), ("ref", x_ref)):
        a = alloc.clone()
        out[name] = _run(
            x,
            weight,
            bias,
            a.transpose(-1, -2),
            seqlens,
            slots_in,
            slots_out,
            has_init,
            k_dim,
            v_dim,
            norm,
        )
        pools[name] = a
    for got, want in zip(out["rows"], out["ref"]):
        assert torch.equal(got, want)
    assert torch.equal(pools["rows"], pools["ref"])
