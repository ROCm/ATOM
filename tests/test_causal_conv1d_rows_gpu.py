# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`causal_conv1d_fn`'s width-4 rows kernel against the reference kernel.

A channel-last `x` (the strided in_proj output, as the GDN layer passes it)
takes `_causal_conv1d_fwd_rows_kernel`; the same values copied channel-first
take `_causal_conv1d_fwd_kernel`. The two must leave the same conv-state pool,
bit for bit. The rows kernel's q, k, v must be the correctly rounded bf16 of an
fp64 conv (with the q / k norm applied to the bf16 conv output, as `l2norm_fwd`
after the conv does), up to its fast SiLU: at most one bf16 step, on almost no
values. A step in a conv output can become two once its head is normalized and
rounded again, so q and k may be two steps off when the norm is fused.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_fn

DEV = "cuda"
HEAD = 128


def _fp64_conv(x, weight, bias, alloc, seqlens, slots_in, has_init, k_dim, norm):
    """q, k, v of an fp64 width-4 conv + SiLU over each sequence, its window
    starting from the slot's state (oldest first) or zeros."""
    x, w, b = x.double(), weight.double(), bias.double()
    outs, start = [], 0
    for i, n in enumerate(seqlens):
        if has_init[i]:
            init = alloc[slots_in[i]].double()
        else:
            init = torch.zeros(3, x.shape[1], dtype=torch.float64, device=DEV)
        win = torch.cat([init, x[start : start + n]])
        y = sum(win[j : j + n] * w[:, j] for j in range(4)) + b
        outs.append(y / (1 + torch.exp(-y)))
        start += n
    y = torch.cat(outs)
    q, k, v = y[:, :k_dim], y[:, k_dim : 2 * k_dim], y[:, 2 * k_dim :]
    if norm:
        q, k = (
            (t := u.to(torch.bfloat16).double().view(-1, HEAD))
            / torch.sqrt((t * t).sum(-1, keepdim=True) + 1e-6)
            for u in (q, k)
        )
    return [t.reshape(y.shape[0], -1) for t in (q, k, v)]


def _bf16_steps(got, ref):
    """|got - bf16(ref)| in bf16 steps at |ref|."""
    step = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    return (got.double() - ref.to(torch.bfloat16).double()).abs() / step


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
def test_rows_kernel_matches_fp64(k_heads, v_heads, norm, fork):
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
    assert torch.equal(pools["rows"], pools["ref"])
    refs = _fp64_conv(
        proj[:, :dim], weight, bias, alloc, seqlens, slots_in, has_init, k_dim, norm
    )
    for i, (got, ref) in enumerate(zip(out["rows"], refs)):
        steps = _bf16_steps(got.reshape(ref.shape), ref)
        assert steps.max() <= (2 if norm and i < 2 else 1)
        assert (steps > 0).double().mean() < 1e-3
