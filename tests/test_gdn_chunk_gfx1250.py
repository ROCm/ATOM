# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 Gluon chunk_h / chunk_o kernels against the Triton kernels they
replace there: varlen with a state pool by slot (eager) or a bucket-shaped plan,
and a fixed-length batch; the output with and without the fused gated RMSNorm."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
    pytest.skip("gfx1250 kernels", allow_module_level=True)

from atom.model_ops.fla_ops import chunk_delta_h, chunk_o  # noqa: E402
from atom.model_ops.fla_ops.chunk import chunk_gated_delta_rule  # noqa: E402
from atom.model_ops.fla_ops.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from atom.model_ops.fla_ops.layernorm_guard import layer_norm_fwd  # noqa: E402
from atom.model_ops.fla_ops.varlen_plan import VarlenPlan  # noqa: E402

DEV = "cuda"


def _inputs(lens, H, Hg, seed=0):
    T = sum(lens)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    row = 2 * Hg * 128 + 2 * H * 128
    proj = torch.randn(T, row, device=DEV, dtype=torch.bfloat16, generator=gen)
    q = proj[:, : Hg * 128].reshape(1, T, Hg, 128).contiguous()
    k = proj[:, Hg * 128 : 2 * Hg * 128].reshape(1, T, Hg, 128).contiguous()
    v = proj[:, 2 * Hg * 128 : 2 * Hg * 128 + H * 128].reshape(1, T, H, 128).contiguous()
    z = proj[:, 2 * Hg * 128 + H * 128 :].view(T, H, 128)
    g = -torch.nn.functional.softplus(torch.randn(1, T, H, device=DEV, generator=gen)) / 2
    beta = torch.rand(1, T, H, device=DEV, generator=gen).to(torch.bfloat16)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=DEV)
    n = len(lens)
    pool = torch.randn(4 * n + 4, H, 128, 128, device=DEV, generator=gen) / 20
    perm = torch.randperm(4 * n + 4, generator=torch.Generator().manual_seed(seed))
    slots = (perm[:n].to(DEV), perm[n : 2 * n].to(DEV), torch.arange(n, device=DEV) % 3 != 2)
    weight = 1 + torch.randn(128, device=DEV, generator=gen) / 10
    return q, k, v, z, g, beta, cu, pool, slots, weight


def _plan(cu, slots):
    n = cu.numel() - 1
    n_max = n + 3

    def padded(t, rows, dtype=None):
        out = torch.zeros((rows, *t.shape[1:]), device=DEV, dtype=dtype or t.dtype)
        out[: t.shape[0]] = t
        return out

    ci64, ci16 = prepare_chunk_indices(cu, 64), prepare_chunk_indices(cu, 16)
    i32 = torch.int32
    plan = VarlenPlan(
        chunk_indices_64=padded(ci64, ci64.shape[0] + 5),
        nt64=torch.tensor([ci64.shape[0]], device=DEV, dtype=i32),
        nt64_max=ci64.shape[0] + 5,
        chunk_indices_16=padded(ci16, ci16.shape[0] + 20),
        nt16=torch.tensor([ci16.shape[0]], device=DEV, dtype=i32),
        nt16_max=ci16.shape[0] + 20,
        chunk_offsets=padded(prepare_chunk_offsets(cu, 64), n_max + 1, i32),
        num_seqs=torch.tensor([n], device=DEV, dtype=i32),
        n_max=n_max,
        state_idx=padded(slots[1], n_max, i32),
        state_idx_in=padded(slots[0], n_max, i32),
        has_initial_state=padded(slots[2], n_max, i32),
    )
    return plan, torch.cat([cu, cu[-1:].repeat(n_max - n)])


def _run(gluon, q, k, v, z, g, beta, cu, pool, slots, weight, path, gate, monkeypatch):
    monkeypatch.setattr(chunk_delta_h, "_use_gfx1250_kernel", lambda _: gluon)
    monkeypatch.setattr(chunk_o, "_use_gfx1250_kernel", lambda _: gluon)
    pool = pool.clone()
    out = torch.empty_like(v)
    kw = {}
    if path == "plan":
        kw["plan"], cu = _plan(cu, slots)
    else:
        kw["state_slots"] = slots
    if gate:
        kw["output_gate"] = (z, weight, 1e-6)
    chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=pool, output_final_state=True,
        cu_seqlens=cu, use_qk_l2norm_in_kernel=True, o=out, **kw,
    )
    return out, pool


@pytest.mark.parametrize("path", ["slots", "plan"])
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize(
    "lens,H,Hg",
    [([16384], 32, 16), ([4000, 37, 1200, 64, 3], 32, 16), ([900, 2100, 30], 8, 4), ([48], 8, 4)],
)
def test_varlen(lens, H, Hg, path, gate, monkeypatch):
    x = _inputs(lens, H, Hg)
    ref_out, ref_pool = _run(False, *x, path, gate, monkeypatch)
    out, pool = _run(True, *x, path, gate, monkeypatch)
    torch.testing.assert_close(pool, ref_pool, rtol=0, atol=1e-6)
    tol = 2 ** -6 * ref_out.abs().amax().item()
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=0, atol=tol)


def test_fixed_length_batch(monkeypatch):
    gen = torch.Generator(device=DEV).manual_seed(3)
    B, T, Hg, H = 3, 1000, 4, 8
    q, k = (torch.randn(B, T, Hg, 128, device=DEV, dtype=torch.bfloat16, generator=gen) for _ in "qk")
    v = torch.randn(B, T, H, 128, device=DEV, dtype=torch.bfloat16, generator=gen)
    g = -torch.nn.functional.softplus(torch.randn(B, T, H, device=DEV, generator=gen)) / 2
    beta = torch.rand(B, T, H, device=DEV, generator=gen).to(torch.bfloat16)
    h0 = torch.randn(B, H, 128, 128, device=DEV, generator=gen) / 20
    res = []
    for gluon in (False, True):
        monkeypatch.setattr(chunk_delta_h, "_use_gfx1250_kernel", lambda _, g=gluon: g)
        monkeypatch.setattr(chunk_o, "_use_gfx1250_kernel", lambda _, g=gluon: g)
        res.append(chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=h0, output_final_state=True,
            use_qk_l2norm_in_kernel=True))
    torch.testing.assert_close(res[1][1], res[0][1], rtol=0, atol=1e-6)
    tol = 2 ** -6 * res[0][0].abs().amax().item()
    torch.testing.assert_close(res[1][0].float(), res[0][0].float(), rtol=0, atol=tol)


def test_fused_gate_matches_layer_norm(monkeypatch):
    """With the gate, the Gluon output is bitwise chunk_o then layer_norm_fwd."""
    q, k, v, z, g, beta, cu, pool, slots, weight = _inputs([3000, 700, 5], 32, 16)
    fused, _ = _run(True, q, k, v, z, g, beta, cu, pool, slots, weight, "slots", True, monkeypatch)
    raw, _ = _run(True, q, k, v, z, g, beta, cu, pool, slots, weight, "slots", False, monkeypatch)
    flat = raw[0].view(raw.shape[1], -1)
    layer_norm_fwd(flat, weight.repeat(32), None, 1e-6, z=z.flatten(1), out=flat, group_size=128,
                   norm_before_gate=True, is_rms_norm=True, rows_per_block=4)
    assert torch.equal(fused, raw)
