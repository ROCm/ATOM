# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 Gluon chunk_h / chunk_o kernels against the Triton kernels they
replace there: varlen with a state pool by slot (eager) or a bucket-shaped plan,
and a fixed-length batch; the output with and without the fused gated RMSNorm.
chunk_h stores h K-major ([.., V, K] storage, returned as a [.., K, V] view); its
value tile widths and ring depths, and the fused KKT kernel's decay rows in place of the gates
(bitwise the same). chunk_o reads K-major h (the Triton kernel too) and its fp8
output is the bf16 output followed by the unfused quant, byte for byte."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
    pytest.skip("gfx1250 kernels", allow_module_level=True)

from atom.model_ops.fla_ops import (
    chunk_delta_h,
    chunk_delta_h_gfx1250,
    chunk_o,
)
from atom.model_ops.fla_ops import (
    fused_kkt_inverse_gfx1250 as fused_kkt,
)
from atom.model_ops.fla_ops.chunk import (
    chunk_gated_delta_rule,
    pop_last_intermediate_states,
)
from atom.model_ops.fla_ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from atom.model_ops.fla_ops.cumsum import chunk_local_cumsum
from atom.model_ops.fla_ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from atom.model_ops.fla_ops.layernorm_guard import layer_norm_fwd
from atom.model_ops.fla_ops.varlen_plan import VarlenPlan

DEV = "cuda"


def _inputs(lens, H, Hg, seed=0):
    T = sum(lens)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    row = 2 * Hg * 128 + 2 * H * 128
    proj = torch.randn(T, row, device=DEV, dtype=torch.bfloat16, generator=gen)
    q = proj[:, : Hg * 128].reshape(1, T, Hg, 128).contiguous()
    k = proj[:, Hg * 128 : 2 * Hg * 128].reshape(1, T, Hg, 128).contiguous()
    v = (
        proj[:, 2 * Hg * 128 : 2 * Hg * 128 + H * 128]
        .reshape(1, T, H, 128)
        .contiguous()
    )
    z = proj[:, 2 * Hg * 128 + H * 128 :].view(T, H, 128)
    g = (
        -torch.nn.functional.softplus(torch.randn(1, T, H, device=DEV, generator=gen))
        / 2
    )
    beta = torch.rand(1, T, H, device=DEV, generator=gen).to(torch.bfloat16)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=DEV)
    n = len(lens)
    pool = torch.randn(4 * n + 4, H, 128, 128, device=DEV, generator=gen) / 20
    perm = torch.randperm(4 * n + 4, generator=torch.Generator().manual_seed(seed))
    slots = (
        perm[:n].to(DEV),
        perm[n : 2 * n].to(DEV),
        torch.arange(n, device=DEV) % 3 != 2,
    )
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
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        o=out,
        keep_intermediate_states=True,
        **kw,
    )
    return out, pool, pop_last_intermediate_states()


@pytest.mark.parametrize("path", ["slots", "plan"])
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize(
    "lens,H,Hg",
    [
        ([16384], 32, 16),
        ([4000, 37, 1200, 64, 3], 32, 16),
        ([900, 2100, 30], 8, 4),
        ([48], 8, 4),
        # many short sequences
        ([1, 2, 3, 5, 17, 63, 64, 65, 127, 128, 129] * 6, 32, 16),
    ],
)
def test_varlen(lens, H, Hg, path, gate, monkeypatch):
    x = _inputs(lens, H, Hg)
    ref_out, ref_pool, ref_h = _run(False, *x, path, gate, monkeypatch)
    out, pool, h = _run(True, *x, path, gate, monkeypatch)
    torch.testing.assert_close(pool, ref_pool, rtol=0, atol=1e-6)
    _check_h(h, ref_h, x[6])
    tol = 2**-6 * ref_out.abs().amax().item()
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=0, atol=tol)


def _check_h(h, ref_h, cu):
    """The K-major chunk states read as [.., K, V]: Triton's row-major h within a bf16 step."""
    assert h.stride(-2) == 1 and h.shape == ref_h.shape
    nt = prepare_chunk_indices(cu, 64).shape[0]
    h, ref_h = h[:, :nt].float(), ref_h[:, :nt].float()
    torch.testing.assert_close(h, ref_h, rtol=2**-7, atol=1e-6)
    assert (h != ref_h).double().mean() < 1e-3


def test_fixed_length_batch(monkeypatch):
    gen = torch.Generator(device=DEV).manual_seed(3)
    B, T, Hg, H = 3, 1000, 4, 8
    q, k = (
        torch.randn(B, T, Hg, 128, device=DEV, dtype=torch.bfloat16, generator=gen)
        for _ in "qk"
    )
    v = torch.randn(B, T, H, 128, device=DEV, dtype=torch.bfloat16, generator=gen)
    g = (
        -torch.nn.functional.softplus(torch.randn(B, T, H, device=DEV, generator=gen))
        / 2
    )
    beta = torch.rand(B, T, H, device=DEV, generator=gen).to(torch.bfloat16)
    h0 = torch.randn(B, H, 128, 128, device=DEV, generator=gen) / 20
    res = []
    for gluon in (False, True):
        monkeypatch.setattr(chunk_delta_h, "_use_gfx1250_kernel", lambda _, g=gluon: g)
        monkeypatch.setattr(chunk_o, "_use_gfx1250_kernel", lambda _, g=gluon: g)
        res.append(
            chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=h0,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        )
    torch.testing.assert_close(res[1][1], res[0][1], rtol=0, atol=1e-6)
    tol = 2**-6 * res[0][0].abs().amax().item()
    torch.testing.assert_close(res[1][0].float(), res[0][0].float(), rtol=0, atol=tol)


def test_fused_gate_matches_layer_norm(monkeypatch):
    """With the gate, the Gluon output is bitwise chunk_o then layer_norm_fwd."""
    monkeypatch.setenv("ATOM_GDN_CHUNK_O_FAST_NORM", "0")
    q, k, v, z, g, beta, cu, pool, slots, weight = _inputs([3000, 700, 5], 32, 16)
    fused, _, _ = _run(
        True, q, k, v, z, g, beta, cu, pool, slots, weight, "slots", True, monkeypatch
    )
    raw, _, _ = _run(
        True, q, k, v, z, g, beta, cu, pool, slots, weight, "slots", False, monkeypatch
    )
    flat = raw[0].view(raw.shape[1], -1)
    layer_norm_fwd(
        flat,
        weight.repeat(32),
        None,
        1e-6,
        z=z.flatten(1),
        out=flat,
        group_size=128,
        norm_before_gate=True,
        is_rms_norm=True,
        rows_per_block=4,
    )
    assert torch.equal(fused, raw)


@pytest.mark.parametrize(
    "lens,H,Hg", [([3000, 700, 5], 32, 16), ([900, 2100, 30], 8, 4)]
)
def test_fast_norm_close_to_exact(lens, H, Hg, monkeypatch):
    """ATOM_GDN_CHUNK_O_FAST_NORM: within a bf16 rounding step of the exact norm."""
    x = _inputs(lens, H, Hg)
    monkeypatch.setenv("ATOM_GDN_CHUNK_O_FAST_NORM", "0")
    ref, ref_pool, _ = _run(True, *x, "slots", True, monkeypatch)
    monkeypatch.setenv("ATOM_GDN_CHUNK_O_FAST_NORM", "1")
    out, pool, _ = _run(True, *x, "slots", True, monkeypatch)
    assert torch.equal(pool, ref_pool)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2**-7, atol=1e-5)


def _force_tile(block_n, buffers, monkeypatch):
    monkeypatch.setattr(
        chunk_delta_h_gfx1250,
        "launch_config",
        lambda *_: (block_n, buffers, chunk_delta_h_gfx1250._NUM_WARPS),
    )


# (value tile width, buffers) launch_config picks: long walks, short walks, one-chunk walks
TILES = [(32, 4), (32, 3), (32, 2), (64, 4), (64, 3), (64, 2), (128, 2)]


@pytest.mark.parametrize("path", ["slots", "plan"])
@pytest.mark.parametrize(
    "block_n,buffers,pool_dtype",
    [(*t, torch.float32) for t in TILES]
    + [(32, 4, torch.bfloat16), (64, 3, torch.bfloat16), (128, 2, torch.bfloat16)],
)
def test_value_tile_widths(block_n, buffers, pool_dtype, path, monkeypatch):
    """Every value tile width and ring depth launch_config picks (early operand reads, fp32 and
    bf16 state pools) against Triton."""
    x = list(_inputs([2100, 300, 37, 5, 129], 32, 16))
    x[7] = x[7].to(pool_dtype)
    ref_out, ref_pool, ref_h = _run(False, *x, path, False, monkeypatch)
    _force_tile(block_n, buffers, monkeypatch)
    out, pool, h = _run(True, *x, path, False, monkeypatch)
    if pool_dtype == torch.float32:
        torch.testing.assert_close(pool, ref_pool, rtol=0, atol=1e-6)
    else:
        torch.testing.assert_close(pool.float(), ref_pool.float(), rtol=2**-7, atol=1e-6)
    _check_h(h, ref_h, x[6])
    tol = 2**-6 * ref_out.abs().amax().item()
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=0, atol=tol)


@pytest.mark.parametrize("path", ["slots", "plan"])
@pytest.mark.parametrize("paired", ["0", "1"])
@pytest.mark.parametrize("block_n,buffers", TILES)
def test_decay_rows_match_gates(block_n, buffers, paired, path, monkeypatch):
    """chunk_h with the fused KKT kernel's decay rows is bitwise chunk_h with its gates: h, v_new,
    final states by slot and fp32 checkpoints, partial last chunks and many short sequences.
    """
    monkeypatch.setenv("ATOM_GDN_CHUNK_H_PAIRED_LOADS", paired)
    _force_tile(block_n, buffers, monkeypatch)
    lens = [1, 2, 3, 17, 63, 64, 65, 127, 128, 129, 1000, 3000]
    _, k, v, _, _, _, cu, pool, slots, _ = _inputs(lens, 32, 16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    gen = torch.Generator(device=DEV).manual_seed(5)
    rows = torch.randn(k.shape[1], 64, device=DEV, generator=gen).to(torch.bfloat16)
    gating = (
        rows[:, :32],
        rows[:, 32:],
        torch.rand(32, device=DEV, generator=gen),
        torch.randn(32, device=DEV, generator=gen),
    )
    kw, nt = {}, prepare_chunk_indices(cu, 64).shape[0]
    if path == "plan":
        kw["plan"], cu = _plan(cu, slots)
        rows_ckpt = kw["plan"].nt64_max
    else:
        kw["state_slots"] = slots
        rows_ckpt = nt
    plan = kw.get("plan")
    g, decay, w, u, _, _ = fused_kkt.fused_kkt_inverse_gfx1250(
        k, v, cu, plan, gating, True
    )
    # checkpoints of every third chunk into slots no sequence reads or writes
    used = set(slots[0].tolist()) | set(slots[1].tolist())
    free = [s for s in range(pool.shape[0]) if s not in used]
    table = torch.full((rows_ckpt,), -1, dtype=torch.int32, device=DEV)
    for i, chunk in enumerate(range(1, nt, 3)[: len(free)]):
        table[chunk] = free[i]
    res = []
    for d in (None, decay):
        p = pool.clone()
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            decay=d,
            initial_state=p,
            output_final_state=True,
            cu_seqlens=cu,
            checkpoint_slots=(p, table),
            **kw,
        )
        res.append((h[:, :nt], v_new, p))
    for a, b in zip(*res):
        assert torch.equal(a, b)


def test_fused_gating_path_decay_flag(monkeypatch):
    """The layer's path (gates computed in the fused KKT kernel): ATOM_GDN_CHUNK_H_DECAY hands its
    decay rows to chunk_h, with bitwise the same output, chunk states and final states as without.
    """
    q, k, v, z, _, _, cu, pool, slots, weight = _inputs([3000, 700, 5, 64, 130], 32, 16)
    gen = torch.Generator(device=DEV).manual_seed(4)
    rows = torch.randn(k.shape[1], 64, device=DEV, generator=gen).to(torch.bfloat16)
    gating = (
        rows[:, :32],
        rows[:, 32:],
        torch.rand(32, device=DEV, generator=gen),
        torch.randn(32, device=DEV, generator=gen),
    )
    calls = []
    kernel = fused_kkt.fused_kkt_inverse_gfx1250

    def spy(*args):
        calls.append(args[-1])
        return kernel(*args)

    monkeypatch.setattr(fused_kkt, "fused_kkt_inverse_gfx1250", spy)
    monkeypatch.setenv("ATOM_GDN_GLUON_INVERSE", "1")
    res = []
    for flag in ("1", "0"):
        monkeypatch.setenv("ATOM_GDN_CHUNK_H_DECAY", flag)
        p, out = pool.clone(), torch.empty_like(v)
        chunk_gated_delta_rule(
            q,
            k,
            v,
            None,
            None,
            initial_state=p,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            o=out,
            keep_intermediate_states=True,
            state_slots=slots,
            output_gate=(z, weight, 1e-6),
            gating=gating,
        )
        res.append((out, p, pop_last_intermediate_states()))
    assert calls == [True, False]
    for a, b in zip(*res):
        assert torch.equal(a, b)


def test_triton_output_reads_k_major_states(monkeypatch):
    """The Triton chunk_o (the fallback beside the Gluon chunk_h) reads K-major h as row-major h."""
    lens, H, Hg = [300, 64, 5, 1000], 8, 4
    q, k, v, _, g, _, cu, _, _, _ = _inputs(lens, H, Hg)
    gen = torch.Generator(device=DEV).manual_seed(1)
    nt = prepare_chunk_indices(cu, 64).shape[0]
    h = (torch.randn(1, nt, H, 128, 128, device=DEV, generator=gen) / 20).to(
        torch.bfloat16
    )
    h_vk = h.transpose(-1, -2).contiguous().transpose(-1, -2)
    monkeypatch.setattr(chunk_o, "_use_gfx1250_kernel", lambda _: False)
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu)
    out = [chunk_o.chunk_fwd_o(q, k, v, x, g, cu_seqlens=cu) for x in (h, h_vk)]
    assert torch.equal(out[0], out[1])


@pytest.mark.parametrize("fast_norm", ["0", "1"])
def test_fp8_output_is_bf16_output_then_quant(fast_norm, monkeypatch):
    """OUT_FP8 (mxfp8_output): fp8 values and e8m0 scales are byte for byte the bf16 output's
    unfused quant; the padding rows past the tokens are left alone."""
    from atom.model_ops.mxfp8_act_quant import alloc_mxfp8, reference

    monkeypatch.setenv("ATOM_GDN_CHUNK_O_FAST_NORM", fast_norm)
    x = _inputs([3000, 700, 5, 64], 32, 16)
    out, pool, _ = _run(True, *x, "slots", True, monkeypatch)
    T, HV = out.shape[1], 32 * 128
    rows = T + 9
    o_fp8, o_scale = alloc_mxfp8(rows, HV, DEV)
    o_fp8.view(torch.uint8).fill_(0x5A)
    o_scale.fill_(0x5A)
    with chunk_o.mxfp8_output(o_fp8, o_scale) as used:
        _, pool8, _ = _run(True, *x, "slots", True, monkeypatch)
    assert used
    assert torch.equal(pool8, pool)
    ref_q, ref_s = reference(out.view(T, HV))
    assert torch.equal(o_fp8[:T].view(torch.uint8), ref_q.view(torch.uint8))
    # both store the byte of (row m, head g) at g * rows + m
    scales = o_scale.view(HV // 128, rows)
    assert torch.equal(
        scales[:, :T], ref_s.view(torch.uint8).flatten().view(HV // 128, T)
    )
    assert (o_fp8[T:].view(torch.uint8) == 0x5A).all()
    assert (scales[:, T:] == 0x5A).all()
