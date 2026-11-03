# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Interior GDN state checkpoints stored by the chunk state kernel
(`chunk_gated_delta_rule(..., checkpoint_slots=...)`) are the exact fp32 state:
the final state of the same forward cut at the checkpoint, bit for bit, on the
gfx1250 and Triton kernels, by slot and under a bucket-shaped plan. The per-chunk
`h` they used to be copied from holds that state rounded to bf16. And the copy
kernel, told the interior states are already stored, copies only the
end-of-sequence states and the conv windows, skipping the padding rows of a
fixed-capacity table."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
    pytest.skip("gfx1250 kernels", allow_module_level=True)

from atom.model_ops.fla_ops import chunk_delta_h, chunk_o
from atom.model_ops.fla_ops.chunk import chunk_gated_delta_rule
from atom.model_ops.fla_ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints
from atom.model_ops.fla_ops.varlen_plan import VarlenPlan

# inputs and plan as in tests/test_gdn_chunk_gfx1250.py
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


LENS, H, HG = [900, 2100, 30], 8, 4
# (row, token offset within the row): interior positions on the 64-token grid
TARGETS = [(0, 64), (0, 640), (1, 128), (1, 2048)]


def _chunk_slots(lens, targets, dests):
    counts = [-(-n // 64) for n in lens]
    offsets = [sum(counts[:i]) for i in range(len(lens))]
    table = torch.full((sum(counts),), -1, dtype=torch.int32)
    for (row, off), dest in zip(targets, dests):
        table[offsets[row] + off // 64] = dest
    return table.to(DEV)


def _forward(gluon, x, path, monkeypatch, cu=None, rows=None, ckpt=None, pool=None):
    q, k, v, _, g, beta, cu_all, pool0, slots, _ = x
    monkeypatch.setattr(chunk_delta_h, "_use_gfx1250_kernel", lambda _: gluon)
    monkeypatch.setattr(chunk_o, "_use_gfx1250_kernel", lambda _: gluon)
    pool = pool0.clone() if pool is None else pool
    cu = cu_all if cu is None else cu
    if rows is not None:
        slots = tuple(s[rows] for s in slots)
    kw = {}
    if path == "plan":
        kw["plan"], cu = _plan(cu, slots)
    else:
        kw["state_slots"] = slots
    T = int(cu[-1])
    chunk_gated_delta_rule(
        q[:, :T],
        k[:, :T],
        v[:, :T],
        g[:, :T],
        beta[:, :T],
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        o=torch.empty_like(v[:, :T]),
        checkpoint_slots=ckpt,
        **kw,
    )
    return pool


@pytest.mark.parametrize("gluon", [True, False])
@pytest.mark.parametrize("path", ["slots", "plan"])
def test_interior_checkpoint_is_the_exact_state(gluon, path, monkeypatch):
    x = _inputs(LENS, H, HG)
    q, k, v, z, g, beta, cu, pool0, slots, w = x
    # destinations: pool slots no sequence reads or writes
    used = set(slots[0].tolist()) | set(slots[1].tolist())
    dests = [s for s in range(pool0.shape[0]) if s not in used][: len(TARGETS)]
    full = pool0.clone()
    _forward(
        gluon,
        x,
        path,
        monkeypatch,
        pool=full,
        ckpt=(full, _chunk_slots(LENS, TARGETS, dests)),
    )
    for (row, off), dest in zip(TARGETS, dests):
        # the same forward over this row's first `off` tokens alone: its final
        # state, written in place to the row's write slot
        start = int(cu[row])
        xs = (
            q[:, start : start + off],
            k[:, start : start + off],
            v[:, start : start + off],
            z,
            g[:, start : start + off],
            beta[:, start : start + off],
            cu,
            pool0,
            slots,
            w,
        )
        cut = _forward(
            gluon,
            xs,
            path,
            monkeypatch,
            cu=torch.tensor([0, off], device=DEV),
            rows=[row],
        )
        want = cut[slots[1][row]]
        torch.testing.assert_close(full[dest], want, rtol=0, atol=0)
    # the checkpoint store leaves every other slot exactly as the plain forward
    plain = _forward(gluon, x, path, monkeypatch)
    keep = [s for s in range(pool0.shape[0]) if s not in dests]
    torch.testing.assert_close(full[keep], plain[keep], rtol=0, atol=0)


def test_copy_kernel_skips_stored_states_and_padding():
    gen = torch.Generator(device=DEV).manual_seed(1)
    Hh, K, V, D, W, state_len = 2, 128, 128, 96, 7, 3
    lens = [200, 130]
    T = sum(lens)
    cu = torch.tensor([0, 200, 330], device=DEV, dtype=torch.int32)
    x = torch.randn(T, 2 * D, device=DEV, generator=gen).to(torch.bfloat16)[:, :D]
    ssm = torch.randn(12, Hh, K, V, device=DEV, generator=gen)
    conv = torch.randn(12, W, D, device=DEV, generator=gen).transpose(-1, -2)
    before = ssm.clone()
    i32 = torch.int32
    # interior target (row 0, off 128 -> slot 5), end target (row 1, off 130,
    # runtime slot 2 -> slot 6), then two padding rows
    rows = torch.tensor([0, 1, 0, 0], device=DEV, dtype=i32)
    dst = torch.tensor([5, 6, -1, -1], device=DEV, dtype=i32)
    offs = torch.tensor([128, 130, 0, 0], device=DEV, dtype=i32)
    is_end = torch.tensor([0, 1, 0, 0], device=DEV, dtype=i32)
    runtime = torch.tensor([0, 2, 0, 0], device=DEV, dtype=i32)
    chunk_offsets = torch.tensor([0, 4, 7], device=DEV, dtype=i32)
    write_state_checkpoints(
        ssm,
        ssm,
        x,
        conv,
        rows,
        dst,
        offs,
        is_end,
        runtime,
        chunk_offsets,
        cu,
        64,
        state_len,
        ssm_from_h=False,
    )
    # the interior state is the state kernel's to store: untouched here
    torch.testing.assert_close(ssm[5], before[5], rtol=0, atol=0)
    torch.testing.assert_close(ssm[6], before[2], rtol=0, atol=0)
    for slot, end in ((5, 0 + 128), (6, 200 + 130)):
        want = x[end - state_len : end].float().t()
        torch.testing.assert_close(
            conv[slot, :, :state_len].float(), want, rtol=0, atol=0
        )
    # padding rows wrote nothing
    others = [s for s in range(12) if s not in (5, 6)]
    torch.testing.assert_close(ssm[others], before[others], rtol=0, atol=0)
