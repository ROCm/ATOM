# SPDX-License-Identifier: MIT
"""Kimi K3's paged KDA + state-checkpoint path.

Both cases here failed on the code as first written, and neither failure was
visible from an exception or a shape error — a missing normalization and two
stride assumptions each produce a well-formed, wrong answer.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paged KDA kernels are GPU-only"
)


def _i32(vals):
    return torch.tensor(vals, dtype=torch.int32, device="cuda")


def _kda_inputs(T=256, H=2, K=64, V=64, dev="cuda"):
    torch.manual_seed(0)

    def r(*shape, dt=torch.bfloat16):
        return torch.randn(*shape, dtype=dt, device=dev)

    return dict(
        q=r(1, T, H, K),
        k=r(1, T, H, K),
        v=r(1, T, H, V),
        g=r(1, T, H, K),
        beta=r(1, T, H, dt=torch.float32),
        A_log=r(H, K, dt=torch.float32),
        dt_bias=r(H, K, dt=torch.float32),
    )


def test_paged_kda_matches_fla():
    """The paged forward must agree with the op the model uses unpaged.

    ``chunk_kda_paged`` reuses fla's stages directly rather than going through
    ``ChunkKDAFunction.forward``, which is where the QK l2norm lives. Dropping
    it leaves q/k carrying raw magnitudes into the delta rule: output is still
    finite and plausibly scaled, but ~27% off.
    """
    from fla.ops.kda import chunk_kda

    from atom.model_ops.fla_ops.chunk_kda import chunk_kda_paged

    T, H, K, V = 256, 2, 64, 64
    x = _kda_inputs(T, H, K, V)
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    pool = torch.zeros(4, H, V, K, dtype=torch.float32, device="cuda")
    si = torch.tensor([1], dtype=torch.int32, device="cuda")

    o_paged, _, _ = chunk_kda_paged(
        **x,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        state_indices=si,
        dst_indices=si,
        h0_mask=torch.tensor([False], device="cuda"),
    )
    o_ref, s_ref = chunk_kda(
        **x,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        transpose_state_layout=True,
        cu_seqlens=cu,
        disable_recompute=True,
    )

    # bf16 chunked reductions differ in summation order between the two
    # assemblies of the same math; the un-normalized bug is two orders of
    # magnitude larger than this and cannot hide under it.
    assert (o_paged.float() - o_ref.float()).abs().max() < 1e-3
    assert (pool[1].float() - s_ref[0].float()).abs().max() < 1e-4


def test_checkpoint_replay_is_exact_with_noncontiguous_conv():
    """Split-and-replay through a checkpoint, on Kimi's real tensor layouts.

    Two layouts here are load-bearing and neither is contiguous:
    ``conv_state`` is a ``transpose(-1, -2)`` view of a ``[slots, state_len,
    D]`` allocation, and the conv input is a column slice of the fused
    in-projection, so its row stride is the fused width rather than ``D``.
    Deriving either from the shape writes a correctly-shaped checkpoint out of
    the wrong bytes.
    """
    from atom.model_ops.fla_ops.chunk_kda import chunk_kda_paged
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    T, H, K, V, P = 256, 2, 64, 64, 128
    x = _kda_inputs(T, H, K, V)
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    slots, state_len, D = 8, 3, 16
    pool = torch.zeros(slots, H, V, K, dtype=torch.float32, device="cuda")
    conv = torch.zeros(
        slots, state_len, D, dtype=torch.bfloat16, device="cuda"
    ).transpose(-1, -2)
    conv_in = torch.randn(T, D + 5, dtype=torch.bfloat16, device="cuda")[:, :D]
    assert conv.stride(2) != 1 and conv_in.stride(0) != D, "layouts under test"

    RUNTIME, CKPT = 1, 5
    o_full, _, h = chunk_kda_paged(
        **x,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        state_indices=_i32([RUNTIME]),
        dst_indices=_i32([RUNTIME]),
        h0_mask=torch.tensor([False], device="cuda"),
        return_intermediate_states=True,
    )
    final_full = pool[RUNTIME].clone()

    write_state_checkpoints(
        h,
        pool,
        conv_in,
        conv,
        _i32([0]),  # rows
        _i32([CKPT]),  # slots
        _i32([P]),  # offs
        _i32([0]),  # is_end: interior
        _i32([RUNTIME]),
        _i32([0, T // 64]),  # chunk_offsets
        cu,
        64,
    )
    assert torch.equal(conv[CKPT], conv_in[P - state_len : P].transpose(0, 1))

    # Resume the suffix from the checkpoint; P is on the kernel's chunk grid,
    # so this is exact, not merely close.
    pool[RUNTIME].zero_()
    o_suffix, _, _ = chunk_kda_paged(
        q=x["q"][:, P:],
        k=x["k"][:, P:],
        v=x["v"][:, P:],
        g=x["g"][:, P:],
        beta=x["beta"][:, P:],
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, T - P], dtype=torch.int32, device="cuda"),
        state_indices=_i32([CKPT]),
        dst_indices=_i32([RUNTIME]),
        h0_mask=torch.tensor([True], device="cuda"),
    )
    assert torch.equal(o_suffix, o_full[:, P:])
    assert torch.equal(pool[RUNTIME], final_full)
