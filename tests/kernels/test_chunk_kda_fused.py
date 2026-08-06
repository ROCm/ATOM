# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parity tests for ATOM's vendored, fused KDA chunk forward.

Reference is stock ``fla.ops.kda.chunk_kda``. Both the output and the final
state are asserted: a gather/scatter bug can leave the output correct while
corrupting the state, which would only surface on a later token.

The expectation is bitwise equality. The fused and reference paths run the same
arithmetic in the same order -- the changes are pointer arithmetic and buffer
ownership, not math. A discrepancy is a bug to explain, not a tolerance to
loosen.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KDA kernels require a GPU"
)

PAD_SLOT_ID = -1


def _make_inputs(seq_lens, hv=4, k_dim=128, v_dim=128, dtype=torch.bfloat16, seed=0):
    """Build a flattened varlen KDA input set on the GPU.

    Returns a dict of the arguments both paths take, plus ``cu_seqlens``.
    """
    torch.manual_seed(seed)
    dev = "cuda"
    total = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.long, device=dev
    )
    return {
        "q": torch.randn(1, total, hv, k_dim, dtype=dtype, device=dev),
        "k": torch.randn(1, total, hv, k_dim, dtype=dtype, device=dev),
        "v": torch.randn(1, total, hv, v_dim, dtype=dtype, device=dev),
        "g": torch.randn(1, total, hv, k_dim, dtype=dtype, device=dev),
        "beta": torch.randn(1, total, hv, dtype=torch.float32, device=dev),
        "A_log": torch.randn(hv, dtype=torch.float32, device=dev),
        "dt_bias": torch.randn(hv * k_dim, dtype=torch.float32, device=dev),
        "cu_seqlens": cu_seqlens,
    }


def _flags():
    """The exact flag set Kimi-K3's prefill path uses."""
    return {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": True,
        "lower_bound": -5.0,
        "state_v_first": True,
        "disable_recompute": True,
        "output_final_state": True,
    }


def _reference(inp, initial_state):
    """Stock fla, dense initial state, allocating its own output."""
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    return fla_chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=initial_state,
        cu_seqlens=inp["cu_seqlens"],
        **_flags(),
    )


def _fused(inp, initial_state, **extra):
    from atom.model_ops.fla_ops.kda import chunk_kda

    return chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=initial_state,
        cu_seqlens=inp["cu_seqlens"],
        **_flags(),
        **extra,
    )


def _dense_state(n, hv=4, k_dim=128, v_dim=128, seed=1):
    """[N, HV, V, K] fp32 -- the state_v_first layout ssm_state uses."""
    torch.manual_seed(seed)
    return torch.randn(n, hv, v_dim, k_dim, dtype=torch.float32, device="cuda")


def test_vendored_path_matches_fla_with_fusion_off():
    """Case 1: the vendoring itself, independent of any fusion."""
    inp = _make_inputs([64, 128])
    h0 = _dense_state(2)
    ref_o, ref_ht = _reference(inp, h0.clone())
    got_o, got_ht = _fused(inp, h0.clone())
    assert torch.equal(got_o, ref_o)
    assert torch.equal(got_ht, ref_ht)


def test_indexed_gather_and_scatter():
    """Case 2: non-monotonic, non-contiguous slots -- a dense-indexing bug
    cannot pass by coincidence."""
    inp = _make_inputs([64, 96, 128])
    slots = [5, 1, 3]
    cache = _dense_state(8)
    packed = torch.stack([cache[s] for s in slots]).contiguous()

    ref_o, ref_ht = _reference(inp, packed.clone())

    fused_cache = cache.clone()
    idx = torch.tensor(slots, dtype=torch.int32, device="cuda")
    got_o, got_ht = _fused(inp, fused_cache, h0_indices=idx, inplace_final_state=True)
    assert got_ht.data_ptr() == fused_cache.data_ptr(), "inplace must alias"
    assert torch.equal(got_o, ref_o)
    for i, s in enumerate(slots):
        assert torch.equal(fused_cache[s], ref_ht[i]), f"slot {s}"
    # Slots nobody claimed must be untouched.
    for s in set(range(8)) - set(slots):
        assert torch.equal(fused_cache[s], cache[s]), f"slot {s} was clobbered"


def test_mixed_has_initial_state():
    """Case 3: some sequences fresh, some resuming."""
    inp = _make_inputs([64, 128, 64])
    slots = [2, 0, 6]
    cache = _dense_state(8)
    has_init = torch.tensor([True, False, True], device="cuda")

    # Reference: gather, then zero the fresh ones -- what the old path did.
    packed = torch.stack([cache[s] for s in slots]).contiguous()
    packed[~has_init] = 0
    ref_o, ref_ht = _reference(inp, packed)

    fused_cache = cache.clone()
    got_o, _ = _fused(
        inp,
        fused_cache,
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        has_initial_state=has_init,
        inplace_final_state=True,
    )
    assert torch.equal(got_o, ref_o)
    for i, s in enumerate(slots):
        assert torch.equal(fused_cache[s], ref_ht[i]), f"slot {s}"


def test_pad_slot_id_leaves_cache_untouched():
    """Case 4: a -1 slot reads nothing and writes nothing.

    Two sequences: seq 0 is PAD (slot -1), seq 1 uses slot 4.

    Assertability by sequence:
    - Seq 0 (PAD): the kernel treats it as starting from zeros and its output
      rows are computed, but there is no meaningful expected value to compare
      against because the caller contracts that PAD-slot output is unused.
      Output for seq 0 is intentionally left unasserted.
    - Seq 1 (slot 4): initial state is cache[4], output and final state both
      have well-defined expected values. Both are asserted via _reference.

    The "leaves cache untouched" assertion covers all slots except 4.
    """
    inp = _make_inputs([64, 64])
    cache = _dense_state(8)
    before = cache.clone()
    slots = [PAD_SLOT_ID, 4]

    # Reference: seq 1 only, using cache[4] as its initial state.
    # The reference receives a 2-seq batch (slot -1 starts from zeros, slot 4
    # from cache[4]), matching the kernel's behavior.
    packed = torch.zeros(2, *cache.shape[1:], dtype=cache.dtype, device=cache.device)
    packed[1] = cache[4]
    ref_o, ref_ht = _reference(inp, packed)

    fused_cache = cache.clone()
    got_o, _ = _fused(
        inp,
        fused_cache,
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        inplace_final_state=True,
    )

    # Output for seq 1 (second 64 rows) must match reference.
    # Output for seq 0 (PAD) is intentionally unasserted -- caller ignores it.
    assert torch.equal(
        got_o[:, 64:, :, :], ref_o[:, 64:, :, :]
    ), "seq 1 output mismatch"

    # Final state for slot 4 must match reference seq 1 final state.
    assert torch.equal(fused_cache[4], ref_ht[1]), "slot 4 final state mismatch"

    # All slots other than 4 must be completely untouched (slot -1 has no mapping).
    for s in set(range(8)) - {4}:
        assert torch.equal(fused_cache[s], before[s]), f"slot {s} was clobbered"


def test_varlen_with_ragged_tail():
    """Case 5: a sequence length that is not a multiple of chunk_size (64)."""
    inp = _make_inputs([64, 100, 37])
    slots = [0, 1, 2]
    cache = _dense_state(4)
    packed = torch.stack([cache[s] for s in slots]).contiguous()

    ref_o, ref_ht = _reference(inp, packed.clone())
    fused_cache = cache.clone()
    got_o, _ = _fused(
        inp,
        fused_cache,
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        inplace_final_state=True,
    )
    assert torch.equal(got_o, ref_o)
    for i, s in enumerate(slots):
        assert torch.equal(fused_cache[s], ref_ht[i]), f"slot {s}"


def test_out_buffer_is_fully_written():
    """Case 6: prove the removed zero-fill is not load-bearing.

    Contract (from chunk_o_gk.py:16-22): in the varlen case the grid tiles
    only [cu_seqlens[0], cu_seqlens[-1]), so rows at t >= cu_seqlens[-1] get
    no program and keep their prior contents.

    The property has two directions:
      1. No sentinel survives in [:total] -- the kernel wrote every row it owns.
      2. The sentinel IS intact in [total:T_buf] -- no out-of-bounds write.

    The public chunk_kda API enforces o.shape == (B, T, HV, V) with T =
    sum(seq_lens), so it is not possible to pass a larger o through it.  The
    tail contract lives at the o-kernel level.  This test calls
    chunk_gla_fwd_kernel_o directly with T_buf > total so both directions can
    be exercised.  The h tensor is a zero-filled stub (the kernel reads it but
    the test goal is buffer coverage, not arithmetic correctness).

    The first-direction assertion (no NaN in [:total]) also covers the
    "o= removal is not load-bearing" goal: if the kernel failed to write any
    row in its owned region the NaN sentinel would survive.
    """
    import triton
    from fla.ops.utils import prepare_chunk_indices

    from atom.model_ops.fla_ops.kda.chunk_o_gk import chunk_gla_fwd_kernel_o

    seq_lens = [64, 96]
    total = sum(seq_lens)  # 160 == cu_seqlens[-1]
    T_buf = 192  # 32-row tail beyond cu_seqlens[-1]; T_buf must be chunk-aligned
    BT = 64
    hv = 4
    k_dim = 128
    v_dim = 128
    H = hv
    dtype = torch.bfloat16
    dev = "cuda"

    cu_seqlens = torch.tensor([0, seq_lens[0], total], dtype=torch.long, device=dev)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices)

    # Inputs dimensioned to T_buf so pointer arithmetic is valid for all tiles.
    q = torch.randn(1, T_buf, hv, k_dim, dtype=dtype, device=dev)
    v = torch.randn(1, T_buf, hv, v_dim, dtype=dtype, device=dev)
    g = torch.randn(1, T_buf, hv, k_dim, dtype=dtype, device=dev)
    A = torch.randn(1, T_buf, hv, BT, dtype=dtype, device=dev)
    # h is [B, NT, HV, V, K] (state_v_first=True layout used by Kimi-K3).
    h = torch.zeros(1, NT, hv, v_dim, k_dim, dtype=dtype, device=dev)

    sentinel = torch.full((1, T_buf, hv, v_dim), float("nan"), dtype=dtype, device=dev)

    def grid(meta):
        return (triton.cdiv(v_dim, meta["BV"]), NT, 1 * hv)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=sentinel,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=k_dim**-0.5,
        T=T_buf,
        H=H,
        HV=hv,
        K=k_dim,
        V=v_dim,
        BT=BT,
        STATE_V_FIRST=True,
    )

    # Direction 1: every row the kernel owns must be overwritten.
    assert not torch.isnan(sentinel[:, :total, :, :]).any(), (
        "kernel left NaN sentinel in its owned region [:total] -- "
        "the removed zero-fill WAS load-bearing"
    )
    # Direction 2: tail rows must be untouched (no out-of-bounds write).
    assert torch.isnan(
        sentinel[:, total:, :, :]
    ).all(), "kernel wrote into the tail region [total:T_buf] -- out-of-bounds write"


def test_kimi_k3_shape():
    """Case 7: the shape actually served -- K=V=128, bf16 in, fp32 state."""
    inp = _make_inputs([512, 1024], hv=4, k_dim=128, v_dim=128)
    cache = _dense_state(16)
    slots = [11, 2]
    packed = torch.stack([cache[s] for s in slots]).contiguous()

    ref_o, ref_ht = _reference(inp, packed.clone())
    fused_cache = cache.clone()
    out = torch.empty_like(inp["v"])
    got_o, _ = _fused(
        inp,
        fused_cache,
        h0_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
        inplace_final_state=True,
        o=out,
    )
    assert torch.equal(got_o, ref_o)
    for i, s in enumerate(slots):
        assert torch.equal(fused_cache[s], ref_ht[i]), f"slot {s}"


def test_2d_indices_rejected():
    """Spec-decode indices must fail loudly rather than mis-index."""
    from atom.model_ops.fla_ops.kda import chunk_kda

    inp = _make_inputs([64, 64])
    cache = _dense_state(4)
    with pytest.raises(ValueError, match="1D"):
        chunk_kda(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            g=inp["g"],
            beta=inp["beta"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            initial_state=cache,
            cu_seqlens=inp["cu_seqlens"],
            h0_indices=torch.zeros(2, 2, dtype=torch.int32, device="cuda"),
            **_flags(),
        )


def test_non_contiguous_out_rejected():
    from atom.model_ops.fla_ops.kda import chunk_kda

    inp = _make_inputs([64, 64])
    cache = _dense_state(4)
    bad = torch.empty(1, 128, 4, 256, dtype=inp["v"].dtype, device="cuda")[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        chunk_kda(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            g=inp["g"],
            beta=inp["beta"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            initial_state=cache,
            cu_seqlens=inp["cu_seqlens"],
            h0_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            inplace_final_state=True,
            o=bad,
            **_flags(),
        )


def test_short_has_initial_state_rejected():
    """A short has_initial_state would be read out of bounds by the kernel."""
    from atom.model_ops.fla_ops.kda import chunk_kda

    inp = _make_inputs([64, 64])
    cache = _dense_state(4)
    with pytest.raises(ValueError, match="has_initial_state has 1 entries"):
        chunk_kda(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            g=inp["g"],
            beta=inp["beta"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            initial_state=cache,
            cu_seqlens=inp["cu_seqlens"],
            h0_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            has_initial_state=torch.ones(1, dtype=torch.bool, device="cuda"),
            **_flags(),
        )
