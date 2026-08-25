# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the in-kernel recurrent-state checkpoint scatter.

``chunk_gated_delta_rule_fwd_h`` optionally scatters the running fp32 state
accumulator into a checkpoint pool at every ``ckpt_every`` tokens, so that
prefix caching can resume a linear-attention sequence from a mid-prompt
boundary. This is the mechanism that decouples *how many* checkpoints a
prefill produces from *how many forward passes* it takes: without it, one
forward leaves exactly one state behind, so a wide prefill chunk would
checkpoint only at its own end -- past any prefix a peer request could share.

Both GDN and KDA route their state recurrence through this one kernel, so
these tests cover the checkpoint path for both.

Coverage:
    * Enabling checkpoints does not perturb ``h`` or ``final_state``.
    * Each checkpoint equals the state the scan held at that token boundary.
    * A ``-1`` slot (pool exhausted) is skipped without corrupting slot 0
      or writing outside the reserved slots.
    * ``ckpt_base`` keeps boundaries at absolute token positions across a
      chunked prefill, so splitting a prompt does not move its checkpoints.
    * Varlen batches place each sequence's boundaries independently.
"""

from __future__ import annotations

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()


# See tests/test_chunk_gated_delta_rule_fused.py: conftest stubs atom.config
# for the GPU-free scheduler tests, but GPU-kernel tests need the real thing.
def _restore_real_atom_modules():
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "atom" or mod_name.startswith("atom."):
            del sys.modules[mod_name]


_restore_real_atom_modules()

from atom.model_ops.fla_ops.chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h,
)

pytestmark = pytest.mark.skipif(not _HAS_CUDA, reason="No GPU available")

DEV = "cuda"
BT = 64  # scan chunk size; checkpoint intervals must be a multiple of it
H, K, V = 2, 64, 64
NUM_SLOTS = 8  # pool size for these tests; slot 7 is a canary


def _inputs(T: int, seed: int = 0, B: int = 1):
    torch.manual_seed(seed)
    return dict(
        k=torch.randn(B, T, H, K, device=DEV, dtype=torch.bfloat16) / 8,
        w=torch.randn(B, T, H, K, device=DEV, dtype=torch.bfloat16) / 8,
        u=torch.randn(B, T, H, V, device=DEV, dtype=torch.bfloat16) / 8,
        g=torch.nn.functional.logsigmoid(
            torch.randn(B, T, H, device=DEV, dtype=torch.float32)
        ),
    )


def _pool():
    return torch.zeros(NUM_SLOTS, H, K, V, device=DEV, dtype=torch.float32)


def _run(inp, *, ckpt=None, slots=None, base=None, every=0, **kw):
    return chunk_gated_delta_rule_fwd_h(
        inp["k"],
        inp["w"],
        inp["u"],
        g=inp["g"],
        output_final_state=True,
        chunk_size=BT,
        ckpt=ckpt,
        ckpt_slots=slots,
        ckpt_base=base,
        ckpt_every=every,
        **kw,
    )


def test_checkpoints_do_not_perturb_h_or_final_state():
    """The scatter is a pure side-write; the recurrence must be untouched."""
    inp = _inputs(512)
    ref_h, _, ref_ht = _run(inp)

    slots = torch.tensor([[0, 1, 2, -1]], device=DEV, dtype=torch.long)
    got_h, _, got_ht = _run(
        inp,
        ckpt=_pool(),
        slots=slots,
        base=torch.zeros(1, device=DEV, dtype=torch.long),
        every=2 * BT,
    )

    assert torch.equal(got_h, ref_h)
    assert torch.equal(got_ht, ref_ht)


def test_each_checkpoint_is_the_state_at_its_token_boundary():
    """ckpt[i] must be the state after (i+1)*every tokens.

    ``h[0, c]`` is the state carried into scan chunk ``c``, i.e. the state
    after ``c * BT`` tokens -- the same quantity, but stored in ``h``'s bf16.
    The checkpoint comes straight off the fp32 accumulator, so it is *more*
    precise than ``h``; equality is asserted after rounding it back to bf16.
    That extra precision is the reason for scattering from the accumulator
    rather than copying out of ``h``.
    """
    inp = _inputs(512)
    ref_h, _, _ = _run(inp)

    ckpt, every = _pool(), 2 * BT
    slots = torch.tensor([[0, 1, 2, -1]], device=DEV, dtype=torch.long)
    _run(
        inp,
        ckpt=ckpt,
        slots=slots,
        base=torch.zeros(1, device=DEV, dtype=torch.long),
        every=every,
    )

    for i_ck, chunk in enumerate((2, 4, 6)):
        want = ref_h[0, chunk]
        assert torch.equal(
            ckpt[i_ck].to(torch.bfloat16), want
        ), f"checkpoint {i_ck} (token {chunk * BT}) != h[0, {chunk}]"


def test_a_minus_one_slot_is_skipped_without_collateral_writes():
    """A boundary the scheduler could not reserve must be a no-op.

    -1 is how the pool reports exhaustion. The kernel folds the skip into
    the store mask and clamps the base pointer, so the danger this guards
    against is a clamped -1 silently landing on slot 0.
    """
    inp = _inputs(512)
    ckpt = _pool()
    # Reserve only the middle boundary; the first and third are unreservable.
    slots = torch.tensor([[-1, 3, -1, -1]], device=DEV, dtype=torch.long)
    _run(
        inp,
        ckpt=ckpt,
        slots=slots,
        base=torch.zeros(1, device=DEV, dtype=torch.long),
        every=2 * BT,
    )

    assert ckpt[3].abs().max().item() > 0.0, "the one reserved slot was not written"
    for i in range(NUM_SLOTS):
        if i != 3:
            assert ckpt[i].abs().max().item() == 0.0, f"slot {i} was written anyway"


def test_ckpt_base_keeps_boundaries_at_absolute_token_positions():
    """Splitting a prefill must not move where its checkpoints land.

    Boundaries are counted in absolute token position via ``ckpt_base``, the
    number of scan chunks the sequence already consumed. A boundary landing
    exactly on a split is written by the *next* pass at ``i_t == 0``, from the
    ``initial_state`` it was handed -- the scatter fires at the top of each
    chunk, so a pass's last scatterable state is the one before its final
    chunk.
    """
    T = 512
    inp = _inputs(T)
    every = 2 * BT
    half = T // 2

    one = _pool()
    _run(
        inp,
        ckpt=one,
        slots=torch.tensor([[0, 1, 2, -1]], device=DEV, dtype=torch.long),
        base=torch.zeros(1, device=DEV, dtype=torch.long),
        every=every,
    )

    split = _pool()
    first = {k: v[:, :half] for k, v in inp.items()}
    second = {k: v[:, half:] for k, v in inp.items()}

    s_a = torch.full((1, 4), -1, device=DEV, dtype=torch.long)
    s_a[0, 0] = 0  # token 128, interior to pass 1
    _, _, mid = _run(
        first,
        ckpt=split,
        slots=s_a,
        base=torch.zeros(1, device=DEV, dtype=torch.long),
        every=every,
    )

    s_b = torch.full((1, 4), -1, device=DEV, dtype=torch.long)
    s_b[0, 1] = 1  # token 256, the split point -- pass 2's i_t == 0
    s_b[0, 2] = 2  # token 384
    _run(
        second,
        ckpt=split,
        slots=s_b,
        base=torch.full((1,), half // BT, device=DEV, dtype=torch.long),
        every=every,
        initial_state=mid,
    )

    for i_ck in range(3):
        assert torch.equal(
            split[i_ck], one[i_ck]
        ), f"checkpoint {i_ck} moved when the prefill was split"


def test_varlen_places_each_sequences_boundaries_independently():
    """A varlen batch mixes requests at different prefill offsets.

    ``ckpt_base`` is per-sequence for exactly this reason: seq 0 starts
    fresh while seq 1 is already two chunks into its prompt, so the same
    ``i_t`` in one launch is a boundary for one and not the other.
    """
    T0, T1 = 384, 256
    inp = _inputs(max(T0, T1))
    every = 2 * BT

    single = _pool()
    _run(
        inp,
        ckpt=single,
        slots=torch.tensor([[0, 1, 2, -1]], device=DEV, dtype=torch.long),
        base=torch.zeros(1, device=DEV, dtype=torch.long),
        every=every,
    )

    # Both sequences replay the same tokens, so equal offsets must yield
    # equal states -- which is what makes the cross-check below meaningful.
    varlen = {
        key: torch.cat([val[0, :T0], val[0, :T1]]).unsqueeze(0)
        for key, val in inp.items()
    }
    cu = torch.tensor([0, T0, T0 + T1], device=DEV, dtype=torch.int32)

    ckpt = _pool()
    slots = torch.full((2, 4), -1, device=DEV, dtype=torch.long)
    slots[0, 0] = 4  # seq 0: token 128
    slots[0, 1] = 5  # seq 0: token 256
    slots[1, 1] = 6  # seq 1: base=2 chunks in, so its i_t=2 is token 256
    _run(
        varlen,
        ckpt=ckpt,
        slots=slots,
        base=torch.tensor([0, 2], device=DEV, dtype=torch.long),
        every=every,
        cu_seqlens=cu,
    )

    assert torch.equal(ckpt[4], single[0])
    assert torch.equal(ckpt[5], single[1])
    # seq 1 is offset by 2 chunks, so its first written boundary replays
    # the token-128 state.
    assert torch.equal(ckpt[6], single[0])
    assert ckpt[7].abs().max().item() == 0.0, "wrote outside the reserved slots"


def test_interval_must_be_a_multiple_of_the_scan_chunk():
    """Boundaries can only land where the scan pauses."""
    inp = _inputs(128)
    with pytest.raises(AssertionError, match="multiple of"):
        _run(
            inp,
            ckpt=_pool(),
            slots=torch.zeros(1, 4, device=DEV, dtype=torch.long),
            base=torch.zeros(1, device=DEV, dtype=torch.long),
            every=BT + 1,
        )
