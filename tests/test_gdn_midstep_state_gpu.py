# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A midstep checkpoint must equal the state a shortened forward would leave.

`readable_midstep` rests on one claim about the chunk kernel: `h[:, j]` is the
recurrent state after `j * 64` tokens, so a checkpoint at an interior boundary
can be sliced out of a full-length forward instead of being produced by ending
a forward there. `BlockManager` acts on that claim by suppressing
`checkpoint_cut` entirely for a readable backend -- the prefill runs whole and
the boundaries are harvested afterwards.

`TestMidstepCheckpoints` in `test_state_checkpoint.py` pins everything around
that claim -- which positions are chosen, reserve/publish/cancel, that the cut
is suppressed -- but it stubs the kernel, so it cannot see the claim itself
fail. If the slice and the shortened run disagree, every checkpoint the
readable path stores is subtly wrong, resuming requests inherit a state their
prefix never produced, and no CPU test notices.

Bit-exactness rather than a tolerance is the right bar: the two arms round the
same fp32 value into the same dtype (`h` is `k.new_empty`, and
`GDNStateMixin._state_dtypes` returns `config.torch_dtype`), so any difference
at all means they are not the same computation.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises the GDN chunk kernel; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.fla_ops.chunk import (
    chunk_gated_delta_rule,
    pop_last_intermediate_states,
)

DEV = "cuda"
DTYPE = torch.bfloat16
CHUNK = 64  # what the kernel materializes `h` at; see `chunk_local_cumsum`
NUM_CHUNKS = 8
H, K, V = 4, 128, 128


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    tokens = NUM_CHUNKS * CHUNK

    def rnd(*shape):
        return torch.randn(*shape, dtype=DTYPE, device=DEV)

    return {
        "q": rnd(1, tokens, H, K),
        "k": rnd(1, tokens, H, K),
        "v": rnd(1, tokens, H, V),
        # Log-space gate, kept negative so the decay is a decay -- the regime
        # a real prompt runs in, and the one where a mis-sliced state would
        # drift rather than saturate.
        "g": -torch.rand(1, tokens, H, dtype=DTYPE, device=DEV),
        "beta": torch.rand(1, tokens, H, dtype=DTYPE, device=DEV),
    }


def _forward(inp, *, upto=None, keep=False):
    """Run the kernel over the first `upto` tokens, or all of them."""
    sl = slice(0, upto) if upto is not None else slice(None)
    tokens = upto if upto is not None else inp["q"].shape[1]
    _, final = chunk_gated_delta_rule(
        q=inp["q"][:, sl],
        k=inp["k"][:, sl],
        v=inp["v"][:, sl],
        g=inp["g"][:, sl],
        beta=inp["beta"][:, sl],
        initial_state=None,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=DEV),
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        keep_intermediate_states=keep,
    )
    return final


def test_every_interior_boundary_matches_a_shortened_forward():
    """The claim `readable_midstep` makes, asked of the kernel directly."""
    inp = _inputs()
    _forward(inp, keep=True)
    h = pop_last_intermediate_states()
    assert h is not None, "keep_intermediate_states did not retain h"
    assert h.shape[1] == NUM_CHUNKS

    for j in range(1, NUM_CHUNKS):
        sliced = h[:, j]
        cut = _forward(inp, upto=j * CHUNK)
        assert torch.equal(
            sliced, cut.reshape(sliced.shape).to(sliced.dtype)
        ), f"boundary at token {j * CHUNK} differs from a forward ending there"


def test_popping_consumes_the_reference():
    """A second pop must come back empty.

    Every GDN layer pops in turn, so a tensor left behind is one a later layer
    reads as its own -- checkpointing one layer's state under another's slot,
    which is silent and survives into the resumed request.
    """
    _forward(_inputs(), keep=True)
    assert pop_last_intermediate_states() is not None
    assert pop_last_intermediate_states() is None


def test_a_forward_that_was_not_asked_keeps_nothing():
    """The default must not pin `h`.

    It is large, and the vLLM/SGLang/rtpllm plugins never pop -- one retained
    after their last forward is held until the process exits.
    """
    _forward(_inputs(), keep=False)
    assert pop_last_intermediate_states() is None


# ── through the copy kernel, the way a server runs it ───────────────────────
#
# The tests above ask the chunk kernel alone, for one sequence starting from
# nothing. A server batches several prefills, most of them resuming from a
# checkpoint, and reads `h` through `write_state_checkpoints` -- whose
# per-sequence base arithmetic (`chunk_offsets[row] + off // 64`,
# `cu_seqlens[row] + off`) is exactly what a single-sequence test cannot see.


def _varlen(lens, seed):
    torch.manual_seed(seed)
    tokens = sum(lens)

    def rnd(*shape):
        return torch.randn(*shape, dtype=DTYPE, device=DEV)

    return {
        "q": rnd(1, tokens, H, K),
        "k": rnd(1, tokens, H, K),
        "v": rnd(1, tokens, H, V),
        "g": -torch.rand(1, tokens, H, dtype=DTYPE, device=DEV),
        "beta": torch.rand(1, tokens, H, dtype=DTYPE, device=DEV),
    }


def _cu(lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32, device=DEV)


def _run(inp, cu, init, keep=False):
    _, final = chunk_gated_delta_rule(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        initial_state=init,
        output_final_state=True,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
        keep_intermediate_states=keep,
    )
    return final


def test_varlen_resumed_rows_match_a_forward_stopped_there():
    """Three resumed prefills of uneven length, every readable position.

    Each row starts from its own initial state, as a request resuming from a
    checkpoint does, so a position is an offset from the row's start -- the
    frame `_checkpoint_targets` hands the kernel. The reference for each target
    is that row alone, from the same state, stopped there.
    """
    from atom.model_ops.fla_ops.index import prepare_chunk_offsets
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    lens = [200, 77, 333]
    inp = _varlen(lens, seed=1)
    cu = _cu(lens)
    init = torch.randn(len(lens), H, K, V, dtype=DTYPE, device=DEV) * 0.1
    final = _run(inp, cu, init.clone(), keep=True)
    h = pop_last_intermediate_states()
    assert h is not None

    # Runtime slots 0..2 hold each row's final state, as the scatter after the
    # chunk kernel leaves them; checkpoint slots follow.
    targets = []  # (row, off, is_end)
    for r, n in enumerate(lens):
        targets += [(r, off, 0) for off in range(CHUNK, n, CHUNK)]
        targets.append((r, n, 1))
    pool = torch.zeros(
        len(lens) + len(targets), *final.shape[1:], dtype=DTYPE, device=DEV
    )
    pool[: len(lens)] = final.to(DTYPE)
    D = 8
    x = torch.randn(sum(lens), D, dtype=DTYPE, device=DEV)
    conv = torch.zeros(pool.shape[0], D, 6, dtype=DTYPE, device=DEV)

    def ix(vals):
        return torch.tensor(vals, dtype=torch.int32, device=DEV)

    slots = list(range(len(lens), len(lens) + len(targets)))
    write_state_checkpoints(
        h,
        pool,
        x,
        conv,
        ix([r for r, _o, _e in targets]),
        ix(slots),
        ix([o for _r, o, _e in targets]),
        ix([e for _r, _o, e in targets]),
        ix([r for r, _o, _e in targets]),
        prepare_chunk_offsets(cu, CHUNK),
        cu,
        CHUNK,
        3,
    )

    starts = cu.tolist()
    for slot, (r, off, _end) in zip(slots, targets):
        sl = slice(starts[r], starts[r] + off)
        alone = {key: t[:, sl] for key, t in inp.items()}
        ref = _run(alone, _cu([off]), init[r : r + 1].clone())
        assert torch.equal(
            pool[slot], ref[0].to(DTYPE)
        ), f"row {r} at offset {off} differs from a forward stopped there"


def test_the_conv_checkpoint_is_what_the_prefill_kernel_resumes_from():
    """Positions `[0, conv_kernel - 1)` of the slot, oldest first.

    The pool is `conv_kernel - 1 + num_spec` = 6 wide; `causal_conv1d_fn`
    reads and writes only the first 3. The reference is that kernel run on
    each row stopped at the target, from the same initial window; the
    checkpoint is sliced from the full batch. Offsets 3..5 are included: they
    are where a 6-wide window used to reach into the previous row.
    """
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints
    from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_fn

    k_dim, v_dim, width, spec = 128, 256, 4, 3
    D = 2 * k_dim + v_dim
    lens = [70, 9, 130]
    cu = _cu(lens)
    torch.manual_seed(2)
    x = torch.randn(sum(lens), D, dtype=DTYPE, device=DEV)
    weight = torch.randn(D, width, dtype=DTYPE, device=DEV)
    # [slots, window, D] as allocated, viewed [slots, D, window] as the
    # attention layer hands it to both kernels.
    alloc = torch.randn(16, width - 1 + spec, D, dtype=DTYPE, device=DEV)
    pool = alloc.transpose(-1, -2)

    targets = [(0, 3, 0), (0, 5, 0), (0, 64, 0), (1, 4, 0), (1, 9, 1), (2, 6, 0)]
    targets += [(2, 128, 0), (2, 130, 1)]
    ckpt_slots = list(range(8, 8 + len(targets)))
    h = torch.zeros(1, 8, 1, 4, 4, dtype=DTYPE, device=DEV)
    ssm = torch.zeros(16, 1, 4, 4, dtype=DTYPE, device=DEV)

    def ix(vals):
        return torch.tensor(vals, dtype=torch.int32, device=DEV)

    write_state_checkpoints(
        h,
        ssm,
        x,
        pool,
        ix([r for r, _o, _e in targets]),
        ix(ckpt_slots),
        ix([o for _r, o, _e in targets]),
        ix([0] * len(targets)),  # SSM half reads `h`; only conv is checked
        ix([0] * len(targets)),
        ix([0, 2, 3, 6]),
        cu,
        CHUNK,
        width - 1,
    )

    starts = cu.tolist()
    for slot, (r, off, _end) in zip(ckpt_slots, targets):
        ref_alloc = alloc.clone()
        ref_pool = ref_alloc.transpose(-1, -2)
        # Row r alone in runtime slot r, resumed from that slot's window.
        causal_conv1d_fn(
            x[starts[r] : starts[r] + off].T,
            weight,
            None,
            conv_states=ref_pool,
            query_start_loc=_cu([off]),
            k_dim_size=k_dim,
            v_dim_size=v_dim,
            cache_indices=ix([r]),
            has_initial_state=torch.tensor([True], device=DEV),
            activation="silu",
        )
        assert torch.equal(
            pool[slot, :, : width - 1], ref_pool[r, :, : width - 1]
        ), f"row {r} at offset {off}: conv window differs"
