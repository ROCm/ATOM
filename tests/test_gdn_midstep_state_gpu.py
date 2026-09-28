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
