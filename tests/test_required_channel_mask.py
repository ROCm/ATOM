# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which logits row gets which constraint, and the state a step leaves behind.

The properties are asserted against the pure-torch references on CPU: a
verification row is constrained as if the drafts before it were accepted, a
carried request starts from the previous forward's state, and the state after
a step counts only the ids it emitted. `test_kernels_match_reference` then
holds the Triton kernels to the references, and needs a GPU.
"""

import random

import pytest
import torch

pytest.importorskip("triton", reason="needs the Triton GPU kernel library")

from atom.model_ops.required_channel import (
    DeviceChannel,
    advance_states,
    advance_states_reference,
    constrain_logits,
    constrain_logits_reference,
)
from atom.sampling_params import UNCONSTRAINED, RequiredChannel

V = 64
# after = 10 11 12, opening = 20 21, end ids 2 and 3. States: 0-2 matching
# `after`, 3 forces 20, 4 forces 21, 5 is a `20` the model wrote itself; once
# 20 21 is written either way the channel is open and UNCONSTRAINED.
CH = RequiredChannel(after=(10, 11, 12), opening=(20, 21), opens_at=2, end_ids=(2, 3))
# The same, opened by a lone `20`: no own-start state at all.
CH_OPENS_AT_ONE = RequiredChannel(
    after=(10, 11, 12), opening=(20, 21), opens_at=1, end_ids=(2, 3)
)
NEG_INF = float("-inf")


def _logits(rows):
    return torch.arange(rows * V, dtype=torch.float32).view(rows, V) / 7.0


def _kind(row, original):
    """What happened to one row: untouched, ends banned, or one id forced."""
    if torch.equal(row, original):
        return "free"
    finite = torch.isfinite(row).nonzero().view(-1).tolist()
    if len(finite) == 1:
        (forced,) = finite
        assert row[forced] == original[forced], "a forced logit keeps its value"
        return ("force", forced)
    banned = (~torch.isfinite(row)).nonzero().view(-1).tolist()
    assert banned == list(CH.end_ids)
    assert torch.equal(row[torch.isfinite(row)], original[torch.isfinite(row)])
    return "ban"


def test_decode_rows_follow_their_state():
    states = torch.tensor([0, 2, 3, 4, 5, UNCONSTRAINED], dtype=torch.int32)
    logits = _logits(len(states))
    original = logits.clone()
    base = constrain_logits_reference(CH, states, logits)
    assert base.tolist() == states.tolist()
    kinds = [_kind(logits[i], original[i]) for i in range(len(states))]
    assert kinds == ["ban", "ban", ("force", 20), ("force", 21), "ban", "free"]


def test_a_forced_id_the_model_ruled_out_still_leaves_a_finite_row():
    """All -inf would be a NaN softmax: the forced logit is then 0."""
    logits = _logits(1)
    logits[0, 20] = NEG_INF
    constrain_logits_reference(CH, torch.tensor([3], dtype=torch.int32), logits)
    assert torch.isfinite(logits[0]).nonzero().view(-1).tolist() == [20]
    assert logits[0, 20] == 0


def test_verification_rows_walk_the_drafts_before_them():
    """Request 0 starts after `10 11` and drafts `12 20 21`: its rows are
    constrained at states 2, 3, 4, and its bonus row comes after the opening,
    unconstrained. Request 1 has no drafts; request 2 is unconstrained."""
    states = torch.tensor([2, 0, UNCONSTRAINED], dtype=torch.int32)
    # Row layout of the step: request 0 holds rows 0-3, request 1 row 4,
    # request 2 rows 5-6. A row's draft is the input id after it.
    input_ids = torch.tensor([11, 12, 20, 21, 7, 5, 6, 0], dtype=torch.int32)
    target_logits_indices = torch.tensor([0, 1, 2, 5], dtype=torch.int32)
    cu_num_draft_tokens = torch.tensor([3, 3, 4], dtype=torch.int32)
    target = _logits(4)
    bonus = _logits(3)
    t0, b0 = target.clone(), bonus.clone()
    constrain_logits_reference(
        CH,
        states,
        bonus,
        target_logits=target,
        cu_num_draft_tokens=cu_num_draft_tokens,
        input_ids=input_ids,
        target_logits_indices=target_logits_indices,
    )
    assert [_kind(target[r], t0[r]) for r in range(4)] == [
        "ban",
        ("force", 20),
        ("force", 21),
        "free",
    ]
    assert [_kind(bonus[i], b0[i]) for i in range(3)] == ["free", "ban", "free"]


def test_carried_requests_start_from_the_previous_forward():
    """The host's state is a forward stale for a carried request; the GPU's
    copy is not. A request the host caught up with keeps the host's."""
    host = torch.tensor([0, 0, UNCONSTRAINED], dtype=torch.int32)
    carried_from = torch.tensor([1, -1, -1], dtype=torch.int32)
    carried = torch.tensor([UNCONSTRAINED, 3], dtype=torch.int32)
    logits = _logits(3)
    original = logits.clone()
    base = constrain_logits_reference(
        CH, host, logits, carried_from=carried_from, carried_states=carried
    )
    assert base.tolist() == [3, 0, UNCONSTRAINED]
    assert _kind(logits[0], original[0]) == ("force", 20)
    assert _kind(logits[1], original[1]) == "ban"


def test_state_after_a_step_counts_only_emitted_ids():
    base = torch.tensor([1, 0, 4, UNCONSTRAINED], dtype=torch.int32)
    sampled = torch.tensor(
        [[11, 12, -1, -1], [10, -1, -1, -1], [21, 10, 11, 12], [10, 11, 12, -1]],
        dtype=torch.int32,
    )
    assert advance_states_reference(CH, base, sampled).tolist() == [
        3,
        1,
        UNCONSTRAINED,  # the last forced id opens the channel, for good
        UNCONSTRAINED,
    ]
    # A single id per row; `10` after `10` restarts the match rather than
    # dropping it, and a `20` the model writes itself starts the channel's
    # own opening.
    plain = torch.tensor([10, 20, 21, 7], dtype=torch.int32)
    assert advance_states_reference(CH, base, plain).tolist() == [
        1,
        5,
        UNCONSTRAINED,
        UNCONSTRAINED,
    ]


def _random_step(rng, device, dtype, bs, max_drafts, channel=CH, vocab_size=300):
    """A verification step with ragged drafts drawn mostly from the channel's
    own ids, so every state comes up."""
    vocab = (2, 3, 10, 11, 12, 20, 21, 5)
    lens = [rng.randint(0, max_drafts) for _ in range(bs)]
    cu = torch.tensor(lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    input_ids, target_idx, pos = [], [], 0
    for n in lens:
        span = [rng.choice(vocab) for _ in range(n + 1)]
        target_idx += [pos + j for j in range(n)]
        input_ids += span
        pos += n + 1
    input_ids.append(0)
    choices = [UNCONSTRAINED, *range(channel.num_states)]
    states = [rng.choice(choices) for _ in range(bs)]
    num_prev = bs + 2
    carried_from = [rng.choice([-1, *range(num_prev)]) for _ in range(bs)]
    carried = [rng.choice(choices) for _ in range(num_prev)]

    def t(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    target = torch.randn(max(len(target_idx), 1), vocab_size, device=device)
    return {
        "states": t(states),
        "last_logits": torch.randn(bs, vocab_size, device=device).to(dtype),
        "carried_from": t(carried_from),
        "carried_states": t(carried),
        "target_logits": target[: len(target_idx)].to(dtype),
        "cu_num_draft_tokens": cu.to(device),
        "input_ids": t(input_ids),
        "target_logits_indices": t(target_idx),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("channel", [CH, CH_OPENS_AT_ONE], ids=["own2", "own1"])
# One block; nine, the last ragged; twenty, more than a request's programs.
@pytest.mark.parametrize("vocab_size", [300, 9000, 20000])
def test_kernels_match_reference(dtype, seed, channel, vocab_size):
    rng = random.Random(seed)
    device_channel = DeviceChannel.of(channel, "cuda")
    for drafts in (False, True):
        step = _random_step(
            rng,
            "cuda",
            dtype,
            bs=9,
            max_drafts=4,
            channel=channel,
            vocab_size=vocab_size,
        )
        if not drafts:
            for k in ("target_logits", "cu_num_draft_tokens", "input_ids"):
                step[k] = None
            step["target_logits_indices"] = None
        ref = {k: (v.cpu().clone() if v is not None else None) for k, v in step.items()}

        base = constrain_logits(device_channel, **step)
        ref_base = constrain_logits_reference(channel, **ref)
        assert base.cpu().tolist() == ref_base.tolist()
        assert torch.equal(step["last_logits"].cpu(), ref["last_logits"])
        if drafts:
            assert torch.equal(step["target_logits"].cpu(), ref["target_logits"])

        width = 5
        sampled = torch.tensor(
            [
                [rng.choice((10, 11, 12, 20, 21, 7)) for _ in range(width)]
                for _ in range(len(base))
            ],
            dtype=torch.int32,
        )
        for row in sampled:
            row[rng.randint(1, width) :] = -1
        out = advance_states(device_channel, base, sampled.cuda())
        assert (
            out.cpu().tolist()
            == advance_states_reference(channel, ref_base, sampled).tolist()
        )


# Kimi-K3's own ids and vocabulary: forced ids far past one `BLOCK_V` tile,
# rows wider than one, and no request carried over.
K3 = RequiredChannel(
    after=(163588, 12092, 163589),
    opening=(163587, 25385, 163589, 163587, 10257, 4453, 878),
    opens_at=3,
    end_ids=(163585, 163586),
)
K3_VOCAB = 163840


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernels_match_reference_on_k3s_vocab_without_a_carry(dtype):
    device_channel = DeviceChannel.of(K3, "cuda")
    states = [UNCONSTRAINED, *range(K3.num_states)]
    forced_end = len(K3.after) + len(K3.opening)
    logits = torch.randn(len(states), K3_VOCAB, device="cuda").to(dtype)
    # A forced id the model ruled out: its row's last forced state.
    logits[states.index(forced_end - 1), K3.opening[-1]] = NEG_INF
    ref = logits.cpu().clone()
    st = torch.tensor(states, dtype=torch.int32)

    base = constrain_logits(device_channel, st.cuda(), logits)
    ref_base = constrain_logits_reference(K3, st, ref)
    assert base.cpu().tolist() == ref_base.tolist() == states
    assert torch.equal(logits.cpu(), ref)
    assert torch.isfinite(logits.float()).any(dim=1).all()

    # Each row writes the id its state waits for or is forced to: every state
    # steps on, and the last forced one and the last own-start one open it.
    def next_id(s):
        if s < len(K3.after):
            return K3.after[s]
        if s < forced_end:
            return K3.opening[s - len(K3.after)]
        return K3.opening[s - forced_end + 1]

    sampled = torch.tensor(
        [[5]] + [[next_id(s)] for s in range(K3.num_states)], dtype=torch.int32
    )
    out = advance_states(device_channel, base, sampled.cuda()).cpu().tolist()
    assert out == advance_states_reference(K3, ref_base, sampled).tolist()
    assert out[states.index(forced_end - 1)] == UNCONSTRAINED
    assert out[-1] == UNCONSTRAINED
