# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_DRAFT_PRESENCE_PENALTY: `DraftPresencePenalty` against a torch reference.

A chain of draft passes (staged once by `begin`, then `apply` + argmax + `record`
per pass, the order the recorded passes run) must pick, at every pass, the argmax
of the logits less the request's penalty on its mask's tokens and on the drafts
the chain already made -- and leave rows without a penalty untouched."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.presence_penalty import DraftPresencePenalty, PresencePenaltyState

DEV = "cuda"


class _Step:
    def __init__(self, slots, penalties):
        self.slots = torch.tensor(slots, dtype=torch.int32, device=DEV)
        self.penalties = torch.tensor(penalties, dtype=torch.float32, device=DEV)


@pytest.mark.parametrize("rows,passes", [(1, 3), (3, 3), (5, 4), (8, 7)])
def test_draft_chain_matches_the_penalized_argmax(rows, passes):
    vocab = 4099
    gen = torch.Generator(device=DEV).manual_seed(rows * 31 + passes)
    state = PresencePenaltyState(vocab, max_num_seqs=8, max_extra=8, device=DEV)
    # A few tokens already emitted per slot.
    slots = list(range(rows))
    for s in slots:
        state.mask[s, torch.randint(0, vocab, (40,), generator=gen, device=DEV)] = 1
    pens = [1.5 if r % 3 != 2 else 0.0 for r in range(rows)]
    dp = DraftPresencePenalty(state, max_rows=16, max_drafts=passes, device=DEV)
    dp.begin(_Step(slots, pens), rows)
    drafted = [[] for _ in range(rows)]
    for _ in range(passes):
        # Logits that favour the emitted tokens, so the penalty decides the pick.
        logits = torch.randn(rows, vocab, generator=gen, device=DEV)
        logits += 1.2 * state.mask[:rows].float()
        logits = logits.to(torch.bfloat16)
        ref = logits.float().clone()
        for r in range(rows):
            present = state.mask[r].bool().clone()
            for t in drafted[r]:
                present[t] = True
            ref[r] -= pens[r] * present.float()
        expect = ref.to(torch.bfloat16).float().argmax(-1)
        dp.apply(logits)
        ids = logits.float().argmax(-1).to(torch.int32)
        dp.record(ids)
        assert torch.equal(ids.long(), expect), (ids.tolist(), expect.tolist())
        for r in range(rows):
            drafted[r].append(int(ids[r]))
    assert dp.pos[:rows].tolist() == [passes] * rows


def test_begin_without_a_step_penalizes_nothing():
    vocab = 1024
    state = PresencePenaltyState(vocab, max_num_seqs=4, max_extra=8, device=DEV)
    state.mask[0, :512] = 1
    dp = DraftPresencePenalty(state, max_rows=4, max_drafts=3, device=DEV)
    dp.begin(_Step([0], [1.5]), 1)
    dp.begin(None, 1)
    logits = torch.zeros(1, vocab, dtype=torch.bfloat16, device=DEV)
    dp.apply(logits)
    assert torch.count_nonzero(logits) == 0
