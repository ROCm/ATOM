# SPDX-License-Identifier: MIT
"""DSpark's accepted gfx1250 TP4 fence, including the tagged mHC workspace."""
import torch
from atom.models.deepseek_v41.mono.gfx1250.dspark_step_begin import MISSING
from atom.models.deepseek_v41.mono.gfx1250.dspark_step_begin import STATE_WORDS
from atom.models.deepseek_v41.mono.gfx1250.dspark_step_begin import step_begin

class StepMailboxes:
    def __init__(self, peers, *scratch, debug=False):
        self.peers = peers
        self._scratch = scratch
        self._debug = debug
        self._state = torch.zeros(STATE_WORDS, dtype=torch.int64, device=peers.bytes.device)

    def begin_step(self):
        step_begin(self._scratch, self.peers, self._state, self._debug)

    def missing_ranks(self):
        missing = self._state[MISSING:MISSING + self.peers.addresses.numel()]
        ranks = [rank for rank, value in enumerate(missing.tolist()) if value]
        missing.zero_()
        return ranks
