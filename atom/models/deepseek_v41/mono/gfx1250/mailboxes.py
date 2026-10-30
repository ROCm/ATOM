# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Use the TP4 wave32 step fence without changing other mono backends."""

from atom.mono.runtime.mailboxes import StepMailboxes as SharedStepMailboxes

from .step_begin import step_begin


class StepMailboxes(SharedStepMailboxes):
    def begin_step(self):
        step_begin(self._scratch, self.peers, self._state, self._debug)
