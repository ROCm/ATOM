# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Switches as constants. ``PREFILL_MSA_SCHED``: the kill switch of the fp16 kernel's scheduling ports, which run only
for ``sched=True`` while it is on and change no arithmetic."""

PREFILL_MSA_SCHED = True
