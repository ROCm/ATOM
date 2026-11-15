# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Retry spacing shared by the gfx950 tagged-mailbox implementations."""

# s_sleep uses 64-clock units: 16 is about 1024 clocks. On gfx950, s_sleep(1)
# hung with a concurrent copy kernel (see tools/mono_mailbox_copy_repro.py and
# ROCm/ATOM#3369). This is an empirical mitigation, not an ordering guarantee
# or a proved hardware root cause. Do not lower it without rerunning the repro.
POLL_SLEEP_UNITS = 16
