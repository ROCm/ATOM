# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Retry spacing shared by the gfx950 tagged-mailbox implementations."""

# s_sleep uses 64-clock units: 16 is about 1024 clocks. On gfx950, s_sleep(1)
# hung with a concurrent copy kernel (see tools/mono_mailbox_copy_repro.py and
# ROCm/ATOM#3369). This is an empirical mitigation, not an ordering guarantee
# or a proved hardware root cause.
# Evidence (reproducer `--mode side`; synthetic, finite sampling, 5-7 untested):
#   1: 0/6 pass; 4: 0/6 (every run hung, rounds 2-22);
#   8: 128 rounds 44/44, but 1024 rounds 3/4 (one hang at round 583);
#   16: 128 rounds 44/44, 1024 rounds 12/12.
# 128-round runs cannot tell 8 from 16. TPOT cost was measured at 16 only
# (+0.5% on V4.1); 8 was not measured. A pass is not proof of safety.
# To lower it, rerun `--mode side --repeats 1024` (timeout >= 300s) on several
# GPUs, at least 12 runs per value alongside 16 in the same batch; 128 rounds
# is not enough. Any hang rejects the value.
POLL_SLEEP_UNITS = 16
