# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Retry spacing shared by the gfx950 tagged-mailbox implementations.

The value is read once, at import, from ``ATOM_MONO_POLL_SLEEP_UNITS`` (unset
or empty: ``DEFAULT``), so it is a plain Python constant in every kernel build.
Spawned workers inherit the environment: ``Config`` sets it to ``COPY_SAFE``
before they spawn when a KV transfer or offload connector runs copies beside
the Mono kernels and the variable is unset. It is not in this file's bytes, so every
source digest that hashes this file also hashes ``CACHE_TAG``: builds with
different values never share a JIT cache entry.

This module imports nothing from ATOM but the variable's vocabulary
(``polling_env``) so ops, sync and the runtime can all use it without an
import cycle.
"""

import os

from atom.mono.polling_env import ENV, units

# s_sleep uses 64-clock units: 16 is about 1024 clocks. On gfx950, s_sleep(1)
# hung with a concurrent copy kernel (see tools/mono_mailbox_copy_repro.py and
# ROCm/ATOM#3369). This is an empirical mitigation, not an ordering guarantee
# or a proved hardware root cause.
# Evidence (reproducer `--mode side`; synthetic, finite sampling, 5-7 untested):
#   1: 0/6 pass; 4: 0/6 (every run hung, rounds 2-22);
#   8: 128 rounds 44/44, but 1024 rounds 3/4 (one hang at round 583);
#   16: 128 rounds 44/44, 1024 rounds 12/12.
# 128-round runs cannot tell 8 from 16. A pass is not proof of safety.
# To lower COPY_SAFE, rerun `--mode side --repeats 1024` (timeout >= 300s) on
# several GPUs, at least 12 runs per value alongside 16 in the same batch; 128
# rounds is not enough. Any hang rejects the value.
#
# Override: ATOM_MONO_POLL_SLEEP_UNITS=N. N=1 restores main's instructions
# exactly (`s_sleep 1` in sync.py / step_begin.py, `s_nop 0` in spin_pause and
# the V4 helpers' poll). Below COPY_SAFE beside concurrent KV copies, 1 and 4
# hung and 8 hung once (above); `Config` warns once at startup.
POLL_SLEEP_UNITS = units(os.environ.get(ENV))
# `spin_pause` and the V4 helpers' poll: main had `s_nop 0` there
MAIN_POLLING = POLL_SLEEP_UNITS == 1
SPIN_PAUSE_ASM = "s_nop 0" if MAIN_POLLING else f"s_sleep {POLL_SLEEP_UNITS}"
CACHE_TAG = f"poll_sleep_units={POLL_SLEEP_UNITS}".encode()
