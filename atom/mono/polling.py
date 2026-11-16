# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Retry spacing shared by the gfx950 tagged-mailbox implementations.

The value is read once, at import, from ``ATOM_MONO_POLL_SLEEP_UNITS`` (unset
or empty: 16), so it is a plain Python constant in every kernel build. Spawned
workers inherit the environment. It is not in this file's bytes, so every
source digest that hashes this file also hashes ``CACHE_TAG``: builds with
different values never share a JIT cache entry.

This module imports nothing from ATOM so ops, sync and the runtime can all use
it without an import cycle.
"""

import os

ENV = "ATOM_MONO_POLL_SLEEP_UNITS"
DEFAULT = 16
# s_sleep's immediate is 16 bits, but gfx9 / CDNA (gfx950 included) sleeps for
# about 64 * SIMM16[6:0] clocks: the assembler takes 128 and the wave sleeps as
# for 0. 0 is no backoff at all. So only 1..127 mean what they say.
MAX = 127


def _units(raw):
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT
    try:
        units = int(raw)
    except ValueError:
        units = None
    if units is None or not 1 <= units <= MAX:
        raise ValueError(
            f"{ENV}={raw!r}: expected an integer in 1..{MAX} (s_sleep units "
            f"of 64 clocks; unset means {DEFAULT})"
        )
    return units


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
# To lower the default, rerun `--mode side --repeats 1024` (timeout >= 300s) on
# several GPUs, at least 12 runs per value alongside 16 in the same batch; 128
# rounds is not enough. Any hang rejects the value.
#
# Override: ATOM_MONO_POLL_SLEEP_UNITS=N. N=1 restores main's instructions
# exactly (`s_sleep 1` in sync.py / step_begin.py, `s_nop 0` in spin_pause and
# the V4 helpers' poll), for a regression on a deployment with no concurrent
# KV copies (no offload or P/D connector). Below 16 beside such copies, 1 and 4
# hung and 8 hung once (above); `Config` warns once at startup.
POLL_SLEEP_UNITS = _units(os.environ.get(ENV))
# `spin_pause` and the V4 helpers' poll: main had `s_nop 0` there
MAIN_POLLING = POLL_SLEEP_UNITS == 1
SPIN_PAUSE_ASM = "s_nop 0" if MAIN_POLLING else f"s_sleep {POLL_SLEEP_UNITS}"
CACHE_TAG = f"poll_sleep_units={POLL_SLEEP_UNITS}".encode()
