# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""``ATOM_MONO_POLL_SLEEP_UNITS``: its name, its values and their parse, read
from no environment. ``atom.mono.polling`` reads this process's value once,
at import; ``Config`` sets the variable before anything imports it, so it
needs the name without the read."""

ENV = "ATOM_MONO_POLL_SLEEP_UNITS"
# without concurrent KV copies (the evidence in ``polling`` is about them only)
DEFAULT = 4
# beside concurrent KV copies (offload or P/D connectors)
COPY_SAFE = 16
# s_sleep's immediate is 16 bits, but gfx9 / CDNA (gfx950 included) sleeps for
# about 64 * SIMM16[6:0] clocks: the assembler takes 128 and the wave sleeps as
# for 0. 0 is no backoff at all. So only 1..127 mean what they say.
MAX = 127


def units(raw: str | None) -> int:
    """The spacing ``raw`` (the variable's text; unset or empty: ``DEFAULT``)
    sets, s_sleep units of 64 clocks; a value outside 1..``MAX`` raises."""
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT
    try:
        value = int(raw)
    except ValueError:
        value = None
    if value is None or not 1 <= value <= MAX:
        raise ValueError(
            f"{ENV}={raw!r}: expected an integer in 1..{MAX} (s_sleep units "
            f"of 64 clocks; unset means {DEFAULT})"
        )
    return value
