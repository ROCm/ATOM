# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi programs' hand-off contracts and memory model, checked on their builds (compile-only, no GPU needed).

Composed outside ``hi.emit.compile.serving()``, every build of ``hi_compose.HiPrograms`` is traced (FlyDSL's disk
cache off) and its trace held to the hand-off contract (``hi.plan.check.check``: every poll waits on a region an
earlier stage or the host puts) and the memory model (``check_memory``), raising ``ContractError`` on a violation.
The engine's builds at load skip both (``serving``: FlyDSL's disk cache on) and rely on this check of the same
programs.

Usage: python -m atom.models.deepseek_v41.mono.hi_contract CONFIG.json [WIDTH ...] [--fits]
(``CONFIG.json``: the model's ``config.json``; widths default to every served one, ``hi_compose.ROWS``: the 108
programs and their fits, about 10 minutes on 8 CPUs). The Linear prefetch fits are derived and held to the frozen
table (``hi.blocks.gfx950.linear_fits``); ``--fits`` prints the table instead (to regenerate it).
"""

from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

from atom.models.deepseek_v41.config import build_attention_topology
from atom.models.deepseek_v41.mono import hi_compose
from atom.models.deepseek_v41.mono.hi_gate import MAX_CTX

TP = 4


def config(path: str) -> SimpleNamespace:
    """The text config of a DeepSeek-V4.1 ``config.json``."""
    with open(path) as f:
        raw = json.load(f)
    return SimpleNamespace(**raw.get("text_config", raw))


def check(cfg, widths=None, tp: int = TP, fits: bool = True, kv_dtype: str = "bf16") -> hi_compose.HiPrograms:  # fmt: skip
    """Every hi program of ``widths`` (default ``hi_compose.ROWS``) at ``tp`` built for gfx950 compile-only, its trace
    checked (module docstring), on ATOM's KV pool of ``kv_dtype``; -> the composition. Raises ``ContractError`` on a violation. ``fits``: the Linear
    prefetch fits derived by their probes, not read from the frozen table (``linear_fits``), and each held to the
    table's (``ContractError`` on a difference; a plan the table lacks: ``missing_fits``).
    """
    from .hi.blocks.gfx950 import linear, linear_fits

    os.environ.setdefault("ARCH", "gfx950")
    os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx950")
    topology = build_attention_topology(cfg)[: cfg.num_hidden_layers]
    widths = list(hi_compose.ROWS) if widths is None else list(widths)
    frozen, linear.FROZEN = linear.FROZEN, not fits
    try:
        progs = hi_compose.HiPrograms(
            cfg, topology, widths, MAX_CTX, tp, kv_dtype=kv_dtype
        )
    finally:
        linear.FROZEN = frozen
    if fits:
        from .hi.plan.check import ContractError

        bad = {d: (v, linear_fits.FITS[d]) for d, v in linear.DERIVED.items()
               if d in linear_fits.FITS and linear_fits.FITS[d] != v}  # fmt: skip
        if bad:
            raise ContractError(
                f"Linear fits differ from the frozen table (derived, frozen): {bad}"
            )
    return progs


def missing_fits() -> dict:
    """The fits derived in this process (``check``) that the frozen table lacks: plan digest -> depth."""
    from .hi.blocks.gfx950 import linear, linear_fits

    return {d: v for d, v in linear.DERIVED.items() if d not in linear_fits.FITS}


def fits_module(fits: dict) -> str:
    """``linear_fits.py``'s text for ``fits`` (plan digest -> depth), with this FlyDSL."""
    import importlib.metadata as md

    lines = [f"    {d!r}: {v!r}," for d, v in sorted(fits.items())]
    return "\n".join(
        [
            "# SPDX-License-Identifier: MIT",
            "# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.",
            '"""The gfx950 Linear prefetch depths the served compositions derive (``linear._fit``: the deepest that builds',
            "without spills, with its VGPR margin), frozen: the plan's digest (``linear.plan_digest``) -> depth (None: no",
            "depth builds). Derived by the probes with FlyDSL ``FLYDSL`` for MI355X; another FlyDSL derives its own. The",
            "contract check derives them again and holds them to this table; regenerate with",
            "``python -m atom.models.deepseek_v41.mono.hi_contract CONFIG.json --fits``.",
            '"""',
            "",
            f"FLYDSL = {md.version('flydsl')!r}",
            "",
            "FITS: dict = {",
            *lines,
            "}",
            "",
        ]
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    write = "--fits" in argv
    argv = [a for a in argv if a != "--fits"]
    t0 = time.time()
    progs = check(config(argv[0]), [int(w) for w in argv[1:]] or None)
    if (
        write
    ):  # the frozen table, every fit of the composition (the derived ones it lacked included)
        from .hi.blocks.gfx950 import linear

        print(fits_module(dict(linear.DERIVED)), end="")
        return 0
    print(f"{len(progs.kernels())} hi programs built and checked in {time.time() - t0:.0f} s; "
          f"{len(missing_fits())} fits not in the frozen table")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
