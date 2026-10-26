# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A runner's kernels, one build per step width."""

import logging

from atom.mono.runtime.compile import compile_kernels, scratch_bytes
from atom.mono.runtime.consensus import tp_agree

logger = logging.getLogger("atom")


class WidthBuilds:
    """``build(width)`` on a width's first step, its kernels (``kernels(built)``:
    ``(launcher, abi)`` pairs) compiled there without a launch: that step is a
    warmup forward before the width's graph capture, which then only records.

    Collective: every rank prepares the same width, and all agree they hold it,
    so a rank whose build failed keeps the others from launching a kernel that
    would wait on it for ever. A width refused once (a build that does not fit)
    stays refused: its steps take the original path without building again.

    A kernel with scratch refuses its width, with a warning: a persistent kernel
    needs every CTA resident at once, and one whose waves need scratch can be
    dispatched only in part (the scratch the runtime holds for the grid), its
    resident CTAs then waiting for ever on the rest."""

    def __init__(self, build, kernels, group, what: str) -> None:
        self._build, self._kernels = build, kernels
        self._group, self._what = group, what
        self._built = {}
        self._refused = set()

    def prepare(self, width: int) -> bool:
        """Whether every rank holds the ``width`` build.

        ``width`` is whatever the model keys its builds by, not necessarily a
        row count: Kimi-K3 packs the rows *and* the query length into one int
        (``kimi_k3.mono.runner.width_key``), so a 16-row step at query length 8
        is width 280. Reporting it as "rows" sent one reader hunting for a
        280-row step that does not exist.
        """
        if width in self._built:
            return True
        if width in self._refused:
            return False
        built = None
        try:
            built = self._build(width)
            launchers = compile_kernels(self._kernels(built))
            if not self._scratch_free(launchers, width):
                built = None
        except Exception:
            logger.exception(
                "%s: building the width-%d kernels failed", self._what, width
            )
            built = None
        if not tp_agree(built is not None, self._group):
            self._refused.add(width)
            return False
        self._built[width] = built
        logger.info("%s kernels built for width %d", self._what, width)
        return True

    def _scratch_free(self, launchers, width: int) -> bool:
        for launcher in launchers:
            scratch = scratch_bytes(launcher)
            if scratch:
                logger.warning(
                    "%s: the width-%d kernel needs %d B of scratch a lane, which"
                    " can leave its CTAs partly resident and waiting for ever:"
                    " width-%d steps take the original path",
                    self._what,
                    width,
                    scratch,
                    width,
                )
                return False
        return True

    def __getitem__(self, width: int):
        return self._built[width]
