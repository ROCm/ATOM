# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A runner's kernels, one build per step width."""

import logging

import torch.distributed as dist

from atom.mono.runtime.compile import compile_kernels, kernel_names, scratch_bytes
from atom.mono.runtime.consensus import MonoUnsupported, tp_agree

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
    resident CTAs then waiting for ever on the rest. ``lean(built, spilled)``,
    when given, first rebuilds the kernels whose launchers spilled with fewer
    registers (a build returned with those swapped, or None when it has none
    leaner), every rank together: the width is refused only if those spill too."""

    def __init__(self, build, kernels, group, what: str, lean=None) -> None:
        self._build, self._kernels, self._lean = build, kernels, lean
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
        built, scratch = self._compiled(width, lambda: self._build(width))
        spilled = [launcher for launcher, n in scratch.items() if n]
        # any rank's spill sends every rank through the lean rebuild together,
        # if every rank has a leaner build (``lean`` None: nothing to swap)
        if self._lean is not None and not tp_agree(not spilled, self._group):
            lean = self._lean(built, spilled) if spilled else built
            if tp_agree(lean is not None, self._group):
                logger.info(
                    "%s: %s need scratch, rebuilt lean",
                    self._what,
                    ", ".join(sorted({n for f in spilled for n in kernel_names(f)})),
                )
                built, scratch = self._compiled(width, lambda: lean)
        if built is not None and not self._scratch_free(scratch):
            built = None
        if not tp_agree(built is not None, self._group):
            self._refused.add(width)
            return False
        self._built[width] = built
        names = sorted({n for launcher in scratch for n in kernel_names(launcher)})
        logger.info("%s kernels built: %s", self._what, ", ".join(names))
        return True

    def _compiled(self, width: int, make) -> tuple:
        """``make()``'s build and its compiled launchers' scratch bytes a lane
        (``{launcher: bytes}``), or ``(None, {})`` if building, loading or
        reading a binary failed on any rank (collective)."""
        built, kernels = None, []
        try:
            built = make()
            kernels = self._kernels(built)
            compile_kernels(self._share(kernels))
        except MonoUnsupported as why:  # a build that does not fit, named
            logger.warning("%s: width %d refused: %s", self._what, width, why)
            built = None
        except Exception:
            logger.exception(
                "%s: building the width-%d kernels failed", self._what, width
            )
            built = None
        # every rank's share in FlyDSL's disk cache before any rank reads it
        if not tp_agree(built is not None, self._group):
            return None, {}
        try:
            launchers = compile_kernels(kernels)
            return built, {launcher: scratch_bytes(launcher) for launcher in launchers}
        except Exception:
            logger.exception(
                "%s: loading the width-%d kernels failed", self._what, width
            )
            return None, {}

    def _share(self, kernels: list) -> list:
        """This rank's share of a width's kernels to compile first (rank r:
        every ``world``-th from r). The ranks build the same binaries, so
        compiling them all in one order put every rank on one FlyDSL cache lock
        a kernel at a time; shared out, they compile side by side and each
        rank then loads the rest from the disk cache."""
        if self._group is None:
            return kernels
        rank = dist.get_rank(self._group)
        return kernels[rank :: dist.get_world_size(self._group)]

    def _scratch_free(self, scratch: dict) -> bool:
        for launcher, n in scratch.items():
            if n:
                logger.warning(
                    "%s: %s needs %d B of scratch a lane, which can leave its"
                    " CTAs partly resident and waiting for ever: its width's"
                    " steps take the original path",
                    self._what,
                    ", ".join(kernel_names(launcher)),
                    n,
                )
                return False
        return True

    def __getitem__(self, width: int):
        return self._built[width]
