# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A model's mono runner, as its routing wrapper holds it."""

import logging

import torch

from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")


class LazyRunner:
    """``create()`` on the first eligible step outside a graph capture (a warmup
    forward: allocation and the peer handshake cannot be captured), then each
    step width prepared on its first step (``runner.prepare(rows)``).

    The two refusals are NOT the same. ``create()`` failing is a deployment
    verdict -- the model or the runtime cannot serve mono at all -- and turns
    mono off for good. ``prepare(width)`` failing is about that one width, and
    must leave every other width alone: a step that pads to an uncaptured
    width would otherwise disable mono for the widths that do build. Both
    calls are collective, so every rank decides alike (the routing inputs must
    be TP-uniform)."""

    def __init__(self, create, what: str) -> None:
        self._create, self.what = create, what
        self.runner = None
        self.enabled = True
        # widths already reported; the runners memoize their own refusals, so
        # without this the per-step retry would log the same line every step
        self._reported: set[tuple] = set()

    def ready(self, rows: int, *shape) -> bool:
        """Whether this ``rows``-row step runs on mono. ``shape``: anything else
        that picks the build (a model whose kernels specialize on more than the
        row count passes it through to its ``prepare``)."""
        if not self.enabled:
            return False
        if self.runner is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            try:
                self.runner = self._create()
            except MonoUnsupported as why:
                # exc_info: a refusal raised deliberately carries no cause, but
                # one converted from another failure (consensus.bind_agreed
                # chains the original) would otherwise lose its traceback and
                # read as a plain "mono is off" line.
                logger.warning(
                    "%s off: %s", self.what, why, exc_info=why.__cause__ is not None
                )
                self.enabled = False
                return False
            logger.info("%s on", self.what)
        if not self.runner.prepare(rows, *shape):
            # This width only. `prepare` memoizes its refusals, so the retry on
            # the next step of the same width is a set lookup, and a width that
            # does build is unaffected.
            key = (rows, *shape)
            if key not in self._reported:
                self._reported.add(key)
                logger.warning(
                    "%s declined the %d-row build; other widths unaffected",
                    self.what,
                    rows,
                )
            return False
        return True
