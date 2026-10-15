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
    step width prepared on its first step (``runner.prepare(rows)``). Any refusal
    turns mono off for good; both calls are collective, so every rank decides
    alike (the routing inputs must be TP-uniform)."""

    def __init__(self, create, what: str) -> None:
        self._create, self.what = create, what
        self.runner = None
        self.enabled = True

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
            logger.warning("%s off: the %d-row kernel build", self.what, rows)
            self.enabled = False
            return False
        return True
