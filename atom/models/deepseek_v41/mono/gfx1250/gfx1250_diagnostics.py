# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bounded deferred mono acceptance using the existing output D2H queue."""

import torch


def _diagnostics(cache, rows):
    stores = cache.__dict__.setdefault("_gfx1250_k1_diagnostics", {})
    if "_gfx1250_diagnostic_sticky" not in cache.__dict__:
        cache._gfx1250_diagnostic_sticky = torch.zeros(
            1, dtype=torch.int32, device=cache.pool.device
        )
    if rows not in stores:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K1 runner] warm diagnostics before capture")
        n = cache.geometry.layers + 1
        stores[rows] = (
            torch.zeros(n, dtype=torch.int32, device=cache.pool.device),
            torch.empty(n, dtype=torch.int32, device="cpu", pin_memory=True),
        )
    return stores[rows]


def _raise_diagnostic_values(cache, values):
    if values[-1]:
        cache._gfx1250_k1_failed = True
        local = [(i, code) for i, code in enumerate(values[:-1]) if code]
        context = getattr(cache, "_gfx1250_decode", None)
        detail = context.sync_detail.cpu().tolist() if context is not None else None
        raise RuntimeError(
            f"[gfx1250 mhtjstest] TP diagnostic {values[-1]}, local {local}; "
            f"local wait [code,wanted,observed,source,producer_epoch,stage,published_stage,reserved,norm_wave0..7]: {detail}; "
            "discard output and stop dispatch"
        )


class DeferredDiagnostics:
    """The existing D2HQueue, bounded to two snapshots and one deferred step.

    The current step's copy overlaps CPU input preparation; before publishing
    deferred tokens the preceding step is accepted. A device sticky summary
    survives row-width switches and preserves failure until the host drains it.
    Every source is protected from the next replay by a GPU copy-done event.
    """

    def __init__(self, stream, layers):
        from collections import deque

        from atom.model_engine.output_transfer import D2HQueue

        self.queue = D2HQueue(layers + 1, stream, max_pending=2)
        self.caches = deque()
        self.accept_count = 0
        self.staged = False

    def stage(self, metadata, rows):
        if self.staged:
            raise RuntimeError(
                "[gfx1250 mhtjstest] diagnostics must be accepted once per output"
            )
        self.accept_count = len(self.caches)
        if metadata.step.decode and rows in (1, 2):
            cache = metadata.cache
            codes, _ = _diagnostics(cache, rows)
            self.queue.send(codes, protect_source=True)
            self.caches.append(cache)
        self.staged = True

    def accept(self):
        if not self.staged:
            raise RuntimeError("[gfx1250 mhtjstest] no staged diagnostic output")
        for _ in range(self.accept_count):
            values = self.queue.receive()
            cache = self.caches.popleft()
            _raise_diagnostic_values(cache, values)
        self.staged = False

    def poll_completed(self):
        while self.caches and self.queue._pending[0].copy_done.query():
            values = self.queue.receive()
            cache = self.caches.popleft()
            _raise_diagnostic_values(cache, values)

    def drain(self):
        while self.caches:
            values = self.queue.receive()
            cache = self.caches.popleft()
            _raise_diagnostic_values(cache, values)
        self.staged = False
