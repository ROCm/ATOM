# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The single launch's decode round for V4.1's metadata builder (``model_hooks.DecodeRound``,
``DeepseekV41MetadataBuilder.use_decode_round``): the hi path's device round (``round.device_round``) on the steps the
single launch certainly serves.

The launch derives every input that depends on the previous verdict from the step's positions on the device (the
visible counts and the compress / write plans: ``step_plans``; the window rows, key metadata and RoPE rows already
were). So on a step the launch serves:

- ``LEAN``: the builder skips the host's compress plans and step planners (nothing the launch runs reads them);
- ``DEVICE`` (the same request as the served step before it, which left a verdict): the host lays the step out as if no
  draft had been rejected, ``device_round.advance`` rewrites its positions from that verdict on the stream before any
  reader, and the runner takes the previous status after the launch -- the host prepares the next step while the
  current one runs (TileRT's host enqueueing ahead).

"Certainly serves" is decided on the host before the layout, from the scheduler's batch (TP-uniform): the request
shapes ``SingleStepModel.single_step_runner`` and ``StepInputs.precheck`` serve. A step decided here that the launch
then declines raises (``SingleStepModel.step_override``): its layout has no plans for the production path.

The launch reads Engram's staged rows (``looks_up_engram_rows`` False): the staging runs in full on its steps.
"""

from __future__ import annotations

import numpy as np

from atom.model_engine.model_hooks import DecodeRound
from atom.models.deepseek_v41.mono.gfx1250.config import MAX_REQUESTS, ROWS_PER_REQ
from atom.models.deepseek_v41.mono.gfx1250.tables import BCOLS, ILW
from atom.utils.forward_context import RoundService


class _Hook:
    """What ``device_round`` reads of a hook: the launch serves every layer, from its own plans (``lean_host``);
    ``pos``: each request's first position of the last served step (the runner's, written by its preparation).
    """

    lean_host = True
    runs_every_layer = True
    bmax = MAX_REQUESTS

    def __init__(self, model) -> None:
        self.model = model
        self.split = {
            n * ROWS_PER_REQ: (n, ROWS_PER_REQ) for n in range(1, MAX_REQUESTS + 1)
        }

    def ready(self, rows) -> bool:
        """The launch serves ``rows``: its runner exists (built by the first step it served, inside the forward that
        has the step's metadata; never from here) and the width is built."""
        return self.model.single_runner is not None and self.model.step_ready(rows)

    @property
    def served_widths(self):
        return {w for w in self.split if self.ready(w)}

    @property
    def pos(self):
        return self.model.single_runner.first_positions


class SingleStepRound(DecodeRound):
    looks_up_engram_rows = False

    def __init__(self, model, metadata_builder) -> None:
        # (imported here: it imports Triton)
        from atom.models.deepseek_v41.mono.round import device_round

        self.model, self.builder = model, metadata_builder
        self.hook = _Hook(model)
        self.device = device_round.install(metadata_builder, self.hook)

    def _serves(self, batch, max_q_len) -> bool:
        """The batch is one the launch serves (``single_step_runner``'s and ``precheck``'s batch conditions)."""
        if not hasattr(batch, "any_logprobs"):
            # a graph capture's synthetic batch: it records the production path
            return False
        n = batch.total_seqs_num
        if (
            batch.is_dummy_run
            or getattr(batch, "total_seqs_num_prefill", 0)
            or not 1 <= n <= MAX_REQUESTS
            or max_q_len != ROWS_PER_REQ
            or getattr(batch, "structured_requests", None)
            or batch.any_logprobs
            or getattr(batch, "required_channel", None) is not None
            or getattr(batch, "multimodal_data", None)
            or getattr(batch, "replaying", None)
            or (np.asarray(batch.top_ks[:n]) != -1).any()
        ):
            return False
        return self.hook.ready(n * ROWS_PER_REQ)

    def consumes_device_verdict(self, batch, max_seqlen_q) -> bool:
        if self.device is None:
            return False
        # every forward: the device round keeps its history of the requests it served
        on = self.device.consumes(batch, max_seqlen_q)
        if on and not self._serves(batch, max_seqlen_q):
            state = self.device.state
            state.on = state.first = False
            state.advance_from = None
            on = False
        return on

    def service(self, batch, spans, running_bs, running_tokens, *, max_q_len, decode, tentative, is_prefill,
                resumes, image_rows) -> RoundService:  # fmt: skip
        served = (
            decode
            and tentative
            and not is_prefill
            and not image_rows
            and resumes is None
            and running_tokens == running_bs * ROWS_PER_REQ
            and len(spans) <= 8
            and self._serves(batch, max_q_len)
        )
        if served:
            end = max((span.end for span in spans), default=1)
            block = self.builder.cache.geometry.block_size
            served = end <= ILW and -(-end // block) <= BCOLS
        on = self.device is not None and self.device.state.on
        if not served:
            if on:
                raise RuntimeError(
                    "V4.1 single-launch step: a device-round step the launch does not serve"
                )
            return RoundService.NONE
        return RoundService.DEVICE if on else RoundService.LEAN

    def laid_out(self, metadata) -> None:
        if self.device is not None:
            self.device.laid_out(metadata)

    def committed(self, metadata, anchors) -> bool:
        if self.device is not None:
            self.device.committed(metadata, anchors)
        return False
