# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi path's host work of a decode round: the decode round its install gives V4.1's metadata builder
(``model_engine.model_hooks.DecodeRound``, ``DeepseekV41MetadataBuilder.use_decode_round``).

One object (``HiRound``) answers every point the builder asks a decode round at:

- ``consumes_device_verdict`` before the step's input ids, every forward: whether the step advances from the previous
  verdict on the device (``device_round``), the builder's answer to the runner (which then defers its status wait);
- ``service`` before a step's layout, the step's one routing decision (``AttentionMetaData.served_by_round``:
  ``FORWARD`` a step the hook's forward serves, ``LEAN`` laid out without what only ATOM's replaced layers read,
  ``lean_host``; ``DEVICE`` its positions advanced on the device, ``device_round``), which also publishes a served
  step's sampling rows for the verify head (``HiLayers.set_head_sampling``);
- ``laid_out`` after it (the device round's advance);
- ``committed`` before a served step's speculative commit, at the anchors the runner commits: True when they are the
  in-graph verdict's (``post_verify``: the runner took the forward's head output, and the kernel committed them),
  which also hands the device round the verdict its next step advances from.

``HiDecodeModel.forward``, Engram staging and the DSpark proposer read the builder's decision
(``served_by_round``), so they agree on every step. ``ModelRunner`` calls none of it: it takes the forward's head
output (``HiDecodeModel.compute_head``) and asks the builder whether to defer its status wait.
"""

from __future__ import annotations

import logging

from atom.model_engine.model_hooks import DecodeRound
from atom.models.deepseek_v41.mono.round import lean_host
from atom.utils.forward_context import RoundService

logger = logging.getLogger("atom")


class HiRound(DecodeRound):
    """The decode round (module docstring) over its parts: ``device`` (``device_round.DeviceRound``), ``pv``
    (``post_verify.PostVerify``)."""

    def __init__(self, hook, device=None, pv=None) -> None:
        self.hook, self.device, self.pv = hook, device, pv
        self.rows_sampling = False  # the head's rows hold a sampled step's (T, top_p)

    def consumes_device_verdict(self, batch, max_seqlen_q) -> bool:
        return self.device is not None and self.device.consumes(batch, max_seqlen_q)

    def service(self, batch, spans, running_bs, running_tokens, *, max_q_len, decode, tentative, is_prefill,
                resumes, image_rows) -> RoundService:  # fmt: skip
        """The step's routing decision, before its layout. ``FORWARD``: a decode at one of the hook's widths and its
        query bucket (``HiLayers.serves_width``), no image rows; ``LEAN`` / ``DEVICE`` on top of it (module
        docstring). Host-side and TP-uniform (the scheduler's batch and the forward's shape). A ragged decode at a
        batch width raises (``lean_host.check_batch_layout``). A served step's sampling rows go up with it
        (``_publish_sampling``)."""
        hook = self.hook
        if not is_prefill:
            lean_host.check_batch_layout(hook, spans, running_tokens, max_q_len)
        on = self.device is not None and self.device.state.on
        if (
            is_prefill
            or not decode
            or image_rows
            or not hook.serves_width(running_tokens, max_q_len)
        ):
            service = RoundService.NONE
        elif not lean_host.lean_step(hook, batch, spans, running_bs, running_tokens, max_q_len, tentative,
                                     is_prefill, resumes):  # fmt: skip
            service = RoundService.FORWARD
        else:
            service = RoundService.DEVICE if on else RoundService.LEAN
        if on and service is not RoundService.DEVICE:
            raise RuntimeError(
                "V4.1 mono: a device-round step whose builder did not take the lean path"
            )
        if service:
            self._publish_sampling(batch)
        return service

    def _publish_sampling(self, batch) -> None:
        """A served step's rows' (T, top_p) for the verify head, request by request on its rows
        (``HiLayers.set_head_sampling``: staged only when they changed). The runner takes the head's ids where they
        stand for the step (``ModelRunner._head_ids_serve``: every row greedy, or sampled from T / top-p alone); a step
        whose rows are all greedy (a capture's too) costs the first test only."""
        temperatures = getattr(batch, "temperatures", None)
        if temperatures is None or not temperatures.any():
            if self.rows_sampling:
                self.hook.set_head_sampling(None, None, None)
                self.rows_sampling = False
            return
        self.rows_sampling = bool(self.hook.set_head_sampling(temperatures, batch.top_ps,
                                                              batch.num_scheduled_tokens))  # fmt: skip

    def laid_out(self, metadata) -> None:
        if metadata.served_by_round >= RoundService.LEAN:
            self.hook.lean_steps += 1
        if self.device is not None:
            self.device.laid_out(metadata)

    def committed(self, metadata, anchors) -> bool:
        if self.device is not None:
            self.device.committed(metadata, anchors)
        return self.pv is not None and self.pv.committed(metadata, anchors)


def install(hook, metadata_builder) -> HiRound:
    """``hook``'s decode round, for ``metadata_builder`` (V4.1's: ``use_decode_round``)."""
    # (imported here: ``service`` imports without Triton)
    from atom.models.deepseek_v41.mono.round import device_round

    lean_host.install(hook)
    device = device_round.install(metadata_builder, hook)
    return HiRound(hook, device=device, pv=getattr(hook, "pv", None))
