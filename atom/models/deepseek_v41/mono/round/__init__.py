# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi path's host work of a decode round: the model's ``decode_round`` (``model_engine.model_hooks.DecodeRound``).

One object (``HiRound``) answers every point the engine calls a decode round at:

- V4.1's metadata builder: ``service`` before a step's layout, the step's one routing decision
  (``AttentionMetaData.served_by_round``: ``FORWARD`` a step the hook's forward serves, ``LEAN`` laid out without what
  only ATOM's replaced layers read, ``lean_host``; ``DEVICE`` its positions advanced on the device, ``device_round``);
  ``laid_out`` after it (the device round's advance); ``committed`` before a served step's speculative commit (the
  in-graph commit, ``post_verify``);
- ``ModelRunner``: ``before_input_ids`` / ``spec_status`` in ``prepare_input_ids`` and ``after_forward`` after the
  target's launch (the device round's verdict wait), ``note_verdict`` after the rejection sampler (and after the hook's
  own verdict, ``post_verify.verdict``), ``decode_ids_filler`` / ``before_staging_mark`` (the prologue's recorder,
  ``prologue``), ``host_arena_reader`` (a served decode step's publications' scatter reads their pinned arenas
  directly).

``HiDecodeModel.forward``, Engram staging and the DSpark proposer read the builder's decision
(``served_by_round``), so they agree on every step.
"""

from __future__ import annotations

import logging

import torch

from atom.model_engine.model_hooks import DecodeRound
from atom.models.deepseek_v41.mono.round import lean_host
from atom.utils.forward_context import RoundService

logger = logging.getLogger("atom")


class HiRound(DecodeRound):
    """The decode round (module docstring) over its parts: ``device`` (``device_round.DeviceRound``), ``pv``
    (``post_verify.PostVerify``), ``pro`` (``prologue.Prologue``)."""

    def __init__(self, hook, device=None, pv=None, pro=None) -> None:
        self.hook, self.device, self.pv, self.pro = hook, device, pv, pro

    # ---- V4.1's metadata builder

    def service(self, batch, spans, running_bs, running_tokens, *, max_q_len, decode, tentative, is_prefill,
                resumes, image_rows) -> RoundService:  # fmt: skip
        """The step's routing decision, before its layout. ``FORWARD``: a decode at one of the hook's widths and its
        query bucket (``HiLayers.serves_width``), no image rows; ``LEAN`` / ``DEVICE`` on top of it (module
        docstring). Host-side and TP-uniform (the scheduler's batch and the forward's shape). A ragged decode at a
        batch width raises (``lean_host.check_batch_layout``)."""
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
        if self.pro is not None:
            # the step metadata's publication, next: zero-copy on a served step only
            self.pro.serve(service is not RoundService.NONE)
        return service

    def laid_out(self, metadata) -> None:
        if metadata.served_by_round >= RoundService.LEAN:
            self.hook.lean_steps += 1
        if self.device is not None:
            self.device.laid_out(metadata)

    def committed(self, metadata) -> bool:
        return self.pv is not None and self.pv.committed(metadata)

    # ---- ModelRunner

    def before_input_ids(self, batch, max_seqlen_q) -> None:
        if self.device is not None:
            self.device.before_input_ids(batch, max_seqlen_q)

    def spec_status(self):
        return None if self.device is None else self.device.spec_status()

    def decode_ids_filler(self):
        pro = self.pro
        if pro is None or not pro.armed or torch.cuda.is_current_stream_capturing():
            return None
        return pro.fill

    def host_arena_reader(self, publication: str):
        from atom.models.deepseek_v41.mono.round import prologue

        if self.pro is None or publication not in prologue.DECODE_PUBLICATIONS:
            return None
        return self.pro

    def before_staging_mark(self) -> None:
        if self.pro is not None:
            self.pro.flush()

    def after_forward(self, token_processor) -> None:
        if self.device is not None:
            self.device.after_forward(token_processor)

    def note_verdict(self, out) -> None:
        if self.device is not None:
            self.device.note_verdict(out)


def install(hook, metadata_builder, runner) -> HiRound:
    """``hook``'s decode round (``HiDecodeModel.decode_round``; the post-verify reads it from ``hook.round``)."""
    # (imported here: ``service`` and ``verify_head`` import without Triton)
    from atom.models.deepseek_v41.mono.round import device_round

    lean_host.install(hook)
    device = device_round.install(metadata_builder, hook, runner)
    rnd = HiRound(hook, device=device, pv=getattr(hook, "pv", None),
                  pro=None if device is None else device.pro)  # fmt: skip
    hook.round = rnd
    return rnd
