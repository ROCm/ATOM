# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi path's host work of a decode round, at the engine's explicit decode-round hooks.

One object (``HiRound``) answers every hook the engine calls a model's decode round at:

- V4.1's metadata builder (``DeepseekV41MetadataBuilder.decode_round``): ``lean_layout`` before a step's layout (the
  lean host prep, ``lean_host``), ``laid_out`` after it (the lean mark and the device round's advance,
  ``device_round``, in that order), ``committed`` before the speculative commit (the
  in-graph commit, ``post_verify``);
- ``ModelRunner`` (the model's ``decode_round``, ``hi_dispatch.HiDecodeModel``): ``before_input_ids`` /
  ``spec_status`` in ``prepare_input_ids`` and ``after_forward`` after the target's launch (the device round's
  verdict wait), ``note_verdict`` after the rejection sampler (and after the hook's own verdict,
  ``post_verify.verdict``), ``fill_decode_ids`` / ``before_staging_mark`` (the prologue's recorder, ``prologue``),
  ``publications_ready`` (the packed publications read their pinned arenas directly);
- the DSpark proposer (``DSparkProposer.anchors_staged``): the in-graph anchor staging (``post_verify``).

Each part is None when its own refusal holds (logged at install), and then that hook does what the engine does
without it.
"""

from __future__ import annotations

import logging

import torch

from atom.models.deepseek_v41.mono.round import device_round, lean_host

logger = logging.getLogger("atom")


class HiRound:
    """The decode round's hooks (module docstring) over its parts: ``lean`` (``lean_host.LeanHost``), ``device``
    (``device_round.DeviceRound``), ``pv`` (``post_verify.PostVerify``), ``pro`` (``prologue.Prologue``).
    """

    def __init__(self, lean=None, device=None, pv=None, pro=None) -> None:
        self.lean, self.device, self.pv, self.pro = lean, device, pv, pro

    # ---- V4.1's metadata builder

    def lean_layout(self, batch, spans, running_bs, running_tokens, **kw) -> bool:
        return self.lean is not None and self.lean.lean_layout(
            batch, spans, running_bs, running_tokens, **kw
        )

    def laid_out(
        self, batch, spans, rows, running_bs, running_tokens, metadata, lean: bool
    ) -> None:
        if self.lean is not None:
            self.lean.laid_out(metadata, lean)
        if self.device is not None:
            self.device.laid_out(running_bs, running_tokens, metadata)

    def committed(self, metadata) -> bool:
        return self.pv is not None and self.pv.committed(metadata)

    # ---- ModelRunner

    def before_input_ids(self, batch, max_seqlen_q) -> None:
        if self.device is not None:
            self.device.before_input_ids(batch, max_seqlen_q)

    def spec_status(self):
        return None if self.device is None else self.device.spec_status()

    def fill_decode_ids(
        self, out, cu, src, prev, draft, *, max_tokens_per_seq, width
    ) -> bool:
        return (
            self.pro is not None
            and self.pro.armed
            and not torch.cuda.is_current_stream_capturing()
            and self.pro.fill(
                out,
                cu,
                src,
                prev,
                draft,
                max_tokens_per_seq=max_tokens_per_seq,
                width=width,
            )
        )

    def before_staging_mark(self) -> None:
        if self.pro is not None:
            self.pro.flush()

    def after_forward(self, processor) -> None:
        if self.device is not None:
            self.device.after_forward(processor)

    def note_verdict(self, out) -> None:
        if self.device is not None:
            self.device.note_verdict(out)

    def publications_ready(self, runner) -> None:
        """Every packed H2D publication of the runner (``PackedCopy``: the step's inputs, V4.1's step metadata) has
        its scatter read the pinned arena through its device address on eager launches, instead of after an H2D
        copy: one launch a publication instead of two (``PackedCopy.read_host_arena``).
        """
        names = []
        for groups in runner._h2d_groups:
            for name, group in groups.items():
                if group._backend is not None:
                    group._backend.read_host_arena()
                    names.append(name)
        logger.info(
            "V4.1 mono hi: packed H2D publications %s read their pinned arenas directly",
            sorted(set(names)),
        )

    # ---- the DSpark proposer

    def anchors_staged(self, running_bs: int) -> bool:
        return self.pv is not None and self.pv.anchors_staged(running_bs)


def install(hook, metadata_builder, runner, drafter) -> HiRound:
    """``hook``'s decode round, set on the builder (``decode_round``) and the proposer (``anchors_staged``); the model
    (``HiDecodeModel.decode_round``) and the post-verify read it from ``hook.round``."""
    lean = lean_host.install(hook)
    device = device_round.install(metadata_builder, hook, runner)
    rnd = HiRound(lean=lean, device=device, pv=getattr(hook, "pv", None),
                  pro=None if device is None else device.pro)  # fmt: skip
    hook.round = rnd
    metadata_builder.decode_round = rnd
    if drafter is not None and hasattr(drafter, "anchors_staged"):
        drafter.anchors_staged = rnd.anchors_staged
    return rnd
