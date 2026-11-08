# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hooks a model may give ``ModelRunner``: what the runner calls, where.

A model takes over part of a step by exposing ``verify_head`` and / or
``decode_round``, instances of the classes below. ``ModelRunner`` resolves them
once, when the model is installed (``resolve``), into its ``verify_head`` /
``decode_round`` attributes: None for every model without them, which then
runs exactly as before. DeepSeek-V4.1's hi path
(``atom/models/deepseek_v41/mono/``) is the one model that has them.
"""

from __future__ import annotations

from collections.abc import Callable

from atom.utils.forward_context import RoundService


class VerifyHead:
    """A head inside the model's forward that picks each row of a speculative
    verify, so the runner reads ids instead of logits."""

    def before_forward(self, runner, batch) -> None:
        """Before the forward (``ModelRunner.run_model``): the step's sampling
        rows, if the head draws them."""

    def ids(self, runner, batch, hidden_states, graph_key=None):
        """After the forward: this verify's ids, int32 [rows], or None (the
        logits path). ``graph_key``: the replayed graph's request bucket."""
        return None

    def captured(self, graph_key) -> None:
        """After a graph capture of request bucket ``graph_key``."""

    def verdict(self, spec, ids, sampler):
        """``postprocess`` on a step whose ids came from the head: (output
        ids, verdict). The runner broadcasts the output ids over TP and takes
        each request's next id from them, as on the logits path."""
        raise NotImplementedError


class DecodeRound:
    """Host work of the decode round the model takes over, called where that
    work sits. Every method defaults to "nothing taken over"."""

    # ---- ModelRunner and its token processor

    def before_input_ids(self, batch, max_seqlen_q) -> None:
        """First thing in ``prepare_input_ids``: the step's batch."""

    def spec_status(self):
        """``prepare_input_ids``' speculative status (rejected, bonus counts,
        a request each of the previous forward), when the round knows it
        without the queue's wait (taken later: ``after_forward``); None: the
        queue's."""
        return None

    def decode_ids_filler(self) -> Callable | None:
        """The fill of this step's deferred decode ids, if the round does it
        (``fill_deferred_decode_ids``' arguments); None: the fill kernel."""
        return None

    def host_arena_reader(self, publication: str):
        """The ``h2d.HostArenaReader`` the packed publication ``publication``
        opts into (``PublicationGroup.read_host_arena``), or None."""
        return None

    def before_staging_mark(self) -> None:
        """Before the owner's staging-reuse event is recorded."""

    def after_forward(self, token_processor) -> None:
        """After the target forward's launch (``ModelRunner.forward``)."""

    def note_verdict(self, out) -> None:
        """After a verify's rejection sampler: its (tokens, verdict)."""

    # ---- the attention metadata builder

    def service(
        self,
        batch,
        spans,
        running_bs,
        running_tokens,
        *,
        max_q_len,
        decode,
        tentative,
        is_prefill,
        resumes,
        image_rows,
    ) -> RoundService:
        """Before a step's layout: how much of it the round takes over
        (``AttentionMetaData.served_by_round``)."""
        return RoundService.NONE

    def laid_out(self, metadata) -> None:
        """After the step's layout and publication."""

    def committed(self, metadata) -> bool:
        """Before the speculative commit of a served step: True when the
        forward committed it already."""
        return False


def resolve(model) -> tuple[VerifyHead | None, DecodeRound | None]:
    """The installed ``model``'s hooks (its ``verify_head`` / ``decode_round``),
    each None when it has none."""
    found = []
    for name, kind in (("verify_head", VerifyHead), ("decode_round", DecodeRound)):
        hook = getattr(model, name, None)
        if hook is not None and not isinstance(hook, kind):
            raise TypeError(f"{type(model).__name__}.{name} is not a {kind.__name__}")
        found.append(hook)
    return found[0], found[1]
