# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Carry a block drafter's target aux hidden states across PP stage boundaries.

DSpark projects its drafting context from a handful of target layers spread
over the whole stack -- Kimi-K3 taps 2, 23, 47, 71 and 89 of 93. The drafter
itself is built only on the LAST pipeline stage (``ModelRunner.__init__`` gates
on ``get_pp_group().is_last_rank``), so at pp2 the first two taps sit on a stage
that has no drafter, and the hooks ``Drafter.arm_aux_capture`` installs land on
``PPMissingLayer`` placeholders, which are never called.

Nothing fails. ``aux_for`` hands the proposer its zero-initialized buffers, the
context projection runs on a tensor that is two fifths zeros, and the draft is
noise: measured on GSM8K at pp2+tp4, acceptance 0.01% over 28k drafted tokens
against ~80% for the same checkpoint at tp8.

So the earlier stages capture their own taps and ship them forward. The rows
ride the existing intermediate-tensor payload under one key, concatenated along
the hidden axis in ``layer_ids`` order -- both sides derive the same column
layout from the same tuple, so the split needs no metadata of its own.

A middle stage relays what it received AND what it captured, which is exactly
"every tap below my ``end_layer``". That falls out of the id partition: a tap
belongs to exactly one stage, ids are strictly increasing, and a stage holds a
contiguous layer range.
"""

import logging

import torch
from aiter.dist.parallel_state import get_pp_group

from atom.models.utils import PPMissingLayer
from atom.spec_decode.drafter import AuxCaptureSpec, _resolve_decoder_layers

logger = logging.getLogger("atom")


class PPAuxRelay:
    """Per-stage half of the aux transfer. One instance per ModelRunner.

    Construct on every stage when speculation is on and pp > 1; the stage's role
    follows from which taps are resident. ``incoming_ids`` and ``own_ids``
    together are the columns this stage sends, in ``layer_ids`` order.
    """

    def __init__(
        self,
        spec: AuxCaptureSpec,
        target_model: torch.nn.Module,
        max_num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        pp = get_pp_group()
        layers = _resolve_decoder_layers(target_model)
        resident = [
            lid
            for lid in spec.layer_ids
            if 0 <= lid < len(layers) and not isinstance(layers[lid], PPMissingLayer)
        ]
        # A tap on the embedding (-1) is produced wherever the embedding is,
        # which is the first stage. Treat it as resident there and relayed
        # everywhere else, same as any layer tap.
        if -1 in spec.layer_ids and pp.is_first_rank:
            resident.insert(0, -1)

        self.spec = spec
        self.layer_ids = spec.layer_ids
        self.hidden_size = spec.hidden_size
        self.own_ids = tuple(resident)
        # Everything below this stage's lowest resident tap comes from upstream.
        # With no resident tap at all, every tap below the stage's own layer
        # range does -- and a stage holding none of them still has to pass on
        # what it was handed.
        first_own = self.own_ids[0] if self.own_ids else None
        self.incoming_ids = tuple(
            lid
            for lid in self.layer_ids
            if lid not in self.own_ids and (first_own is None or lid < first_own)
        )
        self.outgoing_ids = (
            () if pp.is_last_rank else self.incoming_ids + self.own_ids
        )

        # Buffers only where this stage captures but cannot consume: on the last
        # stage the drafter already owns a buffer per tap and the hooks write
        # straight into those.
        self.buffers: dict[int, torch.Tensor] = {}
        if not pp.is_last_rank:
            for lid in self.own_ids:
                self.buffers[lid] = torch.zeros(
                    max_num_tokens, spec.hidden_size, device=device, dtype=dtype
                )

        self._armed = False

    def arm(self, target_model: torch.nn.Module, is_draft_forward) -> None:
        """Install capture hooks for this stage's taps. No-op on the last stage,
        whose drafter armed its own.

        ``is_draft_forward`` is a callable rather than the flag itself: the hook
        has to re-read it at every call. Read once at arm time it would freeze
        whatever the setup step happened to see.
        """
        if self._armed or get_pp_group().is_last_rank or not self.own_ids:
            return
        self._armed = True
        layers = _resolve_decoder_layers(target_model)
        for lid in self.own_ids:
            module = layers[lid]
            hook = self._make_hook(self.buffers[lid], self.spec.extract, is_draft_forward)
            if self.spec.capture == "input":

                def pre_hook(module, inputs, hook=hook):
                    hook(module, (), inputs)

                module.register_forward_pre_hook(pre_hook)
            else:
                module.register_forward_hook(hook)
        logger.info(
            "PPAuxRelay: capturing target layers %s, relaying %s forward",
            list(self.own_ids),
            list(self.outgoing_ids),
        )

    @staticmethod
    def _make_hook(buffer, extract, is_draft_forward):
        def _hook(module, _inputs, output):
            if is_draft_forward():
                return
            tensor = extract(output, module)
            if tensor is None:
                return
            buffer[: tensor.shape[0]].copy_(tensor)

        return _hook

    def pack(self, received: torch.Tensor | None, num_tokens: int):
        """The payload to send on, or None when this stage relays nothing.

        Columns are ``outgoing_ids`` in order: the received block first (already
        in ``incoming_ids`` order) then this stage's own captures.
        """
        if not self.outgoing_ids:
            return None
        parts = []
        if self.incoming_ids:
            if received is None:
                raise RuntimeError(
                    f"PPAuxRelay expected aux rows for target layers "
                    f"{list(self.incoming_ids)} from the previous stage, got none."
                )
            parts.append(received[:num_tokens])
        parts.extend(self.buffers[lid][:num_tokens] for lid in self.own_ids)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def absorb(self, received: torch.Tensor | None, aux_buffers: list) -> None:
        """Write the relayed rows into the drafter's own capture buffers.

        ``aux_buffers`` is ``Drafter._aux_buffers``, indexed by position in
        ``layer_ids`` -- the same order the columns were packed in.
        """
        if not self.incoming_ids:
            return
        if received is None:
            raise RuntimeError(
                f"PPAuxRelay expected aux rows for target layers "
                f"{list(self.incoming_ids)} from the previous stage, got none."
            )
        n = received.shape[0]
        for col, lid in enumerate(self.incoming_ids):
            start = col * self.hidden_size
            buf = aux_buffers[self.layer_ids.index(lid)]
            buf[:n].copy_(received[:, start : start + self.hidden_size])
