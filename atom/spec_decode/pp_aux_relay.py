# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Relay a block drafter's target aux hidden states across PP stages.

The drafter lives only on the last PP stage, so hooks on taps owned by earlier
stages land on ``PPMissingLayer`` and never fire, leaving zero aux buffers and
garbage drafts. Earlier stages capture their own taps and forward them in the
intermediate-tensor payload, concatenated along the hidden axis in the
spec's ``layer_ids`` order, which ``install_aux_taps`` requires strictly
increasing (-1, the embedding, first). Stages own contiguous ascending layer
ranges, so every stage derives the same columns from ``layer_ids`` alone.
"""

import logging

import torch
from aiter.dist.parallel_state import get_pp_group

from atom.models.utils import PPMissingLayer
from atom.spec_decode.drafter import (
    AuxCaptureSpec,
    _aux_hook,
    _resolve_decoder_layers,
    install_aux_taps,
    tap_is_resident,
)
from atom.spec_decode.factory import drafter_class

logger = logging.getLogger("atom")


class PPAuxRelay:
    """Per-stage aux relay, built on every stage when speculating with pp > 1.

    Sends ``incoming_ids + own_ids`` columns, in ascending tap order.
    """

    def __init__(
        self,
        spec: AuxCaptureSpec,
        target_model: torch.nn.Module,
    ) -> None:
        pp = get_pp_group()
        layers = _resolve_decoder_layers(target_model)
        stage_start = next(
            i for i, layer in enumerate(layers) if not isinstance(layer, PPMissingLayer)
        )
        taps = spec.layer_ids

        self.spec = spec
        self.own_ids = tuple(
            lid for lid in taps if tap_is_resident(target_model, layers, lid)
        )
        # Every tap below this stage's first layer was captured upstream, which
        # is exactly what the previous stage sent, even if it owned none itself.
        self.incoming_ids = tuple(
            lid
            for lid in taps
            if not tap_is_resident(target_model, layers, lid) and lid < stage_start
        )
        self.outgoing_ids = () if pp.is_last_rank else self.incoming_ids + self.own_ids

        # The send payload, allocated per forward at its row count (a resident
        # max_num_batched_tokens buffer is ~0.5 GB per rank) by the lowest own
        # tap, which fires first; every own tap captures into its columns and
        # pack() copies the received taps into the leading ones. The last
        # stage's hooks write into the drafter's own buffers instead.
        # The write reads no bookkeeping: inside the compiled forward Dynamo
        # freezes it to the warmup trace.
        # shortcut: a recorded graph would bake the capture-time buffer, so
        # this holds only while PP stages run their target eagerly.
        self._send: torch.Tensor | None = None
        self._col = {lid: col for col, lid in enumerate(self.outgoing_ids)}

    @classmethod
    def build(cls, config, target_model) -> "PPAuxRelay | None":
        """This stage's armed relay, or None when no tap crosses a PP boundary.

        Every stage takes the drafter class's config-only spec, so all agree on
        the columns; a draft with its own contract refuses PP when it arms.
        """
        if get_pp_group().world_size <= 1 or config.speculative_config is None:
            return None
        spec = drafter_class(config).aux_capture_spec(config)
        if spec is None:
            return None
        relay = cls(spec, target_model)
        if not relay.incoming_ids and not relay.outgoing_ids:
            return None
        relay.arm(target_model)
        return relay

    def arm(self, target_model: torch.nn.Module) -> None:
        """Install capture hooks for this stage's taps (no-op on the last stage,
        whose taps the drafter's own ``arm_aux_capture`` hooks)."""
        if get_pp_group().is_last_rank:
            return
        install_aux_taps(
            target_model,
            self.spec,
            self.own_ids,
            lambda lid, extract: _aux_hook(extract, self._capture_into(lid)),
        )
        logger.info(
            "PPAuxRelay: capturing target layers %s, relaying %s forward",
            list(self.own_ids),
            list(self.outgoing_ids),
        )

    def _capture_into(self, lid):
        h, col = self.spec.hidden_size, self._col[lid]
        width = len(self.outgoing_ids) * h

        opens = lid == self.own_ids[0]

        def write(tensor, _off):
            # TBO is refused under PP, so a forward is never split into ubatches.
            if opens:
                self._send = torch.empty(
                    tensor.shape[0], width, device=tensor.device, dtype=tensor.dtype
                )
            self._send[:, col * h : (col + 1) * h].copy_(tensor)

        return write

    def pack(self, received: torch.Tensor | None, num_tokens: int):
        """Payload to send on (received block, then own captures), or None."""
        if not self.outgoing_ids:
            return None
        if self.incoming_ids:
            self._check_received(received)
            if not self.own_ids:
                return received[:num_tokens]
        # Hand the step's buffer over: the send path owns it from here.
        send, self._send = self._send, None
        assert (
            send is not None and send.shape[0] == num_tokens
        ), f"PPAuxRelay: no capture of {num_tokens} rows to send"
        if self.incoming_ids:
            send[:, : received.shape[-1]].copy_(received[:num_tokens])
        return send

    def _check_received(self, received: torch.Tensor | None) -> None:
        if received is None:
            raise RuntimeError(
                f"PPAuxRelay expected aux rows for target layers "
                f"{list(self.incoming_ids)} from the previous stage, got none."
            )
        expected = len(self.incoming_ids) * self.spec.hidden_size
        if received.shape[-1] != expected:
            raise RuntimeError(
                f"PPAuxRelay expected {expected} aux columns for target layers "
                f"{list(self.incoming_ids)}, got {received.shape[-1]}."
            )

    def absorb(self, received: torch.Tensor | None, drafter) -> None:
        """Hand the relayed block to the drafter, which reads it in place."""
        if not self.incoming_ids:
            return
        self._check_received(received)
        drafter.take_relayed_aux(received)
