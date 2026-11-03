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

        def resident(lid):
            # The embedding tap (-1) is resident on the first stage.
            if lid == -1:
                return pp.is_first_rank
            return 0 <= lid < len(layers) and not isinstance(
                layers[lid], PPMissingLayer
            )

        self.spec = spec
        self.layer_ids = spec.layer_ids
        self.hidden_size = spec.hidden_size
        self.own_ids = tuple(lid for lid in taps if resident(lid))
        # Every tap below this stage's first layer was captured upstream, which
        # is exactly what the previous stage sent, even if it owned none itself.
        self.incoming_ids = tuple(
            lid for lid in taps if not resident(lid) and lid < stage_start
        )
        self.outgoing_ids = () if pp.is_last_rank else self.incoming_ids + self.own_ids

        # The send payload, allocated per step at the forward's row count (a
        # resident max_num_batched_tokens buffer is ~0.5 GB per rank): the
        # first own tap to fire allocates it (a TBO ubatch past it grows it),
        # every own tap captures into its columns and pack() copies the
        # received taps into the leading ones. The last stage's hooks write
        # into the drafter's own buffers instead.
        # shortcut: a recorded graph would bake the capture-time buffer, so
        # this holds only while PP stages run their target eagerly.
        self._send: torch.Tensor | None = None
        # own taps captured into `_send`; rebound, never mutated: an in-place
        # add inside the compiled forward makes Dynamo replay it via `update`
        self._filled: frozenset[int] = frozenset()
        self._col = {lid: col for col, lid in enumerate(self.outgoing_ids)}

        self._armed = False

    @classmethod
    def build(cls, config, target_model, drafter) -> "PPAuxRelay | None":
        """This stage's armed relay, or None when no tap crosses a PP boundary.

        Runs on every stage; drafterless stages take the drafter class's
        config-only spec.
        """
        if get_pp_group().world_size <= 1 or config.speculative_config is None:
            return None
        if drafter is not None:
            spec = drafter._aux_capture_spec(target_model)
        else:
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
        if self._armed or get_pp_group().is_last_rank:
            return
        self._armed = True
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
        h, col = self.hidden_size, self._col[lid]
        width = len(self.outgoing_ids) * h

        def write(tensor, off):
            end = off + tensor.shape[0]
            # A tap already captured firing at row 0 starts a new forward:
            # drop whatever an unsent forward left behind.
            if off == 0 and lid in self._filled:
                self._send, self._filled = None, frozenset()
            # Whichever tap fires first allocates, so a skipped tap cannot
            # leave the next one writing into a stale or missing buffer.
            if self._send is None or self._send.shape[0] < end:
                grown = torch.empty(
                    end, width, device=tensor.device, dtype=tensor.dtype
                )
                if self._send is not None:
                    grown[: self._send.shape[0]].copy_(self._send)
                self._send = grown
            self._send[off:end, col * h : (col + 1) * h].copy_(tensor)
            self._filled = self._filled | {lid}

        return write

    def pack(self, received: torch.Tensor | None, num_tokens: int):
        """Payload to send on (received block, then own captures), or None."""
        if not self.outgoing_ids:
            return None
        if self.incoming_ids:
            if received is None:
                raise RuntimeError(
                    f"PPAuxRelay expected aux rows for target layers "
                    f"{list(self.incoming_ids)} from the previous stage, got none."
                )
            self._check_width(received)
            if not self.own_ids:
                return received[:num_tokens]
        # Hand the step's buffer over: the send path owns it from here.
        send, self._send = self._send, None
        assert (
            send is not None and send.shape[0] == num_tokens
        ), f"PPAuxRelay: no capture of {num_tokens} rows to send"
        missing = set(self.own_ids) - self._filled
        assert not missing, f"PPAuxRelay: target layers {sorted(missing)} not captured"
        if self.incoming_ids:
            send[:, : received.shape[-1]].copy_(received[:num_tokens])
        return send

    def _check_width(self, received: torch.Tensor) -> None:
        expected = len(self.incoming_ids) * self.hidden_size
        if received.shape[-1] != expected:
            raise RuntimeError(
                f"PPAuxRelay expected {expected} aux columns for target layers "
                f"{list(self.incoming_ids)}, got {received.shape[-1]}."
            )

    def absorb(self, received: torch.Tensor | None, aux_buffers: list) -> None:
        """Copy relayed rows into ``Drafter._aux_buffers`` (indexed by
        ``layer_ids`` position)."""
        if not self.incoming_ids:
            return
        if received is None:
            raise RuntimeError(
                f"PPAuxRelay expected aux rows for target layers "
                f"{list(self.incoming_ids)} from the previous stage, got none."
            )
        self._check_width(received)
        n = received.shape[0]
        for col, lid in enumerate(self.incoming_ids):
            start = col * self.hidden_size
            buf = aux_buffers[self.layer_ids.index(lid)]
            buf[:n].copy_(received[:, start : start + self.hidden_size])
