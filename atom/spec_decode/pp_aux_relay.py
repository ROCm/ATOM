# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Relay a block drafter's target aux hidden states across PP stages.

The drafter lives only on the last PP stage, so hooks on taps owned by earlier
stages land on ``PPMissingLayer`` and never fire, leaving zero aux buffers and
garbage drafts. Earlier stages capture their own taps and forward them in the
intermediate-tensor payload, concatenated along the hidden axis in ascending
tap order (-1, the embedding, first). Stages own contiguous ascending layer
ranges, so every stage derives the same order from ``layer_ids`` alone,
whatever order the drafter lists them in.
"""

import logging

import torch
from aiter.dist.parallel_state import get_pp_group

from atom.models.utils import PPMissingLayer
from atom.spec_decode.drafter import (
    AuxCaptureSpec,
    _identity_extract,
    _resolve_decoder_layers,
    _resolve_embedding,
)

logger = logging.getLogger("atom")


class PPAuxRelay:
    """Per-stage aux relay, built on every stage when speculating with pp > 1.

    Sends ``incoming_ids + own_ids`` columns, in ascending tap order.
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
        stage_start = next(
            i for i, layer in enumerate(layers) if not isinstance(layer, PPMissingLayer)
        )
        taps = sorted(spec.layer_ids)

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

        # The last stage's hooks write into the drafter's own buffers.
        self.buffers: dict[int, torch.Tensor] = {}
        if not pp.is_last_rank:
            for lid in self.own_ids:
                self.buffers[lid] = torch.zeros(
                    max_num_tokens, spec.hidden_size, device=device, dtype=dtype
                )

        self._armed = False

    def arm(self, target_model: torch.nn.Module, is_draft_forward) -> None:
        """Install capture hooks for this stage's taps (no-op on the last stage).

        ``is_draft_forward`` is a callable so the hook re-reads it per call.
        """
        if self._armed or get_pp_group().is_last_rank or not self.own_ids:
            return
        self._armed = True
        layers = _resolve_decoder_layers(target_model)
        for lid in self.own_ids:
            # Same (module, extract) resolution as `Drafter.arm_aux_capture`:
            # -1 taps the embedding output verbatim, not `layers[-1]`.
            module, extract = (
                (_resolve_embedding(target_model), _identity_extract)
                if lid == -1
                else (layers[lid], self.spec.extract)
            )
            hook = self._make_hook(self.buffers[lid], extract, is_draft_forward)
            if lid != -1 and self.spec.capture == "input":

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
        """Payload to send on (received block, then own captures), or None."""
        if not self.outgoing_ids:
            return None
        parts = []
        if self.incoming_ids:
            if received is None:
                raise RuntimeError(
                    f"PPAuxRelay expected aux rows for target layers "
                    f"{list(self.incoming_ids)} from the previous stage, got none."
                )
            self._check_width(received)
            parts.append(received[:num_tokens])
        parts.extend(self.buffers[lid][:num_tokens] for lid in self.own_ids)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

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
