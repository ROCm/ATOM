# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Carry a DSpark drafter's tapped target layers across pipeline stages.

The drafter lives on the last PP stage, but most of its taps (M3.1: layers
3/17/31/45/59) run on earlier ones, where its hooks would sit on placeholders
and never fire. Each earlier stage taps the layers it runs and sends the
captures on with its hidden states; the last stage hands them to the drafter.
"""

import torch

from atom.distributed.pp_comm import PP_DRAFT_AUX_PREFIX
from atom.models.utils import PPMissingLayer
from atom.spec_decode.drafter import Drafter, _resolve_decoder_layers
from atom.spec_decode.dspark_proposer import DSparkProposer


class PPAuxRelay:
    def __init__(self, config, target_model, drafter, device):
        if getattr(getattr(drafter, "model", None), "target_aux_capture_spec", None):
            raise NotImplementedError("PP cannot relay a draft's own target taps")
        draft_cfg = config.speculative_config.draft_model_hf_config
        self.layer_ids = tuple(int(i) for i in draft_cfg.dspark_target_layer_ids)
        if not self.layer_ids or min(self.layer_ids) < 0:
            raise NotImplementedError(f"cannot relay taps {self.layer_ids} under PP")
        layers = _resolve_decoder_layers(target_model)
        held = [i for i in self.layer_ids if not isinstance(layers[i], PPMissingLayer)]
        # The last stage taps its own layers through the drafter's hooks and
        # must be handed every other one.
        self._expected = set(self.layer_ids) - set(held) if drafter else set()
        self._received: dict[int, torch.Tensor] = {}
        self._captures: dict[int, torch.Tensor] = {}
        for i in held if drafter is None else []:
            self._captures[i] = torch.zeros(
                config.max_num_batched_tokens,
                config.hf_config.hidden_size,
                dtype=config.torch_dtype,
                device=device,
            )
            layers[i].register_forward_hook(
                Drafter.aux_capture_hook(
                    self._captures[i], DSparkProposer._extract_layer_hidden
                )
            )

    def take_received(self, payload) -> None:
        """Pop the earlier stages' taps off a received payload."""
        self._received = {
            i: payload.tensors.pop(f"{PP_DRAFT_AUX_PREFIX}{i}")
            for i in self.layer_ids
            if f"{PP_DRAFT_AUX_PREFIX}{i}" in payload.tensors
        }

    def attach(self, payload) -> None:
        """Send on every tap held so far, received and this stage's own."""
        num_tokens = payload.tensors["hidden_states"].shape[0]
        taps = {i: buf[:num_tokens] for i, buf in self._captures.items()}
        for i, tensor in {**self._received, **taps}.items():
            payload.tensors[f"{PP_DRAFT_AUX_PREFIX}{i}"] = tensor

    def deliver(self, drafter) -> None:
        """On the last stage, hand the drafter the taps it cannot capture."""
        if self._received.keys() != self._expected:
            raise RuntimeError(
                f"expected taps {sorted(self._expected)}, got {sorted(self._received)}"
            )
        for i, tensor in self._received.items():
            drafter.accept_relayed_aux(i, tensor)
