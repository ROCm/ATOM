# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn

from atom.distributed.ulysses_sp import sp_gather_tokens, sp_local_slice
from atom.utils.forward_context import get_forward_context


class SPModelWrapper(nn.Module):
    """Runs the model on this rank's contiguous token range under Ulysses SP.

    Positions stay global: attention finds the real rows by them. Decode
    gathers the ranges back; prefill returns this rank's range, from which the
    LM head reads only each prompt's last row.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ):
        num_tokens = input_ids.shape[0]
        rows = sp_local_slice(num_tokens)
        # Every range has the same length, so the last one may run past the
        # batch into the runner's input buffer (`max_num_batched_tokens`
        # divides by SP); the gather drops those rows. Graphs keep the address.
        local_ids = input_ids.as_strided(
            (rows.stop - rows.start,), (1,), input_ids.storage_offset() + rows.start
        )
        if inputs_embeds is None:
            hidden_states = self.model(local_ids, positions)
        else:
            # A multimodal prefill embeds the whole batch; zero-fill past its end.
            local_embeds = inputs_embeds[rows]
            local_embeds = F.pad(
                local_embeds, (0, 0, 0, local_ids.shape[0] - local_embeds.shape[0])
            )
            hidden_states = self.model(local_ids, positions, inputs_embeds=local_embeds)
        if get_forward_context().context.is_prefill:
            return hidden_states
        return sp_gather_tokens(hidden_states, num_tokens)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
