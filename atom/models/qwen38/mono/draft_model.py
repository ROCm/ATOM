# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8's MTP drafter on mono: ``dispatch.install_mono_decode`` wraps the loaded ``Qwen4ExpMTP``
(``MonoDraftModel``), as V4.1's wraps its DSpark model, and every other attribute passes through to it.

A draft step returns the complete hyper-connection state and the head input (``EagleProposer``'s
``draft_returns_head_input``), the first step's MLP runs on each request's last row only (``selected_rows``), and the
draft ids come from the certificate heads (``heads``). A step of at most ``ROWS`` rows runs ``draft_fallback.forward``
(the mono draft program, ``draft_runner.DraftRunner``, when it serves the step, else the fallback draft), a prefill
of at most ``ROWS`` rows ``draft_fallback.compact_forward``, anything larger ATOM's own layer.
"""

import torch
from torch import nn

from atom.models.qwen38.mono import draft_fallback, heads
from atom.models.qwen38.mono.config import BATCHES, SPEC

ROWS = SPEC * max(BATCHES)  # the widest step the program and the fallback draft take


class MonoDraftModel(nn.Module):
    """The loaded ``Qwen4ExpMTP`` with its draft steps on mono."""

    draft_returns_head_input = True

    def __init__(self, draft: nn.Module, runner) -> None:
        super().__init__()
        self.model = draft
        self._runner = runner  # ``draft_runner.DraftRunner``
        p = draft.model
        assert p.layers[0].tp_size == 1 and draft.lm_head.tp_size == 1
        assert draft.lm_head.bias is None
        assert draft.lm_head.weight.shape == (248320, 2560)
        # before warm-up and graph capture: the fallback draft's one-row experts and the draft heads
        self._experts = draft_fallback.prepare_experts(p)
        self._screen_head, self._batch_heads = heads.prepare(draft.lm_head.weight)

    def _fc_sum(self, input_ids, hidden_states, inputs_embeds):
        """``fc_hidden(norm(state)) + fc_embedding(norm(embed))`` per stream, [n, streams, H] (into the mono draft
        program's registered input when it is large enough: its kernel binds it, no staging copy).
        """
        p = self.model.model
        n = hidden_states.shape[0]
        if inputs_embeds is None:
            inputs_embeds = p.embed_tokens(input_ids)
        embedding = p.fc_embedding(p.pre_fc_norm_embedding(inputs_embeds))
        hidden = p.pre_fc_norm_hidden(hidden_states.reshape(n, -1))
        hidden = p.fc_hidden(hidden.reshape(-1, p.hidden_size))
        inbuf = self._runner.inbuf
        if n <= inbuf.shape[0]:
            return torch.add(hidden.view(n, p.hc_count, p.hidden_size), embedding[:, None, :],
                             out=inbuf[:n].view(n, p.hc_count, p.hidden_size))  # fmt: skip
        return hidden.view(n, p.hc_count, p.hidden_size) + embedding[:, None, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        p = self.model.model
        hidden = self._fc_sum(input_ids, hidden_states, inputs_embeds).flatten(1)
        if hidden.shape[0] <= ROWS:
            hidden = draft_fallback.compact_forward(p.layers[0], positions, hidden)
        else:
            hidden = p.layers[0](positions, hidden, None)
        return hidden.view(-1, p.hc_count, p.hidden_size)

    def forward_with_head(
        self, input_ids, positions, hidden_states, selected_rows=None
    ):
        """(state, head input) of a draft step; with ``selected_rows`` (the first step) only those rows'."""
        n = hidden_states.shape[0]
        if n <= ROWS and (selected_rows is None or n == SPEC * selected_rows.numel()):
            hidden = self._fc_sum(input_ids, hidden_states, None).flatten(1)
            return draft_fallback.forward(
                self.model.model,
                self._runner,
                self._experts,
                positions,
                hidden,
                selected_rows,
            )
        state = self(input_ids, positions, hidden_states)
        if selected_rows is not None:
            state = torch.index_select(state, 0, selected_rows)
        return state, self.model._head_input(state)

    def compute_draft_ids_from_head(self, head_input, *, out):
        """1 row: the global-certificate head; 2 / 4 rows: the batched W4 head; others: ATOM's argmax head. The ids
        are the BF16 head's argmax (ties low) on every route."""
        if (
            head_input.ndim == 2
            and head_input.shape[0] in self._batch_heads
            and head_input.shape[1] == 2560
            and head_input.dtype == torch.bfloat16
            and head_input.is_contiguous()
            and out.shape == (head_input.shape[0],)
            and out.dtype == torch.int32
            and out.is_contiguous()
        ):
            return self._batch_heads[head_input.shape[0]].run(head_input, out=out)
        if head_input.shape == (1, 2560):
            return self._screen_head.run(head_input, out=out)
        return self.model.lm_head.compute_argmax_token(head_input, out=out)

    def compute_draft_ids(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0, *, out: torch.Tensor
    ):
        return self.compute_draft_ids_from_head(
            self.model._head_input(hidden_states), out=out
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
