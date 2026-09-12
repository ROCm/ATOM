# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from atom.spec_decode.draft_graph import DraftGraph, StagedInput
from atom.spec_decode.eagle_proposer import EagleProposer
from atom.utils.forward_context import get_forward_context


class DeepseekV4Proposer(EagleProposer):
    """DeepSeek-V4 serial MTP specialization.

    V4 step 0 consumes the target's full ``mtp_k + 1`` decode rectangle and
    carries an mHC residual.  That contract is not shared by the generic Eagle
    path, so its graph declaration and replay eligibility live here.
    """

    def _declare_draft_graphs(self):
        mid_step_graphs = super()._declare_draft_graphs()
        self.step0 = None

        draft_hf = self.speculative_config.draft_model_hf_config
        parallel = self.config.parallel_config
        hc = getattr(draft_hf, "hc_mult", None)
        if (
            self.runner.use_mrope
            or parallel.data_parallel_size != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1
            or getattr(self.config, "prefill_context_parallel_size", 1) != 1
            or getattr(self.config, "enable_tbo", False)
            or getattr(self.config, "enable_expert_parallel", False)
            or hc is None
        ):
            return mid_step_graphs

        q = self.mtp_k + 1
        self.step0 = DraftGraph(
            forward=self._step0_forward,
            epilogue=self._step0_head,
            capture_epilogue=True,
            tokens_per_seq=q,
            inputs={
                "input_ids": StagedInput(shape=(q,), dtype=torch.int32),
                "positions": StagedInput(shape=(q,), dtype=torch.int64),
                "hidden_states": StagedInput(
                    shape=(q, hc, draft_hf.hidden_size), dtype=self.dtype
                ),
                "last_token_indices": StagedInput(dtype=torch.int32),
            },
            warmup_inputs=self._step0_warmup_inputs,
        )
        return (self.step0, *mid_step_graphs)

    def _step0_forward(self, running_bs, *, input_ids, positions, hidden_states, **_):
        return self.model(
            input_ids=input_ids.flatten(0, 1),
            positions=positions.flatten(0, 1),
            hidden_states=hidden_states.flatten(0, 1),
        )

    def _step0_head(self, out, running_bs, *, last_token_indices, **_):
        sample_hidden = torch.index_select(out, 0, last_token_indices)
        return sample_hidden, self.model.compute_draft_ids(sample_hidden)

    def _step0_warmup_inputs(self, running_bs, *, positions, last_token_indices, **_):
        q = self.mtp_k + 1
        context = get_forward_context().context
        positions.copy_(context.positions[: running_bs * q].view(running_bs, q) + 1)
        last_token_indices.copy_(
            torch.arange(
                q - 1, running_bs * q, q, device=self.device, dtype=torch.int32
            )
        )

    def _select_step0_graph(
        self, scheduled_bs, input_ids, positions, hidden_states, last_token_indices
    ) -> DraftGraph | None:
        graph = self.step0
        if graph is None or not graph.is_captured(scheduled_bs):
            return None

        forward_context = get_forward_context()
        context = forward_context.context
        attn_metadata = forward_context.attn_metadata
        q = self.mtp_k + 1
        if (
            context.is_prefill
            or scheduled_bs != context.running_bs
            or input_ids.shape != (scheduled_bs * q,)
            or positions.shape != (scheduled_bs * q,)
            or hidden_states.shape[0] != scheduled_bs * q
            or last_token_indices.shape != (scheduled_bs,)
            or input_ids.dtype != torch.int32
            or positions.dtype != torch.int64
            or hidden_states.dtype != self.dtype
            or last_token_indices.dtype != torch.int32
            or not input_ids.is_contiguous()
            or not positions.is_contiguous()
            or not hidden_states.is_contiguous()
            or attn_metadata.max_seqlen_q != q
        ):
            return None
        return graph

    def _stage_step0_graph_inputs(
        self,
        graph,
        running_bs,
        input_ids,
        positions,
        hidden_states,
        last_token_indices,
    ):
        q = self.mtp_k + 1
        return graph.stage(
            running_bs,
            {
                "input_ids": input_ids.view(running_bs, q),
                "positions": positions.view(running_bs, q),
                "hidden_states": hidden_states.view(
                    running_bs, q, *hidden_states.shape[1:]
                ),
                "last_token_indices": last_token_indices,
            },
        )
