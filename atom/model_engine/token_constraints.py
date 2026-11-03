# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sampling constraints shared by required channels and structured output.

Backends determine allowed/forbidden token IDs; this boundary owns when their
masks run and when accepted tokens advance their state. Required channels keep
their fused GPU state machine, while grammars keep packed masks and overlapped
token copies. Neither backend is converted to a Python list of vocabulary IDs.
"""

import torch

from atom.model_ops.required_channel import (
    DeviceChannel,
    advance_states,
    constrain_logits,
)


def prepare_token_constraints(runner, batch, metadata):
    """After deferred input assembly, before forward: overlap grammar D2H."""
    if getattr(batch, "structured_requests", None):
        runner.structured_output.prepare(batch, metadata)


def apply_required_channel(runner, batch, logits, metadata=None, target=None):
    """GPU-only force/deny backend, also used by disaggregated prefill."""
    channel = batch.required_channel
    if channel is None:
        return None
    device_channel = runner._device_channels.get(channel)
    if device_channel is None:
        device_channel = DeviceChannel.of(channel, runner.device)
        runner._device_channels[channel] = device_channel
    tp = runner.tokenID_processor
    bs = batch.total_seqs_num
    carried = tp.carries_over and tp.prev_channel_states is not None
    base = constrain_logits(
        device_channel,
        runner.forward_vars["required_channel_states"].gpu[:bs],
        logits,
        carried_from=tp.decode_src.gpu[:bs] if carried else None,
        carried_states=tp.prev_channel_states if carried else None,
        target_logits=target,
        cu_num_draft_tokens=None if metadata is None else metadata.cu_num_draft_tokens,
        input_ids=None if metadata is None else metadata.input_ids,
        target_logits_indices=(
            None if metadata is None else metadata.target_logits_indices
        ),
    )
    return device_channel, base


class TokenConstraints:
    """One constrained sampling step, committed only after token broadcasts.

    The no-constraint path returns None without allocating a step or GPU buffers.
    Logprobs preserve the pre-mask model distribution. Speculative channel masks
    operate on gathered copies, as the original verifier does.
    """

    @classmethod
    def apply(cls, runner, batch, logits, metadata, need_logprobs=False):
        requests = getattr(batch, "structured_requests", None)
        if not requests and batch.required_channel is None:
            return None
        step = cls()
        step.grammar = getattr(runner, "structured_output", None) if requests else None
        if requests and step.grammar is None:
            raise ValueError("Structured output must be prepared before enqueue")
        if requests and logits.shape[-1] != step.grammar.vocab_size:
            raise ValueError(
                "Target logits do not match the prepared structured vocabulary"
            )
        bs = batch.total_seqs_num
        step.premask_lse = (
            torch.logsumexp(logits[:bs].float(), dim=-1)
            if need_logprobs and metadata is None
            else None
        )
        step.prefixes = step.grammar.apply(logits, batch, metadata) if requests else {}
        if metadata is None:
            step.logits = logits[:bs]
            step.target = None
        else:
            step.logits = torch.index_select(logits, 0, metadata.bonus_logits_indices)
            step.target = torch.index_select(logits, 0, metadata.target_logits_indices)
        step.channel = apply_required_channel(
            runner, batch, step.logits, metadata, step.target
        )
        return step

    def commit(self, sampled_tokens, batch, deferred):
        """Consume synchronized, accepted IDs, never unverified draft IDs."""
        channel_states = (
            advance_states(*self.channel, sampled_tokens)
            if self.channel is not None and deferred
            else None
        )
        if self.prefixes:
            self.grammar.commit(sampled_tokens, batch, self.prefixes)
        return channel_states
