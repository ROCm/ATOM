# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sampling constraints shared by required channels and structured output.

A request carries at most one rule: a grammar (packed masks, overlapped token
copies) or a required channel (fused GPU state machine). This boundary owns
when their masks run and when accepted tokens advance their state. Neither
backend is converted to a Python list of vocabulary IDs.
"""

import torch

from atom.model_ops.required_channel import DeviceChannel


def has_constraints(batch):
    """Query the batch without creating instances or GPU work."""
    return batch is not None and (
        bool(batch.structured_requests) or batch.required_channel is not None
    )


def needs_constraint_snapshot(batch, metadata):
    """Whether logprob capture must copy the logits before they are masked."""
    return bool(batch.structured_requests) or (
        batch.required_channel is not None and metadata is None
    )


def prepare_token_constraints(runner, batch, metadata):
    """After deferred input assembly, before forward: overlap grammar work."""
    if batch.structured_requests:
        runner.structured_output.prepare(batch, metadata)


def release_token_constraints(runner, request_ids):
    if runner.structured_output is not None:
        runner.structured_output.release(request_ids)


def constraint_errors(runner):
    grammar = runner.structured_output
    return dict(grammar.errors) if grammar is not None else None


def apply_required_channel(runner, batch, logits, metadata=None, target=None):
    """GPU-only required-channel backend, also used by disaggregated prefill."""
    channel = batch.required_channel
    if channel is None:
        return None
    from atom.model_ops.triton_required_channel import constrain_logits

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
    operate on gathered copies.
    """

    def __init__(self, runner, batch, logits, metadata):
        self.runner, self.batch, self.metadata = runner, batch, metadata
        self.source = logits
        self.logits = self.target = None
        self.grammar = self.channel = None
        self.prefixes = {}

    @classmethod
    def apply(cls, runner, batch, logits, metadata):
        if not has_constraints(batch):
            return None
        step = cls(runner, batch, logits, metadata)
        if batch.structured_requests:
            step.grammar = runner.structured_output
            if step.grammar is None:
                raise ValueError("Structured output must be prepared before enqueue")
            if logits.shape[-1] != step.grammar.vocab_size:
                raise ValueError(
                    "Target logits do not match the prepared structured vocabulary"
                )
            step.prefixes = step.grammar.apply(logits, batch, metadata)
        step.gather()
        if batch.required_channel is not None:
            step.channel = apply_required_channel(
                runner, batch, step.logits, metadata, step.target
            )
        return step

    def commit(self, sampled_tokens, batch, deferred):
        """Consume synchronized, accepted IDs, never unverified draft IDs.

        Returns the channel states a deferred step carries on the GPU.
        """
        channel_states = None
        if self.channel is not None and deferred:
            from atom.model_ops.triton_required_channel import advance_states

            channel_states = advance_states(*self.channel, sampled_tokens)
        if self.prefixes:
            self.grammar.commit(sampled_tokens, batch, self.prefixes)
        return channel_states

    def gather(self):
        if self.metadata is None:
            self.logits = self.source[: self.batch.total_seqs_num]
        else:
            self.logits = torch.index_select(
                self.source, 0, self.metadata.bonus_logits_indices
            )
            self.target = torch.index_select(
                self.source, 0, self.metadata.target_logits_indices
            )
