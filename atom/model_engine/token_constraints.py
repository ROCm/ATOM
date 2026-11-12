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
        GrammarProvider.prepare(runner, batch, metadata)


def apply_required_channel(runner, batch, logits, metadata=None, target=None, preserve_masks=False):
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
        preserve_masks=preserve_masks,
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
    def apply(cls, runner, batch, logits, metadata):
        requests = getattr(batch, "structured_requests", None)
        if not requests and batch.required_channel is None:
            return None
        step = cls()
        step.runner, step.batch, step.metadata = runner, batch, metadata
        step.source = logits
        step.logits = step.target = None
        step.grammar = None
        step.prefixes = {}
        step.channel = None
        step.invalid = step.target_invalid = None
        step.providers = []
        if requests:
            step.providers.append(GrammarProvider())
        if batch.required_channel is not None:
            step.providers.append(ChannelProvider())
        for provider in step.providers:
            provider.apply(step)
        step.gather()
        return step

    def commit(self, sampled_tokens, batch, deferred, failure):
        """Consume synchronized, accepted IDs, never unverified draft IDs."""
        channel_states = None
        for provider in self.providers:
            result = provider.commit(self, sampled_tokens, batch, deferred, failure)
            if result is not None:
                channel_states = result
        return channel_states

    def gather(self):
        if self.logits is not None:
            return
        if self.metadata is None:
            self.logits = self.source[:self.batch.total_seqs_num]
        else:
            self.logits = torch.index_select(self.source, 0, self.metadata.bonus_logits_indices)
            self.target = torch.index_select(self.source, 0, self.metadata.target_logits_indices)

    def failed(self, sampled):
        if self.invalid is None:
            return None
        if self.metadata is None:
            return self.invalid
        tokens = sampled.reshape(self.logits.shape[0], -1)
        md = self.metadata
        end = md.cu_num_draft_tokens
        start = torch.cat((torch.zeros_like(end[:1]), end[:-1]))
        width = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        rows = start[:, None] + width
        draft = width < (end - start)[:, None]
        bad = self.invalid[:, None].expand_as(tokens)
        if self.target_invalid.numel():
            target_bad = self.target_invalid[
                rows.clamp(max=self.target_invalid.numel() - 1).long()
            ]
            bad = torch.where(draft, target_bad, bad)
        return (bad & (tokens >= 0)).any(dim=1)


class GrammarProvider:
    @staticmethod
    def prepare(runner, batch, metadata):
        runner.structured_output.prepare(batch, metadata)

    def apply(self, step):
        step.grammar = getattr(step.runner, "structured_output", None)
        if step.grammar is None:
            raise ValueError("Structured output must be prepared before enqueue")
        if step.source.shape[-1] != step.grammar.vocab_size:
            raise ValueError("Target logits do not match the prepared structured vocabulary")
        step.prefixes = step.grammar.apply(step.source, step.batch, step.metadata)

    def commit(self, step, tokens, batch, deferred, failure):
        if step.prefixes:
            step.grammar.commit(tokens if failure is None else _error_tokens(tokens, failure),
                                batch, step.prefixes)


class ChannelProvider:
    def apply(self, step):
        step.gather()
        requests = getattr(step.batch, "structured_requests", None)
        preserve = False
        if requests:
            preserve = torch.zeros(step.batch.total_seqs_num, dtype=torch.bool,
                                   device=step.logits.device)
            preserve[list(requests)] = True
        step.channel = apply_required_channel(step.runner, step.batch, step.logits,
                                              step.metadata, step.target, preserve)
        if requests and step.channel is not None:
            intersection = preserve & (step.channel[1] >= 0)
            step.invalid = _guard_empty_rows(step.logits, intersection)
            if step.target is not None:
                end = step.metadata.cu_num_draft_tokens
                counts = end - torch.cat((torch.zeros_like(end[:1]), end[:-1]))
                owners = torch.repeat_interleave(intersection, counts.long(),
                                                output_size=step.target.shape[0])
                step.target_invalid = _guard_empty_rows(step.target, owners)

    def commit(self, step, tokens, batch, deferred, failure):
        if step.channel is not None and deferred:
            return advance_states(*step.channel, tokens)


def _guard_empty_rows(logits, selected=None):
    if selected is None:
        selected = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    invalid = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    if logits.is_cuda:
        from atom.model_ops.required_channel_gpu import guard_empty_rows
        return guard_empty_rows(logits, selected)
    rows = selected.nonzero().flatten()
    subset = logits.index_select(0, rows)
    bad = ~torch.isfinite(subset).any(dim=-1)
    invalid[rows] = bad
    logits[rows, 0] = torch.where(bad, 0.0, subset[:, 0])
    return invalid


def _error_tokens(tokens, failure):
    mask = failure if tokens.ndim == 1 else failure[:, None]
    return tokens.masked_fill(mask, -2)
