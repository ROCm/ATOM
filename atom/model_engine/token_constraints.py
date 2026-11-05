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
from atom.model_ops.token_constraints import TokenRuleBuffer


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
        preserve_masks=bool(getattr(batch, "structured_requests", None)),
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
        rules = getattr(batch, "token_constraints", None)
        if not requests and batch.required_channel is None and not rules:
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
        step.invalid = None
        step.target_invalid = None
        step.metadata = metadata
        if rules:
            buffer = getattr(runner, "token_rule_buffer", None)
            if buffer is None:
                runner.token_rule_buffer = buffer = TokenRuleBuffer()
            buffer.apply(
                rules,
                step.logits,
                step.target,
                runner.forward_vars["positions"].gpu,
                runner.forward_vars["cu_seqlens_q"].gpu,
                metadata,
            )
        # A sparse deny on the unmasked model row leaves other finite logits,
        # just like the standalone channel's EOS deny path. Only intersections,
        # allow lists, or a vocabulary-sized deny can remove every candidate.
        check_empty = rules and (
            requests
            or step.channel is not None
            or any(rule.mode == "allow" for _, _, rule in rules)
            or sum(len(rule.token_ids) for _, _, rule in rules) >= logits.shape[-1]
        )
        if check_empty or (requests and step.channel is not None):
            # Conflicting hard rules must fail, never silently relax a rule or
            # let an all-infinite row produce NaNs. The safe ID stays GPU-only;
            # failure travels with the existing asynchronous output copy.
            step.invalid = _guard_empty_rows(step.logits)
            if step.target is not None:
                step.target_invalid = _guard_empty_rows(step.target)
        return step

    def commit(self, sampled_tokens, batch, deferred, failure):
        """Consume synchronized, accepted IDs, never unverified draft IDs."""
        channel_states = (
            advance_states(*self.channel, sampled_tokens)
            if self.channel is not None and deferred
            else None
        )
        if self.prefixes:
            output = (
                sampled_tokens
                if failure is None
                else _error_tokens(sampled_tokens, failure)
            )
            self.grammar.commit(output, batch, self.prefixes)
        return channel_states

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


def _guard_empty_rows(logits):
    invalid = ~torch.isfinite(logits).any(dim=-1)
    logits[:, 0] = torch.where(invalid, 0.0, logits[:, 0])
    return invalid


def _error_tokens(tokens, failure):
    mask = failure if tokens.ndim == 1 else failure[:, None]
    return tokens.masked_fill(mask, -2)
