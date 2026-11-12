# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sampling constraints shared by required channels and structured output.

Backends determine allowed/forbidden token IDs; this boundary owns when their
masks run and when accepted tokens advance their state. Required channels keep
their fused GPU state machine, while grammars keep packed masks and overlapped
token copies. Neither backend is converted to a Python list of vocabulary IDs.
"""

import torch

from atom.constraint_parser import ConstraintBinding, register_constraint_request_owner
from atom.model_ops.required_channel import (
    DeviceChannel,
    advance_states,
    constrain_logits,
)

_PROVIDERS = []


def register_constraint_provider(provider):
    """Register an adapter in mask order, without changing the executor.

    Adapters own native field interpretation and resource lifecycle. The
    executor only binds, prepares, applies and commits accepted token output.
    """
    if any(existing.backend == provider.backend for existing in _PROVIDERS):
        raise ValueError(f"Duplicate constraint backend: {provider.backend}")
    if not callable(getattr(provider, "hard_requests", None)):
        raise TypeError(
            "Constraint adapters must declare hard_requests(binding, batch)"
        )
    if not hasattr(provider, "mask_stage") or not hasattr(provider, "commit_order"):
        raise TypeError("Constraint adapters must declare mask_stage and commit_order")
    owns_request = getattr(provider, "owns_request", None)
    if owns_request is not None:
        register_constraint_request_owner(provider.backend, owns_request)
    _PROVIDERS.append(provider)
    _PROVIDERS.sort(key=lambda adapter: adapter.mask_stage)


def has_constraints(batch):
    """Query adapters without creating bindings, instances or GPU work."""
    return batch is not None and any(provider.enabled(batch) for provider in _PROVIDERS)


def needs_constraint_snapshot(batch, metadata):
    return any(
        provider.enabled(batch) and provider.masks_source(metadata)
        for provider in _PROVIDERS
    )


def _bindings(batch):
    for provider in _PROVIDERS:
        if not provider.enabled(batch):
            continue
        binding = provider.bind(batch)
        if binding is None:
            raise ValueError(
                f"Enabled constraint adapter {provider.backend} did not bind"
            )
        yield provider, binding


def prepare_token_constraints(runner, batch, metadata):
    """After deferred input assembly, before forward: overlap backend work."""
    for provider, binding in _bindings(batch):
        provider.prepare(runner, batch, metadata, binding)


def release_token_constraints(runner, request_ids):
    """Release per-request backend resources through the same adapter contract."""
    for provider in _PROVIDERS:
        provider.release(runner, request_ids)


def constraint_errors(runner):
    errors = None
    for provider in _PROVIDERS:
        collect = getattr(provider, "errors", None)
        current = collect(runner) if collect is not None else None
        if current is not None:
            if errors is None:
                errors = {}
            errors.update(current)
    return errors


def apply_required_channel(
    runner, batch, logits, metadata=None, target=None, preserve_masks=False
):
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
        if not has_constraints(batch):
            return None
        bindings = tuple(_bindings(batch))
        step = cls()
        step.runner, step.batch, step.metadata = runner, batch, metadata
        step.source = logits
        step.logits = step.target = None
        step.grammar = None
        step.prefixes = {}
        step.channel = None
        step.invalid = step.target_invalid = None
        step.providers = bindings
        step.rule_counts = None
        if len(bindings) > 1:
            step.rule_counts = torch.zeros(
                batch.total_seqs_num, dtype=torch.int32, device=logits.device
            )
            for provider, binding in bindings:
                indices = provider.hard_requests(binding, batch)
                if indices:
                    step.rule_counts[indices] += 1
        for provider, binding in step.providers:
            previous, previous_target = step.invalid, step.target_invalid
            # Providers may reuse their failure output in place. Preserve the
            # accumulated failures before exposing those tensors to the next
            # adapter; these are row-sized flags, never vocabulary masks.
            if previous is not None:
                previous = previous.clone()
            if previous_target is not None:
                previous_target = previous_target.clone()
            provider.apply(step, binding)
            step.invalid = _merge_invalid(previous, step.invalid)
            step.target_invalid = _merge_invalid(previous_target, step.target_invalid)
        step.gather()
        if len(bindings) > 1:
            selected = step.rule_counts > 1
            step.invalid = _merge_invalid(
                step.invalid, _guard_empty_rows(step.logits, selected)
            )
            if step.target is not None:
                end = metadata.cu_num_draft_tokens
                counts = end - torch.cat((torch.zeros_like(end[:1]), end[:-1]))
                owners = torch.repeat_interleave(
                    selected, counts.long(), output_size=step.target.shape[0]
                )
                step.target_invalid = _merge_invalid(
                    step.target_invalid, _guard_empty_rows(step.target, owners)
                )
        return step

    def commit(self, sampled_tokens, batch, deferred, failure):
        """Consume synchronized, accepted IDs, never unverified draft IDs."""
        channel_states = None
        for provider, binding in sorted(
            self.providers, key=lambda item: item[0].commit_order
        ):
            result = provider.commit(
                self, sampled_tokens, batch, deferred, failure, binding
            )
            if result is not None:
                if channel_states is not None:
                    raise ValueError("Multiple constraint adapters returned GPU carry")
                channel_states = result
        return channel_states

    def gather(self):
        if self.logits is not None:
            return
        if self.metadata is None:
            self.logits = self.source[: self.batch.total_seqs_num]
        else:
            self.logits = torch.index_select(
                self.source, 0, self.metadata.bonus_logits_indices
            )
            self.target = torch.index_select(
                self.source, 0, self.metadata.target_logits_indices
            )

    def failed(self, sampled):
        if self.invalid is None and self.target_invalid is None:
            return None
        invalid = self.invalid
        if invalid is None:
            invalid = torch.zeros(
                self.logits.shape[0], dtype=torch.bool, device=self.logits.device
            )
        if self.metadata is None:
            return invalid
        tokens = sampled.reshape(self.logits.shape[0], -1)
        md = self.metadata
        end = md.cu_num_draft_tokens
        start = torch.cat((torch.zeros_like(end[:1]), end[:-1]))
        width = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        rows = start[:, None] + width
        draft = width < (end - start)[:, None]
        bad = invalid[:, None].expand_as(tokens)
        if self.target_invalid is not None and self.target_invalid.numel():
            target_bad = self.target_invalid[
                rows.clamp(max=self.target_invalid.numel() - 1).long()
            ]
            bad = torch.where(draft, target_bad, bad)
        return (bad & (tokens >= 0)).any(dim=1)


class GrammarProvider:
    backend = "grammar"
    mask_stage = 0
    commit_order = 10

    @staticmethod
    def hard_requests(binding, batch):
        return list(binding.value)

    @staticmethod
    def enabled(batch):
        return bool(getattr(batch, "structured_requests", None))

    @staticmethod
    def masks_source(metadata):
        return True

    @staticmethod
    def bind(batch):
        requests = getattr(batch, "structured_requests", None)
        return ConstraintBinding("grammar", requests) if requests else None

    @staticmethod
    def prepare(runner, batch, metadata, binding):
        runner.structured_output.prepare(batch, metadata)

    @staticmethod
    def errors(runner):
        grammar = getattr(runner, "structured_output", None)
        return grammar.errors if grammar is not None else None

    @staticmethod
    def release(runner, request_ids):
        grammar = getattr(runner, "structured_output", None)
        if grammar is not None:
            grammar.release(request_ids)

    @staticmethod
    def apply(step, binding):
        step.grammar = getattr(step.runner, "structured_output", None)
        if step.grammar is None:
            raise ValueError("Structured output must be prepared before enqueue")
        if step.source.shape[-1] != step.grammar.vocab_size:
            raise ValueError(
                "Target logits do not match the prepared structured vocabulary"
            )
        step.prefixes = step.grammar.apply(step.source, step.batch, step.metadata)

    @staticmethod
    def commit(step, tokens, batch, deferred, failure, binding):
        if step.prefixes:
            step.grammar.commit(
                tokens if failure is None else _error_tokens(tokens, failure),
                batch,
                step.prefixes,
            )


class ChannelProvider:
    backend = "channel"
    mask_stage = 1
    commit_order = 0

    @staticmethod
    def hard_requests(binding, batch):
        # Effective states are resolved from GPU carry during apply.
        return []

    @staticmethod
    def enabled(batch):
        return getattr(batch, "required_channel", None) is not None

    @staticmethod
    def masks_source(metadata):
        return metadata is None

    @staticmethod
    def bind(batch):
        channel = getattr(batch, "required_channel", None)
        return ConstraintBinding("channel", channel) if channel is not None else None

    @staticmethod
    def prepare(runner, batch, metadata, binding):
        # Channel state is assembled with the batch and carried on the GPU.
        pass

    @staticmethod
    def release(runner, request_ids):
        # Immutable device tables are server-wide; request carry belongs to
        # tokenIDProcessor and follows its existing batch/output lifetime.
        pass

    @staticmethod
    def apply(step, binding):
        step.gather()
        preserve = False if step.rule_counts is None else step.rule_counts > 0
        step.channel = apply_required_channel(
            step.runner, step.batch, step.logits, step.metadata, step.target, preserve
        )
        if step.channel is not None and step.rule_counts is not None:
            step.rule_counts += (step.channel[1] >= 0).to(step.rule_counts.dtype)

    @staticmethod
    def commit(step, tokens, batch, deferred, failure, binding):
        if step.channel is not None and deferred:
            return advance_states(*step.channel, tokens)


def _merge_invalid(previous, current):
    if current is None:
        return previous
    return current if previous is None else previous | current


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


register_constraint_provider(GrammarProvider)
register_constraint_provider(ChannelProvider)
