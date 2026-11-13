# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sampling constraints shared by required channels and structured output.

Backends determine allowed/forbidden token IDs; this boundary owns when their
masks run and when accepted tokens advance their state. Required channels keep
their fused GPU state machine, while grammars keep packed masks and overlapped
token copies. Neither backend is converted to a Python list of vocabulary IDs.
"""

import logging

import torch

from atom.model_ops.required_channel import DeviceChannel

logger = logging.getLogger("atom")

_PROVIDERS = []


def register_constraint_provider(provider):
    """Register an adapter in mask order, without changing the executor.

    Adapters own native field interpretation and resource lifecycle. The
    executor only prepares, applies and commits accepted token output.
    """
    if any(existing.backend == provider.backend for existing in _PROVIDERS):
        raise ValueError(f"Duplicate constraint backend: {provider.backend}")
    _PROVIDERS.append(provider)
    _PROVIDERS.sort(key=lambda adapter: adapter.mask_stage)


def has_constraints(batch):
    """Query adapters without creating instances or GPU work."""
    return batch is not None and any(provider.enabled(batch) for provider in _PROVIDERS)


def needs_constraint_snapshot(batch, metadata):
    return any(
        provider.enabled(batch) and provider.masks_source(metadata)
        for provider in _PROVIDERS
    )


def _enabled_providers(batch):
    for provider in _PROVIDERS:
        if provider.enabled(batch):
            yield provider


def prepare_token_constraints(runner, batch, metadata):
    """After deferred input assembly, before forward: overlap backend work."""
    for provider in _enabled_providers(batch):
        provider.prepare(runner, batch, metadata)


def release_token_constraints(runner, request_ids):
    """Release per-request backend resources through the same adapter contract.

    Every adapter is released even when another one fails, so one adapter's
    error cannot leave a later adapter's request state (for example a grammar
    matcher) allocated. The first failure is raised after all releases ran.
    """
    first_error = None
    for provider in _PROVIDERS:
        try:
            provider.release(runner, request_ids)
        except Exception as error:
            if first_error is None:
                first_error = error
            else:
                logger.exception(
                    "Constraint adapter %s release failed", provider.backend
                )
    if first_error is not None:
        raise first_error


def constraint_errors(runner):
    errors = None
    for provider in _PROVIDERS:
        current = provider.errors(runner)
        if current is not None:
            if errors is None:
                errors = {}
            errors.update(current)
    return errors


def apply_required_channel(runner, batch, logits, metadata=None, target=None):
    """GPU-only required-channel backend, also used by disaggregated prefill."""
    from atom.model_ops.triton_required_channel import constrain_logits

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
    operate on gathered copies.
    """

    def __init__(self, runner, batch, logits, metadata, providers):
        self.runner, self.batch, self.metadata = runner, batch, metadata
        self.source = logits
        self.logits = self.target = None
        self.grammar = self.channel = None
        self.prefixes = {}
        self.providers = providers

    @classmethod
    def apply(cls, runner, batch, logits, metadata):
        if not has_constraints(batch):
            return None
        step = cls(runner, batch, logits, metadata, tuple(_enabled_providers(batch)))
        for provider in step.providers:
            provider.apply(step)
        step.gather()
        return step

    def commit(self, sampled_tokens, batch, deferred):
        """Consume synchronized, accepted IDs, never unverified draft IDs."""
        channel_states = None
        for provider in sorted(self.providers, key=lambda item: item.commit_order):
            result = provider.commit(self, sampled_tokens, batch, deferred)
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


class GrammarProvider:
    backend = "grammar"
    mask_stage = 0
    commit_order = 10

    @staticmethod
    def enabled(batch):
        return bool(batch.structured_requests)

    @staticmethod
    def masks_source(metadata):
        return True

    @staticmethod
    def prepare(runner, batch, metadata):
        runner.structured_output.prepare(batch, metadata)

    @staticmethod
    def errors(runner):
        grammar = runner.structured_output
        return grammar.errors if grammar is not None else None

    @staticmethod
    def release(runner, request_ids):
        grammar = runner.structured_output
        if grammar is not None:
            grammar.release(request_ids)

    @staticmethod
    def apply(step):
        step.grammar = step.runner.structured_output
        if step.grammar is None:
            raise ValueError("Structured output must be prepared before enqueue")
        if step.source.shape[-1] != step.grammar.vocab_size:
            raise ValueError(
                "Target logits do not match the prepared structured vocabulary"
            )
        step.prefixes = step.grammar.apply(step.source, step.batch, step.metadata)

    @staticmethod
    def commit(step, tokens, batch, deferred):
        if step.prefixes:
            step.grammar.commit(tokens, batch, step.prefixes)


class ChannelProvider:
    backend = "channel"
    mask_stage = 1
    commit_order = 0

    @staticmethod
    def enabled(batch):
        return batch.required_channel is not None

    @staticmethod
    def masks_source(metadata):
        return metadata is None

    @staticmethod
    def prepare(runner, batch, metadata):
        # Channel state is assembled with the batch and carried on the GPU.
        pass

    @staticmethod
    def errors(runner):
        return None

    @staticmethod
    def release(runner, request_ids):
        # Immutable device tables are server-wide; request carry belongs to
        # tokenIDProcessor and follows its batch/output lifetime.
        pass

    @staticmethod
    def apply(step):
        step.gather()
        step.channel = apply_required_channel(
            step.runner, step.batch, step.logits, step.metadata, step.target
        )

    @staticmethod
    def commit(step, tokens, batch, deferred):
        if step.channel is not None and deferred:
            from atom.model_ops.triton_required_channel import advance_states

            return advance_states(*step.channel, tokens)


register_constraint_provider(GrammarProvider)
register_constraint_provider(ChannelProvider)
