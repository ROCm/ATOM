# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native-layer indexer routing; importable without AITER or FlyDSL."""

from torch import nn

from atom.models.deepseek_v41.mono.indexer_build import MAX_ROWS
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context


def step_supported(input_ids, inputs_embeds):
    rows = input_ids.numel()
    fwd = get_forward_context()
    step = getattr(fwd.attn_metadata, "step", None)
    return (
        inputs_embeds is None
        and 0 < rows <= MAX_ROWS
        and fwd.context is not None
        and not fwd.context.is_prefill
        and fwd.ubatch_slices is None
        and step is not None
        and step.decode
        and step.running_tokens == rows
        and getattr(fwd.attn_metadata, "image_mask", None) is None
    )


class MonoIndexerModel(nn.Module):
    """Activate the native layer hook for one supported decode forward."""

    def __init__(self, wrapped, raw, config, *, workspace=None):
        super().__init__()
        self.model = wrapped

        def create():
            from aiter.dist.parallel_state import get_tp_group

            from atom.models.deepseek_v41.mono.indexer import MonoIndexer

            return MonoIndexer(
                raw,
                config.max_model_len,
                config.index_cache_dtype == "fp4",
                config.index_cache_layout,
                get_tp_group().cpu_group,
                replicated=not config.enable_dp_attention,
                workspace=workspace,
            )

        self._mono = LazyRunner(create, "V4.1 Mono indexer")
        for block in raw.layers:
            if block.attn.indexer is not None:
                block.attn.indexer._mono = self._mono

    def forward(self, input_ids, positions, inputs_embeds=None):
        # The layer runner owns its standalone launch and buffers. Preparing
        # the native hook here would allocate and clear an unused second set.
        if self.model.uses_layer_mono(input_ids, inputs_embeds):
            return self.model(input_ids, positions)
        active = step_supported(input_ids, inputs_embeds) and self._mono.ready(
            input_ids.numel()
        )
        if active:
            md = get_forward_context().attn_metadata
            self._mono.runner.begin_step(input_ids.numel(), md.cache, md.step)
            self._mono.runner.engram_staged = md.engram_embeddings
            self._mono.runner.active = True
        try:
            if inputs_embeds is None:
                return self.model(input_ids, positions)
            return self.model(input_ids, positions, inputs_embeds=inputs_embeds)
        finally:
            if active:
                self._mono.runner.active = False
                self._mono.runner.engram_staged = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
