# SPDX-License-Identifier: MIT
"""Bind target and draft modules consistently for allocation and imported pools."""

import logging

logger = logging.getLogger(__name__)


def kv_cache_bindings(runner):
    models = [("target", runner.model, runner.attn_metadata_builder)]
    if runner.config.speculative_config and hasattr(runner, "drafter"):
        models.append(
            ("draft", runner.drafter.model, getattr(runner, "draft_kv_builder", None))
        )
    for name, model, builder in models:
        logger.info("Binding KV cache for the %s model", name)
        for module in model.modules():
            tensor = (
                builder.build_kv_cache_tensor(module) if builder is not None else None
            )
            if tensor is None and builder is not runner.attn_metadata_builder:
                tensor = runner.attn_metadata_builder.build_kv_cache_tensor(module)
            yield module, tensor
