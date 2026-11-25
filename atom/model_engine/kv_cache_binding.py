# SPDX-License-Identifier: MIT
"""Yield target and draft KV views for allocation and imported-pool callers."""

import logging

logger = logging.getLogger(__name__)


def kv_cache_bindings(runner):
    """Yield modules and KV views for callers to key by ``module.layer_num``.

    Try an independent draft's builder first. Builders return None for unknown
    modules; fall back to the target builder only if it was not already tried.
    """
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
