# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which GLM-5.2 forwards take the mono path.

``GlmMoeDsaForCausalLM`` is an uncompiled outer class (``@support_torch_compile``
sits on the inner ``DeepseekV2Model``), so the routing lives in its forward --
the MiniMax-M3 arrangement, not V4.1's ``mono_decode_installers`` wrapper. See
``models.registry.mono_decode_installers``' own note on which models take which.

The deployment is checked once (``config.config_refusal``) and so is the built
model's layer layout (``_layout_refusal``). A step is then routed only when it
is an MTP verify of a batch in ``AGENTX_BATCHES``. Every per-step read is
TP-uniform: the kernels wait in-kernel on every peer rank, so ranks that
disagreed would wait on each other for ever (a per-rank flag such as
``is_dummy_run`` must never gate this).
"""

from __future__ import annotations

import logging

import torch

from atom.models.glm52.mono.config import AGENTX_BATCHES, config_refusal, native_config
from atom.models.glm52.mono.contract import (
    kv_caches_ok,
    mono_layers,
    shared_sparse_buffer,
    step_refusal,
)
from atom.models.glm52.mono.runner import Glm52MonoRunner
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _layout_refusal(model) -> str | None:
    """Why the built model's layers are not a layout mono can run, or None.

    Both reads are of the module tree, identical on every TP rank, and both are
    about *order*, which ``contract.mono_layers`` and ``shared_sparse_buffer``
    do not check: mono owns a suffix of the layers, so a dense layer after the
    first MoE one would be left reading a residual mono already folded in; and
    an IndexShare layer reuses the sparse indices a preceding full layer wrote,
    so one that comes first would select over an index buffer nothing has
    filled.
    """
    layers = list(model.layers[model.start_layer : model.end_layer])
    first_moe = next(
        (i for i, layer in enumerate(layers) if hasattr(layer.mlp, "experts")),
        len(layers),
    )
    if first_moe == len(layers) or any(
        not hasattr(layer.mlp, "experts") for layer in layers[first_moe:]
    ):
        return "MoE layers are not a suffix"
    seen_full_indexer = False
    for layer in layers:
        indexer = layer.self_attn.indexer
        if indexer is not None and not layer.self_attn.skip_topk:
            seen_full_indexer = True
        elif not seen_full_indexer:
            return "shared IndexShare layer precedes a full layer"
    return None


class MonoDecode:
    """Routes supported decode steps of one ``GlmMoeDsaForCausalLM``."""

    def __init__(self, causal_lm, atom_config) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        # created on its first eligible step (``LazyRunner``)
        self._mono = LazyRunner(
            lambda: Glm52MonoRunner(causal_lm, atom_config), "GLM-5.2 mono decode"
        )
        why = config_refusal(atom_config) or _layout_refusal(causal_lm.model)
        if why is not None:
            logger.info("GLM-5.2 MonoKernel off: %s", why)
            self._mono.enabled = False

    def supports(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors,
        inputs_embeds,
    ) -> bool:
        if not self._mono.enabled or intermediate_tensors is not None:
            return False
        if self._lm.model.aux_hidden_state_layers:
            return False
        samples = input_ids.numel()  # a cudagraph's pad rows included
        fwd = get_forward_context()
        context, metadata = fwd.context, fwd.attn_metadata
        if context is None or metadata is None or context.is_prefill:
            return False
        config = self._atom_config
        spec = config.speculative_config
        query_length = metadata.max_seqlen_q
        shard = native_config(
            samples=samples,
            tp_size=config.tensor_parallel_size,
            kv_cache_dtype=config.kv_cache_dtype,
            mtp=spec is not None and spec.method == "mtp",
            query_length=query_length,
        )
        if shard is None:
            return False
        batch = getattr(context, "scheduled_bs", None)
        batch = samples // query_length if batch is None else batch
        if batch not in AGENTX_BATCHES:
            return False
        if (
            step_refusal(
                fwd=fwd,
                context=context,
                metadata=metadata,
                samples=samples,
                positions=positions,
                inputs_embeds=inputs_embeds,
                hidden_size=shard.hidden,
            )
            is not None
        ):
            return False
        try:
            shared = shared_sparse_buffer(self._lm.model.layers)
        except (KeyError, MonoUnsupported):
            return False
        if not kv_caches_ok(
            fwd,
            mono_layers(self._lm.model),
            config.kv_cache_dtype,
            shard,
            shared,
            samples,
        ):
            return False
        # the runner and its width build, both collective: ``samples`` is
        # TP-uniform
        return self._mono.ready(samples)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._mono.runner.forward(input_ids, positions, inputs_embeds)
