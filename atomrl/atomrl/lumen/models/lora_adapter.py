# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Lightweight LoRA adapter for Megatron-Core parallel linear layers.

This module provides a TP-aware LoRA injection that works with Megatron-Core's
``ColumnParallelLinear`` and ``RowParallelLinear`` without requiring
``megatron-bridge`` or ``transformer_engine``.  It is used to enable LoRA
in VERL Megatron RL training on ROCm (MI300X).

Unlike wrapper-based LoRA (which replaces modules), this implementation uses
**forward hooks** and injects ``_lora_a`` / ``_lora_b`` as nn.Parameter
attributes on the original module.  This preserves the module tree structure
so that ``bridge.load_weights()`` still finds base weights under their
original state-dict keys.

TP communication strategy
-------------------------
* **ColumnParallelLinear** base: output is already sharded across TP ranks.
  ``_lora_a`` is full ``(rank, input_size)`` and ``_lora_b`` is sharded
  ``(output_per_partition, rank)`` — no extra collective needed.
* **RowParallelLinear** base: input is sharded.  ``_lora_a`` is sharded
  ``(rank, input_per_partition)`` so the intermediate ``x @ lora_a.T``
  is a *partial* sum that must be all-reduced before ``lora_b`` applies.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn


def _make_lora_hook(is_row_parallel: bool, tp_group=None, dropout: float = 0.0):
    """Return a forward hook that adds the LoRA residual to the base output."""
    drop = nn.Dropout(dropout) if dropout > 0 else None

    def hook(module, args, output):
        x = args[0].to(module._lora_a.dtype)
        if drop is not None:
            x = drop(x)
        lora_out = x @ module._lora_a.T
        if is_row_parallel:
            dist.all_reduce(lora_out, group=tp_group)
        lora_out = lora_out @ module._lora_b.T * module._lora_scaling

        if isinstance(output, tuple):
            return (output[0] + lora_out.to(output[0].dtype), *output[1:])
        return output + lora_out.to(output.dtype)

    return hook


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> None:
    """Inject LoRA parameters and a forward hook into a single linear module.

    The original module is left in place; ``_lora_a``, ``_lora_b``, and
    ``_lora_scaling`` are added as attributes.  A forward hook computes
    the low-rank residual.

    Parameters
    ----------
    module : nn.Module
        A Megatron ``ColumnParallelLinear`` or ``RowParallelLinear``.
    rank, alpha, dropout : int, float, float
        LoRA hyper-parameters.
    """
    from megatron.core.tensor_parallel.layers import RowParallelLinear

    is_row_parallel = isinstance(module, RowParallelLinear)

    weight = module.weight
    in_features = weight.shape[1]
    out_features = weight.shape[0]

    module._lora_a = nn.Parameter(
        torch.empty(rank, in_features, dtype=torch.bfloat16, device=weight.device),
    )
    module._lora_b = nn.Parameter(
        torch.zeros(out_features, rank, dtype=torch.bfloat16, device=weight.device),
    )
    nn.init.kaiming_uniform_(module._lora_a, a=math.sqrt(5))
    module._lora_scaling = alpha / rank

    tp_group = None
    if is_row_parallel:
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()

    module.register_forward_hook(_make_lora_hook(is_row_parallel, tp_group, dropout))


def apply_megatron_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target: str = "attention_mlp",
) -> None:
    """Apply LoRA to a Megatron GPTModel via forward hooks.

    Freezes all base parameters, then injects LoRA hooks on target layers.

    Parameters
    ----------
    model : nn.Module
        A Megatron ``GPTModel`` (unwrapped, before DDP).
    rank, alpha, dropout : int, float, float
        LoRA hyper-parameters.
    target : str
        ``"attention"`` — QKV + output projection only.
        ``"attention_mlp"`` — attention + MLP (default).
        ``"all"`` — attention + MLP + embedding + output layer.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "decoder") and model.decoder is not None:
        for layer in model.decoder.layers:
            inject_lora(layer.self_attention.linear_qkv, rank, alpha, dropout)
            inject_lora(layer.self_attention.linear_proj, rank, alpha, dropout)
            if (
                target in ("all", "attention_mlp")
                and hasattr(layer, "mlp")
                and layer.mlp is not None
            ):
                inject_lora(layer.mlp.linear_fc1, rank, alpha, dropout)
                inject_lora(layer.mlp.linear_fc2, rank, alpha, dropout)

    if (
        target == "all"
        and hasattr(model, "embedding")
        and model.embedding is not None
        and hasattr(model.embedding, "word_embeddings")
    ):
        inject_lora(model.embedding.word_embeddings, rank, alpha, dropout)
    if (
        target == "all"
        and hasattr(model, "output_layer")
        and model.output_layer is not None
    ):
        inject_lora(model.output_layer, rank, alpha, dropout)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(
        f"[LUMEN] LoRA applied (rank={rank}, alpha={alpha}) — "
        f"trainable: {trainable:,} / {total:,} ({pct:.2f}%)",
        flush=True,
    )
