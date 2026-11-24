# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""LLaMA 3.1 pretraining components for Lumen.

Two training backends are available:

- **Megatron** (``atomrl.lumen.models.llama31.megatron``)
  Pretraining using Megatron-LM-AMD with TP/PP/CP/VP/SP parallelism,
  Lumen attention (AITER / Triton / FP8), FP8 hybrid training.

- **FSDP** (``atomrl.lumen.models.llama31.fsdp``)
  Pretraining using PyTorch FSDP + HuggingFace LlamaForCausalLM,
  with LoRA (PEFT), FP8 training, and a standard PyTorch training loop.

Both backends share the same :class:`PretrainTextDataset`.

For backward compatibility, the Megatron APIs are re-exported at this level::

    from atomrl.lumen.models.llama31 import lumen_gpt_builder  # Megatron
"""

from atomrl.lumen.models.llama31.dataset import PretrainTextDataset

_MEGATRON_EXPORTS = (
    "add_pretrain_args",
    "apply_fp8_training",
    "apply_lora",
    "forward_step",
    "get_batch",
    "loss_func",
    "reset_fp8_state",
    "lumen_gpt_builder",
    "train_valid_test_datasets_provider",
)

__all__ = [
    "PretrainTextDataset",
    "add_pretrain_args",
    "apply_fp8_training",
    "apply_lora",
    "forward_step",
    "get_batch",
    "loss_func",
    "lumen_gpt_builder",
    "reset_fp8_state",
    "train_valid_test_datasets_provider",
]


def __getattr__(name):
    """Load optional Megatron APIs only when a caller requests one."""
    if name not in _MEGATRON_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from atomrl.lumen.models.llama31 import megatron

    value = getattr(megatron, name)
    globals()[name] = value
    return value
