# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""LLaMA2 model components for Lumen.

Two training backends are available:

- **Megatron** (``atomrl.lumen.models.llama2.megatron``)
  Full-featured SFT using Megatron-LM-AMD with TP/PP/CP/VP/SP parallelism,
  Lumen attention, and the ``pretrain`` driver.

- **FSDP** (``atomrl.lumen.models.llama2.fsdp``)
  SFT using PyTorch FSDP + HuggingFace LlamaForCausalLM, with LoRA (PEFT),
  FP8 training, and a standard PyTorch training loop.

Both backends share the same :class:`LLaMA2SFTDataset` (packed sequences,
answer-only loss masking, jsonl format).

For backward compatibility, the Megatron APIs are re-exported at this level::

    from atomrl.lumen.models.llama2 import lumen_gpt_builder  # Megatron
"""

from atomrl.lumen.models.llama2.dataset import LLaMA2SFTDataset
from atomrl.lumen.models.llama2.megatron import (
    add_finetune_args,
    apply_fp8_training,
    apply_lora,
    forward_step,
    get_batch,
    loss_func,
    lumen_gpt_builder,
    reset_fp8_state,
    train_valid_test_datasets_provider,
)

__all__ = [
    # Shared
    "LLaMA2SFTDataset",
    # Megatron (backward-compat re-exports)
    "add_finetune_args",
    "apply_fp8_training",
    "apply_lora",
    "forward_step",
    "get_batch",
    "loss_func",
    "lumen_gpt_builder",
    "reset_fp8_state",
    "train_valid_test_datasets_provider",
]
