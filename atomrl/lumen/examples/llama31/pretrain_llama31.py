# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""LLaMA 3.1 Pretraining — unified entry point.

Supports two backends, selected via ``--backend``:

- **megatron** — Megatron-LM-AMD ``pretrain`` driver with TP/PP/CP/VP/SP.
  Requires ``pip install megatron-lm``.
- **fsdp** — PyTorch FSDP + HuggingFace Transformers.
  Requires ``pip install transformers peft``.

Usage:
    # Megatron backend (default)
    torchrun --nproc_per_node=8 pretrain_llama31.py --backend megatron <args>

    # FSDP backend
    torchrun --nproc_per_node=8 pretrain_llama31.py --backend fsdp <args>

See run_pretrain.sh for a complete launch example.
"""

import sys

from atomrl.lumen.models.utils import peek_backend


def _run_megatron():
    # Import TL megatron FIRST so _install_fused_layer_norm_patch() runs before
    # megatron.training loads TransformerBlock/FusedLayerNorm.
    from atomrl.lumen.models.llama31.megatron import (
        add_pretrain_args,
        apply_fp8_training,
        apply_lora,
        forward_step,
        lumen_gpt_builder,
        train_valid_test_datasets_provider,
    )
    from atomrl.lumen.models.megatron import make_lumen_model_provider
    from atomrl.lumen.patches import apply_training_patches
    from megatron.core.enums import ModelType
    from megatron.training import pretrain

    model_provider = make_lumen_model_provider(
        lumen_gpt_builder,
        lora_applier=apply_lora,
        fp8_applier=apply_fp8_training,
    )
    apply_training_patches(names={"fp8_param_gather_hook", "mxfp4_weight_cache_hook"})

    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_pretrain_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )


def _run_fsdp():
    from atomrl.lumen.models.llama31.fsdp import FSDPTrainer, get_args

    args = get_args()
    trainer = FSDPTrainer(args)
    trainer.train()


if __name__ == "__main__":
    backend = peek_backend()

    if backend == "megatron":
        _run_megatron()
    elif backend == "fsdp":
        _run_fsdp()
    else:
        print(f"ERROR: Unknown backend '{backend}'. Use 'megatron' or 'fsdp'.")
        sys.exit(1)
