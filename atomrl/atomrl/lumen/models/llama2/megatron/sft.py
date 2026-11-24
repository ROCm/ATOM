# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""LLaMA2 Supervised Fine-Tuning components for Megatron-LM-AMD.

This module provides the building blocks for SFT on LLaMA2 models using
Megatron-LM-AMD as the training backbone and Lumen for the
core dot-product attention (AITER / Triton / FP8).

Features:
    - Full fine-tuning or LoRA (parameter-efficient fine-tuning, with A2A comm opt)
    - FP8 quantised training (weight/activation quantisation via Lumen)
    - Lumen attention backends: AITER, Triton, Triton-FP8
    - Fine-grained MXFP8 block configuration (6 independent block sizes)
    - Context Parallelism, Tensor Parallelism, Pipeline Parallelism, VP, SP
    - Packed sequences with cross-sample attention boundary tracking (seq_start_id)
    - Answer-only loss masking for SFT
    - Synthetic warmup with FP8 state reset
    - Early stopping based on validation loss target

Example::

    from atomrl.lumen.models.llama2.megatron import (
        add_finetune_args,
        forward_step,
        lumen_gpt_builder,
        train_valid_test_datasets_provider,
    )
"""

import os

import torch
from megatron.core import tensor_parallel
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    is_first_or_last_pipeline_stage,
)

from atomrl.lumen.models.llama2.dataset import LLaMA2SFTDataset

# Re-export shared symbols so existing callers are not broken.
from atomrl.lumen.models.megatron import (  # noqa: F401
    add_common_megatron_args,
    apply_fp8_training,
    apply_lora,
    loss_func,
    lumen_gpt_builder,
    make_forward_step,
    reset_fp8_state,
)
from atomrl.lumen.patches.builders import apply_args_patches

__all__ = [
    "LLaMA2SFTDataset",
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


# ---------------------------------------------------------------------------
# Dataset provider
# ---------------------------------------------------------------------------


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, validation, and test SFT datasets."""
    args = get_args()

    tokenizer_obj = get_tokenizer()
    is_hf = hasattr(tokenizer_obj, "_tokenizer") and hasattr(
        tokenizer_obj._tokenizer, "encode"
    )
    raw_tokenizer = tokenizer_obj._tokenizer if is_hf else tokenizer_obj

    train_path = args.train_data_path[0] if args.train_data_path else None
    valid_path = args.valid_data_path[0] if args.valid_data_path else None
    test_path = args.test_data_path[0] if args.test_data_path else None

    print_rank_0("> building train, validation, and test SFT datasets ...")

    shuffle_train = bool(int(os.environ.get("LUMEN_SHUFFLE_TRAIN", "0")))
    train_ds = LLaMA2SFTDataset(
        train_val_test_num_samples[0],
        train_path,
        args.seq_length,
        raw_tokenizer,
        is_hf,
        shuffle=shuffle_train,
        seed=args.seed,
    )
    valid_ds = LLaMA2SFTDataset(
        train_val_test_num_samples[1],
        valid_path,
        args.seq_length,
        raw_tokenizer,
        is_hf,
    )
    test_ds = LLaMA2SFTDataset(
        train_val_test_num_samples[2],
        test_path,
        args.seq_length,
        raw_tokenizer,
        is_hf,
    )

    print_rank_0("> finished creating SFT datasets ...")
    return train_ds, valid_ds, test_ds


# ---------------------------------------------------------------------------
# Batch construction (SFT-specific: answer-only loss masking)
# ---------------------------------------------------------------------------


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch with answer-only loss masking and packed sequence params."""
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    args = get_args()

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(
        ["input_ids", "loss_mask"], data, torch.int64
    )

    tokens_ = data_b["input_ids"]
    tokens = tokens_[:, : args.seq_length].contiguous()
    labels = tokens_[:, 1 : args.seq_length + 1].contiguous()
    answer_loss_mask = data_b["loss_mask"][:, 1 : args.seq_length + 1].contiguous()

    tokenizer = get_tokenizer()
    if hasattr(tokenizer, "_tokenizer") and hasattr(
        tokenizer._tokenizer, "eos_token_id"
    ):
        eod_token = tokenizer._tokenizer.eos_token_id
    else:
        eod_token = tokenizer.eod

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_token,
        eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        False,
    )
    loss_mask = loss_mask * answer_loss_mask.to(dtype=loss_mask.dtype)

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


# ---------------------------------------------------------------------------
# Forward step (bound to SFT get_batch)
# ---------------------------------------------------------------------------

forward_step = make_forward_step(get_batch, loss_func, zero_last_loss_mask=True)


# ---------------------------------------------------------------------------
# Extra CLI arguments
# ---------------------------------------------------------------------------


def add_finetune_args(parser):
    """Add finetune-specific arguments."""
    apply_args_patches(parser, names={"common_megatron_args"})
    return parser
