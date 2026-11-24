# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""LLaMA 3.1 Pretraining components for Megatron-LM-AMD.

This module provides the building blocks for pretraining LLaMA 3.1 models
using Megatron-LM-AMD as the training backbone and Lumen for
the core dot-product attention (AITER / Triton / FP8).

Features:
    - LLaMA 3.1 8B architecture (GQA, RoPE theta=500000)
    - FP8 hybrid training (weight/activation quantisation via Lumen)
    - FP8 attention (MXFP8 / FP8 blockwise via Lumen)
    - Lumen attention backends: AITER, Triton, Triton-FP8
    - Fine-grained MXFP8 block configuration (6 independent block sizes)
    - Context Parallelism, Tensor Parallelism, Pipeline Parallelism, VP, SP
    - Distributed optimizer with FP8 param gather
    - Cosine annealing LR with warmup
    - Synthetic warmup with FP8 state reset
    - Early stopping based on validation loss target

Example::

    from atomrl.lumen.models.llama31.megatron import (
        add_pretrain_args,
        forward_step,
        lumen_gpt_builder,
        train_valid_test_datasets_provider,
    )
"""

import torch
from megatron.core import tensor_parallel
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    is_first_or_last_pipeline_stage,
)

from atomrl.lumen.models.llama31.dataset import PretrainTextDataset, check_sample_budget

# Re-export shared symbols so existing callers are not broken.
from atomrl.lumen.models.megatron import (  # noqa: F401  # Compatibility re-exports preserve the legacy module API.
    add_common_megatron_args,
    apply_fp8_training,
    apply_lora,
    enable_fp4_for_parallel_linear,
    enable_fp8_for_parallel_linear,
    loss_func,
    make_forward_step,
    reset_fp8_state,
)
from atomrl.lumen.models.megatron import lumen_gpt_builder as _lumen_gpt_builder_generic
from atomrl.lumen.models.utils import safe_add_argument
from atomrl.lumen.patches.builders import apply_args_patches

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


# ---------------------------------------------------------------------------
# LLaMA 3.1 architecture constants
# ---------------------------------------------------------------------------

LLAMA31_CONFIGS = {
    "8b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "seq_length": 8192,
        "max_position_embeddings": 131072,
        "rotary_base": 500000,
    },
}


# ---------------------------------------------------------------------------
# Custom GPT builder (thin wrapper with model-specific log message)
# ---------------------------------------------------------------------------


def lumen_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None):
    return _lumen_gpt_builder_generic(
        args,
        pre_process,
        post_process,
        vp_stage=vp_stage,
        config=config,
        model_name="LLaMA 3.1",
    )


# ---------------------------------------------------------------------------
# Dataset provider
# ---------------------------------------------------------------------------


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, validation, and test pretraining datasets."""
    args = get_args()

    tokenizer_obj = get_tokenizer()
    is_hf = hasattr(tokenizer_obj, "_tokenizer") and hasattr(
        tokenizer_obj._tokenizer, "encode"
    )
    raw_tokenizer = tokenizer_obj._tokenizer if is_hf else tokenizer_obj

    train_path = args.train_data_path[0] if args.train_data_path else None
    valid_path = args.valid_data_path[0] if args.valid_data_path else None
    test_path = args.test_data_path[0] if args.test_data_path else None

    print_rank_0("> building train, validation, and test pretraining datasets ...")

    # Only the training set can wrap. Validation and test never do: a held-out
    # loss averaged over duplicated samples is not a held-out loss.
    repeat_corpus = getattr(args, "lumen_repeat_corpus", False)
    train_ds = PretrainTextDataset(
        train_path,
        args.seq_length,
        raw_tokenizer,
        is_hf,
        max_samples=train_val_test_num_samples[0],
        allow_repeat=repeat_corpus,
    )
    # Safe to raise from here only because this provider is marked
    # is_distributed, so every rank builds its own shard and reaches the same
    # verdict. Were Megatron to build on rank 0 and broadcast, raising here
    # would hang the other ranks in that broadcast instead of failing the job.
    budget_note = check_sample_budget(
        train_ds.n_samples_read, train_val_test_num_samples[0], repeat_corpus
    )
    if budget_note:
        print_rank_0(f"> WARNING: {budget_note}")
    valid_ds = PretrainTextDataset(
        valid_path,
        args.seq_length,
        raw_tokenizer,
        is_hf,
        max_samples=train_val_test_num_samples[1],
    )
    test_ds = PretrainTextDataset(
        test_path,
        args.seq_length,
        raw_tokenizer,
        is_hf,
        max_samples=train_val_test_num_samples[2],
    )

    print_rank_0("> finished creating pretraining datasets ...")
    return train_ds, valid_ds, test_ds


# ---------------------------------------------------------------------------
# Batch construction (standard LM — all tokens contribute)
# ---------------------------------------------------------------------------


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch for pretraining (standard LM, all tokens contribute)."""
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    args = get_args()

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(["input_ids", "labels"], data, torch.int64)

    tokens = data_b["input_ids"].contiguous()
    labels = data_b["labels"].contiguous()

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
# Forward step (bound to pretraining get_batch)
# ---------------------------------------------------------------------------

forward_step = make_forward_step(get_batch, loss_func)


# ---------------------------------------------------------------------------
# Extra CLI arguments
# ---------------------------------------------------------------------------


def add_pretrain_args(parser):
    """Add pretrain-specific arguments."""
    # Pre-register args whose defaults differ from the shared module before
    # apply_args_patches (safe_add_argument skips duplicates).
    safe_add_argument(
        parser,
        "--lumen-fp8-quant-type",
        type=str,
        default="blockwise",
        choices=[
            "dynamic",
            "delayed",
            "blockwise",
            "blockwise2d",
            "per_token",
            "none",
            "mxfp8",
        ],
        help="FP8 quantisation type for FP8 attention backends.",
    )
    safe_add_argument(
        parser,
        "--lumen-repeat-corpus",
        action="store_true",
        help="Let the training set wrap to fill --train-iters when the corpus is "
        "shorter than the run. Off by default: without it a short corpus is a "
        "startup error rather than a run that silently repeats data.",
    )
    safe_add_argument(
        parser,
        "--linear-fp8-amax-algo",
        type=str,
        default="most_recent",
        choices=["max", "most_recent"],
    )
    safe_add_argument(parser, "--linear-fp8-amax-history", type=int, default=4)

    apply_args_patches(
        parser,
        names={"common_megatron_args", "llama_pretrain_args"},
    )

    return parser
