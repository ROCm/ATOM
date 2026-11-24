# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Megatron-LM-AMD backend for LLaMA 3.1 pretraining."""

from atomrl.lumen.models.llama31.megatron.pretrain import (
    add_pretrain_args,
    apply_fp8_training,
    apply_lora,
    enable_fp4_for_parallel_linear,
    enable_fp8_for_parallel_linear,
    forward_step,
    get_batch,
    loss_func,
    lumen_gpt_builder,
    reset_fp8_state,
    train_valid_test_datasets_provider,
)

__all__ = [
    "add_pretrain_args",
    "apply_fp8_training",
    "apply_lora",
    "enable_fp4_for_parallel_linear",
    "enable_fp8_for_parallel_linear",
    "forward_step",
    "get_batch",
    "loss_func",
    "lumen_gpt_builder",
    "reset_fp8_state",
    "train_valid_test_datasets_provider",
]
