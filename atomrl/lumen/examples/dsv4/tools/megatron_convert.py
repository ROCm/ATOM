# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Megatron helpers for HF→torch_dist checkpoint conversion."""

from __future__ import annotations

import importlib
import logging
import os
import random
from functools import partial

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.config import set_experimental_flag
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.training.global_vars import set_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

logger = logging.getLogger(__name__)
_HF_ALIASES_REGISTERED: set[str] = set()


def register_hf_config_aliases() -> None:
    if "deepseek_v4" in _HF_ALIASES_REGISTERED:
        return
    module = importlib.import_module(
        "transformers.models.deepseek_v3.configuration_deepseek_v3"
    )
    base_config = module.DeepseekV3Config
    compat_config = type(
        "DeepseekV4Config",
        (base_config,),
        {"model_type": "deepseek_v4", "__module__": __name__},
    )
    if "deepseek_v4" in CONFIG_MAPPING_NAMES:
        AutoConfig.register("deepseek_v4", compat_config, exist_ok=True)
    else:
        AutoConfig.register("deepseek_v4", compat_config, exist_ok=False)
    _HF_ALIASES_REGISTERED.add("deepseek_v4")


def set_default_megatron_args(args):
    args.use_distributed_optimizer = (
        args.optimizer is None or args.optimizer.lower() == "adam"
    ) and not getattr(args, "debug_disable_optimizer", False)
    args.bf16 = not args.fp16
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1":
        args.dist_ckpt_save_pre_mcore_014 = True
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info(
            "--tokenizer-model not set, use --hf-checkpoint as tokenizer model."
        )
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    if not hasattr(args, "dsv4_dsa_topk_backend"):
        args.dsv4_dsa_topk_backend = "torch"

    return args


def _set_random_seed(seed_: int, data_parallel_random_init: bool = False) -> None:
    seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
    if data_parallel_random_init:
        seed = seed + (10 * mpu.get_data_parallel_rank())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)


def init_megatron_for_convert(args) -> None:
    set_args(args)
    if args.enable_experimental:
        logger.info("Enable megatron experimental")
        set_experimental_flag(True)

    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
        context_parallel_size=args.context_parallel_size,
        hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
        expert_model_parallel_size=args.expert_model_parallel_size,
        num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else "tp-cp-ep-pp-dp",
        create_gloo_process_groups=args.enable_gloo_process_groups,
    )

    if args.rank == 0:
        logger.info(f"> setting random seeds to {args.seed} ...")
    _set_random_seed(args.seed, args.data_parallel_random_init)

    register_hf_config_aliases()

    from megatron.training.tokenizer.tokenizer import _build_tokenizer

    _build_tokenizer(args)

    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )

    if args.deterministic_mode:
        if args.rank == 0:
            logger.info("> running in deterministic mode")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

    if args.tp_comm_overlap:
        from megatron.training.initialize import _initialize_tp_communicators

        _initialize_tp_communicators()


def get_convert_model_provider(args):
    del args
    from atomrl.lumen.models.dsv4.megatron.pretrain import (
        dsv4_gpt_builder,
        dsv4_model_provider,
    )
    from model_provider import model_provider as megatron_model_provider

    return partial(
        dsv4_model_provider,
        megatron_model_provider,
        dsv4_gpt_builder,
    )
