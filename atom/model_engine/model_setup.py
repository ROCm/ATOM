# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Build a rank's model, in the steps ModelRunner takes: the process state
building needs (``init_model_env``), the model (``get_model``, with
``build_model`` and ``load_weights`` for its loader) and the drafter
(``construct_drafter``). All but the last need no ModelRunner, so a process
that only loads can use them too."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from aiter import init_dist_env
from aiter.dist.utils import get_distributed_init_method
from torch import nn

from atom.config import (
    Config,
    local_device_index,
    set_current_atom_config,
    worker_global_rank,
)
from atom.distributed.simulated_tp import apply_simulated_tp, reject_simulated_tp
from atom.distributed.ulysses_sp import check_aiter_sp_api, sp_is_enabled
from atom.model_engine.tp_multinode import managed_store_client
from atom.model_loader.loader import load_model
from atom.models.registry import support_model_arch_dict
from atom.spec_decode.factory import build_drafter
from atom.utils import resolve_obj_by_qualname
from atom.utils.distributed.utils import (
    raise_timeout_if_dp_spans_nodes,
    raise_timeout_if_model_spans_nodes,
)

if TYPE_CHECKING:
    from atom.model_engine.model_runner import ModelRunner
    from atom.spec_decode.drafter import Drafter

logger = logging.getLogger("atom")

# What get_model calls for the model, loaded or to be filled by its caller:
# (config, model class, device) -> model.
ModelLoader = Callable[[Config, type, torch.device], nn.Module]


def init_model_env(config: Config, rank: int) -> torch.device:
    """Set up this process for building its rank's model: the current
    config, its device and its parallel groups (their ranks decide what a
    rank loads). Returns the device."""
    # Read just below (sp_is_enabled) and by model construction.
    set_current_atom_config(config)
    if sp_is_enabled():
        check_aiter_sp_api()
    # `rank` is this worker's local index (0..tp*pcp-1) within its stage.
    dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
    pp_rank = config.parallel_config.pipeline_parallel_rank
    pp_size = config.pipeline_parallel_size
    if pp_size > 1:
        # Reject before any collective can wait for nonexistent ranks.
        reject_simulated_tp(config, "pipeline parallel")
    # tp_world_size: how many GPUs this stage actually occupies.
    stage_span = config.tp_world_size * config.pcp_group_size
    local_device_rank = local_device_index(config, rank)
    num_gpus = torch.cuda.device_count()
    if local_device_rank >= num_gpus:
        raise ValueError(
            f"Calculated local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}). "
        )

    device = torch.device(f"cuda:{local_device_rank}")
    logger.info(
        f"ModelRunner rank={rank}, dp_rank_local={dp_rank_local}, "
        f"pp_rank={pp_rank}, local_device_rank={local_device_rank}, "
        f"device={device}"
    )

    torch.cuda.set_device(device)
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.port)
    distributed_init_method = get_distributed_init_method(
        config.parallel_config.data_parallel_master_ip,
        config.parallel_config.data_parallel_base_port,
    )
    dp_size = config.parallel_config.data_parallel_size
    world_size = dp_size * pp_size * stage_span
    global_rank = worker_global_rank(config, rank)
    raise_timeout_if_dp_spans_nodes(config.parallel_config)
    raise_timeout_if_model_spans_nodes(config)
    if (
        config.parallel_config._managed_distributed_store
        and not torch.distributed.is_initialized()
    ):
        # Preserve AITER's environment setup when preinitializing its group:
        # the GPUs this node runs, which is fewer than the world when the TP
        # group or the pipeline spans nodes.
        if config.is_multinode_tp:
            visible = config.tp_size_local
        elif config.is_multinode_pp:
            visible = config.pp_size_local * stage_span
        else:
            visible = world_size
        os.environ.setdefault("HIP_VISIBLE_DEVICES", ",".join(map(str, range(visible))))
        store = managed_store_client(config)
        torch.distributed.init_process_group(
            backend="nccl", store=store, rank=global_rank, world_size=world_size
        )
    # AITER reuses the default group and creates the model-parallel groups.
    if config.pipeline_parallel_size > 1:
        from atom.distributed.pp_comm import init_pp_aware_dist_env

        init_pp_aware_dist_env(
            tensor_model_parallel_size=config.tensor_parallel_size,
            pipeline_model_parallel_size=pp_size,
            global_rank=global_rank,
            world_size=world_size,
            distributed_init_method=distributed_init_method,
            backend="nccl",
            data_parallel_size=dp_size,
            prefill_context_model_parallel_size=config.pcp_group_size,
            # Every group's device. On one node it equals global_rank (PP
            # excludes DP); a stage on a later node of a pipeline that
            # spans nodes has a global rank past its node's GPUs.
            local_rank=local_device_rank,
        )
        # Before anything is built from the split.
        from atom.models.utils import agree_pp_layer_partition

        agree_pp_layer_partition(config.hf_config.num_hidden_layers)
    else:
        # The group spans the devices that exist; apply_simulated_tp then
        # makes it *report* the logical width so layers shard that many ways.
        init_dist_env(
            config.tp_world_size,
            rankID=rank,
            backend="nccl",
            distributed_init_method=distributed_init_method,
            # This node's physical device index. Without it aiter derives a
            # local rank from the DP-scaled global rank, which overruns the
            # device list on every node after the first.
            local_rank=local_device_rank,
            data_parallel_size=config.parallel_config.data_parallel_size,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
            prefill_context_model_parallel_size=config.pcp_group_size,
            decode_context_parallel_size=getattr(
                config, "decode_context_parallel_size", 1
            ),
        )
        apply_simulated_tp(config)
    return device


def get_model(config: Config, device: torch.device, load: ModelLoader) -> nn.Module:
    """This rank's model, after ``init_model_env``: ``load`` builds it and
    gets its weights (``build_model`` then ``load_weights``, or a builder
    whose weights come later). Leaves the config's dtype as the default."""
    torch.set_default_dtype(config.torch_dtype)
    model_class = resolve_obj_by_qualname(
        support_model_arch_dict[config.hf_config.architectures[0]]
    )
    # The model construction depends on quant_config,
    # so we must complete the remapping for layers before constructing the model.
    config.quant_config.remap_layer_name(
        config.hf_config,
        packed_modules_mapping=getattr(model_class, "packed_modules_mapping", {}),
        quant_exclude_name_mapping=getattr(
            model_class, "quant_exclude_name_mapping", {}
        ),
    )
    return load(config, model_class, device)


def build_model(config: Config, model_class: type, device: torch.device) -> nn.Module:
    """Construct the model on ``device``, without its weights."""
    torch.set_default_device(device)
    try:
        return model_class(config)
    finally:
        torch.set_default_device(None)


def load_weights(model: nn.Module, config: Config) -> set[str]:
    """Load the checkpoint into a built model and process it for serving.
    Returns the names of the parameters loaded or created at load."""
    return load_model(
        model,
        config.model,
        config.hf_config,
        config.load_dummy,
        load_fused_expert_weights_fn=getattr(model, "load_fused_expert_weights", None),
    )


def construct_drafter(
    config: Config, device: torch.device, runner: ModelRunner
) -> Drafter:
    """Construct the drafter for ``runner`` on ``device``, without its
    weights."""
    from atom.utils.backends import set_model_tag

    torch.set_default_device(device)
    try:
        with set_model_tag("drafter"):
            return build_drafter(config, device, runner)
    finally:
        torch.set_default_device(None)
