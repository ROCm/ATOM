# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Build a rank's model: the process state building it needs
(``init_model_env``), then the model with its weights (``get_model``).
ModelRunner serves the model they build; the weight cache daemon only loads
it."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

import torch
from aiter import init_dist_env
from aiter.dist.utils import get_distributed_init_method
from torch import nn

from atom.config import Config, local_device_index, set_current_atom_config
from atom.distributed.simulated_tp import apply_simulated_tp, reject_simulated_tp
from atom.distributed.ulysses_sp import check_aiter_sp_api, sp_is_enabled
from atom.model_loader.loader import load_model
from atom.model_loader.loading_core import rank_tag
from atom.models.registry import support_model_arch_dict
from atom.utils import resolve_obj_by_qualname
from atom.utils.distributed.utils import raise_timeout_if_dp_spans_nodes

logger = logging.getLogger("atom")

# How get_model gets the weights: (config, model class, device) -> model.
ModelLoader = Callable[[Config, type, torch.device], nn.Module]


def init_model_env(
    config: Config, rank: int, *, device_communicators: bool = True
) -> torch.device:
    """Set up this process for building its rank's model: the current
    config, its device and its parallel groups (their ranks decide what it
    loads). ``rank``: this worker's local index
    (0..tp*pcp-1) within its stage. ``device_communicators=False`` builds the
    groups without RCCL and the custom and quick all-reduce buffers (about
    9 GiB per GPU), for a process that runs no device collectives."""
    set_current_atom_config(config)
    if sp_is_enabled():
        check_aiter_sp_api()
    dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
    pp_rank = config.parallel_config.pipeline_parallel_rank
    pp_size = config.pipeline_parallel_size
    if pp_size > 1:
        # Reject before any collective can wait for nonexistent ranks.
        reject_simulated_tp(config, "pipeline parallel")
        if not device_communicators:
            raise NotImplementedError(
                "pipeline parallel groups always build device communicators"
            )
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
    dp_rank = config.parallel_config.data_parallel_rank
    global_rank = (dp_rank * pp_size + pp_rank) * stage_span + rank
    raise_timeout_if_dp_spans_nodes(config.parallel_config)
    if (
        config.parallel_config._managed_distributed_store
        and not torch.distributed.is_initialized()
    ):
        # Preserve AITER's environment setup when preinitializing its group.
        os.environ.setdefault(
            "HIP_VISIBLE_DEVICES", ",".join(map(str, range(world_size)))
        )
        store = torch.distributed.TCPStore(
            config.parallel_config.data_parallel_master_ip,
            config.parallel_config.data_parallel_base_port,
            is_master=False,
        )
        torch.distributed.init_process_group(
            backend="nccl", store=store, rank=global_rank, world_size=world_size
        )
    # AITER reuses the default group and creates the model-parallel groups.
    if config.pipeline_parallel_size > 1:
        from atom.distributed.pp_comm import init_pp_aware_dist_env

        # No local_rank here, unlike the non-PP branch below. Safe only
        # because PP is single-node today: CoreManager rejects multi-node
        # DP when pp_size > 1, and asserts PP+DP out entirely, so
        # global_rank is already the physical device index. Revisit if
        # either restriction is lifted.
        init_pp_aware_dist_env(
            tensor_model_parallel_size=config.tensor_parallel_size,
            pipeline_model_parallel_size=pp_size,
            global_rank=global_rank,
            world_size=world_size,
            distributed_init_method=distributed_init_method,
            backend="nccl",
            data_parallel_size=dp_size,
            prefill_context_model_parallel_size=config.pcp_group_size,
        )
    else:
        # The group spans the devices that exist; apply_simulated_tp then
        # makes it *report* the logical width so layers shard that many ways.
        dcp_size = getattr(config, "decode_context_parallel_size", 1)
        if device_communicators:
            init_dist_env(
                config.tp_world_size,
                rankID=rank,
                backend="nccl",
                distributed_init_method=distributed_init_method,
                # This node's physical device index. Without it aiter derives a
                # local rank from the DP-scaled global rank, which overruns the
                # device list on every node after the first.
                local_rank=local_device_rank,
                data_parallel_size=dp_size,
                data_parallel_rank=dp_rank,
                prefill_context_model_parallel_size=config.pcp_group_size,
                decode_context_parallel_size=dcp_size,
            )
        else:
            # init_dist_env's two steps; custom groups ({}: none) leave the
            # standard groups without device communicators.
            from aiter.dist.parallel_state import (
                ensure_model_parallel_initialized,
                init_distributed_environment,
            )

            init_distributed_environment(
                world_size=config.tp_world_size * config.pcp_group_size,
                rank=rank,
                distributed_init_method=distributed_init_method,
                backend="nccl",
                local_rank=local_device_rank,
                data_parallel_size=dp_size,
                data_parallel_rank=dp_rank,
            )
            ensure_model_parallel_initialized(
                config.tp_world_size,
                1,
                decode_context_model_parallel_size=dcp_size,
                data_parallel_size=dp_size,
                prefill_context_model_parallel_size=config.pcp_group_size,
                custom_group_config={},
            )
        apply_simulated_tp(config)
    return device


def get_model(
    config: Config, device: torch.device, load: ModelLoader | None = None
) -> nn.Module:
    """This rank's model with its weights, after ``init_model_env``. ``load``
    gets the weights; ``load_from_disk`` by default. Leaves the config's
    dtype as the default dtype."""
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
    return (load or load_from_disk)(config, model_class, device)


def build_model(config: Config, model_class: type, device: torch.device) -> nn.Module:
    """Construct the model on ``device``, without its weights."""
    torch.set_default_device(device)
    try:
        return model_class(config)
    finally:
        torch.set_default_device(None)


def load_weights(model: nn.Module, config: Config) -> None:
    """Load the checkpoint into a built model and process it for serving."""
    load_model(
        model,
        config.model,
        config.hf_config,
        config.load_dummy,
        load_fused_expert_weights_fn=getattr(model, "load_fused_expert_weights", None),
    )


def load_from_disk(
    config: Config, model_class: type, device: torch.device
) -> nn.Module:
    model = build_model(config, model_class, device)
    t0 = time.perf_counter()
    load_weights(model, config)
    logger.info(
        f"[{rank_tag()}] Model load done: {config.model} "
        f"(weights loaded in {time.perf_counter() - t0:.2f}s)"
    )
    return model
