# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os

import torch

from aiter import init_dist_env
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method
from atom.model_engine.model_runner import ModelRunner
from atom.rollout.memory_manager import MemoryManagerMixin
from atom.rollout.weight_updater import WeightUpdaterMixin

logger = logging.getLogger("atom")


class RLHFModelRunner(ModelRunner, WeightUpdaterMixin, MemoryManagerMixin):
    """ModelRunner with RLHF extensions (weight sync + memory lifecycle + DP isolation).

    Used when ATOM is driven by an external RLHF framework.
    Pure inference deployments use the base ModelRunner, which carries no
    RLHF-specific code.
    """

    # Environment variable whose value is a comma-separated list of physical
    # GPU indices assigned to this DP rank (e.g. "2,3").  When set, each DP
    # rank's ModelRunners form an independent NCCL world with TP only.
    # Frameworks may set this via their own env vars; the adapter layer is
    # responsible for mapping to VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER before constructing the
    # runner.
    DP_DEVICE_MAP_ENV = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"

    def _setup_device_and_distributed(self, rank: int, config):
        """Override to set up DP-isolated NCCL worlds.

        Each DP rank's ModelRunners form an independent NCCL world scoped
        to TP only. Device assignment is derived from config (dp_rank_local
        and tensor_parallel_size) rather than environment variables, which
        may not survive multiprocessing spawn boundaries reliably.
        """
        if config.parallel_config.data_parallel_size <= 1:
            device_map = os.environ.get(self.DP_DEVICE_MAP_ENV)
            if device_map is None:
                return super()._setup_device_and_distributed(rank, config)

        dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
        local_device_rank = dp_rank_local * config.tensor_parallel_size + rank
        dp_port = config.parallel_config.data_parallel_base_port + dp_rank_local * 100
        num_gpus = torch.cuda.device_count()

        if local_device_rank >= num_gpus:
            raise ValueError(
                f"local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}), "
                f"dp_rank_local={dp_rank_local}, tp_rank={rank}"
            )

        self.device = torch.device(f"cuda:{local_device_rank}")
        logger.info(
            f"RLHFModelRunner rank={rank}, local_device_rank={local_device_rank}, "
            f"device={self.device} (DP isolated)"
        )

        if "HIP_VISIBLE_DEVICES" not in os.environ:
            os.environ["HIP_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(num_gpus)
            )

        torch.cuda.set_device(self.device)
        os.environ["MASTER_ADDR"] = config.master_addr
        os.environ["MASTER_PORT"] = str(config.port)
        distributed_init_method = get_distributed_init_method(
            config.parallel_config.data_parallel_master_ip,
            dp_port,
        )
        init_dist_env(
            config.tensor_parallel_size,
            rankID=rank,
            backend="nccl",
            distributed_init_method=distributed_init_method,
            data_parallel_size=1,
            data_parallel_rank=0,
            local_rank=local_device_rank,
        )

        # DP is handled at the EngineCore level (DPEngineCoreProc), not
        # within ModelRunner. Override so downstream code (get_dp_padding,
        # _preprocess/sync_dp_for_tbo) sees dp_size=1 and skips cross-DP
        # collectives that would fail on the isolated process group.
        config.parallel_config.data_parallel_size = 1

        # aiter's init_dist_env creates a signal tensor on device=rankID
        # (TP rank). When DP isolation remaps devices, recreate it on
        # the correct device.
        if config.tensor_parallel_size > 1:
            tp_grp = get_tp_group()
            ca_comm = tp_grp.device_communicator.ca_comm
            signal = torch.zeros(
                config.tensor_parallel_size * 64,
                dtype=torch.int64,
                device=self.device,
            )
            ca_comm.signal = signal
            ca_comm.register_input_buffer(signal)
            ca_comm.buffer = ca_comm._pool["input"].tensor
