# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import inspect
import logging
import os
from typing import Optional

import numpy as np
import torch

from aiter import init_dist_env
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.rollout.memory_manager import MemoryManagerMixin
from atom.rollout.weight_updater import WeightUpdaterMixin
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class RLHFModelRunner(ModelRunner, WeightUpdaterMixin, MemoryManagerMixin):
    """ModelRunner with RLHF extensions (weight sync + memory lifecycle + DP isolation).

    Used when ATOM is driven by an external RLHF framework (veRL or TorchSpec).
    Pure inference deployments use the base ModelRunner, which carries no
    RLHF-specific code.

    TorchSpec hidden states extraction is opt-in via ``configure_hidden_states()``.
    When not configured (``_extract_mode=False``), all code paths are identical
    to the base veRL behavior.
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

    _extract_mode: bool = False

    def _model_forward_accepts_capture_arg(self) -> bool:
        signature = inspect.signature(self.model.forward)
        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return "capture_hidden_state_layers" in signature.parameters

    def _find_decoder_layers_for_capture(self):
        candidates = (
            ("model.layers", self.model, ("model", "layers")),
            (
                "language_model.model.layers",
                self.model,
                ("language_model", "model", "layers"),
            ),
            (
                "model.language_model.model.layers",
                self.model,
                ("model", "language_model", "model", "layers"),
            ),
        )
        for name, root, path in candidates:
            module = root
            for attr in path:
                module = getattr(module, attr, None)
                if module is None:
                    break
            if module is not None:
                logger.info(f"{self.label}: using {name} for hidden states hooks")
                return module
        return None

    def _register_hidden_state_hooks(self) -> bool:
        if self._model_forward_accepts_capture_arg():
            return False

        layers = self._find_decoder_layers_for_capture()
        if layers is None:
            logger.warning(
                f"{self.label}: model forward does not accept "
                "capture_hidden_state_layers and no decoder layers were found "
                "for hook-based capture"
            )
            return False

        self._hidden_state_hook_handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                if not getattr(self, "_hook_capture_enabled", False):
                    return
                if layer_idx not in self._aux_layer_ids:
                    return

                if isinstance(output, tuple) and len(output) >= 2:
                    hidden_states, residual = output[0], output[1]
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    if residual is not None:
                        hidden_states = hidden_states + residual
                else:
                    hidden_states = output

                self._hook_captured_hidden_states[layer_idx] = hidden_states.detach()

            return hook

        for layer_idx in sorted(self._aux_layer_ids):
            if layer_idx >= len(layers):
                logger.warning(
                    f"{self.label}: skip hidden-state hook for layer {layer_idx}; "
                    f"model only has {len(layers)} layers"
                )
                continue
            handle = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            self._hidden_state_hook_handles.append(handle)

        return bool(self._hidden_state_hook_handles)

    def configure_hidden_states(self, aux_layer_ids, mooncake_config):
        """Enable hidden states extraction for TorchSpec.

        Called via utility command. When not called, ``_extract_mode`` stays
        ``False`` and all TorchSpec code paths are skipped.

        Args:
            aux_layer_ids: Layer indices whose post-layer residual stream
                should be captured (e.g. ``[1, 15, 28]``).
            mooncake_config: Dict with Mooncake connection parameters
                (``local_hostname``, ``metadata_server``, ``protocol``, etc.).
        """
        from torchspec.config.mooncake_config import MooncakeConfig
        from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

        self._aux_layer_ids = set(aux_layer_ids)
        mc_cfg = MooncakeConfig(**mooncake_config)
        self._mooncake_store = EagleMooncakeStore(mc_cfg)
        self._extract_mode = True
        self._captured_hidden_states: Optional[dict[int, torch.Tensor]] = None
        self._captured_last_hidden_states: Optional[torch.Tensor] = None
        self._hook_capture_enabled = False
        self._hook_captured_hidden_states: dict[int, torch.Tensor] = {}
        self._use_hook_capture = self._register_hidden_state_hooks()
        logger.info(
            f"{self.label}: hidden states extraction enabled, "
            f"aux_layers={sorted(self._aux_layer_ids)}, "
            f"hook_capture={self._use_hook_capture}"
        )
        return sorted(self._aux_layer_ids)

    def run_model(
        self,
        input_ids: torch.Tensor,
        batch: Optional[ScheduledBatch] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._extract_mode:
            return super().run_model(input_ids, batch)

        forward_context = get_forward_context()
        context = forward_context.context

        if context.is_prefill:
            positions = context.positions
            if getattr(self, "_use_hook_capture", False):
                self._hook_captured_hidden_states = {}
                self._hook_capture_enabled = True
                try:
                    hidden_states = self.model(input_ids, positions)
                finally:
                    self._hook_capture_enabled = False
                captured = self._hook_captured_hidden_states
            else:
                result = self.model(
                    input_ids,
                    positions,
                    capture_hidden_state_layers=self._aux_layer_ids,
                )
                if isinstance(result, tuple):
                    hidden_states, captured = result
                else:
                    hidden_states = result
                    captured = {}
            logits = self.model.compute_logits(hidden_states)
            self._captured_hidden_states = captured
            self._captured_last_hidden_states = hidden_states.detach()
            return logits, hidden_states

        return super().run_model(input_ids, batch)

    @torch.inference_mode()
    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        result = super().forward(batch)

        if self._extract_mode and self._captured_hidden_states is not None:
            self._store_hidden_states(batch)
            self._captured_hidden_states = None
            self._captured_last_hidden_states = None

        return result

    def _store_hidden_states(self, batch: ScheduledBatch):
        """Write captured hidden states to Mooncake store, one entry per request."""
        captured = self._captured_hidden_states
        last_hs = self._captured_last_hidden_states
        if not captured:
            return

        sorted_layers = sorted(captured.keys())
        aux_hs = torch.cat([captured[lid] for lid in sorted_layers], dim=-1)

        num_scheduled = batch.num_scheduled_tokens
        offsets = np.cumsum(num_scheduled)
        start = 0
        for i, req_id in enumerate(batch.req_ids):
            end = int(offsets[i])
            ext_id = batch.external_request_ids[i]
            if ext_id is None:
                logger.warning(
                    f"{self.label}: skipping hidden states for req {req_id} "
                    f"(no external_request_id)"
                )
                start = end
                continue

            seq_input_ids = torch.from_numpy(
                batch.scheduled_tokens[start:end].astype(np.int64)
            ).to(self.device)

            self._mooncake_store.put(
                key=ext_id,
                hidden_states=aux_hs[start:end],
                input_ids=seq_input_ids,
                last_hidden_states=last_hs[start:end],
            )
            start = end
