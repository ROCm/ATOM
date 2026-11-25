# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM RolloutReplica adapter for verl.

Bridges verl's ``RolloutReplica`` interface to AtomRL's
``ATOMRayServer`` so that verl can use ATOM as a rollout backend.

Key design: ATOM's async engine deadlocks when flooded with concurrent
``generate()`` calls (shm ring buffer overflow). This adapter wraps
ATOMRayServer in a thin Ray actor that converts verl's per-request
``generate()`` into ATOM's ``generate_batch()`` call, and serializes
all requests to avoid the deadlock.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from verl.workers.rollout.replica import RolloutReplica, TokenOutput

logger = logging.getLogger(__name__)


def _make_verl_server_cls():
    """Create ATOMRayServer subclass with verl-compatible generate() signature.

    Done lazily to avoid importing ATOMRayServer at module level.
    """
    from atomrl.engine.inference.atom_ray_server import ATOMRayServer

    class _ATOMServerForVerl(ATOMRayServer):
        """ATOMRayServer subclass that adapts generate() for verl's calling convention."""

        async def generate(
            self,
            request_id: str = "",
            prompt_ids: list[int] | None = None,
            prompt: list[int] | None = None,
            sampling_params: dict | None = None,
            **kwargs,
        ):
            actual_prompt = prompt_ids or prompt or []
            result = await super().generate(
                prompt=actual_prompt,
                sampling_params=sampling_params or {},
                request_id=request_id or None,
            )
            token_ids = result.get("token_ids", []) if isinstance(result, dict) else []
            logprobs = result.get("logprobs") if isinstance(result, dict) else None
            return TokenOutput(
                token_ids=token_ids,
                log_probs=[float(x) for x in logprobs] if logprobs else None,
                stop_reason="completed",
            )

    return _ATOMServerForVerl


class ATOMRolloutReplica(RolloutReplica):
    """verl ``RolloutReplica`` backed by AtomRL's ``ATOMRayServer``."""

    async def launch_servers(self):
        """Launch a single ATOMRayServer with ``max_concurrency=1``.

        verl sends many concurrent ``generate()`` calls, but ATOM's engine
        core deadlocks when its shm ring buffer is flooded. Setting
        ``max_concurrency=1`` on the Ray actor serialises all calls so only
        one ``generate`` runs at a time — slower but correct.

        Only replica_rank 0 launches; all others share the same server.
        """
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from verl.utils.device import get_resource_name

        if not hasattr(ATOMRolloutReplica, "_shared_handle"):
            ATOMRolloutReplica._shared_handle = None
            ATOMRolloutReplica._server_ready = asyncio.Event()

        if self.replica_rank != 0:
            await ATOMRolloutReplica._server_ready.wait()
            self._server_handle = ATOMRolloutReplica._shared_handle
            self.servers = [self._server_handle]
            logger.info(
                "ATOMRolloutReplica[%d]: reusing server from replica 0",
                self.replica_rank,
            )
            return

        worker_infos = await asyncio.gather(
            *[
                w.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[
                            get_resource_name()
                        ][0],
                    )
                )
                for w in self.workers
            ]
        )

        model_path = (
            self.model_config.path
            if hasattr(self.model_config, "path")
            else str(self.model_config)
        )
        engine_kwargs = self._build_engine_kwargs()

        node_id = worker_infos[0][0]
        gpu_ids_str = str(worker_infos[0][1])

        env_vars = {
            "CUDA_VISIBLE_DEVICES": gpu_ids_str,
            "HIP_VISIBLE_DEVICES": gpu_ids_str,
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "MASTER_PORT": "29500",
            "LUMEN_REPLICA_RANK": "0",
        }
        for key in ("ATOM_ISOLATE_TORCH_COMPILE_CACHE", "TORCHDYNAMO_DISABLE"):
            if key in os.environ:
                env_vars[key] = os.environ[key]

        ServerCls = _make_verl_server_cls()
        remote_cls = ray.remote(ServerCls)
        handle = remote_cls.options(
            num_gpus=0,
            num_cpus=1,
            name="lumen-atom-verl-server-0",
            max_concurrency=1,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            ),
            runtime_env={"env_vars": env_vars},
        ).remote(
            model_name=model_path,
            engine_kwargs=engine_kwargs,
            replica_rank=0,
            base_seed=self.config.seed if hasattr(self.config, "seed") else None,
        )

        await handle.launch.remote()
        logger.info(
            "ATOMRolloutReplica: ATOMRayServer launched on GPU %s (max_concurrency=1)",
            gpu_ids_str,
        )

        self._server_handle = handle
        self.servers = [handle]

        ATOMRolloutReplica._shared_handle = handle
        ATOMRolloutReplica._server_ready.set()

    async def wake_up(self):
        # ATOM manages its own weight lifecycle; skip verl's wake/sleep
        # to avoid weight corruption in Ray actor environment.
        pass

    async def sleep(self):
        # Skip sleep — ATOM server runs on a dedicated GPU and doesn't
        # need to release memory for the training engine.
        pass

    async def abort_all_requests(self, reject_request: bool = False):
        """Drain in-flight requests; ATOM does not reject newly submitted work.

        ``reject_request`` is retained only for verl API compatibility.
        """
        await asyncio.gather(
            *[s.wait_for_requests_to_drain.remote() for s in self.servers]
        )

    async def resume_generation(self):
        pass

    async def clear_kv_cache(self):
        await asyncio.gather(*[s.reset_prefix_cache.remote() for s in self.servers])

    async def release_kv_cache(self):
        await self.sleep()

    async def resume_kv_cache(self):
        await self.wake_up()

    async def start_profile(self, **kwargs):
        pass

    async def stop_profile(self):
        pass

    def _build_engine_kwargs(self) -> dict[str, Any]:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "tensor_parallel_size": cfg.tensor_model_parallel_size,
            "dtype": str(cfg.dtype),
            "enforce_eager": getattr(cfg, "enforce_eager", True),
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "max_model_len": cfg.max_model_len,
            "max_num_batched_tokens": cfg.max_num_batched_tokens,
            "max_num_seqs": cfg.max_num_seqs,
            "enable_chunked_prefill": getattr(cfg, "enable_chunked_prefill", True),
        }
        if cfg.quantization:
            kwargs["quantization"] = cfg.quantization
        if hasattr(cfg, "enable_prefix_caching"):
            kwargs["enable_prefix_caching"] = cfg.enable_prefix_caching
        extra = getattr(cfg, "engine_kwargs", None)
        if extra and hasattr(extra, "get"):
            atom_extra = extra.get("atom", {})
            if hasattr(atom_extra, "items"):
                kwargs.update(dict(atom_extra.items()))
        return kwargs
