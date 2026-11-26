# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""VLLMHttpEngine: trainer-facing adapter over verl-style Ray rollout replicas.

Wraps a :class:`VLLMReplicaManager` (colocated ``VLLMRayServer`` actors) and a
:class:`RolloutClient` load balancer so the trainer drives online rollout, KV
sleep/wake, and IPC weight sync through one object -- analogous to how the
FIFO-transport ``VLLMEngine`` is used, but token-in/token-out and Ray-driven.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RolloutClient:
    """Round-robin / even-split request router across rollout replicas.

    Mirrors the role of verl's ``GlobalRequestLoadBalancer`` for the synchronous
    batch-generation path: shard the (prompt x num_generations) requests evenly
    across the DP replicas, run them concurrently, and reassemble in order.
    """

    def __init__(self, servers: list) -> None:
        self.servers = servers

    def generate(
        self,
        prompt_token_ids_list: list[list[int]],
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        import ray

        n = len(self.servers)
        if n == 0:
            raise RuntimeError("RolloutClient has no rollout replicas.")

        buckets: list[list[list[int]]] = [[] for _ in range(n)]
        index_map: list[list[int]] = [[] for _ in range(n)]
        for idx, pids in enumerate(prompt_token_ids_list):
            b = idx % n
            buckets[b].append(pids)
            index_map[b].append(idx)

        pending = []
        for b in range(n):
            if buckets[b]:
                pending.append(
                    (
                        b,
                        self.servers[b].generate_batch.remote(
                            buckets[b], sampling_params
                        ),
                    )
                )

        results: list[dict[str, Any] | None] = [None] * len(prompt_token_ids_list)
        for b, ref in pending:
            out = ray.get(ref)
            for local_i, res in enumerate(out):
                results[index_map[b][local_i]] = res
        return results  # type: ignore[return-value]


class VLLMHttpEngine:
    """verl-aligned rollout engine: Ray actors + AsyncLLM + ZMQ IPC weight sync.

    ``enable_sleep`` toggles KV sleep/wake between rollout and training. On this
    ROCm box vLLM's cumem sleep is broken (frees no memory + corrupts weights),
    so with abundant VRAM (256GB MI300X, vLLM reserves only gpu_memory_utilization
    of it) we keep the engines resident by default and skip sleep entirely.
    """

    def __init__(
        self, manager, sleep_level: int = 2, enable_sleep: bool = False
    ) -> None:
        self.manager = manager
        self.sleep_level = int(sleep_level)
        self.enable_sleep = bool(enable_sleep)
        self.client = RolloutClient(manager.servers)
        self._sleeping = False

    @property
    def is_sleeping(self) -> bool:
        """Whether the rollout replicas are currently in sleep mode."""
        return self._sleeping

    # -- generation ------------------------------------------------------
    def generate_tokens(
        self,
        prompt_token_ids_list: list[list[int]],
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self.client.generate(prompt_token_ids_list, sampling_params)

    # -- memory management (verl sleep/wake) -----------------------------
    def sleep(self, level: int | None = None) -> None:
        if not self.enable_sleep:
            return
        self.manager.drain_all()
        self.manager.sleep_all(level if level is not None else self.sleep_level)
        self._sleeping = True

    def wake(self, tags: list[str] | None = None) -> None:
        if not self.enable_sleep:
            return
        self.manager.wake_all(tags)
        self._sleeping = False

    def shutdown(self) -> None:
        self.manager.shutdown()
