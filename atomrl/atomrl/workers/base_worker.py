# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Abstract base class for Ray-side AtomRL workers."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GPU visibility setup (verl-aligned) — must run before ``import torch``.
#
# 1. Normalise HIP/ROCR → CUDA_VISIBLE_DEVICES (verl worker.py L231-271).
# 2. If NOSET is active (rollout co-location), discover the assigned GPU via
#    ``ray.get_runtime_context().get_accelerator_ids()`` (verl worker.py L273-281).
# 3. ``import torch`` + ``torch.cuda.set_device(0)`` so that downstream
#    module-level code (e.g. ``lumen/__init__.py`` → ``torch.cuda.current_device()``)
#    finds an initialised CUDA context.
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys


def _ray_fallback_errors(ray_module) -> tuple[type[BaseException], ...]:
    """Expected Ray metadata failures before an actor is fully initialized."""
    errors: list[type[BaseException]] = [
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
    ]
    ray_error = getattr(getattr(ray_module, "exceptions", None), "RayError", None)
    if isinstance(ray_error, type) and issubclass(ray_error, BaseException):
        errors.append(ray_error)
    return tuple(errors)


def _get_ray_gpu_ids(ray_module=None, *, runtime_context: bool = True) -> list:
    """Return assigned GPU ids, or an empty list while Ray metadata is unavailable."""
    if ray_module is None:
        try:
            import ray as ray_module
        except (ImportError, OSError):
            return []
    try:
        # ``get_runtime_context`` is an auto-init Ray API.  Calling it while
        # the driver is importing worker classes starts a local Ray instance
        # before RayCluster can configure its GPU resources.
        is_initialized = getattr(ray_module, "is_initialized", None)
        if is_initialized is not None and not is_initialized():
            return []
        if runtime_context:
            return ray_module.get_runtime_context().get_accelerator_ids().get("GPU", [])
        return ray_module.get_gpu_ids()
    except _ray_fallback_errors(ray_module):
        return []


def _setup_gpu_visibility_early() -> None:
    """Normalise GPU visibility to CUDA_VISIBLE_DEVICES only (verl-aligned).

    On ROCm, ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES form a two-layer
    filter: ROCR narrows first, then HIP selects within that subset.  Setting
    both to the same physical ID (e.g. "3") causes double-filtering — HIP
    looks for device 3 inside a set that only contains device 0 → crash.

    verl's solution (worker.py L240-271): pop HIP and ROCR, keep only
    CUDA_VISIBLE_DEVICES.  PyTorch ROCm maps it internally.
    """
    cuda_val = _os.environ.get("CUDA_VISIBLE_DEVICES")
    hip_val = _os.environ.get("HIP_VISIBLE_DEVICES")
    rocr_val = _os.environ.get("ROCR_VISIBLE_DEVICES")

    # Normalise HIP → CUDA, remove HIP (verl L240-254)
    if hip_val:
        _os.environ.pop("HIP_VISIBLE_DEVICES", None)
        if not cuda_val:
            cuda_val = hip_val
            _os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val

    # Normalise ROCR → CUDA, remove ROCR (verl L256-271)
    if rocr_val:
        _os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        if not cuda_val:
            cuda_val = rocr_val
            _os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val

    # Ray-API fallback (verl L273-281). Reached both when NOSET is active
    # (rollout co-location asks Ray not to set the variables) and when Ray's
    # accelerator detection failed and left them unset on its own. The latter
    # is not hypothetical: every actor then sees all 8 GPUs, lands on card 0,
    # and the training step dies with "Multiple ranks detected using the same
    # GPU on this node". Ray's *assignment* is still right, so restore the
    # invariant every consumer downstream assumes -- ``setup_distributed``'s
    # ``local_rank=0``, ``init_model``'s ``model.to(local_device)``, FSDP2's
    # DeviceMesh all take this actor's only device to be ``cuda:0``.
    _NOSET_VARS = (
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    )
    is_noset = any(_os.environ.get(v) for v in _NOSET_VARS)
    if not cuda_val:
        gpu_ids = _get_ray_gpu_ids()
        if gpu_ids:
            # Join them all: a multi-GPU actor must not be truncated to one.
            cuda_val = ",".join(str(g) for g in gpu_ids)
            _os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val

    print(
        f"[base_worker] GPU visibility: CUDA_VIS={_os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
        f"HIP_VIS={_os.environ.get('HIP_VISIBLE_DEVICES', 'unset')} "
        f"ROCR_VIS={_os.environ.get('ROCR_VISIBLE_DEVICES', 'unset')} "
        f"NOSET={is_noset}",
        file=_sys.stderr,
        flush=True,
    )


_setup_gpu_visibility_early()

# Now import torch and initialise the CUDA context so that downstream
# module-level code (lumen/__init__.py → torch.cuda.current_device()) works.
import torch as _torch

if _torch.cuda.is_available():
    _torch.cuda.set_device(0)

del _setup_gpu_visibility_early

import logging
from abc import ABC, abstractmethod
from typing import Any


def get_nested_config(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return ``config[k1][k2]...`` when present, else ``default``."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


class BaseWorker(ABC):
    """Single-process worker contract used by Ray actors.

    The controller stays in one process; each worker actor inherits from this
    class, owns local devices, and exchanges batches via :class:`DataProto`.
    """

    def __init__(
        self, rank: int, world_size: int, config: dict[str, Any] | None = None
    ) -> None:
        # GPU visibility is settled at module import (_setup_gpu_visibility_early),
        # which is the only point early enough: torch caches the device list at its
        # first CUDA call, and this module initialises the CUDA context on import.
        self.rank = rank
        self.world_size = world_size
        self.config: dict[str, Any] = dict(config or {})
        self._log = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Ensure worker logs include rank for multi-actor debugging."""
        if not self._log.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter(
                fmt=f"[rank{self.rank}/{self.world_size}] %(name)s: %(levelname)s %(message)s"
            )
            handler.setFormatter(fmt)
            self._log.addHandler(handler)
        self._log.setLevel(logging.INFO)

    @abstractmethod
    def init_model(self) -> None:
        """Allocate models, optimizers, and device state."""

    def get_dp_rank(self) -> int:
        """Return this worker's data-parallel rank for dispatch routing."""
        return self.rank

    def get_is_collect(self) -> bool:
        """Whether this worker's results should be collected (vs. duplicates from TP/PP peers)."""
        return True

    # ------------------------------------------------------------------
    # Distributed rendezvous (Ray controller path, verl-aligned)
    # ------------------------------------------------------------------
    def get_node_ip(self) -> str:
        """Return this actor's node IP (used to pick the dist master address)."""
        try:
            import ray

            return ray.util.get_node_ip_address()
        except (ImportError, OSError):
            import socket

            return socket.gethostbyname(socket.gethostname())
        except _ray_fallback_errors(ray):
            import socket

            return socket.gethostbyname(socket.gethostname())

    def find_free_port(self) -> int:
        """Bind an ephemeral TCP port on this actor's node and return it."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def get_colocation_info(self) -> dict[str, Any]:
        """Return node id + physical GPU ids so a rollout replica can be pinned
        to this actor's GPU (verl HYBRID colocation)."""
        import ray

        ctx = ray.get_runtime_context()
        gpu_ids = _get_ray_gpu_ids(ray, runtime_context=False)
        return {
            "rank": self.rank,
            "node_id": ctx.get_node_id(),
            "gpu_ids": [str(g) for g in gpu_ids],
        }

    def setup_distributed(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        local_rank: int = 0,
        backend: str = "cpu:gloo,cuda:nccl",
        timeout_s: int = 7200,
    ) -> bool:
        """Join the cross-actor ``torch.distributed`` group for FSDP sharding.

        verl-aligned: each Ray actor is pinned to one GPU (Ray sets
        ``CUDA_VISIBLE_DEVICES`` so its device is ``cuda:0`` => ``local_rank=0``),
        but participates in a single ``world_size``-rank process group so FSDP2
        ``fully_shard`` shards parameters and reduce-scatters gradients across
        all actors. Without this, the FSDP backend falls back to an unsharded
        full replica per actor with no gradient sync (weights diverge).
        """
        import os
        from datetime import timedelta

        import torch

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)

        self.rank = int(rank)
        self.world_size = int(world_size)

        import sys

        def _mem_snap(label):
            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            print(
                f"[MEM_DIAG rank={rank}] {label}: "
                f"total={total/2**30:.2f} GiB, free={free/2**30:.2f} GiB, "
                f"alloc={alloc/2**30:.2f} GiB, "
                f"non_torch={max(0, total - free - alloc)/2**30:.2f} GiB  "
                f"CUDA_VIS={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
                f"HIP_VIS={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')} "
                f"dev_count={torch.cuda.device_count()} "
                f"NCCL_CUMEM={os.environ.get('NCCL_CUMEM_ENABLE', 'unset')}",
                file=sys.stderr,
                flush=True,
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            _mem_snap("BEFORE init_process_group")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=int(timeout_s)),
            )

        if torch.cuda.is_available():
            _mem_snap("AFTER init_process_group")

        self._log.info(
            "setup_distributed: rank=%d world_size=%d master=%s:%s local_rank=%d",
            rank,
            world_size,
            master_addr,
            master_port,
            local_rank,
        )
        return True

    def cleanup(self) -> None:
        """Release GPU memory and tear down runtime hooks."""
        try:
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except (AttributeError, ImportError, OSError, RuntimeError):
            pass
        self._log.info("%s.cleanup: default no-op complete.", self.__class__.__name__)


class RolloutPlacementWorker:
    """Lightweight Ray actor for ATOM replica placement on separate nodes.

    Provides ``get_colocation_info()`` so ``ATOMReplicaManager`` can pin
    replicas to this worker's node/GPU, without loading any model.
    """

    def __init__(self, rank: int, world_size: int, **kwargs: Any) -> None:
        self.rank = rank

    def get_colocation_info(self) -> dict[str, Any]:
        import ray

        ctx = ray.get_runtime_context()
        gpu_ids = _get_ray_gpu_ids(ray, runtime_context=False)
        return {
            "rank": self.rank,
            "node_id": ctx.get_node_id(),
            "gpu_ids": [str(g) for g in gpu_ids],
        }
