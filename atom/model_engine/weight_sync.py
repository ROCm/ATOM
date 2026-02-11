# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import uuid

import torch
from multiprocessing import shared_memory

logger = logging.getLogger("atom")


def load_weights_via_shm(core_mgr, weights, bucket_size_mb: int = 2048):
    """Load weights from a generator/iterator using shared memory.

    Tensor data is written to a POSIX shared-memory segment so that
    ``ModelRunner`` sub-processes can read it directly — **no pickle
    serialisation of tensor payloads** is needed.  Only lightweight metadata
    (parameter names, shapes, dtypes, offsets) travels through the existing
    ``EngineCore`` control path.

    Parameters
    ----------
    core_mgr : CoreManager
        Must expose ``broadcast_utility_command_sync`` and
        ``broadcast_utility_command`` methods.
    weights : Iterable[tuple[str, torch.Tensor]]
        Generator / iterator yielding ``(name, tensor)`` tuples.
    bucket_size_mb : int, optional
        Maximum size of each bucket in MiB (default 2048).
    """
    logger.info("load_weights_via_shm: starting weight update (SHM path)")

    bucket_size = int(bucket_size_mb) << 20  # MiB -> bytes
    shm_name = f"atom_weights_{uuid.uuid4().hex}"
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=bucket_size)
    buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

    total_params = 0
    try:
        offset = 0
        bucket_meta: dict = {}

        for name, tensor in weights:
            # Move to CPU if on GPU
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.contiguous()

            tensor_nbytes = tensor.nbytes

            # If this tensor would overflow the current bucket, flush first
            if bucket_meta and offset + tensor_nbytes > bucket_size:
                core_mgr.broadcast_utility_command_sync(
                    "update_weights_shm",
                    shm_name=shm_name,
                    bucket_meta=bucket_meta,
                    is_last=False,
                )
                total_params += len(bucket_meta)
                bucket_meta = {}
                offset = 0

            # Safety: single tensor larger than bucket – should not happen
            # with a reasonable bucket_size_mb, but guard anyway
            assert tensor_nbytes <= bucket_size, (
                f"Weight '{name}' ({tensor.shape}, {tensor.dtype}) is "
                f"{tensor_nbytes / (1 << 20):.1f} MiB which exceeds "
                f"bucket_size_mb={bucket_size_mb}. Increase bucket_size_mb."
            )

            # Copy raw bytes into the shared-memory buffer
            tensor_bytes = tensor.view(-1).view(torch.uint8)
            buffer[offset : offset + tensor_nbytes].copy_(tensor_bytes)

            bucket_meta[name] = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "offset": offset,
                "nbytes": tensor_nbytes,
            }
            offset += tensor_nbytes

        # Flush remaining parameters (or empty generator)
        if bucket_meta:
            core_mgr.broadcast_utility_command_sync(
                "update_weights_shm",
                shm_name=shm_name,
                bucket_meta=bucket_meta,
                is_last=True,
            )
            total_params += len(bucket_meta)
        else:
            # Generator was empty – just clear KV cache
            logger.warning("load_weights_via_shm: no weights to load")
            core_mgr.broadcast_utility_command("clear_kv_cache")

    finally:
        shm.close()
        shm.unlink()

    logger.info(
        f"load_weights_via_shm: done – {total_params} params via SHM"
    )
