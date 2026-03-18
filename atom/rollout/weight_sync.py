# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import gc
import logging
import os
import time
import uuid
from typing import Callable

import torch
from multiprocessing import shared_memory

logger = logging.getLogger("atom")


def rebuild_ipc_handle(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    """Rebuild a CUDA tensor from its IPC handle.

    When two processes have different CUDA_VISIBLE_DEVICES, the device_id
    in the handle may be wrong. This function fixes it by overriding
    the device_id field (index 6 in the args tuple).

    Args:
        handle: A tuple of (rebuild_function, args) from reduce_tensor().
        device_id: Override the device_id in the handle. If None, use the
            original device_id from the handle.

    Returns:
        The reconstructed CUDA tensor sharing the same GPU memory.
    """
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        list_args[6] = device_id
    return func(*list_args)


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


def load_weights_via_ipc(core_mgr, weights, bucket_size_mb: int = 2048,
                         num_gpus: int = 1):
    """Load weights using CUDA IPC for zero-copy GPU-to-GPU transfer.

    Allocates per-GPU buffers in the caller process, copies weight tensors
    into them, and sends per-GPU CUDA IPC handles to ModelRunner sub-processes.
    Each ModelRunner opens ONLY its own GPU's handle — always same-GPU IPC,
    no cross-GPU ``hipIpcOpenMemHandle``.

    When ``num_gpus == 1``, falls back to a single buffer (original behavior).

    Parameters
    ----------
    core_mgr : CoreManager
        Must expose ``broadcast_utility_command_sync`` method.
    weights : Iterable[tuple[str, torch.Tensor]]
        Generator / iterator yielding ``(name, tensor)`` tuples.
        Tensors should be on GPU for maximum efficiency.
    bucket_size_mb : int, optional
        Maximum size of each bucket in MiB (default 2048).
    num_gpus : int, optional
        Total number of GPUs used by this engine (TP * DP).  When > 1,
        per-GPU IPC buffers are allocated for safe cross-GPU distribution
        on ROCm/MI300X.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    logger.info(
        f"load_weights_via_ipc: starting weight update "
        f"(CUDA IPC path, num_gpus={num_gpus})"
    )
    start_time = time.time()

    bucket_size = int(bucket_size_mb) << 20  # MiB -> bytes

    # Determine the device from the first weight tensor
    device = None
    total_params = 0

    # Allocate a GPU buffer for bucket streaming
    # We need to peek at the first weight to know the device
    weights_iter = iter(weights)
    first_item = None
    try:
        first_item = next(weights_iter)
    except StopIteration:
        logger.warning("load_weights_via_ipc: no weights to load")
        core_mgr.broadcast_utility_command("clear_kv_cache")
        return

    name, tensor = first_item
    if tensor.is_cuda:
        device = tensor.device
    else:
        # Fall back to cuda:0 if the tensor is on CPU
        device = torch.device("cuda:0")
        logger.warning(
            "load_weights_via_ipc: first weight is on CPU, using cuda:0. "
            "For best performance, keep weights on GPU."
        )

    # Primary buffer on the source device (cuda:0 typically)
    buffer = torch.empty(bucket_size, dtype=torch.uint8, device=device)

    # Allocate per-GPU buffers and IPC handles for multi-GPU distribution
    use_per_gpu = num_gpus > 1
    per_gpu_buffers = {}
    per_gpu_ipc_handles = {}
    if use_per_gpu:
        for i in range(num_gpus):
            buf = torch.empty(bucket_size, dtype=torch.uint8, device=f"cuda:{i}")
            per_gpu_buffers[i] = buf
            per_gpu_ipc_handles[i] = reduce_tensor(buf)
        logger.info(
            f"load_weights_via_ipc: allocated {num_gpus} per-GPU IPC "
            f"buffers ({bucket_size / (1 << 20):.1f} MiB each)"
        )
    else:
        # Single-GPU: use the primary buffer's IPC handle
        ipc_handle = reduce_tensor(buffer)

    try:
        import itertools
        all_weights = itertools.chain([(name, tensor)], weights_iter)

        offset = 0
        bucket_meta: dict = {}

        for name, tensor in all_weights:
            tensor = tensor.contiguous()
            if not tensor.is_cuda:
                tensor = tensor.to(device)
            tensor_nbytes = tensor.nbytes

            # If this tensor would overflow the current bucket, flush first
            if bucket_meta and offset + tensor_nbytes > bucket_size:
                torch.cuda.synchronize(device)
                if use_per_gpu:
                    # D2D copy from primary buffer to each per-GPU buffer
                    src_slice = buffer[:offset]
                    for i in range(num_gpus):
                        per_gpu_buffers[i][:offset].copy_(src_slice, non_blocking=True)
                    # Synchronize all GPUs to ensure D2D copies are complete
                    for i in range(num_gpus):
                        torch.cuda.synchronize(i)
                    core_mgr.broadcast_utility_command_sync(
                        "update_weights_ipc",
                        ipc_handle=None,
                        ipc_handles=per_gpu_ipc_handles,
                        bucket_meta=bucket_meta,
                        is_last=False,
                    )
                else:
                    core_mgr.broadcast_utility_command_sync(
                        "update_weights_ipc",
                        ipc_handle=ipc_handle,
                        bucket_meta=bucket_meta,
                        is_last=False,
                    )
                total_params += len(bucket_meta)
                bucket_meta = {}
                offset = 0

            assert tensor_nbytes <= bucket_size, (
                f"Weight '{name}' ({tensor.shape}, {tensor.dtype}) is "
                f"{tensor_nbytes / (1 << 20):.1f} MiB which exceeds "
                f"bucket_size_mb={bucket_size_mb}. Increase bucket_size_mb."
            )

            # Copy weight bytes into the GPU buffer (GPU→GPU, no CPU)
            buffer[offset : offset + tensor_nbytes].copy_(
                tensor.view(-1).view(torch.uint8), non_blocking=True
            )

            bucket_meta[name] = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "offset": offset,
                "nbytes": tensor_nbytes,
            }
            offset += tensor_nbytes

        # Flush remaining parameters
        if bucket_meta:
            torch.cuda.synchronize(device)
            if use_per_gpu:
                src_slice = buffer[:offset]
                for i in range(num_gpus):
                    per_gpu_buffers[i][:offset].copy_(src_slice, non_blocking=True)
                # Synchronize all GPUs to ensure D2D copies are complete
                for i in range(num_gpus):
                    torch.cuda.synchronize(i)
                core_mgr.broadcast_utility_command_sync(
                    "update_weights_ipc",
                    ipc_handle=None,
                    ipc_handles=per_gpu_ipc_handles,
                    bucket_meta=bucket_meta,
                    is_last=True,
                )
            else:
                core_mgr.broadcast_utility_command_sync(
                    "update_weights_ipc",
                    ipc_handle=ipc_handle,
                    bucket_meta=bucket_meta,
                    is_last=True,
                )
            total_params += len(bucket_meta)

    finally:
        del buffer
        del per_gpu_buffers
        del per_gpu_ipc_handles
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info(
        f"load_weights_via_ipc: done – {total_params} params via CUDA IPC "
        f"in {elapsed:.2f}s"
    )
