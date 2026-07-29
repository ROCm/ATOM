# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Reading checkpoint tensors off disk.

Split out of `loader.py`, which imports AITER at module level: the unit test
gate has no AITER build, and the shard-skipping logic here is worth covering
against real files.
"""

import json
import logging
import os
import time
from collections.abc import Callable, Generator
from glob import glob

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from atom.model_loader.weight_utils import (
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# safetensors<=0.7.0 ships a Python `_TYPES` dict missing the `F8_E8M0`
# (MX scale) entry, even though both torch and the safetensors-rust binary
# support it. The mmap'd `safe_open` path goes through Rust and works, but
# the `safetensors.torch.load(bytes)` path used when `ATOM_DISABLE_MMAP=true`
# raises `KeyError: 'F8_E8M0'` on DeepSeek-V4-Pro shards. Register the
# missing dtype string so both paths behave identically.
if "F8_E8M0" not in safetensors.torch._TYPES and hasattr(torch, "float8_e8m0fnu"):
    safetensors.torch._TYPES["F8_E8M0"] = torch.float8_e8m0fnu


_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def _shard_tensor_names(st_file: str) -> list[str] | None:
    """Tensor names in a safetensors file, from its header alone.

    The header is a little-endian u64 byte count followed by that much JSON, so
    this costs one small read and never touches the tensor data.

    Returns None if the header cannot be read, so the caller loads the shard
    anyway and the real reader produces the real diagnostic -- a truncated or
    corrupt file should not be reported as a JSON error from a fast path whose
    only job is to decide whether the file is worth opening.
    """
    try:
        with open(st_file, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return None
            header_len = int.from_bytes(raw_len, "little")
            if not 0 < header_len <= _MAX_SAFETENSORS_HEADER_BYTES:
                return None
            raw_header = f.read(header_len)
            if len(raw_header) != header_len:
                return None
            header = json.loads(raw_header)
    except (OSError, ValueError):
        return None
    if not isinstance(header, dict):
        return None
    return [name for name in header if name != "__metadata__"]


def _init_fastsafetensors(disable_mmap: bool):
    """Resolve the optional fastsafetensors reader and its target device.

    Returns ``(fastsafe_open, device_str)`` or ``(None, "cpu")`` when the
    package is unavailable or disabled. mmap-off (`ATOM_DISABLE_MMAP`) forces
    the plain path since the two are mutually exclusive read strategies.
    """
    if not (envs.ATOM_USE_FASTSAFETENSORS and not disable_mmap):
        return None, "cpu"
    try:
        from fastsafetensors import fastsafe_open
    except ImportError:
        logger.warning(
            "ATOM_USE_FASTSAFETENSORS=1 but fastsafetensors is not installed; "
            "falling back to safetensors.safe_open"
        )
        return None, "cpu"

    device = "cpu"
    requested = envs.ATOM_FASTSAFETENSORS_DEVICE.lower()
    if requested == "auto":
        requested = "cuda" if not envs.ATOM_FASTSAFETENSORS_NOGDS else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ATOM_FASTSAFETENSORS_DEVICE=cuda requested but CUDA/HIP "
                "is not available"
            )
        device = f"cuda:{torch.cuda.current_device()}"
    elif requested != "cpu":
        device = requested
    logger.info(
        "Using fastsafetensors for safetensors reads "
        f"(nogds={envs.ATOM_FASTSAFETENSORS_NOGDS}, device={device})"
    )
    return fastsafe_open, device


def safetensors_weights_iterator(
    model_name_or_path: str,
    disable_mmap: bool = False,
    wants: Callable[[str], bool] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files.

    `wants` lets the caller reject a tensor by name before it is materialized.
    Without it every tensor of every shard is built and then thrown away by the
    caller -- which is what a drafter load does, since it reads the target's
    checkpoint to pick out the MTP block and discards the other ~98%.

    When `ATOM_USE_FASTSAFETENSORS=1` (and mmap is on) shards are read through
    fastsafetensors, whose multi-threaded pread saturates a RAID array that
    single-stream `safe_open` leaves at a fraction of its bandwidth. The
    yielded tensors still materialize before the next iteration, so the
    `(name, tensor)` contract and `wants` filtering are unchanged.
    """
    logger.info(f"disable_mmap: {disable_mmap}")
    fastsafe_open, fastsafe_device = _init_fastsafetensors(disable_mmap)
    path = (
        model_name_or_path
        if os.path.isdir(model_name_or_path)
        else download_weights_from_hf(
            model_name_or_path, None, ["*.safetensors"], ignore_patterns=["original/*"]
        )
    )
    hf_weights_files = filter_duplicate_safetensors_files(
        glob(os.path.join(path, "*.safetensors")), path, SAFE_WEIGHTS_INDEX_NAME
    )
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    # Distributed shard ownership: each TP rank reads one shard of a batch and
    # scatters it to the others, so a shard hits disk once instead of `tp`
    # times. Only meaningful with fastsafetensors; import aiter lazily so the
    # AITER-free unit-test gate keeps importing this module.
    tp_group = None
    tp_world_size = 1
    tp_rank = 0
    fastsafe_dist_load = False
    if fastsafe_open is not None and envs.ATOM_FASTSAFETENSORS_DIST_LOAD:
        try:
            from aiter.dist.parallel_state import get_tp_group

            tp_group = get_tp_group()
            tp_world_size = tp_group.world_size
            tp_rank = tp_group.rank_in_group
            fastsafe_dist_load = tp_world_size > 1
        except Exception:
            logger.exception(
                "ATOM_FASTSAFETENSORS_DIST_LOAD=1 but TP group is unavailable; "
                "falling back to per-rank fastsafetensors loading"
            )
    if fastsafe_dist_load:
        # The distributed scatter broadcasts each shard over the TP
        # `device_group`, which is an RCCL/NCCL communicator: it can only move
        # GPU tensors. A CPU target device (the conservative default) makes
        # `dist.broadcast` raise "No backend type associated with device type
        # cpu", so force reads onto this rank's GPU when dist-load is on.
        if not fastsafe_device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning(
                    "ATOM_FASTSAFETENSORS_DIST_LOAD=1 needs CUDA/HIP for the "
                    "RCCL scatter but none is available; disabling dist-load"
                )
                fastsafe_dist_load = False
            else:
                fastsafe_device = f"cuda:{torch.cuda.current_device()}"
                if tp_rank == 0:
                    logger.info(
                        "ATOM_FASTSAFETENSORS_DIST_LOAD forces device=%s "
                        "(RCCL scatter cannot broadcast CPU tensors)",
                        fastsafe_device,
                    )
    if fastsafe_dist_load and tp_rank == 0:
        logger.info(
            "Using distributed fastsafetensors shard loading across %d TP ranks",
            tp_world_size,
        )

    pbar = tqdm(
        total=len(hf_weights_files),
        desc=f"Loading safetensors shards[{model_name_or_path}]",
        disable=not enable_tqdm,
    )

    batch_size = tp_world_size if fastsafe_dist_load else 1
    for batch_start in range(0, len(hf_weights_files), batch_size):
        st_files = hf_weights_files[batch_start : batch_start + batch_size]

        if fastsafe_open is not None:
            try:
                fastsafe_filenames = st_files
                fastsafe_pg = None
                if fastsafe_dist_load:
                    fastsafe_filenames = {
                        rank: ([st_files[rank]] if rank < len(st_files) else [])
                        for rank in range(tp_world_size)
                    }
                    fastsafe_pg = tp_group.device_group
                shard_start = time.perf_counter()
                with fastsafe_open(
                    filenames=fastsafe_filenames,
                    framework="pt",
                    pg=fastsafe_pg,
                    device=fastsafe_device,
                    nogds=envs.ATOM_FASTSAFETENSORS_NOGDS,
                    debug_log=envs.ATOM_FASTSAFETENSORS_DEBUG,
                ) as f:
                    for name in f.keys():  # noqa: SIM118
                        if wants is None or wants(name):
                            yield name, f.get_tensor(name)
                pbar.update(len(st_files))
                if enable_tqdm:
                    logger.info(
                        "Finished safetensors shards %d-%d/%d in %.2fs: %s",
                        batch_start + 1,
                        batch_start + len(st_files),
                        len(hf_weights_files),
                        time.perf_counter() - shard_start,
                        ", ".join(os.path.basename(s) for s in st_files),
                    )
                continue
            except Exception:
                logger.exception(
                    "fastsafetensors failed for %s; falling back to safe_open",
                    ", ".join(st_files),
                )

        for st_file in st_files:
            if wants is not None:
                names = _shard_tensor_names(st_file)
                if names is not None and not any(map(wants, names)):
                    # Nothing in this shard is wanted -- do not read it. Loading
                    # a drafter reads the target's checkpoint to pick out the MTP
                    # block, which is typically one shard out of dozens.
                    pbar.update(1)
                    continue

            # Advise kernel for sequential read-ahead (mmap optimization)
            if not disable_mmap and hasattr(os, "posix_fadvise"):
                try:
                    fd = os.open(st_file, os.O_RDONLY)
                    file_size = os.fstat(fd).st_size
                    os.posix_fadvise(
                        fd,
                        0,
                        file_size,
                        os.POSIX_FADV_SEQUENTIAL | os.POSIX_FADV_WILLNEED,
                    )
                    os.close(fd)
                except OSError:
                    pass

            if disable_mmap:
                # `safetensors.torch.load` has no partial API, so a shard that
                # holds anything wanted is still deserialized whole.
                with open(st_file, "rb") as f:
                    result = safetensors.torch.load(f.read())
                    for name, param in result.items():
                        if wants is None or wants(name):
                            yield name, param
            else:
                with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                    # `.keys()` is not redundant here: `safe_open` is a Rust
                    # object with no `__iter__`, so iterating it directly raises
                    # TypeError.
                    for name in f.keys():  # noqa: SIM118
                        if wants is None or wants(name):
                            yield name, f.get_tensor(name)
            pbar.update(1)

    pbar.close()
