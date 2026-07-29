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
    """
    logger.info(f"disable_mmap: {disable_mmap}")
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

    iters = tqdm(
        hf_weights_files,
        desc=f"Loading safetensors shards[{model_name_or_path}]",
        disable=not enable_tqdm,
    )
    for st_file in iters:
        if wants is not None:
            names = _shard_tensor_names(st_file)
            if names is not None and not any(map(wants, names)):
                # Nothing in this shard is wanted -- do not read it. Loading a
                # drafter reads the target's checkpoint to pick out the MTP
                # block, which is typically one shard out of dozens.
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
                # `.keys()` is not redundant here: `safe_open` is a Rust object
                # with no `__iter__`, so iterating it directly raises TypeError.
                for name in f.keys():  # noqa: SIM118
                    if wants is None or wants(name):
                        yield name, f.get_tensor(name)
