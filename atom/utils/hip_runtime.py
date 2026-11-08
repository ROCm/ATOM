# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The HIP runtime this process runs, for the ctypes calls torch does not wrap."""

import ctypes
import functools
import logging
import os

import torch

logger = logging.getLogger("atom")


@functools.cache
def hip_runtime():
    """The HIP runtime this process already runs -- the `libamdhip64` torch
    loaded (a ROCm wheel bundles its own, e.g. `torch/lib/libamdhip64.so.7`)
    -- or None.

    Not `CDLL("libamdhip64.so")`: that is the system's unversioned dev link,
    which a wheel-only install lacks (an Engram shard copy would quietly turn
    into an in-place registration) and which can name a second, different
    runtime than torch's, with its own allocations and last error.
    """
    if not torch.version.hip:
        return None
    candidates = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.split(maxsplit=5)[-1].strip()
                name = os.path.basename(path)
                if name.startswith("libamdhip64.so") and path not in candidates:
                    candidates.append(path)
    except OSError:
        pass
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    try:
        candidates += sorted(
            os.path.join(torch_lib, n)
            for n in os.listdir(torch_lib)
            if n.startswith("libamdhip64.so")
        )
    except OSError:
        pass
    candidates.append("libamdhip64.so")
    for path in candidates:
        try:
            hip = ctypes.CDLL(path)
        except OSError:
            continue
        logger.info("HIP runtime: %s", path)
        return hip
    logger.warning(
        "no HIP runtime found (tried %s); an Engram UVA shard is registered in "
        "place instead of copied",
        ", ".join(candidates),
    )
    return None
