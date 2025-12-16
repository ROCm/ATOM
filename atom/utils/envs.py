# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Callable, Any
# Data parallel environment variables

environment_variables: dict[str, Callable[[], Any]] = {
    "ATOM_DP_RANK": lambda: int(os.getenv("ATOM_DP_RANK", "0")),
    "ATOM_DP_RANK_LOCAL": lambda: int(os.getenv("ATOM_DP_RANK_LOCAL", "0")),
    "ATOM_DP_SIZE": lambda: int(os.getenv("ATOM_DP_SIZE", "1")),
    "ATOM_DP_MASTER_IP": lambda: os.getenv("ATOM_DP_MASTER_IP", "127.0.0.1"),
    "ATOM_DP_MASTER_PORT": lambda: int(os.getenv("ATOM_DP_MASTER_PORT", "29500")),
    "ATOM_ENFORCE_EAGER": lambda: os.getenv("ATOM_ENFORCE_EAGER", "0") == "1",
    "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION": lambda: os.getenv("ATOM_ENABLE_DS_QKNORM_QUANT_FUSION", "1") == "1",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION": lambda: os.getenv("ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION", "1") == "1",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION": lambda: os.getenv("ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION", "1") == "1",
    "ATOM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT": lambda: os.getenv("ATOM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT", "1") == "1"
}

def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
