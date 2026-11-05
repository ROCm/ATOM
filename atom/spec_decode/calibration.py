# SPDX-License-Identifier: MIT
"""Offline DSpark calibration artifacts and their deployment compatibility."""

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import torch


def calibration_identity(config, device):
    model = Path(config.model)
    files = ("config.json", "model.safetensors.index.json")
    return {
        "model_files": {
            name: hashlib.sha256((model / name).read_bytes()).hexdigest()
            for name in files
        },
        "model_type": config.hf_config.model_type,
        "tensor_parallel_size": config.tensor_parallel_size,
        "kv_cache_dtype": config.kv_cache_dtype,
        "index_cache_dtype": config.index_cache_dtype,
        "graph": not config.enforce_eager,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }


def validate_calibration(profile, identity, *, width, max_batch):
    if profile.get("version") != 1 or profile.get("identity") != identity:
        raise ValueError(
            "DSpark calibration does not match this model/hardware/runtime"
        )
    if profile.get("draft_width") != width:
        raise ValueError("DSpark calibration draft width mismatch")
    if max_batch > profile.get("max_num_seqs", 0):
        raise ValueError("DSpark calibration does not cover the configured batch size")
    temperatures, sps = profile.get("sts_temperatures", []), profile.get(
        "sps_table", []
    )
    if len(temperatures) != width or len(sps) <= max_batch * (width + 1):
        raise ValueError("DSpark calibration table is incomplete")
    if any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0
        for x in (*temperatures, *sps)
    ):
        raise ValueError("DSpark calibration values must be finite and positive")
    if any(b > a for a, b in pairwise(sps)):
        raise ValueError("DSpark SPS must be monotone non-increasing")


def load_calibration(config, device):
    path = config.dspark.calibration_profile
    profile = json.loads(Path(path).read_text())
    validate_calibration(
        profile,
        calibration_identity(config, device),
        width=config.speculative_config.num_speculative_tokens,
        max_batch=config.max_num_seqs,
    )
    return (
        torch.tensor(profile["sps_table"], dtype=torch.float32, device=device),
        torch.tensor(profile["sts_temperatures"], dtype=torch.float32, device=device),
    )
