# SPDX-License-Identifier: MIT
"""DSpark deployment identity, and the optional calibration_profile interface.

`deployment_identity` keys what a cost measurement is valid for. It serves two
consumers: the startup SPS cache (`startup_sps.py`), and validation of an
externally supplied `calibration_profile` (main's existing interface; no
in-tree tool produces one -- startup SPS plus online STS replace it).
"""

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import torch


def deployment_identity(config, device):
    from atom.model_loader.weight_utils import local_model_dir

    # A hub id joined onto file names is a relative path that does not exist.
    model = Path(local_model_dir(config.model))
    files = ("config.json", "model.safetensors.index.json")
    return {
        "model_files": {
            name: hashlib.sha256((model / name).read_bytes()).hexdigest()
            for name in files
        },
        "model_type": config.hf_config.model_type,
        "tensor_parallel_size": config.tensor_parallel_size,
        "parallelism": {
            "data_parallel_size": getattr(
                getattr(config, "parallel_config", None), "data_parallel_size", 1
            ),
            "dp_attention": getattr(config, "enable_dp_attention", False),
            "expert_parallel": getattr(config, "enable_expert_parallel", False),
            "moe_all2all_backend": getattr(config, "moe_all2all_backend", "auto"),
            "moe_ep_flatten_tp_across_dp": getattr(
                config, "moe_ep_flatten_tp_across_dp", False
            ),
        },
        "kv_cache_dtype": config.kv_cache_dtype,
        "index_cache_dtype": config.index_cache_dtype,
        "graph": not config.enforce_eager,
        "graph_mode": (
            config.compilation_config.cudagraph_mode.name
            if config.compilation_config.cudagraph_mode is not None
            else None
        ),
        "compilation_level": config.compilation_config.level,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }


# Fields a profile may omit: profiles in main's format predate them. A
# profile without `parallelism` was measured on plain TP, so it is admitted
# only there; without the graph fields it is admitted as main admits it.
_PLAIN_TP = {
    "data_parallel_size": 1,
    "dp_attention": False,
    "expert_parallel": False,
    "moe_all2all_backend": "auto",
    "moe_ep_flatten_tp_across_dp": False,
}
_OPTIONAL = ("parallelism", "graph_mode", "compilation_level")


def validate_calibration(profile, identity, *, width, max_batch):
    recorded = profile.get("identity")
    if profile.get("version") != 1 or not isinstance(recorded, dict):
        raise ValueError(
            "DSpark calibration does not match this model/hardware/runtime"
        )
    expected = dict(identity)
    for field in _OPTIONAL:
        if field in recorded:
            continue
        if field == "parallelism" and identity.get(field, _PLAIN_TP) != _PLAIN_TP:
            raise ValueError(
                "DSpark calibration was measured on plain TP and cannot be "
                "reused for DP or EP"
            )
        expected.pop(field, None)
    if recorded != expected:
        raise ValueError(
            "DSpark calibration does not match this model/hardware/runtime"
        )
    if profile.get("draft_width") != width:
        raise ValueError("DSpark calibration draft width mismatch")
    if max_batch > profile.get("max_num_seqs", 0):
        raise ValueError("DSpark calibration does not cover the configured batch size")
    temperatures, sps = (
        profile.get("sts_temperatures", []),
        profile.get("sps_table", []),
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
        deployment_identity(config, device),
        width=config.speculative_config.num_speculative_tokens,
        max_batch=config.max_num_seqs,
    )
    return (
        torch.tensor(profile["sps_table"], dtype=torch.float32, device=device),
        torch.tensor(profile["sts_temperatures"], dtype=torch.float32, device=device),
    )
