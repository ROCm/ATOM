# SPDX-License-Identifier: MIT
"""Combine measured SPS and independently fitted STS into a checked profile."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from atom.spec_decode.calibration import calibration_identity, validate_calibration

from atom.config import get_hf_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--sps", type=Path, required=True)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sps = json.loads(args.sps.read_text())
    sts = json.loads(args.confidence.read_text())
    if not sps["completed"]:
        raise ValueError("SPS measurement did not complete")
    for key in ("tp", "cache_dtype", "graph"):
        if sps[key] != sts[key]:
            raise ValueError(f"SPS and confidence runs disagree on {key}")
    config = SimpleNamespace(
        model=args.model,
        hf_config=get_hf_config(args.model),
        tensor_parallel_size=sps["tp"],
        kv_cache_dtype=sps["cache_dtype"],
        index_cache_dtype="fp8",
        enforce_eager=not sps["graph"],
    )
    identity = calibration_identity(config, torch.device("cuda", 0))
    for key in ("gpu_name", "torch_version", "hip_version"):
        if identity[key] != sps[key]:
            raise ValueError(f"SPS hardware/software mismatch: {key}")
    profile = {
        "version": 1,
        "identity": identity,
        "draft_width": sps["draft_width"],
        "max_num_seqs": sps["max_num_seqs"],
        "sps_table": sps["sps_table"],
        "sts_temperatures": sts["sts_temperatures"],
        "evidence": {
            "sps_sha256": hashlib.sha256(args.sps.read_bytes()).hexdigest(),
            "confidence_sha256": hashlib.sha256(
                args.confidence.read_bytes()
            ).hexdigest(),
            "confidence_workload": sts["workload"],
            "confidence_metrics": sts["cohorts"],
            "sps_measurement": sps["measurement"],
            "sps_token_points": sps["token_points"],
        },
    }
    validate_calibration(profile, identity, width=5, max_batch=sps["max_num_seqs"])
    args.output.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
