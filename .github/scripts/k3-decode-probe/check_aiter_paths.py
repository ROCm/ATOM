"""Check the image against the reviewed A4W4 source without importing GPU code."""

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

expected_env = {
    "VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4": "0",
    "AITER_SITUV2_A8W4": "0",
    "AITER_SITUV2_A4W4": "1",
    "AITER_FLYDSL_STAGE2_FP8": "1",
}
environment = {key: os.environ.get(key) for key in expected_env}
if environment != expected_env:
    raise RuntimeError(f"Unexpected MoE experiment environment: {environment}")
if os.environ.get("AITER_CONFIG_FMOE"):
    raise RuntimeError("This experiment requires the image's default FMOE configs")

spec = importlib.util.find_spec("aiter")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate the image's AITER source")
root = Path(spec.origin).parent
# Pin the reviewed selectors, shuffle, dispatch, registry, and tuned config.
reviewed_sources = {
    "fused_moe.py": "504680065ae94fdc550eaf219aafad45a1e73b9c73caeeb5675ff7fc406d408a",
    "ops/shuffle.py": "38f41cd233c529cba6d18f0654c267c3279df75d50295263ffa3f0df8f427b0b",
    "ops/flydsl/moe_kernels.py": "79bc4526068da2418cc7c21501d778fa7c8a46f555bc8720ff4dede2d73f0a60",
    "configs/model_configs/kimik3_a4w4_tuned_fmoe.csv": "a61bccf9b2cc39e2b9b07a6188fe39b7f5e15a00d785b5940b580df924778d35",
    "ops/flydsl/kernels/mxfp4_gemm_common.py": "8972e9d4f3b1611d945ffa4b6d4200b2402e5bd2e4149cd6995b851f0232d33d",
    "ops/flydsl/kernels/mxmoe_dispatcher.py": "e4510dd9c990c0d0880cd8630ad28f86101d509bc20bc2531a6e09286dbe95de",
    "jit/core.py": "15df98eaf0b9b87ccf79fa6b7c46c318dbe2e903eb1599c48669b0bb1cf48fd7",
    "ops/flydsl/mxfp4_kname.py": "de98ccb74e83cdcb9dc95da4907f594af15a2cc99be197552d0b89cf7484330b",
    "utility/fp4_utils.py": "6f4ecb4dd9eb22d30653264c851d000ce95b0ec8bc1fb40718b72c96eb3b70eb",
}
source_hashes = {
    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in reviewed_sources
}
# Match the runtime shape across the default merge inputs without calling AITER.
tuned_name = "configs/model_configs/kimik3_a4w4_tuned_fmoe.csv"
shape_numbers = {
    "cu_num": 256,
    "model_dim": 3584,
    "inter_dim": 384,
    "expert": 896,
    "topk": 16,
    "use_g1u1": 1,
    "doweight_stage1": 0,
}
shape_strings = {
    "act_type": "ActivationType.Situv2",
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float4_e2m1fn_x2",
    "q_dtype_w": "torch.float4_e2m1fn_x2",
    "q_type": "QuantType.per_1x32",
}
config_files = [root / "configs/tuned_fmoe.csv"]
config_files.extend(
    p
    for p in (root / "configs/model_configs").glob("*tuned_fmoe*.csv")
    if p.is_file() and "untuned" not in p.name
)
matching_rows = []
for path in config_files:
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if any(
                row.get(key, "").strip() != value
                for key, value in shape_strings.items()
            ):
                continue
            if any(
                float(row.get(key) or -1) != value
                for key, value in shape_numbers.items()
            ):
                continue
            # Legacy CU256 configs map to gfx950 in the reviewed merge code.
            if row.get("gfx", "").strip() not in ("", "0", "nan", "None", "gfx950"):
                continue
            matching_rows.append(
                {
                    "source": str(path.relative_to(root)),
                    "token": int(row["token"]),
                    "stage1": row["kernelName1"],
                    "stage2": row["kernelName2"],
                }
            )
expected_tokens = {
    1,
    2,
    3,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
}
config_matches = (
    len(matching_rows) == len(expected_tokens)
    and {row["token"] for row in matching_rows} == expected_tokens
    and all(row["source"] == tuned_name for row in matching_rows)
)

versions = {}
for distribution in ("amd_aiter", "aiter"):
    try:
        versions[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
mismatched = [
    name for name, wanted in reviewed_sources.items() if source_hashes[name] != wanted
]
print(
    "A4W4-PREFLIGHT "
    + json.dumps(
        {
            "distributions": versions,
            "source_root": str(root),
            "source_hashes": source_hashes,
            "environment": environment,
            "mismatched_sources": mismatched,
            "matching_config_rows": matching_rows,
            "config_matches_reviewed_source": config_matches,
            "scope": "Source and switches only; runtime dispatch and accuracy still require GPU validation.",
        },
        sort_keys=True,
    ),
    flush=True,
)
if mismatched:
    raise RuntimeError(f"Image AITER sources require review: {mismatched}")

if not config_matches:
    raise RuntimeError("Image K3 FMOE rows differ from the reviewed default merge")
