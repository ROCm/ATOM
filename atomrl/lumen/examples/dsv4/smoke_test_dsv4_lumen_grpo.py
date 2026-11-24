# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen DSV4 4-layer GRPO full finetune on 8×MI308X.

Uses Miles ``train.py`` (Ray + Megatron actor) with Lumen ``get_dsv4_spec``.
Default: ``--debug-train-only`` + pre-generated GSM8K rollout (no SGLang), same
training mode as Miles ``smoke_test_dsv4_mi308x.py`` — full-parameter GRPO update,
not Megatron pretrain / mock LM loss.

Set ``DEBUG_TRAIN_ONLY=0`` to attempt live SGLang colocated rollout (experimental on ROCm).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MILES_DIR = Path(os.environ.get("MILES_DIR", "/workspace/miles"))
LUMEN_DIR = Path(os.environ.get("LUMEN_DIR", "/workspace/Lumen"))
sys.path.insert(0, str(MILES_DIR))

from scripts.amd.run_deepseek_v4 import (
    ScriptArgs,
    _download_dataset,
    _ensure_4layer_model_type,
    _train,
)

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("RCCL_MSCCL_ENABLE", "0")

FAKE_ROLLOUT_DATA = os.environ.get(
    "FAKE_ROLLOUT_DATA",
    f"{os.environ.get('MODEL_DIR', '/root/models')}/fake_rollout.pt",
)


def _resolve_ref_load(model_dir: str, model_name: str, hc_mult: int) -> str:
    for suffix in (f"_torch_dist_hc{hc_mult}", "_torch_dist"):
        candidate = Path(model_dir) / f"{model_name}{suffix}"
        if (candidate / "latest_checkpointed_iteration.txt").exists():
            return str(candidate)
    return str(Path(model_dir) / f"{model_name}_torch_dist_hc{hc_mult}")


def _lumen_extra_args() -> str:
    model_name = os.environ.get("MODEL_NAME", "DeepSeek-V4-Flash-FP8-4layer")
    model_dir = os.environ.get("MODEL_DIR", "/root/models")
    hc_mult = int(os.environ.get("DSV4_HC_MULT", "2"))
    num_rollout = os.environ.get("NUM_ROLLOUT", "10")
    debug_only = os.environ.get("DEBUG_TRAIN_ONLY", "1") != "0"

    parts = [
        f"--ref-load {_resolve_ref_load(model_dir, model_name, hc_mult)}",
        "--spec atomrl.lumen.models.dsv4.megatron.spec get_dsv4_spec",
        "--transformer-impl local",
        "--miles-dsa-topk-backend torch",
        f"--dsv4-hc-mult {hc_mult}",
    ]

    if debug_only:
        parts.extend(
            [
                "--debug-train-only",
                f"--load-debug-rollout-data {FAKE_ROLLOUT_DATA}",
                "--use-rollout-logprobs",
                "--no-offload-train",
                "--no-offload-rollout",
                f"--num-rollout {num_rollout}",
                "--num-steps-per-rollout 1",
                "--ci-test",
                "--disable-weights-backuper",
                "--check-weight-update-allow-quant-error",
                "--ci-disable-logprobs-checker",
                "--ci-disable-kl-checker",
            ]
        )
    else:
        parts.extend(
            [
                f"--num-rollout {num_rollout}",
                "--num-steps-per-rollout 1",
                "--colocate",
            ]
        )

    extra = os.environ.get("DSV4_GRPO_EXTRA_ARGS", "").strip()
    if extra:
        parts.append(extra)
    return " ".join(parts)


def _runtime_env_json() -> str:
    payload = {
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        "LUMEN_DIR": str(LUMEN_DIR),
    }
    return json.dumps(payload)


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name=os.environ.get("MODEL_NAME", "DeepSeek-V4-Flash-FP8-4layer"),
        task=os.environ.get("TASK", "gsm8k"),
        enable_eval=os.environ.get("ENABLE_EVAL", "0") == "1",
        num_nodes=1,
        num_gpus_per_node=int(os.environ.get("NUM_GPUS", "8")),
        model_dir=os.environ.get("MODEL_DIR", "/root/models"),
        save_dir=os.environ.get("MODEL_DIR", "/root/models"),
        data_dir=os.environ.get("DATA_DIR", "/root/datasets"),
        megatron_path=os.environ.get(
            "MEGATRON_PATH", "/opt/dsv4-bootstrap/Megatron-LM"
        ),
        skip_saving=True,
        use_fault_tolerance=False,
        fp8_training=os.environ.get("FP8_TRAINING", "0") == "1",
        optimizer_offload=os.environ.get("OPTIMIZER_OFFLOAD", "1") == "1",
        enable_r3=False,
        extra_env_vars=_runtime_env_json(),
        extra_args=_lumen_extra_args(),
    )


def _prepare_checkpoint(args: ScriptArgs) -> None:
    if os.environ.get("SKIP_PREPARE", "0") == "1":
        print("[finetune] SKIP_PREPARE=1 — skipping checkpoint prep")
        return
    sys.path.insert(0, str(LUMEN_DIR / "examples" / "dsv4"))
    from prepare_dsv4_4layer_checkpoint import prepare as lumen_prepare  # noqa: WPS433

    lumen_prepare(
        model_dir=args.model_dir,
        model_name=args.model_name,
        megatron_path=args.megatron_path,
        hc_mult=int(os.environ.get("DSV4_HC_MULT", "2")),
        miles_dir=MILES_DIR,
    )


def _prepare_rollout_data(args: ScriptArgs) -> None:
    if os.environ.get("DEBUG_TRAIN_ONLY", "1") == "0":
        print("[finetune] live rollout mode — skipping fake rollout.pt generation")
        return

    sys.path.insert(0, str(LUMEN_DIR / "examples" / "dsv4"))
    from prepare_dsv4_fake_rollout import prepare as prepare_rollout  # noqa: WPS433

    prepare_rollout(
        output_path=FAKE_ROLLOUT_DATA,
        model_dir=args.model_dir,
        model_name=args.model_name,
        data_dir=args.data_dir,
        miles_dir=str(MILES_DIR),
    )


def prepare(args: ScriptArgs) -> None:
    _ensure_4layer_model_type(args)
    _download_dataset(args)
    _prepare_checkpoint(args)
    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"
    _prepare_rollout_data(args)


def execute(args: ScriptArgs) -> None:
    _train(args)


if __name__ == "__main__":
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(var, None)

    if not MILES_DIR.is_dir():
        print(f"[finetune] ERROR: MILES_DIR not found: {MILES_DIR}", file=sys.stderr)
        sys.exit(1)

    args = _args()
    print(f"[finetune] model      = {args.model_name}")
    print(
        "[finetune] spec       = atomrl.lumen.models.dsv4.megatron.spec get_dsv4_spec"
    )
    print(
        f"[finetune] mode       = {'debug-train-only (GRPO actor)' if os.environ.get('DEBUG_TRAIN_ONLY', '1') != '0' else 'colocate GRPO + SGLang rollout'}"
    )
    print(
        f"[finetune] ref_load   = {_resolve_ref_load(args.model_dir, args.model_name, int(os.environ.get('DSV4_HC_MULT', '2')))}"
    )
    print(f"[finetune] num_rollout= {os.environ.get('NUM_ROLLOUT', '10')}")

    prepare(args)
    execute(args)
