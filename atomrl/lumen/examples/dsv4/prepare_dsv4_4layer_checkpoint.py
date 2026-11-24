# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Prepare HF → BF16 → Megatron torch_dist checkpoint for DSV4 4-layer pretrain smoke.

Uses Miles ``convert_hf_to_torch_dist.py`` via shell (no ``train.py`` / Ray head).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

MILES_DIR = Path(os.environ.get("MILES_DIR", "/workspace/miles"))
LUMEN_DIR = Path(os.environ.get("LUMEN_DIR", "/workspace/Lumen"))
ATOMRL_DIR = Path(os.environ.get("ATOMRL_DIR", "/workspace/ATOM/atomrl"))


def _run(cmd: str) -> None:
    print(f"[prepare] $ {cmd}")
    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)


def _patch_hc_mult(model_dir: Path, hc_mult: int = 2) -> None:
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("hc_mult", 4) != hc_mult:
        cfg["hc_mult"] = hc_mult
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[prepare] patched config.json hc_mult → {hc_mult}")


def _fp8_to_bf16(fp8_dir: Path, bf16_dir: Path, miles_dir: Path) -> None:
    sentinel = bf16_dir / "model.safetensors.index.json"
    if sentinel.exists():
        print(f"[prepare] skip FP8→BF16 ({sentinel} exists)")
        return
    _run(
        f"python {miles_dir}/tools/fp8_cast_bf16.py "
        f"--input-fp8-hf-path {fp8_dir} "
        f"--output-bf16-hf-path {bf16_dir}"
    )


def _convert_torch_dist(
    *,
    miles_dir: Path,
    megatron_path: Path,
    model_type: str,
    bf16_dir: Path,
    torch_dist: Path,
    hc_mult: int,
) -> None:
    tracker = torch_dist / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        print(f"[prepare] skip torch_dist ({tracker} exists)")
        return

    megatron_path = str(megatron_path)
    lumen_spec = "atomrl.lumen.models.dsv4.megatron.spec get_dsv4_spec"
    cmd = (
        f"source {miles_dir}/scripts/models/{model_type}.sh && "
        f"PYTHONPATH={ATOMRL_DIR}:{miles_dir}:{megatron_path}:$PYTHONPATH "
        f"torchrun --nproc-per-node 4 "
        f"{miles_dir}/tools/convert_hf_to_torch_dist.py "
        f"${{MODEL_ARGS[@]}} "
        f"--spec {lumen_spec} "
        f"--dsv4-hc-mult {hc_mult} "
        f"--hf-checkpoint {bf16_dir} "
        f"--save {torch_dist} "
        f"--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size 1 "
        f"--expert-model-parallel-size 1 "
        f"--expert-tensor-parallel-size 1 "
        f"--context-parallel-size 1 "
    )
    print(f"[prepare] BF16 → Megatron torch_dist ({torch_dist}) ...")
    _run(cmd)


def prepare(
    *,
    model_dir: str = "/root/models",
    model_name: str = "DeepSeek-V4-Flash-FP8-4layer",
    megatron_path: str | None = None,
    hc_mult: int = 2,
    miles_dir: Path | None = None,
) -> Path:
    miles_dir = miles_dir or MILES_DIR
    megatron_path = Path(
        megatron_path or os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")
    )
    model_root = Path(model_dir)
    hf_dir = model_root / model_name
    bf16_dir = model_root / f"{model_name}-bf16"
    torch_dist = model_root / f"{model_name}_torch_dist_hc{hc_mult}"
    model_type = "deepseek-v4-flash-4layer"

    if not hf_dir.is_dir():
        _run(f"mkdir -p {model_root}")
        _run(f"hf download Pinaster/DeepSeek-V4-Flash-FP8-4layer --local-dir {hf_dir}")

    cfg_path = hf_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("model_type") == "deepseek_ref":
            cfg["model_type"] = "deepseek_v4"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            print("[prepare] patched model_type deepseek_ref → deepseek_v4")

    _patch_hc_mult(hf_dir, hc_mult=hc_mult)
    _fp8_to_bf16(hf_dir, bf16_dir, miles_dir)
    _convert_torch_dist(
        miles_dir=miles_dir,
        megatron_path=megatron_path,
        model_type=model_type,
        bf16_dir=bf16_dir,
        torch_dist=torch_dist,
        hc_mult=hc_mult,
    )
    return torch_dist


def main() -> None:
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(var, None)

    if not MILES_DIR.is_dir():
        print(f"[prepare] ERROR: MILES_DIR not found: {MILES_DIR}", file=sys.stderr)
        sys.exit(1)

    ckpt = prepare(
        model_dir=os.environ.get("MODEL_DIR", "/root/models"),
        megatron_path=os.environ.get("MEGATRON_PATH"),
        hc_mult=int(os.environ.get("DSV4_HC_MULT", "2")),
    )
    print(f"[prepare] checkpoint ready: {ckpt}")


if __name__ == "__main__":
    main()
