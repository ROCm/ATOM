#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Prepare HF → BF16 → Megatron torch_dist for DSV4 Flash full model (43 layers).

Uses Miles convert_hf_to_torch_dist.py. For 16-GPU training layout, convert with
TP4/PP4/EP4 on 2 nodes (same as miles prepare-spmd).

If checkpoint already exists (e.g. from Miles), conversion is skipped.
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
    print(f"[prepare-full] $ {cmd}")
    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)


def _fp8_to_bf16(fp8_dir: Path, bf16_dir: Path, miles_dir: Path) -> None:
    sentinel = bf16_dir / "model.safetensors.index.json"
    if sentinel.exists():
        print(f"[prepare-full] skip FP8→BF16 ({sentinel} exists)")
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
    bf16_dir: Path,
    torch_dist: Path,
    nnodes: int,
    nproc_per_node: int,
    master_addr: str,
    master_port: int,
    node_rank: int,
) -> None:
    tracker = torch_dist / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        print(f"[prepare-full] skip torch_dist ({tracker} exists)")
        return

    megatron_path = str(megatron_path)
    lumen_spec = "atomrl.lumen.models.dsv4.megatron.spec get_dsv4_spec"
    cmd = (
        f"source {miles_dir}/scripts/models/deepseek-v4-flash.sh && "
        f"PYTHONPATH={ATOMRL_DIR}:{miles_dir}:{megatron_path}:$PYTHONPATH "
        f"torchrun --nnodes {nnodes} --nproc_per_node {nproc_per_node} "
        f"--node_rank {node_rank} --master_addr {master_addr} --master_port {master_port} "
        f"{miles_dir}/tools/convert_hf_to_torch_dist.py "
        f"${{MODEL_ARGS[@]}} "
        f"--spec {lumen_spec} "
        f"--hf-checkpoint {bf16_dir} "
        f"--save {torch_dist} "
        f"--tensor-model-parallel-size 4 "
        f"--pipeline-model-parallel-size 4 "
        f"--decoder-first-pipeline-num-layers 11 "
        f"--decoder-last-pipeline-num-layers 10 "
        f"--expert-model-parallel-size 4 "
        f"--expert-tensor-parallel-size 1 "
        f"--context-parallel-size 1 "
        f"--sequence-parallel "
    )
    print(f"[prepare-full] BF16 → Megatron torch_dist ({torch_dist}) ...")
    _run(cmd)


def prepare(
    *,
    model_dir: str = "/root/models",
    model_name: str = "DeepSeek-V4-Flash-FP8",
    megatron_path: str | None = None,
    miles_dir: Path | None = None,
    nnodes: int = 2,
    nproc_per_node: int = 8,
    master_addr: str = "127.0.0.1",
    master_port: int = 29501,
    node_rank: int = 0,
) -> Path:
    miles_dir = miles_dir or MILES_DIR
    megatron_path = Path(
        megatron_path or os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")
    )
    model_root = Path(model_dir)
    hf_dir = model_root / model_name
    bf16_dir = model_root / f"{model_name}-bf16"
    torch_dist = model_root / f"{model_name}_torch_dist"

    if not hf_dir.is_dir():
        _run(f"mkdir -p {model_root}")
        _run(f"hf download sgl-project/DeepSeek-V4-Flash-FP8 --local-dir {hf_dir}")

    cfg_path = hf_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("model_type") == "deepseek_ref":
            cfg["model_type"] = "deepseek_v4"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            print("[prepare-full] patched model_type deepseek_ref → deepseek_v4")

    _fp8_to_bf16(hf_dir, bf16_dir, miles_dir)
    _convert_torch_dist(
        miles_dir=miles_dir,
        megatron_path=megatron_path,
        bf16_dir=bf16_dir,
        torch_dist=torch_dist,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        master_addr=master_addr,
        master_port=master_port,
        node_rank=node_rank,
    )
    return torch_dist


def main() -> None:
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(var, None)

    if not MILES_DIR.is_dir():
        print(
            f"[prepare-full] ERROR: MILES_DIR not found: {MILES_DIR}", file=sys.stderr
        )
        sys.exit(1)

    ckpt = prepare(
        model_dir=os.environ.get("MODEL_DIR", "/root/models"),
        megatron_path=os.environ.get("MEGATRON_PATH"),
        nnodes=int(os.environ.get("NNODES", "2")),
        nproc_per_node=int(os.environ.get("NPROC_PER_NODE", "8")),
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=int(os.environ.get("MASTER_PORT", "29501")),
        node_rank=int(os.environ.get("NODE_RANK", "0")),
    )
    print(f"[prepare-full] checkpoint ready: {ckpt}")


if __name__ == "__main__":
    main()
