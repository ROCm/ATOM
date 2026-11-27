# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Prepare HF → BF16 → Megatron torch_dist checkpoint for DSV4.

Profile via ``DSV4_PROFILE=4layer|flash`` (default ``4layer``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

LUMEN_DIR = Path(os.environ.get("LUMEN_DIR", "/workspace/Lumen"))
ATOMRL_DIR = Path(os.environ.get("ATOMRL_DIR", "/workspace/ATOM/atomrl"))
TOOLS_DIR = LUMEN_DIR / "examples/dsv4/tools"
PROFILE = os.environ.get("DSV4_PROFILE", "4layer").lower()


def _run(cmd: str, *, tag: str = "prepare") -> None:
    print(f"[{tag}] $ {cmd}")
    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)


def _patch_hc_mult(model_dir: Path, hc_mult: int) -> None:
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("hc_mult", 4) != hc_mult:
        cfg["hc_mult"] = hc_mult
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[prepare] patched config.json hc_mult → {hc_mult}")


def _patch_model_type(hf_dir: Path) -> None:
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("model_type") == "deepseek_ref":
        cfg["model_type"] = "deepseek_v4"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print("[prepare] patched model_type deepseek_ref → deepseek_v4")


def _fp8_to_bf16(fp8_dir: Path, bf16_dir: Path) -> None:
    sentinel = bf16_dir / "model.safetensors.index.json"
    if sentinel.exists():
        print(f"[prepare] skip FP8→BF16 ({sentinel} exists)")
        return
    _run(
        f"python {TOOLS_DIR}/fp8_cast_bf16.py "
        f"--input-fp8-hf-path {fp8_dir} "
        f"--output-bf16-hf-path {bf16_dir}"
    )


def _convert_4layer(
    *, megatron_path: Path, bf16_dir: Path, torch_dist: Path, hc_mult: int
) -> None:
    tracker = torch_dist / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        print(f"[prepare] skip torch_dist ({tracker} exists)")
        return
    args_sh = LUMEN_DIR / "examples/dsv4/dsv4_megatron_args.sh"
    cmd = (
        f"DSV4_PROFILE=4layer source {args_sh} && "
        f"PYTHONPATH={ATOMRL_DIR}:{megatron_path}:$PYTHONPATH "
        f"torchrun --nproc-per-node 4 "
        f"{TOOLS_DIR}/convert_hf_to_torch_dist.py "
        f"${{DSV4_MODEL_ARGS[@]}} "
        f"--spec atomrl.lumen.models.dsv4.megatron.spec get_dsv4_spec "
        f"--dsv4-hc-mult {hc_mult} "
        f"--hf-checkpoint {bf16_dir} "
        f"--save {torch_dist} "
        f"--tensor-model-parallel-size 1 "
        f"--pipeline-model-parallel-size 1 "
        f"--expert-model-parallel-size 1 "
        f"--expert-tensor-parallel-size 1 "
        f"--context-parallel-size 1 "
    )
    print(f"[prepare] BF16 → torch_dist ({torch_dist}) ...")
    _run(cmd)


def _convert_flash(
    *,
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
        print(f"[prepare] skip torch_dist ({tracker} exists)")
        return
    args_sh = LUMEN_DIR / "examples/dsv4/dsv4_megatron_args.sh"
    cmd = (
        f"DSV4_PROFILE=flash source {args_sh} && "
        f"PYTHONPATH={ATOMRL_DIR}:{megatron_path}:$PYTHONPATH "
        f"torchrun --nnodes {nnodes} --nproc_per_node {nproc_per_node} "
        f"--node_rank {node_rank} --master_addr {master_addr} --master_port {master_port} "
        f"{TOOLS_DIR}/convert_hf_to_torch_dist.py "
        f"${{DSV4_MODEL_ARGS[@]}} "
        f"--spec atomrl.lumen.models.dsv4.megatron.spec get_dsv4_spec "
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
    print(f"[prepare] BF16 → torch_dist ({torch_dist}) ...")
    _run(cmd)


def prepare(
    *,
    profile: str | None = None,
    model_dir: str | None = None,
    model_name: str | None = None,
    megatron_path: str | None = None,
    hc_mult: int | None = None,
    nnodes: int | None = None,
    nproc_per_node: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
    node_rank: int | None = None,
) -> Path:
    profile = (profile or PROFILE).lower()
    megatron_path = Path(
        megatron_path or os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")
    )
    model_root = Path(model_dir or os.environ.get("MODEL_DIR", "/root/models"))

    if profile == "4layer":
        model_name = model_name or os.environ.get(
            "MODEL_NAME", "DeepSeek-V4-Flash-FP8-4layer"
        )
        hc_mult = int(
            hc_mult if hc_mult is not None else os.environ.get("DSV4_HC_MULT", "2")
        )
        hf_dir = model_root / model_name
        bf16_dir = model_root / f"{model_name}-bf16"
        torch_dist = model_root / f"{model_name}_torch_dist_hc{hc_mult}"
        if not hf_dir.is_dir():
            _run(f"mkdir -p {model_root}")
            _run(
                f"hf download Pinaster/DeepSeek-V4-Flash-FP8-4layer --local-dir {hf_dir}"
            )
        _patch_model_type(hf_dir)
        _patch_hc_mult(hf_dir, hc_mult=hc_mult)
        _fp8_to_bf16(hf_dir, bf16_dir)
        _convert_4layer(
            megatron_path=megatron_path,
            bf16_dir=bf16_dir,
            torch_dist=torch_dist,
            hc_mult=hc_mult,
        )
        return torch_dist

    if profile == "flash":
        model_name = model_name or os.environ.get("MODEL_NAME", "DeepSeek-V4-Flash-FP8")
        hf_dir = model_root / model_name
        bf16_dir = model_root / f"{model_name}-bf16"
        torch_dist = model_root / f"{model_name}_torch_dist"
        if not hf_dir.is_dir():
            _run(f"mkdir -p {model_root}")
            _run(f"hf download sgl-project/DeepSeek-V4-Flash-FP8 --local-dir {hf_dir}")
        _patch_model_type(hf_dir)
        _fp8_to_bf16(hf_dir, bf16_dir)
        _convert_flash(
            megatron_path=megatron_path,
            bf16_dir=bf16_dir,
            torch_dist=torch_dist,
            nnodes=int(nnodes if nnodes is not None else os.environ.get("NNODES", "2")),
            nproc_per_node=int(
                nproc_per_node
                if nproc_per_node is not None
                else os.environ.get("NPROC_PER_NODE", "8")
            ),
            master_addr=master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1"),
            master_port=int(
                master_port
                if master_port is not None
                else os.environ.get("MASTER_PORT", "29501")
            ),
            node_rank=int(
                node_rank if node_rank is not None else os.environ.get("NODE_RANK", "0")
            ),
        )
        return torch_dist

    raise ValueError(f"Unknown DSV4_PROFILE={profile!r}; expected 4layer or flash")


def main() -> None:
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(var, None)
    if not TOOLS_DIR.is_dir():
        print(f"[prepare] ERROR: tools dir missing: {TOOLS_DIR}", file=sys.stderr)
        sys.exit(1)
    ckpt = prepare()
    print(f"[prepare] checkpoint ready: {ckpt}")


if __name__ == "__main__":
    main()
