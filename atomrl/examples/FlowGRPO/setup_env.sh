#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# One-time setup inside the container: pinned sources, the RL stack (without touching the image's ROCm torch / vLLM /
# triton), pinned models and the OCR dataset. Safe to rerun.
# Lumen actor patches come from the in-tree atomrl/lumen checkout.
# Usage (container, root): bash setup_env.sh
set -euo pipefail
: "${WORK:?}"
SRC=$WORK/src
ATOMRL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ATOM_ROOT="$(cd "${ATOMRL_ROOT}/.." && pwd)"
LUMEN_DIR="${LUMEN_DIR:-${ATOMRL_ROOT}/Lumen}"
mkdir -p "$SRC" "$WORK/models" "$WORK/data/ocr" "$WORK/env"

clone() {  # name url commit
  [ -d "$SRC/$1/.git" ] || git clone -q "$2" "$SRC/$1"
  git config --global --get-all safe.directory | grep -qx "$SRC/$1" || git config --global --add safe.directory "$SRC/$1"
  git -C "$SRC/$1" checkout -q --detach "$3"
}
clone verl https://github.com/verl-project/verl.git fefb080262e1c015a0ea05f958822a6a512dc795
clone verl-omni https://github.com/verl-project/verl-omni.git 9ab544d81f93cbf49754aabacfa67f6fcf01b505
clone VeOmni https://github.com/ByteDance-Seed/VeOmni.git 573848a00fcd7329c2411346c6f4a983e9f67e3f
clone vllm-omni https://github.com/vllm-project/vllm-omni.git 444485650b19b792403b12976f1b0eeb2ac1451c
clone flow_grpo https://github.com/yifan123/flow_grpo.git 879042cf5707f8b90daa98d147d7deac2317c5da
[ -d "${LUMEN_DIR}/examples/perf-veomni/veomni_patches" ] || {
  echo "In-tree Lumen patches not found at ${LUMEN_DIR}" >&2
  exit 1
}

# The image's ROCm stack is frozen; transformers 5.10.1 is the only release accepted by verl, verl-omni and vllm-omni.
python3 - > "$WORK/env/constraints.txt" <<'PY'
import importlib.metadata as m
for p in ["torch", "torchvision", "torchaudio", "triton", "vllm", "flash_attn", "amdsmi", "numpy"]:
    try:
        print(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError:
        pass
PY
cat >> "$WORK/env/constraints.txt" <<'EOF'
transformers==5.10.1
pyarrow>=19.0.0,<25.0.0
tensordict>=0.8.0,<=0.10.0,!=0.9.0
diffusers==0.38.0
accelerate>=1.14.0
EOF
PIP=(python3 -m pip install -q --no-cache-dir -c "$WORK/env/constraints.txt")
command -v ffmpeg >/dev/null || { apt-get update -q && apt-get install -y -q --no-install-recommends ffmpeg git jq; }
"${PIP[@]}" \
  "transformers==5.10.1" "diffusers==0.38.0" "av>=14.0.0" "omegaconf>=2.3.0" "safetensors>=0.8.0" \
  "soundfile>=0.13.1" "cache-dit==1.5.0" "tqdm>=4.66.0" "torchsde>=0.2.6" "imageio[ffmpeg]>=2.37.2" \
  "x-transformers>=2.12.2" "einops>=0.8.1" "prettytable>=3.8.0" "aenum==3.1.16" "pyzmq>=25.0.0" \
  "janus>=1.0.0" "msgpack>=1.0.0" "pydantic>=2.1.0" "pydantic-settings>=2.0.0" "gguf>=0.10.0" "torchmetrics==1.9.0"
"${PIP[@]}" \
  cachetools codetiming datasets dill hydra-core pandas peft "pyarrow>=19.0.0,<25.0.0" pybind11 pylatexenc \
  "ray[default]" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata wandb \
  "packaging>=20.0" uvicorn fastapi latex2sympy2_extended math_verify tensorboard "TransferQueue==0.1.8" \
  Levenshtein "qwen-vl-utils>=0.0.14"
"${PIP[@]}" "blobfile>=3.0.0" "tiktoken>=0.9.0" psutil timm "matplotlib>=3.7" setuptools librosa audioread
# ATOM metadata pins Transformers 5.16.1. FlowGRPO keeps its validated 5.10.1
# stack above and therefore installs the unified checkout with --no-deps.
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ATOM=0.1.0 \
  LUMEN_BUILD_EXTENSIONS=1 \
  python3 -m pip install -q --no-cache-dir --no-deps --no-build-isolation -e "$ATOM_ROOT"
# onnxruntime CPU and ROCm wheels share one import path.
if python3 -m pip show onnxruntime >/dev/null 2>&1; then python3 -m pip uninstall -y -q onnxruntime; fi
for repo in verl vllm-omni VeOmni verl-omni; do
  python3 -m pip install -q --no-cache-dir --no-deps --no-build-isolation -e "$SRC/$repo"
done

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
w = os.environ["WORK"]
for repo, rev in [("Qwen/Qwen-Image", "75e0b4be04f60ec59a75f475837eced720f823b6"),
                  ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")]:
    snapshot_download(repo_id=repo, revision=rev, local_dir=f"{w}/models/{repo.split('/')[1]}")
PY

cd "$SRC/verl-omni"
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir "$SRC/flow_grpo/dataset/ocr" --output_dir "$WORK/data/ocr"

python3 - <<'PY'
import os, importlib.metadata as m, pyarrow.parquet as pq
w = os.environ["WORK"]
print({p: m.version(p) for p in ["torch", "vllm", "transformers", "diffusers", "ray", "veomni", "verl", "verl-omni", "vllm-omni"]})
print({s: pq.read_metadata(f"{w}/data/ocr/{s}.parquet").num_rows for s in ["train", "test"]}, "(expect 19653 / 1018)")
print("SETUP OK")
PY
