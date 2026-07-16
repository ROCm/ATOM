#!/usr/bin/env bash
set -euo pipefail

LMMS_EVAL_REF="${LMMS_EVAL_REF:-4510f3e9c04be1952b71caea331822e6ddf73d2f}"
LMMS_EVAL_REPO="${LMMS_EVAL_REPO:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
LMMS_EVAL_SRC_DIR="${LMMS_EVAL_SRC_DIR:-/opt/lmms-eval}"

snapshot_protected_versions() {
  python3 - <<'PY'
import importlib.metadata as md

for name in ("torch", "torchvision", "torchaudio", "transformers", "vllm"):
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        version = "<not-installed>"
    print(f"{name}=={version}")
PY
}

before="$(mktemp)"
after="$(mktemp)"
trap 'rm -f "${before}" "${after}"' EXIT

echo "=== Protected package versions before lmms-eval install ==="
snapshot_protected_versions | tee "${before}"

echo "=== Installing lmms-eval from source with --no-deps ==="
echo "LMMS_EVAL_REPO=${LMMS_EVAL_REPO}"
echo "LMMS_EVAL_REF=${LMMS_EVAL_REF}"
echo "LMMS_EVAL_SRC_DIR=${LMMS_EVAL_SRC_DIR}"
if [[ -d "${LMMS_EVAL_SRC_DIR}/.git" ]]; then
  git -C "${LMMS_EVAL_SRC_DIR}" fetch --all --tags
else
  rm -rf "${LMMS_EVAL_SRC_DIR}"
  git clone "${LMMS_EVAL_REPO}" "${LMMS_EVAL_SRC_DIR}"
fi
git -C "${LMMS_EVAL_SRC_DIR}" checkout "${LMMS_EVAL_REF}"
python3 -m pip install --no-deps --force-reinstall -e "${LMMS_EVAL_SRC_DIR}"

# Install only lmms-eval runtime dependencies that are safe to add without
# letting pip resolve/upgrade the ROCm torch stack. Keep torch, torchvision,
# torchaudio, transformers, vllm, numpy, and triton out of this list.
echo "=== Installing lmms-eval helper packages with --no-deps ==="
python3 -m pip install --no-deps \
  "accelerate>=0.29.1" \
  "av<16.0.0" \
  "datasets>=2.19.0" \
  "evaluate>=0.4.0" \
  "jsonlines" \
  "aiohttp" \
  "numexpr" \
  "peft>=0.2.0" \
  "pybind11>=2.6.2" \
  "pytablewriter" \
  "sacrebleu>=1.5.0" \
  "scikit-learn>=0.24.1" \
  "timm" \
  "einops" \
  "ftfy" \
  "openai" \
  "httpx>=0.23.3" \
  "httpcore" \
  "h11" \
  "anyio" \
  "sniffio" \
  "distro" \
  "jiter" \
  "certifi" \
  "idna" \
  "typing-extensions" \
  "opencv-python-headless" \
  "hf-transfer" \
  "nltk" \
  "sentencepiece" \
  "yt-dlp" \
  "pycocoevalcap" \
  "tqdm-multiprocess" \
  "transformers-stream-generator" \
  "zstandard" \
  "pillow" \
  "pyyaml" \
  "sympy" \
  "latex2sympy2" \
  "mpmath" \
  "Jinja2" \
  "openpyxl" \
  "loguru" \
  "tenacity>=8.3.0" \
  "math-verify" \
  "wandb==0.25.0" \
  "tqdm>4" \
  "tiktoken" \
  "pydantic" \
  "pydantic-core" \
  "annotated-types>=0.6.0" \
  "typing-inspection>=0.4.2" \
  "packaging" \
  "pre-commit" \
  "zss" \
  "protobuf" \
  "python-dotenv" \
  "qwen-vl-utils>=0.0.14" \
  "decord"

# Let pip resolve this small pure-Python client stack. Installing these with
# --no-deps is brittle because openai/pydantic/httpx add runtime dependencies
# such as sniffio, pydantic-core, and typing-inspection.
echo "=== Installing OpenAI client helper dependencies ==="
python3 -m pip install \
  "openai" \
  "httpx>=0.23.3" \
  "pydantic" \
  "tqdm>4" \
  "tenacity>=8.3.0"

echo "=== Protected package versions after lmms-eval install ==="
snapshot_protected_versions | tee "${after}"

if ! diff -u "${before}" "${after}"; then
  echo "ERROR: lmms-eval installation changed protected package versions." >&2
  exit 1
fi

python3 - <<'PY'
from pathlib import Path
import importlib.metadata as md
import anyio
import httpx
import lmms_eval
import openai
import pydantic
import sniffio
import tenacity
import tqdm
from lmms_eval.models.chat.openai import OpenAICompatible

root = Path(lmms_eval.__file__).resolve().parent
template = root / "tasks" / "prismm_bench" / "_default_template_yaml"
if not template.exists():
    raise SystemExit(f"ERROR: missing lmms-eval task template: {template}")
print(f"Verified lmms-eval source install: {root}")
print(f"Verified lmms-eval OpenAI backend import: {OpenAICompatible.__name__}")
print(
    "Verified OpenAI backend dependencies: "
    f"openai={md.version('openai')}, httpx={md.version('httpx')}, "
    f"anyio={md.version('anyio')}, sniffio={md.version('sniffio')}, "
    f"pydantic={md.version('pydantic')}, tqdm={md.version('tqdm')}, "
    f"tenacity={md.version('tenacity')}"
)
PY

echo "lmms-eval install verified: protected package versions are unchanged."
