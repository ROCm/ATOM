#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
# Lumen — Llama3.1-8B pretrain (BF16 or FP8 delayed/hybrid)
#
# Usage:
#   PRECISION=bf16 bash run_pretrain_llama31_8b.sh
#   PRECISION=fp8  bash run_pretrain_llama31_8b.sh   # default
#
# All paths are relative to this script; no absolute/user-specific paths.
# Override any of IMAGE / MBS / GBS / SEQ_LEN / TRAIN_STEPS / SEED via env.
###############################################################################
set -euo pipefail

# Repo root = two levels up from examples/llama31/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUMEN_DIR="${LUMEN_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ATOM_ROOT="${ATOM_ROOT:-$(cd "${LUMEN_DIR}/../.." && pwd)}"

PRECISION="${PRECISION:-fp8}"          # bf16 | fp8
IMAGE="${IMAGE:-lumen:dev}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${LUMEN_DIR}/examples/llama31/tokenizer}"
source "${LUMEN_DIR}/examples/scripts/ensure_tokenizer.sh"
example_tokenizer llama31 "${TOKENIZER_DIR}" || exit 1
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
CONTAINER_NAME="${CONTAINER_NAME:-lumen_llama31_8b_${PRECISION}}"

MBS="${MBS:-2}"
GBS="${GBS:-128}"
SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
SEED="${SEED:-1234}"

mkdir -p "${RESULTS_DIR}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ---- FP8-only env switches (validated forward optimizations) ----------------
FP8_ENV=()
if [ "${PRECISION}" = "fp8" ]; then
    FP8_ENV=(
        -e LUMEN_FUSED_QUANT_TRANSPOSE_CPP=1
        -e LUMEN_FUSED_QUANT_AMAX=1
        -e LUMEN_FUSED_QUANT_SCALE=1
        -e LUMEN_FUSED_CAST_TRANSPOSE=1
        -e LUMEN_FUSED_CAST_TRANSPOSE_V2=1
        -e LUMEN_FUSED_SWIGLU_QUANT=1
        -e LUMEN_FUSED_NORM_QUANT=1
        -e LUMEN_FUSED_NORM_QUANT_V2=1
        -e LUMEN_TRANSPOSE_CACHE=1
        -e LUMEN_FAST_QUANT_DISPATCH=1
        -e LUMEN_WEIGHT_QUANT_ONCE=1
    )
fi

# ---- FP8-only training flags -----------------------------------------------
FP8_ARGS=()
if [ "${PRECISION}" = "fp8" ]; then
    FP8_ARGS=(
        --linear-fp8
        --lumen-linear
        --fp8-format hybrid
        --linear-fp8-scaling delayed
        --linear-fp8-amax-algo max
        --linear-fp8-amax-history 1024
    )
fi

docker run --rm --init \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --ipc=host --network=host \
    --security-opt=seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    --shm-size 16G \
    -v "${ATOM_ROOT}:/workspace/ATOM" \
    -v "${LUMEN_DIR}:/workspace/Lumen" \
    ${AITER_SRC:+-v "${AITER_SRC}:/opt/aiter"} \
    ${AITER_JIT_DIR:+-v "${AITER_JIT_DIR}:/aiter-jit"} \
    ${AITER_JIT_DIR:+-e AITER_JIT_DIR=/aiter-jit} \
    -v "${TOKENIZER_DIR}:/tokenizer:ro" \
    -v "${RESULTS_DIR}:/results" \
    -e PYTHONPATH=/workspace/ATOM/atomrl:/workspace/Lumen:/workspace/megatron_lm:/opt/atomrl/megatron_lm \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e GPU_MAX_HW_QUEUES=8 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=lo \
    -e NCCL_DEBUG=WARN \
    -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
    -e OMP_NUM_THREADS=1 \
    -e TORCHDYNAMO_DISABLE=1 \
    -e USE_HIPBLASLT=1 \
    -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -e LUMEN_PREFER_HIPBLASLT=1 \
    -e LUMEN_FUSED_SWIGLU=1 \
    -e LUMEN_FUSED_RESIDUAL_NORM=1 \
    -e LUMEN_FUSED_RES_BWD=1 \
    -e LUMEN_SKIP_BACKEND_SYNC=1 \
    "${FP8_ENV[@]}" \
    -e MBS="${MBS}" -e GBS="${GBS}" -e SEQ_LEN="${SEQ_LEN}" \
    -e TRAIN_STEPS="${TRAIN_STEPS}" -e SEED="${SEED}" \
    "${IMAGE}" \
    bash -c '
set -euo pipefail

# Dockerfile.lumen already installs AtomRL editable at this bind-mounted path.
# Reinstalling the root ATOM project misplaces the native Lumen extension.
cd /workspace/ATOM/atomrl

LUMEN_ROOT="/workspace/Lumen"
PRETRAIN_DIR="${LUMEN_ROOT}/examples/llama31"
PATCH_SCRIPT="${LUMEN_ROOT}/examples/dsv4/patch_megatron_source.py"
DATA_DIR="/results/mock_data"
TRAIN_JSONL="${DATA_DIR}/mock_train.jsonl"
mkdir -p "${DATA_DIR}"

python -c "import megatron" 2>/dev/null || { echo "ERROR: megatron not found in image"; exit 1; }

MEGATRON_ROOT="${MEGATRON_ROOT:-/workspace/megatron_lm}"
PYTHONPATH="${LUMEN_ROOT}" python3 "${PATCH_SCRIPT}" "${MEGATRON_ROOT}" --tag llama

python - <<PYEOF
import os, json, random
seq = int(os.environ["SEQ_LEN"])
gbs = int(os.environ["GBS"])
steps = int(os.environ["TRAIN_STEPS"])
need_chunks = gbs * (steps + 5)
need_tokens = int(need_chunks * (seq + 1) * 1.2)
random.seed(int(os.environ["SEED"]))
path = "${TRAIN_JSONL}"
words_per_doc = 4000
docs = need_tokens // words_per_doc + 1
with open(path, "w") as f:
    for _ in range(docs):
        toks = [str(random.randint(1, 127999)) for _ in range(words_per_doc)]
        f.write(json.dumps({"text": " ".join(toks)}) + "\n")
print(f"[mock-data] wrote {docs} docs to {path}")
PYEOF

cd "${PRETRAIN_DIR}"
set -x
torchrun --nproc_per_node=8 --nnodes=1 pretrain_llama31.py \
    --backend megatron \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --use-rotary-position-embeddings \
    --rotary-base 500000 \
    --no-position-embedding \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --micro-batch-size "${MBS}" \
    --global-batch-size "${GBS}" \
    --train-iters "${TRAIN_STEPS}" \
    --lr 1.0e-5 --min-lr 0.0 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /tokenizer \
    --train-data-path "${TRAIN_JSONL}" \
    --valid-data-path "${TRAIN_JSONL}" \
    --test-data-path "${TRAIN_JSONL}" \
    --split 98,1,1 \
    --seed "${SEED}" \
    --eval-iters 1 \
    --eval-interval "${TRAIN_STEPS}" \
    --save-interval 1000000 \
    --log-interval 1 \
    --lumen-attn-backend csrc \
    '"${FP8_ARGS[*]}"' \
    2>&1 | tee "/results/lumen_llama31_8b_'"${PRECISION}"'.log"
'

echo ""
echo "[DONE] log: ${RESULTS_DIR}/lumen_llama31_8b_${PRECISION}.log"
