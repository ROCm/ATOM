#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Smoke-test patch registry changes across small models (2 train steps each).
#
# Usage:
#   bash examples/patch_smoke_test_all_models.sh
#   RUN=llama2 bash examples/patch_smoke_test_all_models.sh   # single model
#
# Requires: ROCm 8×GPU and IMAGE set to a compatible Lumen training image.
# Logs: ${LOG_DIR}/patch_smoke_<model>_<timestamp>.log

set -euo pipefail

LUMEN_DIR="${LUMEN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${LUMEN_DIR}/examples/scripts/ensure_tokenizer.sh"
IMAGE="${IMAGE:?Set IMAGE to a compatible Lumen training image}"
LOG_DIR="${LOG_DIR:-${LUMEN_DIR}/examples/results/patch_smoke}"
RUN="${RUN:-all}"   # all | patch | llama2 | llama31 | qwen3 | dsv4 | lora

MBS="${MBS:-1}"
GBS="${GBS:-8}"
SEQ_LEN="${SEQ_LEN:-512}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
PRECISION="${PRECISION:-bf16}"

DATA_ROOT="${DATA_ROOT:-${HOME}/dsv4-data}"
MILES_DIR="${MILES_DIR:-${HOME}/miles}"

mkdir -p "${LOG_DIR}"

ts() { date +%Y%m%d_%H%M%S; }

docker_common=(
    --rm --init
    --device /dev/dri --device /dev/kfd
    --group-add video --group-add render
    --ipc=host --network=host
    --security-opt=seccomp=unconfined
    --cap-add=SYS_PTRACE
    --shm-size 32G
    -v "${LUMEN_DIR}:/workspace/Lumen"
    -e HSA_NO_SCRATCH_RECLAIM=1
    -e HIP_FORCE_DEV_KERNARG=1
    -e NCCL_IB_DISABLE=1
    -e NCCL_SOCKET_IFNAME=lo
    -e NCCL_DEBUG=WARN
    -e CUDA_DEVICE_MAX_CONNECTIONS=8
    -e OMP_NUM_THREADS=1
    -e TORCHDYNAMO_DISABLE=1
    -e USE_HIPBLASLT=1
    -e TORCH_BLAS_PREFER_HIPBLASLT=1
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
)

run_patch_only() {
    local log="${LOG_DIR}/patch_tags_$(ts).log"
    echo "=== PATCH TAG SMOKE → ${log}"
    docker run "${docker_common[@]}" "${IMAGE}" bash -c '
set -euo pipefail
cd /workspace/Lumen
PATCH=examples/dsv4/patch_megatron_source.py
MEG=/workspace/megatron_lm
DSV=/opt/dsv4-bootstrap/Megatron-LM

echo "--- llama (3 patches) ---"
PYTHONPATH=/workspace/ATOM/atomrl python3 "$PATCH" "$MEG" --tag llama --dry-run

echo "--- llama,lora (7 patches) ---"
PYTHONPATH=/workspace/ATOM/atomrl python3 "$PATCH" "$MEG" --tag llama,lora --dry-run

echo "--- dsv4 default SOURCE (Megatron bootstrap) ---"
PYTHONPATH=/workspace/ATOM/atomrl python3 "$PATCH" "$DSV" --dry-run 2>&1 | tail -20

echo "--- IMPORT smoke ---"
PYTHONPATH=/workspace/ATOM/atomrl python3 -c "
from atomrl.lumen.patches.registry import PatchRegistry
r = PatchRegistry()
print(f\"registered patches: {len(PatchRegistry.all())}\")
"

echo "PATCH TAG SMOKE OK"
' 2>&1 | tee "${log}"
}

run_llama_family() {
    local name="$1" layers="$2" hidden="$3" ffn="$4" heads="$5" rotary_base="$6" tokenizer_host="$7"
    example_tokenizer "${name%%-*}" "${tokenizer_host}" || return 1
    local log="${LOG_DIR}/${name}_$(ts).log"
    local results="${LUMEN_DIR}/examples/${name%%-*}/results"
    mkdir -p "${results}"

    echo "=== ${name} pretrain smoke → ${log}"
    docker run --name "lumen_smoke_${name}" "${docker_common[@]}" \
        -v "${tokenizer_host}:/tokenizer:ro" \
        -v "${results}:/results" \
        -e MBS="${MBS}" -e GBS="${GBS}" -e SEQ_LEN="${SEQ_LEN}" \
        -e TRAIN_STEPS="${TRAIN_STEPS}" -e SEED=1234 \
        -e LAYERS="${layers}" -e HIDDEN="${hidden}" -e FFN="${ffn}" -e HEADS="${heads}" \
        -e ROTARY_BASE="${rotary_base}" -e MODEL_NAME="${name}" \
        "${IMAGE}" bash -c '
set -euo pipefail
LUMEN_ROOT=/workspace/Lumen
PRETRAIN_DIR="${LUMEN_ROOT}/examples/llama31"
PATCH_SCRIPT="${LUMEN_ROOT}/examples/dsv4/patch_megatron_source.py"
MEGATRON_ROOT=/workspace/megatron_lm
DATA_DIR=/results/mock_data
TRAIN_JSONL=/results/mock_data/mock_train.jsonl
mkdir -p "${DATA_DIR}"

python -c "import megatron; import torch; print(f\"megatron+torch {torch.__version__}\")"

PYTHONPATH="${LUMEN_ROOT}" python3 "${PATCH_SCRIPT}" "${MEGATRON_ROOT}" --tag llama

python - <<'PYEOF'
import os, json, random
seq = int(os.environ["SEQ_LEN"]); gbs = int(os.environ["GBS"]); steps = int(os.environ["TRAIN_STEPS"])
need_tokens = int(gbs * (steps + 5) * (seq + 1) * 1.2)
random.seed(1234)
path = "/results/mock_data/mock_train.jsonl"
words_per_doc = 4000
docs = need_tokens // words_per_doc + 1
with open(path, "w") as f:
    for _ in range(docs):
        toks = [str(random.randint(1, 31999)) for _ in range(words_per_doc)]
        f.write(json.dumps({"text": " ".join(toks)}) + "\n")
print(f"[mock-data] {docs} docs → {path}")
PYEOF

cd "${PRETRAIN_DIR}"
torchrun --nproc_per_node=8 --nnodes=1 pretrain_llama31.py \
    --backend megatron \
    --num-layers "${LAYERS}" \
    --hidden-size "${HIDDEN}" \
    --ffn-hidden-size "${FFN}" \
    --num-attention-heads "${HEADS}" \
    --seq-length "${SEQ_LEN}" \
    --max-position-embeddings "${SEQ_LEN}" \
    --use-rotary-position-embeddings \
    --rotary-base "${ROTARY_BASE}" \
    --no-position-embedding \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --no-masked-softmax-fusion --attention-softmax-in-fp32 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --micro-batch-size "${MBS}" --global-batch-size "${GBS}" \
    --train-iters "${TRAIN_STEPS}" \
    --lr 1e-5 --min-lr 0.0 --lr-decay-style cosine --lr-warmup-iters 1 \
    --weight-decay 0.1 --clip-grad 1.0 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --bf16 --no-gradient-accumulation-fusion --use-distributed-optimizer --overlap-grad-reduce \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model /tokenizer \
    --train-data-path "${TRAIN_JSONL}" --valid-data-path "${TRAIN_JSONL}" --test-data-path "${TRAIN_JSONL}" \
    --split 98,1,1 --seed 1234 --eval-iters 1 --eval-interval "${TRAIN_STEPS}" \
    --save-interval 1000000 --log-interval 1 \
    --lumen-attn-backend csrc

echo "=== ${MODEL_NAME} SMOKE OK ==="
' 2>&1 | tee "${log}"
}

run_qwen3() {
    local log="${LOG_DIR}/qwen3-8b_$(ts).log"
    local tok="${LUMEN_DIR}/examples/qwen3/tokenizer"
    example_tokenizer qwen3 "${tok}" || return 1
    local results="${LUMEN_DIR}/examples/qwen3/results"
    mkdir -p "${results}"
    echo "=== qwen3-8b pretrain smoke → ${log}"
    docker run --name lumen_smoke_qwen3 "${docker_common[@]}" \
        -v "${tok}:/tokenizer:ro" -v "${results}:/results" \
        -e MBS="${MBS}" -e GBS="${GBS}" -e SEQ_LEN="${SEQ_LEN}" -e TRAIN_STEPS="${TRAIN_STEPS}" \
        "${IMAGE}" bash -c '
set -euo pipefail
LUMEN_ROOT=/workspace/Lumen
PRETRAIN_DIR="${LUMEN_ROOT}/examples/llama31"
PATCH_SCRIPT="${LUMEN_ROOT}/examples/dsv4/patch_megatron_source.py"
MEGATRON_ROOT=/workspace/megatron_lm
DATA_DIR=/results/mock_data
TRAIN_JSONL=/results/mock_data/mock_train.jsonl
mkdir -p "${DATA_DIR}"

PYTHONPATH="${LUMEN_ROOT}" python3 "${PATCH_SCRIPT}" "${MEGATRON_ROOT}" --tag llama

python - <<'PYEOF'
import os, json, random
seq = int(os.environ["SEQ_LEN"]); gbs = int(os.environ["GBS"]); steps = int(os.environ["TRAIN_STEPS"])
need_tokens = int(gbs * (steps + 5) * (seq + 1) * 1.2)
random.seed(1234)
path = "/results/mock_data/mock_train.jsonl"
words_per_doc = 4000
docs = need_tokens // words_per_doc + 1
with open(path, "w") as f:
    for _ in range(docs):
        toks = [str(random.randint(1, 151643)) for _ in range(words_per_doc)]
        f.write(json.dumps({"text": " ".join(toks)}) + "\n")
print(f"[mock-data] {docs} docs → {path}")
PYEOF

cd "${PRETRAIN_DIR}"
torchrun --nproc_per_node=8 --nnodes=1 pretrain_llama31.py \
    --backend megatron \
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 --num-attention-heads 32 \
    --num-query-groups 8 --group-query-attention \
    --seq-length "${SEQ_LEN}" --max-position-embeddings "${SEQ_LEN}" \
    --use-rotary-position-embeddings --rotary-base 1000000 --no-position-embedding \
    --normalization RMSNorm --swiglu --untie-embeddings-and-output-weights --disable-bias-linear \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --no-masked-softmax-fusion --attention-softmax-in-fp32 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --micro-batch-size "${MBS}" --global-batch-size "${GBS}" --train-iters "${TRAIN_STEPS}" \
    --lr 1e-5 --min-lr 0.0 --lr-decay-style cosine --lr-warmup-iters 1 \
    --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --bf16 --no-gradient-accumulation-fusion --use-distributed-optimizer --overlap-grad-reduce \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model /tokenizer \
    --train-data-path "${TRAIN_JSONL}" --valid-data-path "${TRAIN_JSONL}" --test-data-path "${TRAIN_JSONL}" \
    --split 98,1,1 --seed 1234 --eval-iters 1 --eval-interval "${TRAIN_STEPS}" \
    --save-interval 1000000 --log-interval 1 --lumen-attn-backend csrc
echo "=== qwen3-8b SMOKE OK ==="
' 2>&1 | tee "${log}"
}

run_dsv4() {
    local log="${LOG_DIR}/dsv4-4layer_$(ts).log"
    echo "=== dsv4-4layer pretrain smoke → ${log}"
    local mounts=(
        -v "${LUMEN_DIR}:/workspace/Lumen"
        -v "${DATA_ROOT}/models:/root/models"
        -v "${DATA_ROOT}/models/miopen-cache:/root/.config/miopen"
    )
    [[ -d "${MILES_DIR}" ]] && mounts+=(-v "${MILES_DIR}:/workspace/miles")

    docker rm -f lumen_smoke_dsv4 2>/dev/null || true
    docker run --rm \
        --name lumen_smoke_dsv4 \
        --device /dev/kfd --device /dev/dri \
        --group-add video --group-add render \
        --ipc=host --network=host \
        --shm-size 128g \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --ulimit memlock=-1 \
        "${mounts[@]}" \
        -e LUMEN_DIR=/workspace/Lumen \
        -e AITER_DIR=/opt/aiter \
        -e BOOTSTRAP_DIR=/opt/dsv4-bootstrap \
        -e WRITABLE_ROOT=/opt/dsv4-runtime \
        -e LUMEN_DSV4_PRETRAIN=1 \
        -e MODEL_DIR=/root/models \
        -e TRAIN_ITERS="${TRAIN_STEPS}" \
        -e SKIP_PREPARE=1 \
        -e LOAD_CKPT=0 \
        -e EVAL_ITERS=1 \
        -e DSV4_HC_MULT=2 \
        -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -e NCCL_NVLS_ENABLE=0 \
        -e TORCHDYNAMO_DISABLE=1 \
        "${IMAGE}" \
        bash /workspace/Lumen/examples/dsv4/run_dsv4_4layer_pretrain_inner.sh \
        2>&1 | tee "${log}"
}

run_lora_patch() {
    local log="${LOG_DIR}/lora_patch_$(ts).log"
    echo "=== llama,lora SOURCE patch smoke (no 70B weights) → ${log}"
    docker run "${docker_common[@]}" "${IMAGE}" bash -c '
set -euo pipefail
cd /workspace/Lumen
PATCH=examples/dsv4/patch_megatron_source.py
MEG=/workspace/megatron_lm

PYTHONPATH=/workspace/ATOM/atomrl python3 "$PATCH" "$MEG" --tag llama,lora

# Verify LoRA-related SOURCE markers exist in patched tree
python3 - <<PY
from pathlib import Path
meg = Path("/workspace/megatron_lm")
checks = [
    "megatron/core/transformer/transformer_config.py",
    "megatron/core/extensions/transformer_engine.py",
]
for p in checks:
    assert (meg / p).is_file(), f"missing {p}"
print("LoRA SOURCE patch files present")
PY

echo "=== lora PATCH SMOKE OK ==="
' 2>&1 | tee "${log}"
}

echo "Patch smoke suite: RUN=${RUN} IMAGE=${IMAGE} steps=${TRAIN_STEPS} GBS=${GBS} seq=${SEQ_LEN}"
echo "Logs: ${LOG_DIR}"

case "${RUN}" in
    patch)   run_patch_only ;;
    llama2)  run_llama_family llama2-7b 32 4096 11008 32 10000 "${LUMEN_DIR}/examples/llama2/tokenizer" ;;
    llama31) run_llama_family llama31-8b 32 4096 14336 32 500000 "${LUMEN_DIR}/examples/llama31/tokenizer" ;;
    qwen3)   run_qwen3 ;;
    dsv4)    run_dsv4 ;;
    lora)    run_lora_patch ;;
    all)
        run_patch_only
        run_llama_family llama2-7b 32 4096 11008 32 10000 "${LUMEN_DIR}/examples/llama2/tokenizer"
        run_llama_family llama31-8b 32 4096 14336 32 500000 "${LUMEN_DIR}/examples/llama31/tokenizer"
        run_qwen3
        run_dsv4
        run_lora_patch
        ;;
    *) echo "Unknown RUN=${RUN}"; exit 1 ;;
esac

echo ""
echo "[DONE] Patch smoke suite finished. Logs in ${LOG_DIR}"
