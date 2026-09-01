#!/usr/bin/env bash
# Qwen3.8 GSM8K accuracy test for the native ATOM backend.
# Supports dense BF16/block-FP8 and MoE Quark-MXFP4 checkpoints.
#
# Usage:
#   scripts/test_qwen38_gsm8k.sh MODEL [TP_SIZE] [PORT] [SERVER_EXTRA_ARGS...]
#
# Example:
#   HIP_VISIBLE_DEVICES=0 scripts/test_qwen38_gsm8k.sh \
#     /path/to/Qwen3.8-27B-FP8 1 8000
#
# Useful overrides:
#   EVAL_MODE=chat|completion  Default: chat. Chat mode enables Qwen3.8's
#                              official thinking template; completion mode
#                              mirrors the Qwen3.5 ATOM recipe endpoint.
#   NUM_FEWSHOT=5              GSM8K few-shot count used by public evaluations.
#   NUM_CONCURRENT=64          Number of concurrent lm-eval requests.
#   MAX_GEN_TOKS=8192          Per-sample generation limit for thinking mode.
#   GEN_KWARGS=...             lm-eval sampling kwargs; defaults match Qwen's
#                              recommended low-effort thinking-mode parameters.
#   BATCH_SIZE=1               lm-eval request batch size.
#   REQUEST_TIMEOUT=1200       Per-request HTTP timeout for long thinking output.
#   FEWSHOT_AS_MULTITURN=1      Render few-shot examples as separate chat turns.
#   LIMIT=50                   Run a quick subset instead of all 1319 samples.
#   KV_CACHE_DTYPE=fp8         ATOM KV-cache dtype.
#   MAX_MODEL_LEN=32768        Server context limit used for this evaluation.
#   ENABLE_PREFIX_CACHING=0     Enable shared-prefix KV caching when set to 1.
#   RESULT_DIR=/path/to/output Override the timestamped output directory.
#   KEEP_SERVER_ALIVE=1        Leave the server running after evaluation.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-}}"
TP_SIZE="${2:-${TP_SIZE:-1}}"
PORT="${3:-${PORT:-8000}}"
if (( $# >= 3 )); then
    shift 3
else
    shift "$#"
fi
SERVER_EXTRA_ARGS=("$@")

EVAL_MODE="${EVAL_MODE:-chat}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
NUM_CONCURRENT="${NUM_CONCURRENT:-64}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-8192}"
GEN_KWARGS="${GEN_KWARGS:-do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_tokens=1,reasoning_effort=low}"
BATCH_SIZE="${BATCH_SIZE:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-1}"
LIMIT="${LIMIT:-}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${NUM_CONCURRENT}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
KEEP_SERVER_ALIVE="${KEEP_SERVER_ALIVE:-0}"

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/outputs/qwen38_gsm8k/${RUN_TAG}}"
SERVER_LOG="${RESULT_DIR}/atom_server.log"
EVAL_LOG="${RESULT_DIR}/lm_eval.log"
LM_EVAL_OUTPUT="${RESULT_DIR}/lm_eval_results"
SERVER_PID=""

die() {
    echo "ERROR: $*" >&2
    exit 2
}

is_non_negative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_batch_size() {
    [[ "$1" == "auto" ]] || is_positive_integer "$1"
}

[[ -n "${MODEL_PATH}" ]] || die "请传入 Hugging Face 模型 ID 或本地模型目录。"
if [[ -d "${MODEL_PATH}" ]]; then
    [[ -f "${MODEL_PATH}/config.json" ]] \
        || die "模型目录缺少 config.json: ${MODEL_PATH}"
    MODEL_PATH="$(cd -- "${MODEL_PATH}" && pwd)"
elif [[ -e "${MODEL_PATH}" ]]; then
    die "MODEL 必须是 Hugging Face 模型 ID 或模型目录: ${MODEL_PATH}"
fi
is_positive_integer "${TP_SIZE}" || die "TP_SIZE 必须是正整数: ${TP_SIZE}"
is_positive_integer "${PORT}" || die "PORT 必须是正整数: ${PORT}"
is_non_negative_integer "${NUM_FEWSHOT}" || die "NUM_FEWSHOT 必须是非负整数。"
is_positive_integer "${NUM_CONCURRENT}" || die "NUM_CONCURRENT 必须是正整数。"
is_positive_integer "${MAX_GEN_TOKS}" || die "MAX_GEN_TOKS 必须是正整数。"
is_batch_size "${BATCH_SIZE}" || die "BATCH_SIZE 必须是 auto 或正整数。"
is_positive_integer "${REQUEST_TIMEOUT}" || die "REQUEST_TIMEOUT 必须是正整数。"
is_positive_integer "${MAX_MODEL_LEN}" || die "MAX_MODEL_LEN 必须是正整数。"
is_positive_integer "${MAX_NUM_SEQS}" || die "MAX_NUM_SEQS 必须是正整数。"
is_positive_integer "${STARTUP_TIMEOUT}" || die "STARTUP_TIMEOUT 必须是正整数。"
[[ "${FEWSHOT_AS_MULTITURN}" == "0" || "${FEWSHOT_AS_MULTITURN}" == "1" ]] \
    || die "FEWSHOT_AS_MULTITURN 只能是 0 或 1。"
[[ "${ENABLE_PREFIX_CACHING}" == "0" || "${ENABLE_PREFIX_CACHING}" == "1" ]] \
    || die "ENABLE_PREFIX_CACHING 只能是 0 或 1。"
[[ "${EVAL_MODE}" == "chat" || "${EVAL_MODE}" == "completion" ]] \
    || die "EVAL_MODE 只能是 chat 或 completion。"
[[ -z "${LIMIT}" ]] || is_positive_integer "${LIMIT}" \
    || die "LIMIT 必须为空或正整数。"

for command_name in python3 curl lm_eval; do
    command -v "${command_name}" >/dev/null 2>&1 \
        || die "缺少命令 ${command_name}。lm_eval 可通过: pip install 'lm-eval[api]'"
done
command -v setsid >/dev/null 2>&1 || die "缺少 setsid，无法安全管理 ATOM 子进程。"

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Qwen3.8 reuses Qwen3.5 dense/MoE architectures. Accept the official dense
# BF16/block-FP8 and text-only MoE Quark-MXFP4 variants.
python3 - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model = sys.argv[1]
config_path = Path(model) / "config.json"
if config_path.is_file():
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
else:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=True,
    ).to_dict()

architectures = config.get("architectures") or []
model_type = config.get("model_type")
quant_config = config.get("quantization_config") or {}
quant_method = quant_config.get("quant_method")
weight_block_size = quant_config.get("weight_block_size")
global_quant_config = quant_config.get("global_quant_config") or {}
global_weight = global_quant_config.get("weight") or {}

print(f"Model architecture: {architectures}")
print(f"Model type:         {model_type}")
if not quant_config:
    print("Weight format:      BF16 (unquantized checkpoint)")
elif quant_method == "quark":
    print(
        "Weight format:      Quark "
        f"{global_weight.get('dtype')} ({global_weight.get('qscheme')})"
    )
else:
    print(f"Weight format:      {quant_method}, block={weight_block_size}")

supported_architectures = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
}
if not supported_architectures.intersection(architectures):
    raise SystemExit(
        "ERROR: 不支持该 Qwen3.8 architecture: " + repr(architectures)
    )
if model_type not in {"qwen3_5", "qwen3_5_moe_text"}:
    raise SystemExit("ERROR: 不支持该 Qwen3.8 model_type: " + repr(model_type))

is_block_fp8 = quant_method == "fp8" and weight_block_size == [128, 128]
is_quark_mxfp4 = (
    quant_method == "quark"
    and global_weight.get("dtype") == "fp4"
    and global_weight.get("qscheme") == "per_group"
)
if quant_config and not (is_block_fp8 or is_quark_mxfp4):
    raise SystemExit("ERROR: 仅支持官方 BF16、128x128 block FP8 或 Quark-MXFP4。")
PY

if ! python3 -c "import atom" >/dev/null 2>&1; then
    die "当前 Python 环境无法导入 ATOM；请在 ATOM 容器/环境中运行。"
fi

if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    die "端口 ${PORT} 已有服务运行，请更换端口。"
fi

# Defaults follow the Qwen3.5 ATOM recipes. They remain externally overridable.
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION="${ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:-1}"
export ATOM_USE_CUSTOM_ALL_GATHER="${ATOM_USE_CUSTOM_ALL_GATHER:-0}"
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE="${ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE:-0}"

SERVER_CMD=(
    python3 -m atom.entrypoints.openai_server
    --model "${MODEL_PATH}"
    --kv_cache_dtype "${KV_CACHE_DTYPE}"
    -tp "${TP_SIZE}"
    --server-port "${PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)
if [[ "${ENABLE_PREFIX_CACHING}" == "0" ]]; then
    SERVER_CMD+=(--no-enable_prefix_caching)
fi
SERVER_CMD+=("${SERVER_EXTRA_ARGS[@]}")

{
    echo "model_path=${MODEL_PATH}"
    echo "tp_size=${TP_SIZE}"
    echo "port=${PORT}"
    echo "eval_mode=${EVAL_MODE}"
    echo "num_fewshot=${NUM_FEWSHOT}"
    echo "num_concurrent=${NUM_CONCURRENT}"
    echo "max_gen_toks=${MAX_GEN_TOKS}"
    echo "gen_kwargs=${GEN_KWARGS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "request_timeout=${REQUEST_TIMEOUT}"
    echo "fewshot_as_multiturn=${FEWSHOT_AS_MULTITURN}"
    echo "limit=${LIMIT:-full}"
    echo "kv_cache_dtype=${KV_CACHE_DTYPE}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "max_num_seqs=${MAX_NUM_SEQS}"
    echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    echo "enable_prefix_caching=${ENABLE_PREFIX_CACHING}"
    printf "server_command="
    printf "%q " "${SERVER_CMD[@]}"
    printf "\n"
} | tee "${RESULT_DIR}/run_config.txt"

cleanup() {
    local exit_code=$?
    trap - EXIT

    if [[ "${KEEP_SERVER_ALIVE}" == "1" ]]; then
        echo "KEEP_SERVER_ALIVE=1，ATOM 服务保持运行，PID=${SERVER_PID}。"
        exit "${exit_code}"
    fi

    if [[ -n "${SERVER_PID}" ]]; then
        echo "Stopping ATOM server process group ${SERVER_PID}..."
        kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            if ! kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting native ATOM backend..."
setsid "${SERVER_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "ATOM PID=${SERVER_PID}; log=${SERVER_LOG}"

deadline=$((SECONDS + STARTUP_TIMEOUT))
next_progress=$SECONDS
while ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ATOM 服务启动失败，最后 100 行日志：" >&2
        tail -n 100 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "ATOM 服务在 ${STARTUP_TIMEOUT}s 内未就绪，最后 100 行日志：" >&2
        tail -n 100 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if (( SECONDS >= next_progress )); then
        echo "Waiting for ATOM server... elapsed=${SECONDS}s"
        next_progress=$((SECONDS + 30))
    fi
    sleep 5
done
echo "ATOM server is ready."

if [[ "${EVAL_MODE}" == "chat" ]]; then
    MODEL_TYPE="local-chat-completions"
    BASE_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
    EVAL_MODE_ARGS=(
        --apply_chat_template
        --fewshot_as_multiturn "${FEWSHOT_AS_MULTITURN}"
    )
else
    MODEL_TYPE="local-completions"
    BASE_URL="http://127.0.0.1:${PORT}/v1/completions"
    EVAL_MODE_ARGS=()
fi

MODEL_ARGS="model=${MODEL_PATH},base_url=${BASE_URL},num_concurrent=${NUM_CONCURRENT},max_retries=3,timeout=${REQUEST_TIMEOUT},max_gen_toks=${MAX_GEN_TOKS},tokenized_requests=False,trust_remote_code=True"
# ATOM's OpenAI-compatible API enables sampling through temperature > 0;
# it does not expose vLLM's do_sample/min_p/repetition_penalty request fields.
GEN_KWARGS_ARG=()
if [[ -n "${GEN_KWARGS}" ]]; then
    GEN_KWARGS_ARG=(--gen_kwargs "${GEN_KWARGS}")
fi
EVAL_CMD=(
    lm_eval
    --model "${MODEL_TYPE}"
    --model_args "${MODEL_ARGS}"
    --tasks gsm8k
    --num_fewshot "${NUM_FEWSHOT}"
    --batch_size "${BATCH_SIZE}"
    --output_path "${LM_EVAL_OUTPUT}"
    "${GEN_KWARGS_ARG[@]}"
    "${EVAL_MODE_ARGS[@]}"
)

if [[ -n "${LIMIT}" ]]; then
    EVAL_CMD+=(--limit "${LIMIT}")
fi
if [[ "${LOG_SAMPLES}" == "1" ]]; then
    EVAL_CMD+=(--log_samples)
fi

printf "eval_command=" | tee -a "${RESULT_DIR}/run_config.txt"
printf "%q " "${EVAL_CMD[@]}" | tee -a "${RESULT_DIR}/run_config.txt"
printf "\n" | tee -a "${RESULT_DIR}/run_config.txt"

echo "Running GSM8K evaluation..."
set +e
"${EVAL_CMD[@]}" 2>&1 | tee "${EVAL_LOG}"
eval_status=${PIPESTATUS[0]}
set -e
if (( eval_status != 0 )); then
    echo "lm_eval failed with exit code ${eval_status}; log=${EVAL_LOG}" >&2
    exit "${eval_status}"
fi

python3 - "${LM_EVAL_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
candidates = [output_path] if output_path.is_file() else list(output_path.rglob("*.json"))
results = []
for candidate in candidates:
    try:
        with candidate.open(encoding="utf-8") as result_file:
            payload = json.load(result_file)
    except (OSError, json.JSONDecodeError):
        continue
    gsm8k = payload.get("results", {}).get("gsm8k")
    if isinstance(gsm8k, dict):
        results.append((candidate.stat().st_mtime, candidate, gsm8k))

if not results:
    raise SystemExit(f"ERROR: 未在 {output_path} 中找到 GSM8K 结果 JSON。")

_, result_file, gsm8k = max(results)
strict = gsm8k.get("exact_match,strict-match")
flexible = gsm8k.get("exact_match,flexible-extract")
print("========================================")
print(f"Result file:   {result_file}")
print(f"Strict match:  {strict}")
print(f"Flexible EM:   {flexible}")
print("========================================")
PY

echo "All logs and results: ${RESULT_DIR}"
