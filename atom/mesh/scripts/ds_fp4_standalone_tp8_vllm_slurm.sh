#!/usr/bin/env bash
#SBATCH --job-name=ds-fp4-standalone-tp8-vllm
#SBATCH --account=amd-frameworks
#SBATCH --partition=amd-frameworks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=114
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --nodelist=mia1-p02-g44
#SBATCH --output=/it-share/yajizhan/slurm_logs/ds_fp4_standalone_tp8_vllm-%j.out
#SBATCH --error=/it-share/yajizhan/slurm_logs/ds_fp4_standalone_tp8_vllm-%j.err
#
# Self-contained single-node (non-disaggregated) benchmark for DeepSeek-R1 MXFP4
# on vLLM. Mirrors ds_fp4_1p_tp8_1d_tp8_vllm_slurm.sh structure but strips all
# PD/Mooncake plumbing — used to isolate model vs PD-framework regressions.
#
# Workload sweep: ISL:OSL pairs (default "8192:1,1:1024,8192:1024") × CONC list.
#
# Usage:
#   mkdir -p /it-share/yajizhan/slurm_logs
#   sbatch ds_fp4_standalone_tp8_vllm_slurm.sh
#
# Override defaults via env, e.g.:
#   sbatch --export=ALL,ISL_OSL_LIST="8192:1024" ds_fp4_standalone_tp8_vllm_slurm.sh

set -euo pipefail

# ======================== configuration ========================
MODEL_PATH="${MODEL_PATH:-/mnt/models/DeepSeek-R1-0528-MXFP4-V2}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:mesh-vllm-latest}"
CONTAINER="${CONTAINER:-atom_vllm_standalone_${SLURM_JOB_ID}}"

TP="${TP:-8}"
PORT="${PORT:-8000}"

MEM_FRACTION="${MEM_FRACTION:-0.9}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
CUDA_GRAPH_BS_START="${CUDA_GRAPH_BS_START:-1}"
CUDA_GRAPH_BS_END="${CUDA_GRAPH_BS_END:-256}"

# Workload: comma-separated ISL:OSL pairs.
ISL_OSL_LIST="${ISL_OSL_LIST:-8192:1,1:1024,8192:1024}"
CONC_LIST="${CONC_LIST:-1,2,4,8,16}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

LOAD_DUMMY="${LOAD_DUMMY:-}"
WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-1800}"

RUN_GSM8K="${RUN_GSM8K:-auto}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-3}"
GSM8K_NUM_CONCURRENT="${GSM8K_NUM_CONCURRENT:-16}"

LOG_ROOT="${LOG_ROOT:-/it-share/yajizhan/slurm_logs/$(date +%m%d)_ds_fp4_standalone_tp8_vllm_${SLURM_JOB_ID}}"

# ======================== pre-flight ========================
echo "=== Job ${SLURM_JOB_ID} starting on $(hostname) at $(date -Is) ==="
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -ne 1 ]]; then
    echo "ERROR: expected 1 node, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi
NODE="${NODES[0]}"

mkdir -p "${LOG_ROOT}"/{server,bench,gsm8k,scripts}

if [[ "${RUN_GSM8K}" == "auto" ]]; then
    if [[ -n "${LOAD_DUMMY}" ]]; then
        RUN_GSM8K=0
    else
        RUN_GSM8K=1
    fi
fi

# ======================== pre-cleanup ========================
echo "=== pre-cleanup: force-stopping all docker containers on ${NODE} ==="
srun --nodelist="$NODE" --nodes=1 --ntasks=1 --time=00:03:00 bash -c '
    hostname
    running=$(docker ps -q)
    if [[ -n "$running" ]]; then
        echo "  stopping $(echo "$running" | wc -l) running containers:"
        docker ps --format "    {{.ID}} {{.Names}}"
        docker stop -t 0 $running 2>&1 | sed "s/^/    /"
    else
        echo "  no running containers"
    fi
    sleep 2
    used=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | grep -v ": 0$" | head -5)
    if [[ -n "$used" ]]; then
        echo "  WARNING: some GPUs still have VRAM allocated:"
        echo "$used" | sed "s/^/    /"
    else
        echo "  all GPUs free"
    fi
' || echo "[pre-cleanup] WARNING: cleanup had errors (non-fatal)"
echo "=== pre-cleanup done ==="

NODE_IP=$(srun --nodelist="$NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")

if [[ -n "${LOAD_DUMMY}" ]]; then
    LOAD_FORMAT_ARG="--load-format dummy"
else
    LOAD_FORMAT_ARG="--load-format ${LOAD_FORMAT}"
fi

CUDAGRAPH_SIZES=$(seq -s, "${CUDA_GRAPH_BS_START}" "${CUDA_GRAPH_BS_END}")

if [[ -n "${ENFORCE_EAGER}" ]]; then
    COMPILE_ARGS="--enforce-eager"
else
    COMPILE_ARGS="--async-scheduling --compilation-config '{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_SIZES}]}'"
fi

cat <<INFO
=== Configuration ===
NODE     : ${NODE} (IP=${NODE_IP}, TP=${TP}, port=${PORT})
MODEL    : ${MODEL_PATH}
IMAGE    : ${DOCKER_IMAGE}
BACKEND  : vllm (standalone, no PD)
LOAD_DUMMY    : ${LOAD_DUMMY:-<off>}
ENFORCE_EAGER : ${ENFORCE_EAGER:-<off>}
CUDA_GRAPH_BS : ${CUDA_GRAPH_BS_START}-${CUDA_GRAPH_BS_END}
RUN_GSM8K     : ${RUN_GSM8K} (limit=${GSM8K_LIMIT:-all}, fewshot=${GSM8K_NUM_FEWSHOT})
ISL:OSL pairs : ${ISL_OSL_LIST}
CONC_LIST     : ${CONC_LIST}
LOG_ROOT : ${LOG_ROOT}
=====================
INFO

# ======================== generate in-container scripts ========================
GPU_IDS=$(seq -s, 0 $((TP - 1)))

cat > "${LOG_ROOT}/scripts/server.sh" <<'SERVER_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[server] IP=${NODE_IP} TP=${TP} port=${PORT}"

mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${GPU_IDS}
export HF_HUB_CACHE=/mnt/hf_hub_cache
export SAFETENSORS_FAST_GPU=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}

rm -rf /root/.cache

vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --gpu-memory-utilization "${MEM_FRACTION}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --no-enable-prefix-caching \
    ${COMPILE_ARGS} \
    ${LOAD_FORMAT_ARG} \
    2>&1 | tee /workspace/logs/server.log
SERVER_EOF

cat > "${LOG_ROOT}/scripts/gsm8k.sh" <<'GSM8K_EOF'
#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="/workspace/gsm8k_results"

echo "[gsm8k] model=${MODEL_PATH} endpoint=http://127.0.0.1:${PORT}"
echo "[gsm8k] limit=${GSM8K_LIMIT:-all} fewshot=${GSM8K_NUM_FEWSHOT} concurrent=${GSM8K_NUM_CONCURRENT}"

if ! command -v lm_eval >/dev/null 2>&1; then
    echo "[gsm8k] installing lm-eval..."
    pip install 'lm-eval[api]'
fi

RUN_TAG="$(date +%Y%m%d%H%M%S)_gsm8k"
mkdir -p "${RESULT_DIR}"

LIMIT_ARG=""
if [[ -n "${GSM8K_LIMIT}" ]]; then
    LIMIT_ARG="--limit ${GSM8K_LIMIT}"
fi

lm_eval --model local-completions \
    --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=${GSM8K_NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
    --tasks gsm8k \
    --num_fewshot "${GSM8K_NUM_FEWSHOT}" \
    ${LIMIT_ARG} \
    --output_path "${RESULT_DIR}/${RUN_TAG}"

python3 -c "
from pathlib import Path
import json

result_dir = Path('${RESULT_DIR}/${RUN_TAG}')
json_files = list(result_dir.rglob('*.json')) if result_dir.is_dir() else []
if not json_files:
    print('[gsm8k] ERROR: no result JSON found')
    exit(1)

result_file = max(json_files, key=lambda p: p.stat().st_mtime)
data = json.load(open(result_file))
score = data.get('results', {}).get('gsm8k', {}).get('exact_match,flexible-extract', 'N/A')
print('=========================================')
print(f'[gsm8k] exact_match,flexible-extract = {score}')
print('=========================================')
print(json.dumps(data.get('results', {}), indent=2))
"

echo "[gsm8k] results saved to ${RESULT_DIR}/${RUN_TAG}"
GSM8K_EOF

cat > "${LOG_ROOT}/scripts/benchmark.sh" <<'BENCH_EOF'
#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="/workspace/benchmark_results"

echo "[bench] model=${MODEL_PATH} endpoint=http://127.0.0.1:${PORT}"
echo "[bench] ISL:OSL pairs=[${ISL_OSL_LIST}] CONC=[${CONC_LIST}] ratio=${RANDOM_RANGE_RATIO}"

if [[ ! -d /tmp/sglang-benchmark/bench_serving ]]; then
    rm -rf /tmp/sglang-benchmark
    mkdir -p /tmp/sglang-benchmark
    git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/sglang-benchmark/bench_serving
fi

mkdir -p "${RESULT_DIR}"

IFS=',' read -ra PAIRS <<< "${ISL_OSL_LIST}"
IFS=',' read -ra CONCS <<< "${CONC_LIST}"

for PAIR in "${PAIRS[@]}"; do
    ISL="${PAIR%%:*}"
    OSL="${PAIR##*:}"
    EFFECTIVE_RATIO="${RANDOM_RANGE_RATIO}"
    if [[ "${ISL}" -le 1 || "${OSL}" -le 1 ]]; then
        EFFECTIVE_RATIO=1
    fi
    for CONC in "${CONCS[@]}"; do
        RESULT_FILENAME="standalone-vllm-${ISL}-${OSL}-${CONC}-${EFFECTIVE_RATIO}"
        echo ""
        echo "========================================="
        echo "[bench] ISL=${ISL} OSL=${OSL} CONC=${CONC} RATIO=${EFFECTIVE_RATIO}"
        echo "========================================="

        PYTHONDONTWRITEBYTECODE=1 python /tmp/sglang-benchmark/bench_serving/benchmark_serving.py \
            --model="${MODEL_PATH}" \
            --backend=openai \
            --base-url="http://127.0.0.1:${PORT}" \
            --dataset-name=random \
            --random-input-len="${ISL}" \
            --random-output-len="${OSL}" \
            --random-range-ratio "${EFFECTIVE_RATIO}" \
            --num-prompts=$(( CONC * 10 )) \
            --max-concurrency="${CONC}" \
            --trust-remote-code \
            --num-warmups=$(( 2 * CONC )) \
            --request-rate=inf \
            --ignore-eos \
            --save-result \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --result-dir="${RESULT_DIR}" \
            --result-filename="${RESULT_FILENAME}.json"
    done
done

echo ""
echo "========================================="
echo "[bench] summary"
echo "========================================="

python3 -c "
from pathlib import Path
import json

result_dir = Path('${RESULT_DIR}')
json_files = sorted(result_dir.glob('standalone-vllm-*.json'))
if not json_files:
    print('No result files found')
    exit(0)

print(f\"{'Config':<25} {'TTFT(ms)':>10} {'ITL(ms)':>10} {'Throughput(tok/s)':>18}\")
print('-' * 65)
for f in json_files:
    d = json.load(open(f))
    isl = d.get('random_input_len', '?')
    osl = d.get('random_output_len', '?')
    conc = d.get('max_concurrency', '?')
    ttft = d.get('mean_ttft_ms', 0)
    itl = d.get('mean_itl_ms', 0)
    tp = d.get('output_throughput', 0)
    print(f'{isl}/{osl} c={conc:<6} {ttft:>10.1f} {itl:>10.2f} {tp:>18.1f}')
"

echo "[bench] results saved to ${RESULT_DIR}"
BENCH_EOF

chmod +x "${LOG_ROOT}"/scripts/*.sh

for script in "${LOG_ROOT}"/scripts/*.sh; do
    sed -i \
        -e "s|\${NODE_IP}|${NODE_IP}|g" \
        -e "s|\${TP}|${TP}|g" \
        -e "s|\${PORT}|${PORT}|g" \
        -e "s|\${MODEL_PATH}|${MODEL_PATH}|g" \
        -e "s|\${MEM_FRACTION}|${MEM_FRACTION}|g" \
        -e "s|\${KV_CACHE_DTYPE}|${KV_CACHE_DTYPE}|g" \
        -e "s|\${MAX_NUM_BATCHED_TOKENS}|${MAX_NUM_BATCHED_TOKENS}|g" \
        -e "s|\${MAX_MODEL_LEN}|${MAX_MODEL_LEN}|g" \
        -e "s|\${GPU_IDS}|${GPU_IDS}|g" \
        -e "s|\${LOAD_FORMAT_ARG}|${LOAD_FORMAT_ARG}|g" \
        -e "s|\${COMPILE_ARGS}|${COMPILE_ARGS}|g" \
        -e "s|\${ISL_OSL_LIST}|${ISL_OSL_LIST}|g" \
        -e "s|\${CONC_LIST}|${CONC_LIST}|g" \
        -e "s|\${RANDOM_RANGE_RATIO}|${RANDOM_RANGE_RATIO}|g" \
        -e "s|\${GSM8K_LIMIT}|${GSM8K_LIMIT}|g" \
        -e "s|\${GSM8K_NUM_FEWSHOT}|${GSM8K_NUM_FEWSHOT}|g" \
        -e "s|\${GSM8K_NUM_CONCURRENT}|${GSM8K_NUM_CONCURRENT}|g" \
        "$script"
done

echo "[scripts] generated under ${LOG_ROOT}/scripts/"
ls -la "${LOG_ROOT}"/scripts/

# ======================== cleanup trap ========================
cleanup() {
    local rc=$?
    echo ""
    echo "=== cleanup (rc=${rc}) at $(date -Is) ==="
    srun --nodelist="$NODE" --nodes=1 --ntasks=1 --time=00:01:00 bash -c "
        docker logs '${CONTAINER}' > '${LOG_ROOT}/docker_\$(hostname).log' 2>&1 || true
        docker rm -f '${CONTAINER}' >/dev/null 2>&1 || true
        pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
    " || true
    echo "=== cleanup done; logs under ${LOG_ROOT} ==="
}
trap cleanup EXIT
trap 'echo "=== received signal, cleaning up ==="; exit 130' INT TERM

# ======================== helper ========================
launch_container() {
    local node="$1"
    echo "[server] starting container on ${node}"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        set -euo pipefail
        docker rm -f '${CONTAINER}' 2>/dev/null || true
        docker pull '${DOCKER_IMAGE}'
        docker run -d --name '${CONTAINER}' \
            --network host --ipc host --privileged \
            --device /dev/kfd --device /dev/dri \
            --group-add video \
            --cap-add IPC_LOCK --cap-add NET_ADMIN \
            --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
            --shm-size 128G \
            -v /mnt:/mnt \
            -v /it-share:/it-share \
            -v '${LOG_ROOT}/server':/workspace/logs \
            -v '${LOG_ROOT}/bench':/workspace/benchmark_results \
            -v '${LOG_ROOT}/gsm8k':/workspace/gsm8k_results \
            '${DOCKER_IMAGE}' sleep infinity
        docker inspect -f '{{.State.Status}}' '${CONTAINER}'
    "
}

wait_endpoint() {
    local node="$1" url="$2" timeout="$3" name="$4"
    echo "[wait] ${name} -> ${url} (timeout ${timeout}s)"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        deadline=\$(( \$(date +%s) + ${timeout} ))
        while ! curl -sf '${url}' >/dev/null 2>&1; do
            if [[ \$(date +%s) -ge \$deadline ]]; then
                echo '[wait][FAIL] ${name} not ready after ${timeout}s'
                exit 1
            fi
            sleep 10
        done
        echo '[wait][OK] ${name} ready'
    "
}

wait_inference_ready() {
    local node="$1" base_url="$2" model="$3" timeout="$4" name="$5"
    echo "[wait-inference] ${name} -> ${base_url}/v1/completions (timeout ${timeout}s)"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        deadline=\$(( \$(date +%s) + ${timeout} ))
        attempt=0
        while true; do
            attempt=\$((attempt + 1))
            resp=\$(curl -sS -m 120 -X POST '${base_url}/v1/completions' \
                -H 'Content-Type: application/json' \
                -d '{\"model\":\"${model}\",\"prompt\":\"hi\",\"max_tokens\":4,\"temperature\":0}' 2>&1 || true)
            text_len=\$(echo \"\$resp\" | python3 -c 'import sys,json
try:
    d=json.loads(sys.stdin.read())
    print(len(d.get(\"choices\",[{}])[0].get(\"text\",\"\")))
except Exception:
    print(0)' 2>/dev/null || echo 0)
            if [[ \"\$text_len\" -gt 0 ]]; then
                echo \"[wait-inference][OK] ${name} ready (attempt #\${attempt}, text_len=\${text_len})\"
                exit 0
            fi
            if [[ \$(date +%s) -ge \$deadline ]]; then
                echo \"[wait-inference][FAIL] ${name} not ready after ${timeout}s (attempts=\${attempt})\"
                echo \"[wait-inference] last response (truncated): \${resp:0:500}\"
                exit 1
            fi
            sleep 15
        done
    "
}

# ======================== 1. start container ========================
launch_container "$NODE"

# ======================== 2. start server (detached) ========================
echo "[server] launching standalone vllm on ${NODE}"
srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/server.sh'
"

# ======================== 3. wait for server ========================
wait_endpoint "$NODE" "http://${NODE_IP}:${PORT}/v1/models" \
    "$WAIT_SERVER_TIMEOUT" "server-http"

wait_inference_ready "$NODE" "http://${NODE_IP}:${PORT}" \
    "$MODEL_PATH" "$WAIT_SERVER_TIMEOUT" "server-pipeline"

# ======================== 4. run gsm8k accuracy (foreground, optional) ========================
if [[ "${RUN_GSM8K}" == "1" ]]; then
    echo ""
    echo "=== running GSM8K accuracy eval on ${NODE} ==="
    srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
        docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/gsm8k.sh'
    "
else
    echo "=== skipping GSM8K (RUN_GSM8K=${RUN_GSM8K}, LOAD_DUMMY=${LOAD_DUMMY:-<off>}) ==="
fi

# ======================== 5. run benchmark (foreground) ========================
echo ""
echo "=== running benchmark on ${NODE} ==="
srun --nodelist="$NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/benchmark.sh'
"

echo ""
echo "=== done at $(date -Is); results: ${LOG_ROOT}/bench  gsm8k: ${LOG_ROOT}/gsm8k ==="
