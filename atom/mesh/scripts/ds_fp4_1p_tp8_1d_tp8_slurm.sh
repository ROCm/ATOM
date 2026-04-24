#!/usr/bin/env bash
#SBATCH --job-name=ds-fp4-1p-tp8-1d-tp8
#SBATCH --account=amd-frameworks
#SBATCH --partition=amd-frameworks
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=114
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --nodelist=mia1-p02-g42,mia1-p02-g44
#SBATCH --output=/it-share/yajizhan/slurm_logs/ds_fp4_1p_tp8_1d_tp8-%j.out
#SBATCH --error=/it-share/yajizhan/slurm_logs/ds_fp4_1p_tp8_1d_tp8-%j.err
#
# Self-contained 1P+1D PD-disaggregated benchmark for DeepSeek-R1 MXFP4.
#   prefill: TP=8, decode: TP=8, mooncake RDMA KV transfer.
# All server/router/benchmark logic is inline — no external script dependencies.
#
# Usage:
#   mkdir -p /it-share/yajizhan/slurm_logs
#   sbatch ds_fp4_1p_tp8_1d_tp8_slurm.sh
#
# Override defaults via env:
#   sbatch --export=ALL,LOAD_DUMMY=,ISL_LIST="1024,8192" ds_fp4_1p_tp8_1d_tp8_slurm.sh

set -euo pipefail

# ======================== configuration ========================
MODEL_PATH="${MODEL_PATH:-/it-share/models/deepseek-ai/DeepSeek-R1-0528-MXFP4}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:mesh-sglang-latest}"
CONTAINER="${CONTAINER:-atom_sglang_mesh_${SLURM_JOB_ID}}"

PREFILL_TP="${PREFILL_TP:-8}"
DECODE_TP="${DECODE_TP:-8}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
CUDA_GRAPH_BS_START="${CUDA_GRAPH_BS_START:-1}"
CUDA_GRAPH_BS_END="${CUDA_GRAPH_BS_END:-256}"
IB_DEVICE="${IB_DEVICE:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
MESH_BIN="${MESH_BIN:-/usr/local/bin/atom-mesh}"

ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-1,2,4,8,16}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
BACKEND="${BACKEND:-sglang}"

LOAD_DUMMY="${LOAD_DUMMY:-}"
WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-1800}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

# GSM8K accuracy validation. Skipped automatically when LOAD_DUMMY=1 (random
# weights produce garbage output). Override with RUN_GSM8K=1 to force-enable
# or RUN_GSM8K=0 to skip even with real weights.
RUN_GSM8K="${RUN_GSM8K:-auto}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-3}"
GSM8K_NUM_CONCURRENT="${GSM8K_NUM_CONCURRENT:-65}"

LOG_ROOT="${LOG_ROOT:-/it-share/yajizhan/slurm_logs/$(date +%m%d)_ds_fp4_1p_tp8_1d_tp8_${SLURM_JOB_ID}}"

# ======================== pre-flight ========================
echo "=== Job ${SLURM_JOB_ID} starting on $(hostname) at $(date -Is) ==="
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected 2 nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi
PREFILL_NODE="${NODES[0]}"
DECODE_NODE="${NODES[1]}"

mkdir -p "${LOG_ROOT}"/{prefill,decode,router,bench,gsm8k,scripts}

# Resolve "auto": run GSM8K only with real weights.
if [[ "${RUN_GSM8K}" == "auto" ]]; then
    if [[ -n "${LOAD_DUMMY}" ]]; then
        RUN_GSM8K=0
    else
        RUN_GSM8K=1
    fi
fi

# ======================== pre-cleanup ========================
# Force-stop ALL running docker containers on both nodes before starting fresh.
# This ensures no residual sglang/atom-mesh processes are holding GPU memory
# or ports from previous runs (e.g. when `scancel` killed the sbatch shell
# without propagating into the containers).
echo "=== pre-cleanup: force-stopping all docker containers on both nodes ==="
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    srun --nodelist="$node" --nodes=1 --ntasks=1 --time=00:03:00 bash -c '
        hostname
        running=$(docker ps -q)
        if [[ -n "$running" ]]; then
            echo "  stopping $(echo "$running" | wc -l) running containers:"
            docker ps --format "    {{.ID}} {{.Names}}"
            # -t 0 = send SIGKILL immediately, no graceful shutdown
            docker stop -t 0 $running 2>&1 | sed "s/^/    /"
        else
            echo "  no running containers"
        fi
        # Verify GPUs are free
        sleep 2
        used=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | grep -v ": 0$" | head -5)
        if [[ -n "$used" ]]; then
            echo "  WARNING: some GPUs still have VRAM allocated:"
            echo "$used" | sed "s/^/    /"
        else
            echo "  all GPUs free"
        fi
    ' || echo "[pre-cleanup] WARNING: cleanup on $node had errors (non-fatal)"
done
echo "=== pre-cleanup done ==="

PREFILL_IP=$(srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")
DECODE_IP=$(srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")

LOAD_FORMAT_ARG=""
[[ -n "${LOAD_DUMMY}" ]] && LOAD_FORMAT_ARG="--load-format dummy"

cat <<INFO
=== Configuration ===
PREFILL : ${PREFILL_NODE} (IP=${PREFILL_IP}, TP=${PREFILL_TP}, port=${PREFILL_PORT})
DECODE  : ${DECODE_NODE}  (IP=${DECODE_IP},  TP=${DECODE_TP},  port=${DECODE_PORT})
ROUTER  : ${PREFILL_IP}:${ROUTER_PORT}
MODEL   : ${MODEL_PATH}
IMAGE   : ${DOCKER_IMAGE}
LOAD_DUMMY : ${LOAD_DUMMY:-<off>}
RUN_GSM8K  : ${RUN_GSM8K} (limit=${GSM8K_LIMIT:-all}, fewshot=${GSM8K_NUM_FEWSHOT})
ISL/OSL/CONC : ${ISL_LIST} / ${OSL} / ${CONC_LIST}
LOG_ROOT: ${LOG_ROOT}
=====================
INFO

# ======================== generate in-container scripts ========================
# These scripts are written to NFS ($LOG_ROOT/scripts/) and executed inside docker.
# No dependency on any external repo path.

PREFILL_GPU_IDS=$(seq -s, 0 $((PREFILL_TP - 1)))
DECODE_GPU_IDS=$(seq -s, 0 $((DECODE_TP - 1)))

cat > "${LOG_ROOT}/scripts/prefill.sh" <<'PREFILL_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[prefill] IP=${PREFILL_IP} TP=${PREFILL_TP} port=${PREFILL_PORT}"

mkdir -p /workspace/logs
cat > /workspace/mooncake_prefill.json <<MC
{"prefill_url": "${PREFILL_IP}:${PREFILL_PORT}", "protocol": "rdma"}
MC

export HIP_VISIBLE_DEVICES=${PREFILL_GPU_IDS}
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export SGLANG_HOST_IP=${PREFILL_IP}
export MOONCAKE_CONFIG_PATH=/workspace/mooncake_prefill.json
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export MC_TCP_ENABLE_CONNECTION_POOL=true
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 --port "${PREFILL_PORT}" \
    --grpc-mode \
    --trust-remote-code \
    --tp-size "${PREFILL_TP}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --attention-backend aiter \
    --mem-fraction-static "${MEM_FRACTION}" \
    --page-size 1 \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --disable-radix-cache \
    --log-level info \
    --watchdog-timeout 3600 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --disaggregation-ib-device "${IB_DEVICE}" \
    ${LOAD_FORMAT_ARG} \
    2>&1 | tee /workspace/logs/prefill.log
PREFILL_EOF

cat > "${LOG_ROOT}/scripts/decode.sh" <<'DECODE_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[decode] IP=${DECODE_IP} TP=${DECODE_TP} port=${DECODE_PORT}"

mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${DECODE_GPU_IDS}
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export SGLANG_HOST_IP=${DECODE_IP}
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export MC_TCP_ENABLE_CONNECTION_POOL=true
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

TORCHINDUCTOR_COMPILE_THREADS=128 python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 --port "${DECODE_PORT}" \
    --grpc-mode \
    --trust-remote-code \
    --tp-size "${DECODE_TP}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --page-size 1 \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --cuda-graph-bs $(seq ${CUDA_GRAPH_BS_START} ${CUDA_GRAPH_BS_END}) \
    --disable-radix-cache \
    --log-level info \
    --watchdog-timeout 3600 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --disaggregation-ib-device "${IB_DEVICE}" \
    ${LOAD_FORMAT_ARG} \
    2>&1 | tee /workspace/logs/decode.log
DECODE_EOF

cat > "${LOG_ROOT}/scripts/router.sh" <<'ROUTER_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[router] prefill=${PREFILL_IP}:${PREFILL_PORT} decode=${DECODE_IP}:${DECODE_PORT} router=0.0.0.0:${ROUTER_PORT}"

mkdir -p /workspace/logs

${MESH_BIN} launch \
    --host 0.0.0.0 --port "${ROUTER_PORT}" \
    --pd-disaggregation \
    --prefill "grpc://${PREFILL_IP}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
    --decode  "grpc://${DECODE_IP}:${DECODE_PORT}" \
    --policy random \
    --backend sglang \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-health-check \
    --prometheus-port 29100 \
    2>&1 | tee /workspace/logs/router.log
ROUTER_EOF

cat > "${LOG_ROOT}/scripts/gsm8k.sh" <<'GSM8K_EOF'
#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="/workspace/gsm8k_results"

echo "[gsm8k] model=${MODEL_PATH} endpoint=http://127.0.0.1:${ROUTER_PORT}"
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
    --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/completions,num_concurrent=${GSM8K_NUM_CONCURRENT},max_retries=1,tokenized_requests=False,trust_remote_code=True" \
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
print('========================================='
)
print(f'[gsm8k] exact_match,flexible-extract = {score}')
print('========================================='
)
print(json.dumps(data.get('results', {}), indent=2))
"

echo "[gsm8k] results saved to ${RESULT_DIR}/${RUN_TAG}"
GSM8K_EOF

cat > "${LOG_ROOT}/scripts/benchmark.sh" <<'BENCH_EOF'
#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="/workspace/benchmark_results"

echo "[bench] model=${MODEL_PATH} endpoint=http://127.0.0.1:${ROUTER_PORT}"
echo "[bench] ISL=[${ISL_LIST}] OSL=${OSL} CONC=[${CONC_LIST}] ratio=${RANDOM_RANGE_RATIO}"

if [[ ! -d /tmp/sglang-benchmark/bench_serving ]]; then
    rm -rf /tmp/sglang-benchmark
    mkdir -p /tmp/sglang-benchmark
    git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/sglang-benchmark/bench_serving
fi

mkdir -p "${RESULT_DIR}"

IFS=',' read -ra ISLS <<< "${ISL_LIST}"
IFS=',' read -ra CONCS <<< "${CONC_LIST}"

for ISL in "${ISLS[@]}"; do
    for CONC in "${CONCS[@]}"; do
        RESULT_FILENAME="pd-mesh-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}"
        echo ""
        echo "========================================="
        echo "[bench] ISL=${ISL} OSL=${OSL} CONC=${CONC}"
        echo "========================================="

        PYTHONDONTWRITEBYTECODE=1 python /tmp/sglang-benchmark/bench_serving/benchmark_serving.py \
            --model="${MODEL_PATH}" \
            --backend="${BACKEND}" \
            --base-url="http://127.0.0.1:${ROUTER_PORT}" \
            --dataset-name=random \
            --random-input-len="${ISL}" \
            --random-output-len="${OSL}" \
            --random-range-ratio "${RANDOM_RANGE_RATIO}" \
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
json_files = sorted(result_dir.glob('pd-mesh-*.json'))
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

# Now do variable substitution on the generated scripts.
# The heredocs used <<'EOF' (single-quoted) to avoid premature expansion,
# so we need to replace the variable placeholders now.
for script in "${LOG_ROOT}"/scripts/*.sh; do
    sed -i \
        -e "s|\${PREFILL_IP}|${PREFILL_IP}|g" \
        -e "s|\${DECODE_IP}|${DECODE_IP}|g" \
        -e "s|\${PREFILL_TP}|${PREFILL_TP}|g" \
        -e "s|\${DECODE_TP}|${DECODE_TP}|g" \
        -e "s|\${PREFILL_PORT}|${PREFILL_PORT}|g" \
        -e "s|\${DECODE_PORT}|${DECODE_PORT}|g" \
        -e "s|\${ROUTER_PORT}|${ROUTER_PORT}|g" \
        -e "s|\${BOOTSTRAP_PORT}|${BOOTSTRAP_PORT}|g" \
        -e "s|\${MODEL_PATH}|${MODEL_PATH}|g" \
        -e "s|\${MEM_FRACTION}|${MEM_FRACTION}|g" \
        -e "s|\${KV_CACHE_DTYPE}|${KV_CACHE_DTYPE}|g" \
        -e "s|\${CHUNKED_PREFILL_SIZE}|${CHUNKED_PREFILL_SIZE}|g" \
        -e "s|\${MAX_RUNNING_REQUESTS}|${MAX_RUNNING_REQUESTS}|g" \
        -e "s|\${CUDA_GRAPH_BS_START}|${CUDA_GRAPH_BS_START}|g" \
        -e "s|\${CUDA_GRAPH_BS_END}|${CUDA_GRAPH_BS_END}|g" \
        -e "s|\${IB_DEVICE}|${IB_DEVICE}|g" \
        -e "s|\${MESH_BIN}|${MESH_BIN}|g" \
        -e "s|\${PREFILL_GPU_IDS}|${PREFILL_GPU_IDS}|g" \
        -e "s|\${DECODE_GPU_IDS}|${DECODE_GPU_IDS}|g" \
        -e "s|\${LOAD_FORMAT_ARG}|${LOAD_FORMAT_ARG}|g" \
        -e "s|\${ISL_LIST}|${ISL_LIST}|g" \
        -e "s|\${OSL}|${OSL}|g" \
        -e "s|\${CONC_LIST}|${CONC_LIST}|g" \
        -e "s|\${RANDOM_RANGE_RATIO}|${RANDOM_RANGE_RATIO}|g" \
        -e "s|\${BACKEND}|${BACKEND}|g" \
        -e "s|\${GSM8K_LIMIT}|${GSM8K_LIMIT}|g" \
        -e "s|\${GSM8K_NUM_FEWSHOT}|${GSM8K_NUM_FEWSHOT}|g" \
        -e "s|\${GSM8K_NUM_CONCURRENT}|${GSM8K_NUM_CONCURRENT}|g" \
        "$script"
done

echo "[scripts] generated under ${LOG_ROOT}/scripts/"
ls -la "${LOG_ROOT}"/scripts/

# ======================== cleanup trap ========================
# Handle both normal exit and signals (scancel sends SIGTERM).
# We must remove containers to kill child processes (sglang servers),
# otherwise they keep holding GPU memory and ports after scancel.
cleanup() {
    local rc=$?
    echo ""
    echo "=== cleanup (rc=${rc}) at $(date -Is) ==="
    for node in "$PREFILL_NODE" "$DECODE_NODE"; do
        # Use --time to avoid hanging if node is unreachable
        srun --nodelist="$node" --nodes=1 --ntasks=1 --time=00:01:00 bash -c "
            # Save logs before killing
            docker logs '${CONTAINER}' > '${LOG_ROOT}/docker_\$(hostname).log' 2>&1 || true
            # docker rm -f sends SIGKILL to all processes in the container, then removes it
            docker rm -f '${CONTAINER}' >/dev/null 2>&1 || true
            # Belt-and-suspenders: kill any orphaned GPU processes from this run
            pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
            pkill -9 -f 'atom-mesh' 2>/dev/null || true
        " &
    done
    # Wait for both cleanup sruns (with a timeout)
    wait
    echo "=== cleanup done; logs under ${LOG_ROOT} ==="
}
# Trap both EXIT (normal/error) and common signals (scancel = SIGTERM)
trap cleanup EXIT
trap 'echo "=== received signal, cleaning up ==="; exit 130' INT TERM

# ======================== helper ========================
launch_container() {
    local node="$1"
    local role="$2"
    echo "[${role}] starting container on ${node}"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        set -euo pipefail
        docker rm -f '${CONTAINER}' 2>/dev/null || true
        docker pull '${DOCKER_IMAGE}'
        docker run -d --name '${CONTAINER}' \
            --network host --ipc host --privileged \
            --device /dev/kfd --device /dev/dri \
            --group-add video \
            --cap-add IPC_LOCK --cap-add NET_ADMIN \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            --shm-size 128G \
            -v /mnt:/mnt \
            -v /it-share:/it-share \
            -v '${LOG_ROOT}/${role}':/workspace/logs \
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

# Send a real /v1/completions request and require a non-empty response.
# This is the only reliable way to confirm the PD pipeline (prefill -> mooncake
# KV transfer -> decode) is fully operational. The /v1/models endpoint returns
# 200 as soon as the HTTP server starts, long before model weights are loaded
# and CUDA Graph is captured.
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
                -d '{\"model\":\"${model}\",\"prompt\":\"hi\",\"max_tokens\":4,\"temperature\":0,\"ignore_eos\":true}' 2>&1 || true)
            # Parse 'choices[0].text' length; require > 0 to declare ready.
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

# ======================== 1. start containers ========================
launch_container "$PREFILL_NODE" prefill
launch_container "$DECODE_NODE"  decode

# ======================== 2. start prefill server (detached) ========================
echo "[prefill] launching server on ${PREFILL_NODE}"
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill.sh'
"

# ======================== 3. start decode server (detached) ========================
echo "[decode] launching server on ${DECODE_NODE}"
srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/decode.sh'
"

# ======================== 4. wait for servers ========================
wait_endpoint "$PREFILL_NODE" "http://${PREFILL_IP}:${PREFILL_PORT}/v1/models" \
    "$WAIT_SERVER_TIMEOUT" "prefill"
wait_endpoint "$DECODE_NODE"  "http://${DECODE_IP}:${DECODE_PORT}/v1/models" \
    "$WAIT_SERVER_TIMEOUT" "decode"

# ======================== 5. start router (detached) ========================
echo "[router] launching on ${PREFILL_NODE}"
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/router.sh'
"

# Cheap liveness check first: router HTTP server is listening.
wait_endpoint "$PREFILL_NODE" "http://${PREFILL_IP}:${ROUTER_PORT}/v1/models" \
    "$WAIT_ROUTER_TIMEOUT" "router-http"

# Real inference probe: confirms the full prefill->mooncake->decode pipeline
# is operational. /v1/models returning 200 is NOT enough — sglang answers it
# long before model weights are loaded and CUDA Graph capture finishes.
wait_inference_ready "$PREFILL_NODE" "http://${PREFILL_IP}:${ROUTER_PORT}" \
    "$MODEL_PATH" "$WAIT_SERVER_TIMEOUT" "router-pipeline"

# ======================== 6. run gsm8k accuracy (foreground, optional) ========================
if [[ "${RUN_GSM8K}" == "1" ]]; then
    echo ""
    echo "=== running GSM8K accuracy eval on ${PREFILL_NODE} ==="
    srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
        docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/gsm8k.sh'
    "
else
    echo "=== skipping GSM8K (RUN_GSM8K=${RUN_GSM8K}, LOAD_DUMMY=${LOAD_DUMMY:-<off>}) ==="
fi

# ======================== 7. run benchmark (foreground) ========================
echo ""
echo "=== running benchmark on ${PREFILL_NODE} ==="
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/benchmark.sh'
"

echo ""
echo "=== done at $(date -Is); results: ${LOG_ROOT}/bench  gsm8k: ${LOG_ROOT}/gsm8k ==="
