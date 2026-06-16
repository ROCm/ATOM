#!/usr/bin/env bash
#SBATCH --job-name=ds-fp4-atom-1p-tp8-2d-tp8
#SBATCH --account=amd-frameworks
#SBATCH --partition=amd-frameworks
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=114
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --nodelist=mia1-p02-g42,mia1-p02-g44,mia1-p02-g47
#SBATCH --output=/it-share/yajizhan/slurm_logs/ds_fp4_atom_1p_tp8_2d_tp8-%j.out
#SBATCH --error=/it-share/yajizhan/slurm_logs/ds_fp4_atom_1p_tp8_2d_tp8-%j.err
#
# 1P+2D PD-disaggregated benchmark for ATOM native server + atomesh router.
#   - prefill: ATOM kv_producer, TP=8
#   - decode:  ATOM kv_consumer, TP=8 (2 instances, each on its own node)
#   - router:  atomesh launch --backend atom (no bootstrap port)
#   - KV transfer: Mooncake RDMA (atom/kv_transfer/disaggregation/mooncake)
#
# Mesh router learns each prefill's tp_size/dp_size/kv_role by GETing
# http://<prefill>/kv_transfer_info at startup, then awaits each prefill
# response, extracts kv_transfer_params from the top level, enriches with
# remote_dp_size/remote_tp_size, and injects into the decode request.
#
# Usage:
#   mkdir -p /it-share/yajizhan/slurm_logs
#   sbatch ds_fp4_1p_tp8_2d_tp8_atom_slurm.sh

set -euo pipefail

# ======================== configuration ========================
MODEL_PATH="${MODEL_PATH:-/mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:mesh-sglang-latest}"
CONTAINER="${CONTAINER:-atom_atom_mesh_${SLURM_JOB_ID}}"

PREFILL_TP="${PREFILL_TP:-8}"
DECODE_TP="${DECODE_TP:-8}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-2,4,8,16,32,64,128,256}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-1800}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

RUN_GSM8K="${RUN_GSM8K:-1}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-3}"
GSM8K_NUM_CONCURRENT="${GSM8K_NUM_CONCURRENT:-16}"

LOG_ROOT="${LOG_ROOT:-/it-share/yajizhan/slurm_logs/$(date +%m%d)_ds_fp4_atom_1p_tp8_2d_tp8_${SLURM_JOB_ID}}"

# ======================== pre-flight ========================
echo "=== Job ${SLURM_JOB_ID} starting on $(hostname) at $(date -Is) ==="
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -ne 3 ]]; then
    echo "ERROR: expected 3 nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi
PREFILL_NODE="${NODES[0]}"
DECODE_NODE_1="${NODES[1]}"
DECODE_NODE_2="${NODES[2]}"
ALL_NODES=("$PREFILL_NODE" "$DECODE_NODE_1" "$DECODE_NODE_2")

mkdir -p "${LOG_ROOT}"/{prefill,decode_1,decode_2,router,bench,gsm8k,scripts}

# ======================== pre-cleanup ========================
echo "=== pre-cleanup: force-stopping all docker containers on all nodes ==="
for node in "${ALL_NODES[@]}"; do
    srun --nodelist="$node" --nodes=1 --ntasks=1 --time=00:03:00 bash -c '
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
    ' || echo "[pre-cleanup] WARNING: cleanup on $node had errors (non-fatal)"
done
echo "=== pre-cleanup done ==="

PREFILL_IP=$(srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")
DECODE_IP_1=$(srun --nodelist="$DECODE_NODE_1" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")
DECODE_IP_2=$(srun --nodelist="$DECODE_NODE_2" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")

cat <<INFO
=== Configuration ===
PREFILL  : ${PREFILL_NODE}    (IP=${PREFILL_IP},  TP=${PREFILL_TP}, port=${PREFILL_PORT})
DECODE-1 : ${DECODE_NODE_1}   (IP=${DECODE_IP_1}, TP=${DECODE_TP},  port=${DECODE_PORT})
DECODE-2 : ${DECODE_NODE_2}   (IP=${DECODE_IP_2}, TP=${DECODE_TP},  port=${DECODE_PORT})
ROUTER   : ${PREFILL_IP}:${ROUTER_PORT}
MODEL    : ${MODEL_PATH}
IMAGE    : ${DOCKER_IMAGE}
BACKEND  : atom (Mooncake)
HANDSHAKE_PORT : ${HANDSHAKE_PORT}
RUN_GSM8K  : ${RUN_GSM8K} (limit=${GSM8K_LIMIT:-all}, fewshot=${GSM8K_NUM_FEWSHOT})
ISL/OSL/CONC : ${ISL_LIST} / ${OSL} / ${CONC_LIST}
LOG_ROOT : ${LOG_ROOT}
=====================
INFO

# ======================== generate in-container scripts ========================
PREFILL_GPU_IDS=$(seq -s, 0 $((PREFILL_TP - 1)))
DECODE_GPU_IDS=$(seq -s, 0 $((DECODE_TP - 1)))

# Prefill: kv_role=kv_producer. proxy_ip is set to PREFILL_IP so the rank-0
# ZMQ service-discovery ping thread (which tries to register with the
# legacy atom proxy.py) fails silently against a closed port instead of
# crashing on a None host. Mesh router learns workers via the HTTP
# /kv_transfer_info endpoint, not via ZMQ.
cat > "${LOG_ROOT}/scripts/prefill.sh" <<'PREFILL_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[prefill] IP=${PREFILL_IP} TP=${PREFILL_TP} port=${PREFILL_PORT}"
mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${PREFILL_GPU_IDS}
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export ATOM_HOST_IP=${PREFILL_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

# Clear any stale ATOM compile cache from prior runs.
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 --server-port "${PREFILL_PORT}" \
    --trust-remote-code \
    -tp "${PREFILL_TP}" \
    --kv_cache_dtype "${KV_CACHE_DTYPE}" \
    --block-size "${BLOCK_SIZE}" \
    --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${PREFILL_IP}\",\"handshake_port\":${HANDSHAKE_PORT}}" \
    ${EXTRA_SERVER_ARGS} \
    2>&1 | tee /workspace/logs/prefill.log
PREFILL_EOF

# Single decode template — copied to decode_1.sh and decode_2.sh with per-instance
# IP substituted via __DECODE_HANDSHAKE_IP__ before the global sed pass.
cat > "${LOG_ROOT}/scripts/decode.sh.tmpl" <<'DECODE_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[decode] IP=__DECODE_HANDSHAKE_IP__ TP=${DECODE_TP} port=${DECODE_PORT}"
mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${DECODE_GPU_IDS}
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export ATOM_HOST_IP=__DECODE_HANDSHAKE_IP__
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 --server-port "${DECODE_PORT}" \
    --trust-remote-code \
    -tp "${DECODE_TP}" \
    --kv_cache_dtype "${KV_CACHE_DTYPE}" \
    --block-size "${BLOCK_SIZE}" \
    --kv-transfer-config "{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"__DECODE_HANDSHAKE_IP__\",\"handshake_port\":${HANDSHAKE_PORT}}" \
    ${EXTRA_SERVER_ARGS} \
    2>&1 | tee /workspace/logs/decode.log
DECODE_EOF

# Materialize per-instance decode scripts.
sed "s|__DECODE_HANDSHAKE_IP__|${DECODE_IP_1}|g" \
    "${LOG_ROOT}/scripts/decode.sh.tmpl" > "${LOG_ROOT}/scripts/decode_1.sh"
sed "s|__DECODE_HANDSHAKE_IP__|${DECODE_IP_2}|g" \
    "${LOG_ROOT}/scripts/decode.sh.tmpl" > "${LOG_ROOT}/scripts/decode_2.sh"
rm "${LOG_ROOT}/scripts/decode.sh.tmpl"

# Router: NO bootstrap port after --prefill URL (ATOM has no bootstrap server).
# Two --decode flags, one per decode instance.
cat > "${LOG_ROOT}/scripts/router.sh" <<'ROUTER_EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[router] prefill=http://${PREFILL_IP}:${PREFILL_PORT}"
echo "[router] decode-1=http://${DECODE_IP_1}:${DECODE_PORT}"
echo "[router] decode-2=http://${DECODE_IP_2}:${DECODE_PORT}"
echo "[router] router=0.0.0.0:${ROUTER_PORT}"
mkdir -p /workspace/logs

${MESH_BIN} launch \
    --host 0.0.0.0 --port "${ROUTER_PORT}" \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:${PREFILL_PORT}" \
    --decode  "http://${DECODE_IP_1}:${DECODE_PORT}" \
    --decode  "http://${DECODE_IP_2}:${DECODE_PORT}" \
    --policy random \
    --backend atom \
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
    --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/completions,num_concurrent=${GSM8K_NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
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
        RESULT_FILENAME="pd-atomesh-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}"
        echo ""
        echo "========================================="
        echo "[bench] ISL=${ISL} OSL=${OSL} CONC=${CONC}"
        echo "========================================="

        # Use "vllm" backend in bench_serving — it speaks OpenAI /v1/completions
        # which is what mesh router exposes regardless of upstream backend.
        PYTHONDONTWRITEBYTECODE=1 python /tmp/sglang-benchmark/bench_serving/benchmark_serving.py \
            --model="${MODEL_PATH}" \
            --backend=vllm \
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
json_files = sorted(result_dir.glob('pd-atomesh-*.json'))
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

# Substitute build-host variables into the in-container scripts.
for script in "${LOG_ROOT}"/scripts/*.sh; do
    sed -i \
        -e "s|\${PREFILL_IP}|${PREFILL_IP}|g" \
        -e "s|\${DECODE_IP_1}|${DECODE_IP_1}|g" \
        -e "s|\${DECODE_IP_2}|${DECODE_IP_2}|g" \
        -e "s|\${PREFILL_TP}|${PREFILL_TP}|g" \
        -e "s|\${DECODE_TP}|${DECODE_TP}|g" \
        -e "s|\${PREFILL_PORT}|${PREFILL_PORT}|g" \
        -e "s|\${DECODE_PORT}|${DECODE_PORT}|g" \
        -e "s|\${ROUTER_PORT}|${ROUTER_PORT}|g" \
        -e "s|\${HANDSHAKE_PORT}|${HANDSHAKE_PORT}|g" \
        -e "s|\${MODEL_PATH}|${MODEL_PATH}|g" \
        -e "s|\${KV_CACHE_DTYPE}|${KV_CACHE_DTYPE}|g" \
        -e "s|\${BLOCK_SIZE}|${BLOCK_SIZE}|g" \
        -e "s|\${MESH_BIN}|/usr/local/bin/atomesh|g" \
        -e "s|\${PREFILL_GPU_IDS}|${PREFILL_GPU_IDS}|g" \
        -e "s|\${DECODE_GPU_IDS}|${DECODE_GPU_IDS}|g" \
        -e "s|\${EXTRA_SERVER_ARGS}|${EXTRA_SERVER_ARGS}|g" \
        -e "s|\${ISL_LIST}|${ISL_LIST}|g" \
        -e "s|\${OSL}|${OSL}|g" \
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
    for node in "${ALL_NODES[@]}"; do
        srun --nodelist="$node" --nodes=1 --ntasks=1 --time=00:01:00 bash -c "
            docker logs '${CONTAINER}' > '${LOG_ROOT}/docker_\$(hostname).log' 2>&1 || true
            docker rm -f '${CONTAINER}' >/dev/null 2>&1 || true
            pkill -9 -f 'atom.entrypoints.openai_server' 2>/dev/null || true
            pkill -9 -f 'atomesh' 2>/dev/null || true
        " &
    done
    wait
    echo "=== cleanup done; logs under ${LOG_ROOT} ==="
}
trap cleanup EXIT
trap 'echo "=== received signal, cleaning up ==="; exit 130' INT TERM

# ======================== helpers ========================
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
            --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
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

# ======================== 1. start containers ========================
launch_container "$PREFILL_NODE"   prefill
launch_container "$DECODE_NODE_1"  decode_1
launch_container "$DECODE_NODE_2"  decode_2

# ======================== 2. start prefill + decode (detached) ========================
echo "[prefill] launching ATOM kv_producer on ${PREFILL_NODE}"
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill.sh'
"

echo "[decode-1] launching ATOM kv_consumer on ${DECODE_NODE_1}"
srun --nodelist="$DECODE_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/decode_1.sh'
"
echo "[decode-2] launching ATOM kv_consumer on ${DECODE_NODE_2}"
srun --nodelist="$DECODE_NODE_2" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/decode_2.sh'
"

# ======================== 3. wait for servers (HTTP health check) ========================
wait_endpoint "$PREFILL_NODE"  "http://${PREFILL_IP}:${PREFILL_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "prefill-http"
wait_endpoint "$DECODE_NODE_1" "http://${DECODE_IP_1}:${DECODE_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "decode-1-http"
wait_endpoint "$DECODE_NODE_2" "http://${DECODE_IP_2}:${DECODE_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "decode-2-http"

# Verify /kv_transfer_info responds with the right kv_role on each side
# BEFORE starting the router (mesh aborts startup if a prefill worker
# reports kv_role != "kv_producer").
echo ""
echo "=== verifying /kv_transfer_info ==="
verify_kv_info() {
    local role="$1" node="$2" ip="$3" port="$4" want="$5"
    local info="" got="" attempt=0 max_attempts=3
    while [[ $attempt -lt $max_attempts ]]; do
        attempt=$((attempt + 1))
        info=$(srun --nodelist="$node" --nodes=1 --ntasks=1 bash -c \
            "curl -sf http://${ip}:${port}/kv_transfer_info" 2>&1) && break
        echo "[kv_info][${role}] attempt ${attempt}/${max_attempts} failed (rc=$?): ${info:0:200}" >&2
        sleep 5
    done
    if [[ -z "$info" ]]; then
        echo "ERROR: ${role} /kv_transfer_info returned empty after ${max_attempts} attempts" >&2
        return 1
    fi
    echo "[kv_info][${role}] ${info}"
    got=$(echo "$info" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("kv_role",""))' 2>&1) || {
        echo "ERROR: ${role} failed to parse kv_transfer_info JSON: ${info:0:200}" >&2
        return 1
    }
    if [[ "$got" != "$want" ]]; then
        echo "ERROR: ${role} kv_role mismatch: want=${want} got=${got}" >&2
        return 1
    fi
}
verify_kv_info prefill   "$PREFILL_NODE"   "$PREFILL_IP"   "$PREFILL_PORT" kv_producer
verify_kv_info decode-1  "$DECODE_NODE_1"  "$DECODE_IP_1"  "$DECODE_PORT"  kv_consumer
verify_kv_info decode-2  "$DECODE_NODE_2"  "$DECODE_IP_2"  "$DECODE_PORT"  kv_consumer

# ======================== 4. start router (detached) ========================
echo ""
echo "[router] launching atomesh on ${PREFILL_NODE}"
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/router.sh'
"

wait_endpoint "$PREFILL_NODE" "http://${PREFILL_IP}:${ROUTER_PORT}/v1/models" \
    "$WAIT_ROUTER_TIMEOUT" "router-http"

# ======================== 5. smoke completion (catches relay breakage fast) ========================
echo ""
echo "=== smoke completion via mesh router ==="
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' curl -sS -X POST \
        'http://127.0.0.1:${ROUTER_PORT}/v1/completions' \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"${MODEL_PATH}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}'
"

# Block until a real generation succeeds end-to-end (router -> P -> D -> client).
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
    echo "=== skipping GSM8K (RUN_GSM8K=${RUN_GSM8K}) ==="
fi

# ======================== 7. run benchmark (foreground) ========================
echo ""
echo "=== running benchmark on ${PREFILL_NODE} ==="
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/benchmark.sh'
"

echo ""
echo "=== done at $(date -Is); results: ${LOG_ROOT}/bench  gsm8k: ${LOG_ROOT}/gsm8k ==="
