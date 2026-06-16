#!/usr/bin/env bash
#SBATCH --job-name=ds-fp4-atom-2p-dp8ep8-1d-dp8ep8
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
#SBATCH --output=/it-share/yajizhan/slurm_logs/ds_fp4_atom_2p_dp8ep8_1d_dp8ep8-%j.out
#SBATCH --error=/it-share/yajizhan/slurm_logs/ds_fp4_atom_2p_dp8ep8_1d_dp8ep8-%j.err
#
# 2P+1D PD-disaggregated benchmark for ATOM native server + atomesh router.
#   prefill0: ATOM kv_producer, TP=1, DP=8, EP=8 (1 node)
#   prefill1: ATOM kv_producer, TP=1, DP=8, EP=8 (1 node)
#   decode:   ATOM kv_consumer, TP=1, DP=8, EP=8 (1 node)
#   router:   atomesh launch --backend atom (no bootstrap port)
#   KV transfer: Mooncake RDMA (atom/kv_transfer/disaggregation/mooncake)
#
# Mesh router learns each prefill's tp_size/dp_size/kv_role by GETing
# http://<prefill>/kv_transfer_info at startup, then awaits each prefill
# response, extracts kv_transfer_params from the top level, enriches with
# remote_dp_size/remote_tp_size, and injects into the decode request.
#
# Usage:
#   mkdir -p /it-share/yajizhan/slurm_logs
#   sbatch ds_fp4_2p_dp8ep8_1d_dp8ep8_atom_slurm.sh

set -euo pipefail

# ======================== configuration ========================
MODEL_PATH="${MODEL_PATH:-/mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:sglang-v0.5.10-nightly_20260528-mesh-sglang}"
CONTAINER="${CONTAINER:-atom_atom_mesh_${SLURM_JOB_ID}}"

PREFILL_TP="${PREFILL_TP:-1}"
DECODE_TP="${DECODE_TP:-1}"
PREFILL_DP="${PREFILL_DP:-8}"
DECODE_DP="${DECODE_DP:-8}"
PREFILL_EP="${PREFILL_EP:-8}"
DECODE_EP="${DECODE_EP:-8}"
PREFILL0_PORT="${PREFILL0_PORT:-8010}"
PREFILL1_PORT="${PREFILL1_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"

PREFILL_MEM_FRACTION="${PREFILL_MEM_FRACTION:-0.80}"
DECODE_MEM_FRACTION="${DECODE_MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-4096}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-4096}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
MESH_BIN="${MESH_BIN:-/usr/local/bin/atomesh}"

MORI_DISPATCH_DTYPE="${MORI_DISPATCH_DTYPE:-bf16}"
MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-16384}"
MORI_SHMEM_MODE="${MORI_SHMEM_MODE:-ISOLATION}"

ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-1024,4096}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-1800}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

RUN_GSM8K="${RUN_GSM8K:-1}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-3}"
GSM8K_NUM_CONCURRENT="${GSM8K_NUM_CONCURRENT:-65}"

LOG_ROOT="${LOG_ROOT:-/it-share/yajizhan/slurm_logs/$(date +%m%d)_ds_fp4_atom_2p_dp8ep8_1d_dp8ep8_${SLURM_JOB_ID}}"

# ======================== pre-flight ========================
echo "=== Job ${SLURM_JOB_ID} starting on $(hostname) at $(date -Is) ==="
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -ne 3 ]]; then
    echo "ERROR: expected 3 nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi
PREFILL_NODE_0="${NODES[0]}"
PREFILL_NODE_1="${NODES[1]}"
DECODE_NODE="${NODES[2]}"

mkdir -p "${LOG_ROOT}"/{prefill0,prefill1,decode,router,bench,gsm8k,scripts}

# ======================== pre-cleanup ========================
echo "=== pre-cleanup: force-stopping all docker containers on all nodes ==="
for node in "$PREFILL_NODE_0" "$PREFILL_NODE_1" "$DECODE_NODE"; do
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

PREFILL0_IP=$(srun --nodelist="$PREFILL_NODE_0" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")
PREFILL1_IP=$(srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")
DECODE_IP=$(srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")

cat <<INFO
=== Configuration ===
PREFILL0: ${PREFILL_NODE_0} (IP=${PREFILL0_IP}, TP=${PREFILL_TP}, DP=${PREFILL_DP}, EP=${PREFILL_EP}, port=${PREFILL0_PORT})
PREFILL1: ${PREFILL_NODE_1} (IP=${PREFILL1_IP}, TP=${PREFILL_TP}, DP=${PREFILL_DP}, EP=${PREFILL_EP}, port=${PREFILL1_PORT})
DECODE  : ${DECODE_NODE}    (IP=${DECODE_IP},    TP=${DECODE_TP},  DP=${DECODE_DP},  EP=${DECODE_EP},  port=${DECODE_PORT})
ROUTER  : ${PREFILL0_IP}:${ROUTER_PORT}
MODEL   : ${MODEL_PATH}
IMAGE   : ${DOCKER_IMAGE}
BACKEND : atom (Mooncake)
HANDSHAKE_PORT : ${HANDSHAKE_PORT}
MORI    : dtype=${MORI_DISPATCH_DTYPE}, max_tokens=${MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}, shmem=${MORI_SHMEM_MODE}
RUN_GSM8K  : ${RUN_GSM8K} (limit=${GSM8K_LIMIT:-all}, fewshot=${GSM8K_NUM_FEWSHOT})
ISL/OSL/CONC : ${ISL_LIST} / ${OSL} / ${CONC_LIST}
LOG_ROOT: ${LOG_ROOT}
=====================
INFO

# ======================== generate in-container scripts ========================
PREFILL_NUM_GPUS=$((PREFILL_TP * PREFILL_DP))
PREFILL_GPU_IDS=$(seq -s, 0 $((PREFILL_NUM_GPUS - 1)))
DECODE_NUM_GPUS=$((DECODE_TP * DECODE_DP))
DECODE_GPU_IDS=$(seq -s, 0 $((DECODE_NUM_GPUS - 1)))

for idx in 0 1; do
    eval "P_IP=\${PREFILL${idx}_IP}"
    eval "P_PORT=\${PREFILL${idx}_PORT}"
    cat > "${LOG_ROOT}/scripts/prefill${idx}.sh" <<PREFILL_EOF
#!/usr/bin/env bash
set -euo pipefail

echo "[prefill${idx}] IP=${P_IP} TP=${PREFILL_TP} DP=${PREFILL_DP} EP=${PREFILL_EP} port=${P_PORT}"
mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${PREFILL_GPU_IDS}
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export ATOM_HOST_IP=${P_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:\${LD_LIBRARY_PATH:-}

export MORI_SHMEM_MODE=${MORI_SHMEM_MODE}
export MORI_DISPATCH_DTYPE=${MORI_DISPATCH_DTYPE}
export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}

rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \\
    --model "${MODEL_PATH}" \\
    --host 0.0.0.0 --server-port "${P_PORT}" \\
    --trust-remote-code \\
    -tp "${PREFILL_TP}" \\
    -dp "${PREFILL_DP}" \\
    --enable-expert-parallel \\
    --enable-dp-attention \\
    --kv_cache_dtype "${KV_CACHE_DTYPE}" \\
    --block-size "${BLOCK_SIZE}" \\
    --gpu-memory-utilization "${PREFILL_MEM_FRACTION}" \\
    --max-num-seqs "${PREFILL_MAX_NUM_SEQS}" \\
    --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${P_IP}\",\"handshake_port\":${HANDSHAKE_PORT}}" \\
    ${EXTRA_SERVER_ARGS} \\
    2>&1 | tee /workspace/logs/prefill.log
PREFILL_EOF
done

cat > "${LOG_ROOT}/scripts/decode.sh" <<DECODE_EOF
#!/usr/bin/env bash
set -euo pipefail

echo "[decode] IP=${DECODE_IP} TP=${DECODE_TP} DP=${DECODE_DP} EP=${DECODE_EP} port=${DECODE_PORT}"
mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${DECODE_GPU_IDS}
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export ATOM_HOST_IP=${DECODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:\${LD_LIBRARY_PATH:-}

export MORI_SHMEM_MODE=${MORI_SHMEM_MODE}
export MORI_DISPATCH_DTYPE=${MORI_DISPATCH_DTYPE}
export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}

rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \\
    --model "${MODEL_PATH}" \\
    --host 0.0.0.0 --server-port "${DECODE_PORT}" \\
    --trust-remote-code \\
    -tp "${DECODE_TP}" \\
    -dp "${DECODE_DP}" \\
    --enable-expert-parallel \\
    --enable-dp-attention \\
    --kv_cache_dtype "${KV_CACHE_DTYPE}" \\
    --block-size "${BLOCK_SIZE}" \\
    --gpu-memory-utilization "${DECODE_MEM_FRACTION}" \\
    --max-num-seqs "${DECODE_MAX_NUM_SEQS}" \\
    --kv-transfer-config "{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${DECODE_IP}\",\"handshake_port\":${HANDSHAKE_PORT}}" \\
    ${EXTRA_SERVER_ARGS} \\
    2>&1 | tee /workspace/logs/decode.log
DECODE_EOF

cat > "${LOG_ROOT}/scripts/router.sh" <<ROUTER_EOF
#!/usr/bin/env bash
set -euo pipefail

echo "[router] prefill0=http://${PREFILL0_IP}:${PREFILL0_PORT} prefill1=http://${PREFILL1_IP}:${PREFILL1_PORT} decode=http://${DECODE_IP}:${DECODE_PORT} router=0.0.0.0:${ROUTER_PORT}"
mkdir -p /workspace/logs

${MESH_BIN} launch \\
    --host 0.0.0.0 --port "${ROUTER_PORT}" \\
    --pd-disaggregation \\
    --prefill "http://${PREFILL0_IP}:${PREFILL0_PORT}" \\
    --prefill "http://${PREFILL1_IP}:${PREFILL1_PORT}" \\
    --decode  "http://${DECODE_IP}:${DECODE_PORT}" \\
    --policy random \\
    --backend atom \\
    --log-dir /workspace/logs \\
    --log-level info \\
    --disable-health-check \\
    --prometheus-port 29100 \\
    2>&1 | tee /workspace/logs/router.log
ROUTER_EOF

cat > "${LOG_ROOT}/scripts/gsm8k.sh" <<GSMEIGHT_EOF
#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="/workspace/gsm8k_results"

echo "[gsm8k] model=${MODEL_PATH} endpoint=http://127.0.0.1:${ROUTER_PORT}"
echo "[gsm8k] limit=${GSM8K_LIMIT:-all} fewshot=${GSM8K_NUM_FEWSHOT} concurrent=${GSM8K_NUM_CONCURRENT}"

if ! command -v lm_eval >/dev/null 2>&1; then
    echo "[gsm8k] installing lm-eval..."
    pip install 'lm-eval[api]'
fi

RUN_TAG="\$(date +%Y%m%d%H%M%S)_gsm8k"
mkdir -p "\${RESULT_DIR}"

LIMIT_ARG=""
if [[ -n "${GSM8K_LIMIT}" ]]; then
    LIMIT_ARG="--limit ${GSM8K_LIMIT}"
fi

lm_eval --model local-completions \\
    --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/completions,num_concurrent=${GSM8K_NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \\
    --tasks gsm8k \\
    --num_fewshot "${GSM8K_NUM_FEWSHOT}" \\
    \${LIMIT_ARG} \\
    --output_path "\${RESULT_DIR}/\${RUN_TAG}"

python3 -c "
from pathlib import Path
import json

result_dir = Path('\${RESULT_DIR}/\${RUN_TAG}')
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

echo "[gsm8k] results saved to \${RESULT_DIR}/\${RUN_TAG}"
GSMEIGHT_EOF

cat > "${LOG_ROOT}/scripts/benchmark.sh" <<BENCH_EOF
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

mkdir -p "\${RESULT_DIR}"

IFS=',' read -ra ISLS <<< "${ISL_LIST}"
IFS=',' read -ra CONCS <<< "${CONC_LIST}"

for ISL in "\${ISLS[@]}"; do
    for CONC in "\${CONCS[@]}"; do
        RESULT_FILENAME="pd-atomesh-dp8ep8-\${ISL}-${OSL}-\${CONC}-${RANDOM_RANGE_RATIO}"
        echo ""
        echo "========================================="
        echo "[bench] ISL=\${ISL} OSL=${OSL} CONC=\${CONC}"
        echo "========================================="

        PYTHONDONTWRITEBYTECODE=1 python /tmp/sglang-benchmark/bench_serving/benchmark_serving.py \\
            --model="${MODEL_PATH}" \\
            --backend=vllm \\
            --base-url="http://127.0.0.1:${ROUTER_PORT}" \\
            --dataset-name=random \\
            --random-input-len="\${ISL}" \\
            --random-output-len="${OSL}" \\
            --random-range-ratio "${RANDOM_RANGE_RATIO}" \\
            --num-prompts=\$(( CONC * 10 )) \\
            --max-concurrency="\${CONC}" \\
            --trust-remote-code \\
            --num-warmups=\$(( 2 * CONC )) \\
            --request-rate=inf \\
            --ignore-eos \\
            --save-result \\
            --percentile-metrics='ttft,tpot,itl,e2el' \\
            --result-dir="\${RESULT_DIR}" \\
            --result-filename="\${RESULT_FILENAME}.json"
    done
done

echo ""
echo "========================================="
echo "[bench] summary"
echo "========================================="

python3 -c "
from pathlib import Path
import json

result_dir = Path('\${RESULT_DIR}')
json_files = sorted(result_dir.glob('pd-atomesh-dp8ep8-*.json'))
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

echo "[bench] results saved to \${RESULT_DIR}"
BENCH_EOF

chmod +x "${LOG_ROOT}"/scripts/*.sh

echo "[scripts] generated under ${LOG_ROOT}/scripts/"
ls -la "${LOG_ROOT}"/scripts/

# ======================== cleanup trap ========================
cleanup() {
    local rc=$?
    echo ""
    echo "=== cleanup (rc=${rc}) at $(date -Is) ==="
    for node in "$PREFILL_NODE_0" "$PREFILL_NODE_1" "$DECODE_NODE"; do
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
launch_container "$PREFILL_NODE_0" prefill0
launch_container "$PREFILL_NODE_1" prefill1
launch_container "$DECODE_NODE"    decode

# ======================== 2. start prefill + decode servers (detached) ========================
echo "[prefill0] launching ATOM kv_producer on ${PREFILL_NODE_0}"
srun --nodelist="$PREFILL_NODE_0" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill0.sh'
"

echo "[prefill1] launching ATOM kv_producer on ${PREFILL_NODE_1}"
srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill1.sh'
"

echo "[decode] launching ATOM kv_consumer on ${DECODE_NODE}"
srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/decode.sh'
"

# ======================== 3. wait for servers (HTTP health check) ========================
wait_endpoint "$PREFILL_NODE_0" "http://${PREFILL0_IP}:${PREFILL0_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "prefill0-http"
wait_endpoint "$PREFILL_NODE_1" "http://${PREFILL1_IP}:${PREFILL1_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "prefill1-http"
wait_endpoint "$DECODE_NODE"    "http://${DECODE_IP}:${DECODE_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "decode-http"

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
verify_kv_info prefill0  "$PREFILL_NODE_0" "$PREFILL0_IP"  "$PREFILL0_PORT" kv_producer
verify_kv_info prefill1  "$PREFILL_NODE_1" "$PREFILL1_IP"  "$PREFILL1_PORT" kv_producer
verify_kv_info decode    "$DECODE_NODE"    "$DECODE_IP"    "$DECODE_PORT"   kv_consumer

# ======================== 4. start router (detached) ========================
echo ""
echo "[router] launching atomesh on ${PREFILL_NODE_0}"
srun --nodelist="$PREFILL_NODE_0" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/router.sh'
"

wait_endpoint "$PREFILL_NODE_0" "http://${PREFILL0_IP}:${ROUTER_PORT}/v1/models" \
    "$WAIT_ROUTER_TIMEOUT" "router-http"

# ======================== 5. smoke completion (catches relay breakage fast) ========================
echo ""
echo "=== smoke completion via mesh router ==="
srun --nodelist="$PREFILL_NODE_0" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' curl -sS -X POST \
        'http://127.0.0.1:${ROUTER_PORT}/v1/completions' \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"${MODEL_PATH}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}'
"

wait_inference_ready "$PREFILL_NODE_0" "http://${PREFILL0_IP}:${ROUTER_PORT}" \
    "$MODEL_PATH" "$WAIT_SERVER_TIMEOUT" "router-pipeline"

# ======================== 6. run gsm8k accuracy (foreground, optional) ========================
if [[ "${RUN_GSM8K}" == "1" ]]; then
    echo ""
    echo "=== running GSM8K accuracy eval on ${PREFILL_NODE_0} ==="
    srun --nodelist="$PREFILL_NODE_0" --nodes=1 --ntasks=1 bash -lc "
        docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/gsm8k.sh'
    "
else
    echo "=== skipping GSM8K (RUN_GSM8K=${RUN_GSM8K}) ==="
fi

# ======================== 7. run benchmark (foreground) ========================
echo ""
echo "=== running benchmark on ${PREFILL_NODE_0} ==="
srun --nodelist="$PREFILL_NODE_0" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/benchmark.sh'
"

echo ""
echo "=== done at $(date -Is); results: ${LOG_ROOT}/bench  gsm8k: ${LOG_ROOT}/gsm8k ==="
