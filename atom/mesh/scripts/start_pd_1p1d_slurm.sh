#!/usr/bin/env bash
#SBATCH --job-name=atom-pd-1p1d
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
#SBATCH --output=/it-share/yajizhan/slurm_logs/atom-pd-1p1d-%j.out
#SBATCH --error=/it-share/yajizhan/slurm_logs/atom-pd-1p1d-%j.err
#
# Fully automated 1P + 1D PD-disaggregated benchmark for ATOM mesh.
#
# Usage:
#   mkdir -p /it-share/yajizhan/slurm_logs
#   sbatch /it-share/yajizhan/code/ATOM/atom/mesh/scripts/start_pd_1p1d_slurm.sh
#
# Override defaults via env:
#   sbatch --export=ALL,MODEL_PATH=/mnt/models/deepseek-ai/DeepSeek-R1,ISL_LIST="1024,8192" \
#          start_pd_1p1d_slurm.sh
#
# Layout (default):
#   prefill node : first node in $SLURM_JOB_NODELIST  (g42), TP=4, port 8010
#   decode  node : second node                        (g44), TP=8, port 8020
#   router       : runs on prefill node, port 8000
#   benchmark    : runs on prefill node against router

set -euo pipefail

# ---------- configuration ----------
MODEL_PATH="${MODEL_PATH:-/mnt/models/deepseek-ai/DeepSeek-R1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:mesh-sglang-latest}"
CONTAINER="${CONTAINER:-atom_sglang_mesh_${SLURM_JOB_ID}}"

PREFILL_TP="${PREFILL_TP:-4}"
DECODE_TP="${DECODE_TP:-8}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"

ISL_LIST="${ISL_LIST:-1024,8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-16,32,64}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

LOAD_DUMMY="${LOAD_DUMMY:-1}"                        # 1 = skip weight loading (fast validation)
WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-1800}"   # prefill/decode warmup can take a while for DSR1
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

SCRIPTS_DIR="${SCRIPTS_DIR:-/it-share/yajizhan/code/ATOM/atom/mesh/scripts}"
LOG_ROOT="${LOG_ROOT:-/it-share/yajizhan/slurm_logs/atom-pd-1p1d-${SLURM_JOB_ID}}"

# ---------- pre-flight ----------
echo "=== Job ${SLURM_JOB_ID} starting on $(hostname) at $(date -Is) ==="
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "ERROR: expected 2 nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi
PREFILL_NODE="${NODES[0]}"
DECODE_NODE="${NODES[1]}"

mkdir -p "${LOG_ROOT}"/{prefill,decode,router,bench}

PREFILL_IP=$(srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")
DECODE_IP=$(srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")

cat <<EOF
=== Configuration ===
PREFILL_NODE   : ${PREFILL_NODE}  (IP=${PREFILL_IP}, TP=${PREFILL_TP}, port=${PREFILL_PORT})
DECODE_NODE    : ${DECODE_NODE}   (IP=${DECODE_IP},  TP=${DECODE_TP},  port=${DECODE_PORT})
ROUTER         : ${PREFILL_IP}:${ROUTER_PORT}
MODEL_PATH     : ${MODEL_PATH}
DOCKER_IMAGE   : ${DOCKER_IMAGE}
CONTAINER      : ${CONTAINER}
ISL_LIST       : ${ISL_LIST}
OSL            : ${OSL}
CONC_LIST      : ${CONC_LIST}
LOG_ROOT       : ${LOG_ROOT}
=====================
EOF

# ---------- cleanup trap ----------
cleanup() {
    local rc=$?
    echo ""
    echo "=== cleanup (rc=${rc}) at $(date -Is) ==="
    for node in "$PREFILL_NODE" "$DECODE_NODE"; do
        srun --nodelist="$node" --nodes=1 --ntasks=1 bash -c "
            docker logs '${CONTAINER}' > '${LOG_ROOT}/docker_\$(hostname).log' 2>&1 || true
            docker rm -f '${CONTAINER}' >/dev/null 2>&1 || true
        " || true
    done
    echo "=== cleanup done; logs under ${LOG_ROOT} ==="
}
trap cleanup EXIT

# ---------- helper: launch container on a node ----------
launch_container() {
    local node="$1"
    local role="$2"   # prefill | decode
    echo "[${role}] starting container on ${node}"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        set -euo pipefail
        docker rm -f '${CONTAINER}' 2>/dev/null || true
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
            '${DOCKER_IMAGE}' sleep infinity
        # confirm it's alive
        docker inspect -f '{{.State.Status}}' '${CONTAINER}'
    "
}

# ---------- helper: wait for an HTTP endpoint to return /v1/models ----------
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

# ---------- 1. start containers on both nodes ----------
launch_container "$PREFILL_NODE" prefill
launch_container "$DECODE_NODE"  decode

# ---------- 2. start prefill server (detached) ----------
echo "[prefill] launching server on ${PREFILL_NODE}"
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d \
        -e PREFILL_IP='${PREFILL_IP}' \
        -e MODEL_PATH='${MODEL_PATH}' \
        -e PREFILL_TP='${PREFILL_TP}' \
        -e PREFILL_PORT='${PREFILL_PORT}' \
        -e BOOTSTRAP_PORT='${BOOTSTRAP_PORT}' \
        -e LOAD_DUMMY='${LOAD_DUMMY}' \
        '${CONTAINER}' bash '${SCRIPTS_DIR}/start_prefill.sh'
"

# ---------- 3. start decode server (detached) ----------
echo "[decode] launching server on ${DECODE_NODE}"
srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d \
        -e DECODE_IP='${DECODE_IP}' \
        -e MODEL_PATH='${MODEL_PATH}' \
        -e DECODE_TP='${DECODE_TP}' \
        -e DECODE_PORT='${DECODE_PORT}' \
        -e BOOTSTRAP_PORT='${BOOTSTRAP_PORT}' \
        -e LOAD_DUMMY='${LOAD_DUMMY}' \
        '${CONTAINER}' bash '${SCRIPTS_DIR}/start_decode.sh'
"

# ---------- 4. wait for both servers ----------
wait_endpoint "$PREFILL_NODE" "http://${PREFILL_IP}:${PREFILL_PORT}/v1/models" \
    "$WAIT_SERVER_TIMEOUT" "prefill"
wait_endpoint "$DECODE_NODE"  "http://${DECODE_IP}:${DECODE_PORT}/v1/models" \
    "$WAIT_SERVER_TIMEOUT" "decode"

# ---------- 5. start router (detached) on prefill node ----------
echo "[router] launching on ${PREFILL_NODE}"
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d \
        -e PREFILL_IP='${PREFILL_IP}' \
        -e DECODE_IP='${DECODE_IP}' \
        -e PREFILL_PORT='${PREFILL_PORT}' \
        -e DECODE_PORT='${DECODE_PORT}' \
        -e ROUTER_PORT='${ROUTER_PORT}' \
        -e BOOTSTRAP_PORT='${BOOTSTRAP_PORT}' \
        '${CONTAINER}' bash '${SCRIPTS_DIR}/start_router.sh'
"

wait_endpoint "$PREFILL_NODE" "http://${PREFILL_IP}:${ROUTER_PORT}/v1/models" \
    "$WAIT_ROUTER_TIMEOUT" "router"

# ---------- 6. run benchmark (foreground, blocks) ----------
echo ""
echo "=== running benchmark on ${PREFILL_NODE} ==="
srun --nodelist="$PREFILL_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec \
        -e MODEL_PATH='${MODEL_PATH}' \
        -e ROUTER_PORT='${ROUTER_PORT}' \
        -e ISL_LIST='${ISL_LIST}' \
        -e OSL='${OSL}' \
        -e CONC_LIST='${CONC_LIST}' \
        -e RANDOM_RANGE_RATIO='${RANDOM_RANGE_RATIO}' \
        '${CONTAINER}' bash '${SCRIPTS_DIR}/run_benchmark.sh'
"

echo ""
echo "=== done at $(date -Is); results: ${LOG_ROOT}/bench ==="
