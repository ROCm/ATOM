#!/usr/bin/env bash
#SBATCH --job-name=ds-v4-2p-tp8-1d-tp8-atom-mtp-dpa
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
#SBATCH --output=/it-share/yajizhan/slurm_dsv4_logs/ds_v4_2p_tp8_1d_tp8_atom_mtp_dpa-%j.out
#SBATCH --error=/it-share/yajizhan/slurm_dsv4_logs/ds_v4_2p_tp8_1d_tp8_atom_mtp_dpa-%j.err
#
# Self-contained 2P+1D PD-disaggregated benchmark for DeepSeek-V4-Pro
# on ATOM with mooncake RDMA KV transfer and MTP (multi-token prediction).
#   prefill: TP=8 (2 instances), decode: TP=8 (1 instance), --enable-dp-attention.
#   3 nodes total: 2 prefill + 1 decode.
#
# Usage:
#   mkdir -p /it-share/yajizhan/slurm_logs
#   sbatch ds_v4_2p_tp8_1d_tp8_atom_mtp_dpa_slurm.sh

set -euo pipefail

# ======================== configuration ========================
MODEL_PATH="${MODEL_PATH:-/mnt/models/DeepSeek-V4-Pro/}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/atom-dev:latest}"
CONTAINER="${CONTAINER:-atom_mesh_dsv4_2p1d_mtp_dpa_${SLURM_JOB_ID}}"

PREFILL_TP="${PREFILL_TP:-8}"
DECODE_TP="${DECODE_TP:-8}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"

MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-1}"

ISL_LIST="${ISL_LIST:-8192}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-512,768,1024}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

WAIT_SERVER_TIMEOUT="${WAIT_SERVER_TIMEOUT:-1800}"
WAIT_ROUTER_TIMEOUT="${WAIT_ROUTER_TIMEOUT:-300}"

RUN_GSM8K="${RUN_GSM8K:-1}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
GSM8K_NUM_FEWSHOT="${GSM8K_NUM_FEWSHOT:-3}"
GSM8K_NUM_CONCURRENT="${GSM8K_NUM_CONCURRENT:-768}"

LOG_ROOT="${LOG_ROOT:-/it-share/yajizhan/slurm_dsv4_logs/$(date +%m%d)_ds_v4_2p_tp8_1d_tp8_atom_mtp_dpa_${SLURM_JOB_ID}}"

# ======================== pre-flight ========================
echo "=== Job ${SLURM_JOB_ID} starting on $(hostname) at $(date -Is) ==="
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#NODES[@]}" -ne 3 ]]; then
    echo "ERROR: expected 3 nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 1
fi
PREFILL_NODE_1="${NODES[0]}"
PREFILL_NODE_2="${NODES[1]}"
DECODE_NODE="${NODES[2]}"

mkdir -p "${LOG_ROOT}"/{prefill_1,prefill_2,decode,router,bench,gsm8k,scripts}
ln -sfn "${LOG_ROOT}" /it-share/yajizhan/slurm_dsv4_logs/latest_ds_v4_2p1d_mtp_dpa

# ======================== cleanup trap ========================
cleanup() {
    local rc=$?
    echo ""
    echo "=== cleanup (rc=${rc}) at $(date -Is) ==="
    for node in "$PREFILL_NODE_1" "$PREFILL_NODE_2" "$DECODE_NODE"; do
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

# ======================== pre-cleanup ========================
echo "=== pre-cleanup: force-stopping all docker containers on all nodes ==="
for node in "$PREFILL_NODE_1" "$PREFILL_NODE_2" "$DECODE_NODE"; do
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

PREFILL_IP_1=$(srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")
PREFILL_IP_2=$(srun --nodelist="$PREFILL_NODE_2" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")
DECODE_IP=$(srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 \
    bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7; exit}'")

cat <<INFO
=== Configuration ===
PREFILL-1 : ${PREFILL_NODE_1}  (IP=${PREFILL_IP_1}, TP=${PREFILL_TP}, port=${PREFILL_PORT})
PREFILL-2 : ${PREFILL_NODE_2}  (IP=${PREFILL_IP_2}, TP=${PREFILL_TP}, port=${PREFILL_PORT})
DECODE    : ${DECODE_NODE}     (IP=${DECODE_IP},    TP=${DECODE_TP},  port=${DECODE_PORT})
ROUTER    : ${PREFILL_IP_1}:${ROUTER_PORT}
MODEL     : ${MODEL_PATH}
IMAGE     : ${DOCKER_IMAGE}
BACKEND   : atom (PD mooncake KV transfer, DPA + MTP)
MTP       : method=mtp num_speculative_tokens=${NUM_SPEC_TOKENS}
RUN_GSM8K : ${RUN_GSM8K} (limit=${GSM8K_LIMIT:-all}, fewshot=${GSM8K_NUM_FEWSHOT})
ISL/OSL/CONC : ${ISL_LIST} / ${OSL} / ${CONC_LIST}
LOG_ROOT  : ${LOG_ROOT}
=====================
INFO

# ======================== generate in-container scripts ========================
PREFILL_GPU_IDS=$(seq -s, 0 $((PREFILL_TP - 1)))
DECODE_GPU_IDS=$(seq -s, 0 $((DECODE_TP - 1)))

for idx in 1 2; do
    eval "P_IP=\${PREFILL_IP_${idx}}"
    cat > "${LOG_ROOT}/scripts/prefill_${idx}.sh" <<PREFILL_EOF
#!/usr/bin/env bash
set -euo pipefail

echo "[prefill${idx}] IP=${P_IP} TP=${PREFILL_TP} port=${PREFILL_PORT}"

mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${PREFILL_GPU_IDS}
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=${P_IP}
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:\${LD_LIBRARY_PATH:-}

rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \\
    --model "${MODEL_PATH}" \\
    --host 0.0.0.0 --server-port "${PREFILL_PORT}" \\
    --trust-remote-code \\
    -tp "${PREFILL_TP}" \\
    --enable-dp-attention \\
    --kv_cache_dtype "${KV_CACHE_DTYPE}" \\
    --block-size "${BLOCK_SIZE}" \\
    --gpu-memory-utilization "${MEM_FRACTION}" \\
    --max-num-seqs "${MAX_NUM_SEQS}" \\
    --method mtp --num-speculative-tokens "${NUM_SPEC_TOKENS}" \\
    --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${P_IP}\",\"handshake_port\":${HANDSHAKE_PORT}}" \\
    ${EXTRA_SERVER_ARGS} \\
    2>&1 | tee /workspace/logs/prefill.log
PREFILL_EOF
done

cat > "${LOG_ROOT}/scripts/decode.sh" <<DECODE_EOF
#!/usr/bin/env bash
set -euo pipefail

echo "[decode] IP=${DECODE_IP} TP=${DECODE_TP} port=${DECODE_PORT}"

mkdir -p /workspace/logs

export HIP_VISIBLE_DEVICES=${DECODE_GPU_IDS}
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=${DECODE_IP}
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:\${LD_LIBRARY_PATH:-}

rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \\
    --model "${MODEL_PATH}" \\
    --host 0.0.0.0 --server-port "${DECODE_PORT}" \\
    --trust-remote-code \\
    -tp "${DECODE_TP}" \\
    --enable-dp-attention \\
    --kv_cache_dtype "${KV_CACHE_DTYPE}" \\
    --block-size "${BLOCK_SIZE}" \\
    --gpu-memory-utilization "${MEM_FRACTION}" \\
    --max-num-seqs "${MAX_NUM_SEQS}" \\
    --method mtp --num-speculative-tokens "${NUM_SPEC_TOKENS}" \\
    --kv-transfer-config "{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"proxy_ip\":\"${DECODE_IP}\",\"handshake_port\":${HANDSHAKE_PORT}}" \\
    --cudagraph-capture-sizes "[1,2,4,8,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768,784,800,816,832,848,864,880,896,912,928,944,960,976,992,1008,1024]" \\
    ${EXTRA_SERVER_ARGS} \\
    2>&1 | tee /workspace/logs/decode.log
DECODE_EOF

cat > "${LOG_ROOT}/scripts/router.sh" <<ROUTER_EOF
#!/usr/bin/env bash
set -euo pipefail

echo "[router] prefill-1=${PREFILL_IP_1}:${PREFILL_PORT}"
echo "[router] prefill-2=${PREFILL_IP_2}:${PREFILL_PORT}"
echo "[router] decode=${DECODE_IP}:${DECODE_PORT}"
echo "[router] router=0.0.0.0:${ROUTER_PORT}"

mkdir -p /workspace/logs

/usr/local/bin/atomesh launch \\
    --host 0.0.0.0 --port "${ROUTER_PORT}" \\
    --pd-disaggregation \\
    --prefill "http://${PREFILL_IP_1}:${PREFILL_PORT}" \\
    --prefill "http://${PREFILL_IP_2}:${PREFILL_PORT}" \\
    --decode  "http://${DECODE_IP}:${DECODE_PORT}" \\
    --policy random \\
    --backend atom \\
    --log-dir /workspace/logs \\
    --log-level info \\
    --disable-health-check \\
    --disable-circuit-breaker \\
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

mkdir -p "\${RESULT_DIR}"

LIMIT_ARG=""
if [[ -n "${GSM8K_LIMIT}" ]]; then
    LIMIT_ARG="--limit ${GSM8K_LIMIT}"
fi

IFS=',' read -ra GSM8K_CONCS <<< "${GSM8K_NUM_CONCURRENT}"
for GSM8K_CONC in "\${GSM8K_CONCS[@]}"; do
    RUN_TAG="\$(date +%Y%m%d%H%M%S)_gsm8k_v4_2p1d_mtp_dpa_c\${GSM8K_CONC}"
    echo ""
    echo "========================================="
    echo "[gsm8k] running with concurrent=\${GSM8K_CONC}"
    echo "========================================="

    lm_eval --model local-completions \\
        --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${ROUTER_PORT}/v1/completions,num_concurrent=\${GSM8K_CONC},max_retries=3,tokenized_requests=False,trust_remote_code=True" \\
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
print(f'[gsm8k] concurrent=\${GSM8K_CONC} exact_match,flexible-extract = {score}')
print('=========================================')
print(json.dumps(data.get('results', {}), indent=2))
"
done

echo "[gsm8k] all runs done, results saved to \${RESULT_DIR}"
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
        RESULT_FILENAME="pd-atom-v4-2p1d-mtp-dpa-\${ISL}-${OSL}-\${CONC}-${RANDOM_RANGE_RATIO}"
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
json_files = sorted(result_dir.glob('pd-atom-v4-2p1d-mtp-dpa-*.json'))
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

# ======================== helper ========================

detect_nic_type() {
    if [[ -n "${MORI_NIC_TYPE:-}" ]]; then
        echo "$MORI_NIC_TYPE"
        return
    fi
    local bnxt=0 mlx5=0 ionic=0
    if [[ -d /sys/class/infiniband ]]; then
        for dev in /sys/class/infiniband/*; do
            local name
            name=$(basename "$dev")
            case "$name" in
                bnxt_re*) ((bnxt++)) ;;
                mlx5*)    ((mlx5++)) ;;
                ionic*)   ((ionic++)) ;;
                *)
                    local drv
                    drv=$(readlink -f "$dev/device/driver" 2>/dev/null || true)
                    drv=$(basename "$drv" 2>/dev/null || true)
                    case "$drv" in
                        bnxt*)  ((bnxt++)) ;;
                        mlx5*)  ((mlx5++)) ;;
                        ionic*) ((ionic++)) ;;
                    esac
                    ;;
            esac
        done
    fi
    if (( bnxt >= mlx5 && bnxt >= ionic && bnxt > 0 )); then
        echo "bnxt"
    elif (( ionic >= mlx5 && ionic > 0 )); then
        echo "ionic"
    else
        echo "mlx5"
    fi
}

find_host_ibverbs() {
    local candidates=(
        /usr/lib64/libibverbs.so.1
        /lib/x86_64-linux-gnu/libibverbs.so.1
        /usr/lib/x86_64-linux-gnu/libibverbs.so.1
    )
    for c in "${candidates[@]}"; do
        local resolved
        resolved=$(readlink -f "$c" 2>/dev/null || true)
        if [[ -f "$resolved" ]]; then
            echo "$resolved"
            return
        fi
    done
}

nic_mount_flags() {
    local nic_type="$1"
    local flags=()
    case "$nic_type" in
        bnxt)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            for lib in /usr/local/lib/libbnxt_re-rdmav*.so; do
                if [[ -f "$lib" ]]; then
                    flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/libibverbs/$(basename "$lib")")
                fi
            done
            for lib in /usr/local/lib/libbnxt_re.so; do
                if [[ -f "$lib" ]]; then
                    flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                fi
            done
            if [[ -d /etc/libibverbs.d ]]; then
                flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            fi
            ;;
        ionic)
            local host_ibverbs
            host_ibverbs=$(find_host_ibverbs)
            if [[ -n "$host_ibverbs" ]]; then
                flags+=(-v "$host_ibverbs:/lib/x86_64-linux-gnu/libibverbs.so.1")
            fi
            local ionic_dirs=(/usr/local/lib /usr/lib/x86_64-linux-gnu)
            for dir in "${ionic_dirs[@]}"; do
                for lib in "$dir"/libionic*.so; do
                    if [[ -f "$lib" ]]; then
                        local real
                        real=$(readlink -f "$lib")
                        if [[ -f "$real" ]]; then
                            flags+=(-v "$real:$real")
                        fi
                        flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                    fi
                done
            done
            local provider_dir=/usr/lib/x86_64-linux-gnu/libibverbs
            if [[ -d "$provider_dir" ]]; then
                for lib in "$provider_dir"/libionic-rdmav*.so; do
                    if [[ -f "$lib" ]]; then
                        flags+=(-v "$lib:$lib")
                    fi
                done
            fi
            if [[ -d /etc/libibverbs.d ]]; then
                flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            fi
            ;;
        mlx5)
            ;;
    esac
    echo "${flags[@]}"
}

launch_container() {
    local node="$1"
    local role="$2"
    echo "[${role}] starting container on ${node}"
    srun --nodelist="$node" --nodes=1 --ntasks=1 bash -lc "
        set -euo pipefail

        $(declare -f detect_nic_type find_host_ibverbs nic_mount_flags)
        NIC_TYPE=\$(detect_nic_type)
        echo \"[docker] NIC type detected: \${NIC_TYPE} on \$(hostname)\"
        read -ra NIC_MOUNTS <<< \"\$(nic_mount_flags \"\${NIC_TYPE}\")\"
        if [[ \${#NIC_MOUNTS[@]} -gt 0 ]]; then
            echo \"[docker] RDMA mounts: \${NIC_MOUNTS[*]}\"
        else
            echo \"[docker] no out-of-tree RDMA mounts needed\"
        fi

        docker rm -f '${CONTAINER}' 2>/dev/null || true
        docker pull '${DOCKER_IMAGE}'
        docker run -d --name '${CONTAINER}' \
            --network host --ipc host --privileged \
            --device /dev/kfd --device /dev/dri \
            --device /dev/infiniband \
            --group-add video \
            --cap-add IPC_LOCK --cap-add NET_ADMIN \
            --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
            --shm-size 128G \
            -v /mnt:/mnt \
            -v /data:/data \
            -v /it-share:/it-share \
            -v '${LOG_ROOT}/${role}':/workspace/logs \
            -v '${LOG_ROOT}/bench':/workspace/benchmark_results \
            -v '${LOG_ROOT}/gsm8k':/workspace/gsm8k_results \
            \"\${NIC_MOUNTS[@]}\" \
            '${DOCKER_IMAGE}' sleep infinity
        docker inspect -f '{{.State.Status}}' '${CONTAINER}'

        docker exec '${CONTAINER}' bash -c '
            sysctl -w net.core.somaxconn=4096 2>/dev/null || true
            sysctl -w net.ipv4.tcp_max_syn_backlog=4096 2>/dev/null || true
        '
        echo \"[docker] tuned TCP backlog on \$(hostname)\"

        # Fix ionic RDMA ABI mismatch
        HOST_IONIC=\$(ls /usr/lib/x86_64-linux-gnu/libionic.so.1.* 2>/dev/null \
                      | grep -v '\\.a\$' | head -1 || true)
        if [[ -n \"\${HOST_IONIC}\" && -f \"\${HOST_IONIC}\" ]]; then
            IONIC_NAME=\$(basename \"\${HOST_IONIC}\")
            docker cp \"\${HOST_IONIC}\" '${CONTAINER}':/usr/lib/x86_64-linux-gnu/\"\${IONIC_NAME}\"
            docker exec '${CONTAINER}' bash -c \"
                cd /usr/lib/x86_64-linux-gnu
                ln -sf '\${IONIC_NAME}' libionic.so.1
                cp -f '\${IONIC_NAME}' libibverbs/libionic-rdmav34.so 2>/dev/null || true
            \"
            echo \"[docker] patched libionic → \${IONIC_NAME} on \$(hostname)\"
        fi

        # Install latest ATOM from host source
        docker exec '${CONTAINER}' bash -c 'git config --global --add safe.directory /it-share/yajizhan/code/ATOM && pip install -e /it-share/yajizhan/code/ATOM' 2>&1 | tail -5
        echo \"[docker] installed ATOM from host source on \$(hostname)\"
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
launch_container "$PREFILL_NODE_1" prefill_1
launch_container "$PREFILL_NODE_2" prefill_2
launch_container "$DECODE_NODE"    decode

# ======================== 2. start prefill servers (detached) ========================
echo "[prefill-1] launching server on ${PREFILL_NODE_1}"
srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill_1.sh'
"
echo "[prefill-2] launching server on ${PREFILL_NODE_2}"
srun --nodelist="$PREFILL_NODE_2" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/prefill_2.sh'
"

# ======================== 3. start decode server (detached) ========================
echo "[decode] launching server on ${DECODE_NODE}"
srun --nodelist="$DECODE_NODE" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/decode.sh'
"

# ======================== 4. wait for servers ========================
wait_endpoint "$PREFILL_NODE_1" "http://${PREFILL_IP_1}:${PREFILL_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "prefill-1-http"
wait_endpoint "$PREFILL_NODE_2" "http://${PREFILL_IP_2}:${PREFILL_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "prefill-2-http"
wait_endpoint "$DECODE_NODE"    "http://${DECODE_IP}:${DECODE_PORT}/health" \
    "$WAIT_SERVER_TIMEOUT" "decode-http"

# ======================== 4b. verify kv_transfer_info ========================
echo ""
echo "=== verifying /kv_transfer_info ==="
verify_kv_info() {
    local node="$1" ip="$2" port="$3" want="$4" label="$5"
    local info="" got="" attempt=0 max_attempts=3
    while [[ $attempt -lt $max_attempts ]]; do
        attempt=$((attempt + 1))
        info=$(srun --nodelist="$node" --nodes=1 --ntasks=1 \
            bash -c "curl -sf http://${ip}:${port}/kv_transfer_info" 2>&1) && break
        echo "[kv_info][${label}] attempt ${attempt}/${max_attempts} failed (rc=$?): ${info:0:200}" >&2
        sleep 5
    done
    if [[ -z "$info" ]]; then
        echo "ERROR: ${label} /kv_transfer_info returned empty after ${max_attempts} attempts" >&2
        return 1
    fi
    echo "[kv_info][${label}] ${info}"
    got=$(echo "$info" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("kv_role",""))' 2>&1) || {
        echo "ERROR: ${label} failed to parse kv_transfer_info JSON: ${info:0:200}" >&2
        return 1
    }
    if [[ "$got" != "$want" ]]; then
        echo "ERROR: ${label} kv_role mismatch: want=${want} got=${got}" >&2
        return 1
    fi
}
verify_kv_info "$PREFILL_NODE_1" "$PREFILL_IP_1" "$PREFILL_PORT" kv_producer prefill-1
verify_kv_info "$PREFILL_NODE_2" "$PREFILL_IP_2" "$PREFILL_PORT" kv_producer prefill-2
verify_kv_info "$DECODE_NODE"    "$DECODE_IP"    "$DECODE_PORT"  kv_consumer decode

# ======================== 5. start router (detached) ========================
echo "[router] launching on ${PREFILL_NODE_1}"
srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec -d '${CONTAINER}' bash '${LOG_ROOT}/scripts/router.sh'
"

wait_endpoint "$PREFILL_NODE_1" "http://${PREFILL_IP_1}:${ROUTER_PORT}/v1/models" \
    "$WAIT_ROUTER_TIMEOUT" "router-http"

# ======================== 5b. smoke completion ========================
echo ""
echo "=== smoke completion via mesh router ==="
srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' curl -sS -X POST \
        'http://127.0.0.1:${ROUTER_PORT}/v1/completions' \
        -H 'Content-Type: application/json' \
        -d '{\"model\":\"${MODEL_PATH}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}'
"

wait_inference_ready "$PREFILL_NODE_1" "http://${PREFILL_IP_1}:${ROUTER_PORT}" \
    "$MODEL_PATH" "$WAIT_SERVER_TIMEOUT" "router-pipeline"

# ======================== 6. run gsm8k accuracy (foreground, optional) ========================
if [[ "${RUN_GSM8K}" == "1" ]]; then
    echo ""
    echo "=== running GSM8K accuracy eval on ${PREFILL_NODE_1} ==="
    for script in "${LOG_ROOT}"/scripts/gsm8k*.sh; do
        sed -i \
            -e "s|\${MODEL_PATH}|${MODEL_PATH}|g" \
            -e "s|\${ROUTER_PORT}|${ROUTER_PORT}|g" \
            -e "s|\${GSM8K_LIMIT}|${GSM8K_LIMIT}|g" \
            -e "s|\${GSM8K_NUM_FEWSHOT}|${GSM8K_NUM_FEWSHOT}|g" \
            -e "s|\${GSM8K_NUM_CONCURRENT}|${GSM8K_NUM_CONCURRENT}|g" \
            -e "s|\${NUM_SPEC_TOKENS}|${NUM_SPEC_TOKENS}|g" \
            "$script"
    done
    srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
        docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/gsm8k.sh'
    "
else
    echo "=== skipping GSM8K (RUN_GSM8K=${RUN_GSM8K}) ==="
fi

# ======================== 7. run benchmark (foreground) ========================
echo ""
echo "=== running benchmark on ${PREFILL_NODE_1} ==="
srun --nodelist="$PREFILL_NODE_1" --nodes=1 --ntasks=1 bash -lc "
    docker exec '${CONTAINER}' bash '${LOG_ROOT}/scripts/benchmark.sh'
"

echo ""
echo "=== done at $(date -Is); results: ${LOG_ROOT}/bench  gsm8k: ${LOG_ROOT}/gsm8k ==="
