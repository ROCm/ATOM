#!/bin/bash
set -euo pipefail

# PD Disaggregation benchmark test script.
#
# The coordinator (this script) lands on any node from the CI pool.
# It uses itself as the prefill node, then SSH-probes peer nodes to
# find N free ones for decode (N = NUM_DECODE_NODES, default 1).
#
# Env vars (required):
#   PD_NODE_LIST  — comma-separated hostnames of all CI nodes
#
# Env vars (optional):
#   NUM_DECODE_NODES — how many decode nodes to pick (default: 1)
#   PREFILL_ARGS, DECODE_ARGS, PROXY_PORT, PRODUCER_PORT, CONSUMER_PORT,
#   ATOM_DOCKER_IMAGE, ISL, OSL, CONC, RANDOM_RANGE_RATIO, RESULT_FILENAME
#
# Usage:
#   atom_pd_test.sh pick-nodes          # discover self + pick decode nodes
#   atom_pd_test.sh launch-all  <model> # start proxy+producer+consumers
#   atom_pd_test.sh benchmark   <model> # run benchmark against proxy
#   atom_pd_test.sh stop-all            # cleanup everything
#   atom_pd_test.sh dump-logs           # dump all logs

TYPE=${1:-launch-all}
MODEL_PATH=${2:-deepseek-ai/DeepSeek-R1-0528}
EXTRA_ARGS=("${@:3}")

PROXY_PORT=${PROXY_PORT:-10001}
PRODUCER_PORT=${PRODUCER_PORT:-8003}
CONSUMER_BASE_PORT=${CONSUMER_BASE_PORT:-8004}
DISCOVERY_PORT=${DISCOVERY_PORT:-36367}
NUM_DECODE_NODES=${NUM_DECODE_NODES:-1}

PREFILL_ARGS=${PREFILL_ARGS:-"--kv_cache_dtype fp8 -tp 4"}
DECODE_ARGS=${DECODE_ARGS:-"--kv_cache_dtype fp8 -tp 8"}

PROXY_LOG="/tmp/atom_proxy.log"
PRODUCER_LOG="/tmp/atom_producer.log"

DECODE_HOSTS_FILE="/tmp/atom_pd_decode_hosts"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes"

# ── RDMA NIC detection (ported from mori/docker/ci_run.sh) ─────────

detect_nic_type() {
    if [[ -n "${MORI_NIC_TYPE:-}" ]]; then echo "$MORI_NIC_TYPE"; return; fi
    local bnxt=0 mlx5=0 ionic=0
    if [[ -d /sys/class/infiniband ]]; then
        for dev in /sys/class/infiniband/*; do
            local name; name=$(basename "$dev")
            case "$name" in
                bnxt_re*) ((bnxt++)) ;; mlx5*) ((mlx5++)) ;; ionic*) ((ionic++)) ;;
                *)
                    local drv; drv=$(basename "$(readlink -f "$dev/device/driver" 2>/dev/null)" 2>/dev/null || true)
                    case "$drv" in bnxt*) ((bnxt++)) ;; mlx5*) ((mlx5++)) ;; ionic*) ((ionic++)) ;; esac ;;
            esac
        done
    fi
    if (( bnxt >= mlx5 && bnxt >= ionic && bnxt > 0 )); then echo "bnxt"
    elif (( ionic >= mlx5 && ionic > 0 )); then echo "ionic"
    else echo "mlx5"; fi
}

find_host_ibverbs() {
    for c in /usr/lib64/libibverbs.so.1 /lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so.1; do
        local resolved; resolved=$(readlink -f "$c" 2>/dev/null || true)
        if [[ -f "$resolved" ]]; then echo "$resolved"; return; fi
    done
}

nic_mount_flags() {
    local nic_type="$1" flags=()
    case "$nic_type" in
        bnxt)
            local hib; hib=$(find_host_ibverbs)
            [[ -n "$hib" ]] && flags+=(-v "$hib:/lib/x86_64-linux-gnu/libibverbs.so.1")
            for lib in /usr/local/lib/libbnxt_re-rdmav*.so; do
                [[ -f "$lib" ]] && flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/libibverbs/$(basename "$lib")")
            done
            for lib in /usr/local/lib/libbnxt_re.so; do
                [[ -f "$lib" ]] && flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
            done
            [[ -d /etc/libibverbs.d ]] && flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            ;;
        ionic)
            local hib; hib=$(find_host_ibverbs)
            [[ -n "$hib" ]] && flags+=(-v "$hib:/lib/x86_64-linux-gnu/libibverbs.so.1")
            for dir in /usr/local/lib /usr/lib/x86_64-linux-gnu; do
                for lib in "$dir"/libionic*.so; do
                    if [[ -f "$lib" ]]; then
                        local real; real=$(readlink -f "$lib")
                        [[ -f "$real" ]] && flags+=(-v "$real:$real")
                        flags+=(-v "$lib:/usr/lib/x86_64-linux-gnu/$(basename "$lib")")
                    fi
                done
            done
            local pdir=/usr/lib/x86_64-linux-gnu/libibverbs
            if [[ -d "$pdir" ]]; then
                for lib in "$pdir"/libionic-rdmav*.so; do
                    [[ -f "$lib" ]] && flags+=(-v "$lib:$lib")
                done
            fi
            [[ -d /etc/libibverbs.d ]] && flags+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
            ;;
    esac
    echo "${flags[@]}"
}

# ── Node discovery ──────────────────────────────────────────────────

get_local_hostname() {
    hostname -f 2>/dev/null || hostname
}

get_local_ip() {
    hostname -I | awk '{print $1}'
}

check_node_free() {
    local node=$1
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$node" bash -l <<'CHECK_EOF' 2>/dev/null
if docker ps -q -f "name=atom-" 2>/dev/null | grep -q .; then
    echo "BUSY:docker"; exit 1
fi
USED=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l)
if [ "$USED" -gt 0 ]; then
    echo "BUSY:gpu"; exit 1
fi
echo "FREE"; exit 0
CHECK_EOF
}

pick_decode_nodes() {
    local my_hostname num_needed
    my_hostname=$(get_local_hostname)
    num_needed=${NUM_DECODE_NODES}
    local node_list=${PD_NODE_LIST:?"PD_NODE_LIST must be set (comma-separated hostnames)"}

    echo "========== Picking ${num_needed} decode node(s) =========="
    echo "Coordinator (prefill): ${my_hostname}"
    echo "Candidate nodes: ${node_list}"
    echo ""

    > "$DECODE_HOSTS_FILE"
    local found=0

    IFS=',' read -ra NODES <<< "$node_list"
    for node in "${NODES[@]}"; do
        node=$(echo "$node" | xargs)
        if [ "$node" = "$my_hostname" ]; then
            echo "  ${node}: SELF (prefill)"
            continue
        fi
        echo -n "  ${node}: "
        if result=$(check_node_free "$node"); then
            echo "FREE — selected as decode node #$((found + 1))"
            echo "$node" >> "$DECODE_HOSTS_FILE"
            found=$((found + 1))
            if [ "$found" -ge "$num_needed" ]; then
                break
            fi
        else
            echo "BUSY (${result})"
        fi
    done

    echo ""
    if [ "$found" -lt "$num_needed" ]; then
        echo "ERROR: Only found ${found}/${num_needed} free node(s) for decode."
        return 1
    fi
    echo "Selected ${found} decode node(s): $(paste -sd',' "$DECODE_HOSTS_FILE")"
    return 0
}

get_decode_hosts() {
    if [ -n "${DECODE_HOSTS:-}" ]; then
        echo "$DECODE_HOSTS" | tr ',' '\n'
        return
    fi
    if [ -f "$DECODE_HOSTS_FILE" ]; then
        cat "$DECODE_HOSTS_FILE"
        return
    fi
    echo ""
}

get_decode_hosts_csv() {
    get_decode_hosts | paste -sd',' -
}

get_remote_ip() {
    local node=$1
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$node" "hostname -I | awk '{print \$1}'" 2>/dev/null
}

# ── Launch functions ────────────────────────────────────────────────

launch_proxy() {
    local local_ip=$1
    echo "========== Launching Proxy on ${local_ip}:${PROXY_PORT} =========="
    PYTHONUNBUFFERED=1 python -m atom.kv_transfer.disaggregation.proxy \
        --port "$PROXY_PORT" > "$PROXY_LOG" 2>&1 &
    echo $! > /tmp/atom_proxy.pid
    echo "Proxy PID: $(cat /tmp/atom_proxy.pid)"
}

launch_producer() {
    local local_ip=$1
    echo "========== Launching Producer (prefill) on port ${PRODUCER_PORT} =========="
    echo "Model: ${MODEL_PATH}"
    echo "Args: ${PREFILL_ARGS} ${EXTRA_ARGS[*]:-}"

    rm -rf ~/.cache/atom/*

    local kv_config
    kv_config=$(cat <<KVJSON
{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "proxy_ip": "${local_ip}",
    "proxy_ping_port": ${DISCOVERY_PORT},
    "http_port": ${PRODUCER_PORT}
}
KVJSON
)

    ATOM_DISABLE_MMAP=true \
    NCCL_SOCKET_IFNAME=lo \
    AITER_LOG_LEVEL=WARNING \
    PYTHONUNBUFFERED=1 \
    python -m atom.entrypoints.openai_server \
        --model "$MODEL_PATH" \
        --server-port "$PRODUCER_PORT" \
        --kv-transfer-config "$kv_config" \
        $PREFILL_ARGS "${EXTRA_ARGS[@]}" > "$PRODUCER_LOG" 2>&1 &
    echo $! > /tmp/atom_producer.pid
    echo "Producer PID: $(cat /tmp/atom_producer.pid)"
}

launch_consumer_remote() {
    local decode_host=$1
    local local_ip=$2
    local consumer_port=$3
    local container_name=$4
    echo "========== Launching Consumer on ${decode_host}:${consumer_port} (${container_name}) =========="

    local kv_config
    kv_config=$(cat <<KVJSON
{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "proxy_ip": "${local_ip}",
    "proxy_ping_port": ${DISCOVERY_PORT},
    "http_port": ${consumer_port}
}
KVJSON
)

    # shellcheck disable=SC2029,SC2086
    ssh $SSH_OPTS "$decode_host" bash -l <<REMOTE_EOF
set -euo pipefail
echo "=== Decode node (\$(hostname)): starting ${container_name} ==="

containers=\$(docker ps -q -f name=${container_name})
if [ -n "\$containers" ]; then
    docker kill \$containers || true
    docker rm \$containers || true
fi

DEVICE_FLAG=\$(cat /etc/podinfo/gha-render-devices 2>/dev/null || echo "--device /dev/dri")
MODEL_MOUNT=""
[ -d "/models" ] && MODEL_MOUNT="-v /models:/models"
IB_FLAG=""
[ -e "/dev/infiniband" ] && IB_FLAG="--device=/dev/infiniband"

# NIC detection for RDMA userspace libs (ported from mori)
NIC_MOUNTS=""
NIC_TYPE="mlx5"
if [ -d /sys/class/infiniband ]; then
    for dev in /sys/class/infiniband/*; do
        name=\$(basename "\$dev")
        case "\$name" in bnxt_re*) NIC_TYPE="bnxt"; break ;; ionic*) NIC_TYPE="ionic"; break ;; esac
    done
fi
if [ "\$NIC_TYPE" = "bnxt" ]; then
    for c in /usr/lib64/libibverbs.so.1 /lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so.1; do
        resolved=\$(readlink -f "\$c" 2>/dev/null || true)
        if [ -f "\$resolved" ]; then NIC_MOUNTS="\$NIC_MOUNTS -v \$resolved:/lib/x86_64-linux-gnu/libibverbs.so.1"; break; fi
    done
    for lib in /usr/local/lib/libbnxt_re-rdmav*.so; do
        [ -f "\$lib" ] && NIC_MOUNTS="\$NIC_MOUNTS -v \$lib:/usr/lib/x86_64-linux-gnu/libibverbs/\$(basename \$lib)"
    done
    for lib in /usr/local/lib/libbnxt_re.so; do
        [ -f "\$lib" ] && NIC_MOUNTS="\$NIC_MOUNTS -v \$lib:/usr/lib/x86_64-linux-gnu/\$(basename \$lib)"
    done
    [ -d /etc/libibverbs.d ] && NIC_MOUNTS="\$NIC_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"
fi

docker run -dt --device=/dev/kfd \$DEVICE_FLAG \$IB_FLAG \
    \$NIC_MOUNTS \
    \$MODEL_MOUNT \
    -w /workspace --ipc=host --group-add video \
    --shm-size=16G --privileged --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e ATOM_DISABLE_MMAP=true \
    -e NCCL_SOCKET_IFNAME=lo \
    -e AITER_LOG_LEVEL=WARNING \
    --network=host \
    --name ${container_name} \
    ${ATOM_DOCKER_IMAGE:-rocm/atom-dev:latest}

if [ -d "/models" ]; then
    docker exec ${container_name} bash -lc \
        "hf download ${MODEL_PATH} --local-dir /models/${MODEL_PATH}" || true
fi

docker exec ${container_name} bash -lc "rm -rf ~/.cache/atom/*"

docker exec ${container_name} bash -lc \
    "pip install msgpack msgspec quart mooncake-transfer-engine"

docker exec -d ${container_name} bash -lc "
    ATOM_DISABLE_MMAP=true \\
    NCCL_SOCKET_IFNAME=lo \\
    AITER_LOG_LEVEL=WARNING \\
    PYTHONUNBUFFERED=1 \\
    python -m atom.entrypoints.openai_server \\
        --model ${MODEL_PATH} \\
        --server-port ${consumer_port} \\
        --kv-transfer-config '${kv_config}' \\
        ${DECODE_ARGS} \\
        > /tmp/atom_consumer.log 2>&1
"
echo "${container_name} started on \$(hostname)"
REMOTE_EOF
}

# ── Health check / warmup ───────────────────────────────────────────

wait_for_health() {
    local name=$1 url=$2 max_retries=${3:-45} interval=${4:-60}
    echo "Waiting for ${name} at ${url} ..."
    for ((i=1; i<=max_retries; i++)); do
        if curl -sf "$url" -o /dev/null --max-time 10; then
            echo "${name} is healthy after $((i * interval))s"
            return 0
        fi
        echo "  ${name} not ready ($i/$max_retries), retrying in ${interval}s..."
        sleep "$interval"
    done
    echo "ERROR: ${name} did not become healthy after $((max_retries * interval))s"
    return 1
}

wait_for_remote_health() {
    local decode_host=$1 name=$2 port=$3 max_retries=${4:-45} interval=${5:-60}
    echo "Waiting for ${name} on ${decode_host}:${port} ..."
    for ((i=1; i<=max_retries; i++)); do
        # shellcheck disable=SC2086
        if ssh $SSH_OPTS "$decode_host" \
            "curl -sf http://localhost:${port}/health -o /dev/null --max-time 10" 2>/dev/null; then
            echo "${name} is healthy after $((i * interval))s"
            return 0
        fi
        echo "  ${name} not ready ($i/$max_retries), retrying in ${interval}s..."
        sleep "$interval"
    done
    echo "ERROR: ${name} did not become healthy after $((max_retries * interval))s"
    return 1
}

warmup_endpoint() {
    local name=$1 url=$2 max_retries=${3:-10} interval=${4:-30}
    echo "Warming up ${name} at ${url} ..."
    for ((i=1; i<=max_retries; i++)); do
        if curl -sf "${url}/v1/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"'"$MODEL_PATH"'","prompt":"hi","max_tokens":1}' \
            -o /dev/null --max-time 120; then
            echo "${name} warmup completed"
            return 0
        fi
        echo "  Warmup attempt $i/$max_retries failed, retrying in ${interval}s..."
        sleep "$interval"
    done
    echo "ERROR: ${name} warmup failed after $((max_retries * interval))s"
    return 1
}

warmup_remote_endpoint() {
    local decode_host=$1 name=$2 port=$3 max_retries=${4:-10} interval=${5:-30}
    echo "Warming up ${name} on ${decode_host}:${port} ..."
    for ((i=1; i<=max_retries; i++)); do
        # shellcheck disable=SC2086
        if ssh $SSH_OPTS "$decode_host" \
            "curl -sf http://localhost:${port}/v1/completions \
                -H 'Content-Type: application/json' \
                -d '{\"model\":\"${MODEL_PATH}\",\"prompt\":\"hi\",\"max_tokens\":1}' \
                -o /dev/null --max-time 120" 2>/dev/null; then
            echo "${name} warmup completed"
            return 0
        fi
        echo "  Warmup attempt $i/$max_retries failed, retrying in ${interval}s..."
        sleep "$interval"
    done
    echo "ERROR: ${name} warmup failed after $((max_retries * interval))s"
    return 1
}

# ── Cleanup ─────────────────────────────────────────────────────────

stop_local() {
    echo "========== Stopping local processes =========="
    for pidfile in /tmp/atom_proxy.pid /tmp/atom_producer.pid; do
        if [ -f "$pidfile" ]; then
            kill "$(cat "$pidfile")" 2>/dev/null || true
            rm -f "$pidfile"
        fi
    done
    pkill -f 'atom.entrypoints' || true
    pkill -f 'atom.kv_transfer.disaggregation.proxy' || true
    sleep 2
    pkill -9 -f 'multiprocessing.spawn' || true
    pkill -9 -f 'multiprocessing.resource_tracker' || true

    echo "Waiting for local GPU memory to release..."
    for i in $(seq 1 60); do
        USED_GPUS=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk '{print $NF}' | awk '$1 > 0' | wc -l 2>/dev/null || echo "0")
        if [ "$USED_GPUS" -eq 0 ]; then
            echo "Local GPU memory released after ${i}s"
            break
        fi
        [ "$i" -eq 60 ] && echo "WARNING: Local GPU memory still in use after 60s"
        sleep 1
    done
}

stop_remote_all() {
    echo "========== Stopping all remote consumers =========="
    local idx=0
    while IFS= read -r decode_host; do
        [ -z "$decode_host" ] && continue
        local container_name="atom-pd-consumer-${idx}"
        echo "Stopping ${container_name} on ${decode_host} ..."
        # shellcheck disable=SC2086
        ssh $SSH_OPTS "$decode_host" bash -l <<REMOTE_EOF || true
docker stop ${container_name} 2>/dev/null || true
docker rm ${container_name} 2>/dev/null || true
echo "${container_name} stopped"
REMOTE_EOF
        idx=$((idx + 1))
    done < <(get_decode_hosts)
}

# ── Commands ────────────────────────────────────────────────────────

if [ "$TYPE" == "pick-nodes" ]; then
    pick_decode_nodes
    local_ip=$(get_local_ip)
    echo ""
    echo "========== Node Assignment =========="
    echo "  Prefill: $(get_local_hostname) (${local_ip})"
    idx=0
    while IFS= read -r dh; do
        [ -z "$dh" ] && continue
        dip=$(get_remote_ip "$dh")
        echo "  Decode #${idx}: ${dh} (${dip})"
        idx=$((idx + 1))
    done < <(get_decode_hosts)
    echo "DECODE_HOSTS=$(get_decode_hosts_csv)" >> "${GITHUB_OUTPUT:-/dev/null}"
    echo "LOCAL_IP=${local_ip}" >> "${GITHUB_OUTPUT:-/dev/null}"
fi


if [ "$TYPE" == "launch-all" ]; then
    if [ -z "${DECODE_HOSTS:-}" ] && [ ! -f "$DECODE_HOSTS_FILE" ]; then
        pick_decode_nodes
    fi
    local_ip=$(get_local_ip)

    echo ""
    echo "========== PD Disaggregation: Launching All Components =========="
    echo "Prefill:      $(get_local_hostname) (${local_ip})"
    echo "Decode nodes: $(get_decode_hosts_csv)"
    echo "Model:        ${MODEL_PATH}"
    echo ""

    launch_proxy "$local_ip"
    sleep 2

    # Launch consumer on each decode node (different port per node)
    idx=0
    while IFS= read -r decode_host; do
        [ -z "$decode_host" ] && continue
        consumer_port=$((CONSUMER_BASE_PORT + idx))
        container_name="atom-pd-consumer-${idx}"
        launch_consumer_remote "$decode_host" "$local_ip" "$consumer_port" "$container_name"
        idx=$((idx + 1))
    done < <(get_decode_hosts)

    launch_producer "$local_ip"

    # Wait for all to be healthy
    wait_for_health "Producer" "http://localhost:${PRODUCER_PORT}/health"
    idx=0
    while IFS= read -r decode_host; do
        [ -z "$decode_host" ] && continue
        consumer_port=$((CONSUMER_BASE_PORT + idx))
        wait_for_remote_health "$decode_host" "Consumer-${idx}" "$consumer_port"
        idx=$((idx + 1))
    done < <(get_decode_hosts)

    # Warmup all
    warmup_endpoint "Producer" "http://localhost:${PRODUCER_PORT}"
    idx=0
    while IFS= read -r decode_host; do
        [ -z "$decode_host" ] && continue
        consumer_port=$((CONSUMER_BASE_PORT + idx))
        warmup_remote_endpoint "$decode_host" "Consumer-${idx}" "$consumer_port"
        idx=$((idx + 1))
    done < <(get_decode_hosts)

    echo ""
    echo "========== All PD components are ready =========="
    echo "  Proxy:    localhost:${PROXY_PORT}"
    echo "  Producer: localhost:${PRODUCER_PORT} ($(get_local_hostname))"
    idx=0
    while IFS= read -r decode_host; do
        [ -z "$decode_host" ] && continue
        echo "  Consumer-${idx}: ${decode_host}:$((CONSUMER_BASE_PORT + idx))"
        idx=$((idx + 1))
    done < <(get_decode_hosts)
fi


if [ "$TYPE" == "benchmark" ]; then
    echo ""
    echo "========== Running PD Benchmark =========="
    ATOM_CLIENT_LOG="${ATOM_CLIENT_LOG:-/tmp/atom_client.log}"
    RESULT_FILENAME=${RESULT_FILENAME:-pd_benchmark_result}

    python -m atom.benchmarks.benchmark_serving \
        --model="$MODEL_PATH" --backend=vllm --base-url="http://localhost:${PROXY_PORT}" \
        --dataset-name=random \
        --random-input-len="$ISL" --random-output-len="$OSL" --random-range-ratio="$RANDOM_RANGE_RATIO" \
        --max-concurrency="$CONC" \
        --num-prompts="${NUM_PROMPTS_OVERRIDE:-$(( CONC * 10 ))}" \
        --trust-remote-code \
        --num-warmups="$(( CONC * 2 ))" \
        --request-rate=inf --ignore-eos \
        --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
        --result-dir=. --result-filename="${RESULT_FILENAME}.json" \
        ${BENCH_EXTRA_ARGS:-} \
        2>&1 | tee "$ATOM_CLIENT_LOG"

    if [ -f "${RESULT_FILENAME}.json" ]; then
        RESULT_PATH="${RESULT_FILENAME}.json" python3 - <<'PY'
import json
import os
import re

result_path = os.environ["RESULT_PATH"]
with open(result_path, encoding="utf-8") as f:
    d = json.load(f)

d["random_input_len"] = int(os.environ.get("ISL", 0))
d["random_output_len"] = int(os.environ.get("OSL", 0))
d["benchmark_backend"] = "ATOM-PD"
d["pd_mode"] = True

prefill_args = os.environ.get("PREFILL_ARGS", "")
decode_args = os.environ.get("DECODE_ARGS", "")

tp_match = re.search(r"(?:^|\s)-tp\s+(\d+)", prefill_args)
d["prefill_tp"] = int(tp_match.group(1)) if tp_match else 1

tp_match = re.search(r"(?:^|\s)-tp\s+(\d+)", decode_args)
decode_tp = int(tp_match.group(1)) if tp_match else 1
d["decode_tp"] = decode_tp
d["tensor_parallel_size"] = decode_tp

num_decode = int(os.environ.get("NUM_DECODE_NODES", "1"))
d["num_decode_nodes"] = num_decode
d["total_gpus"] = d["prefill_tp"] + decode_tp * num_decode
d["data_parallel_size"] = 1
d["enable_dp_attention"] = False

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2)
PY
    fi
fi


if [ "$TYPE" == "stop-all" ]; then
    echo ""
    echo "========== PD Disaggregation: Stopping All Components =========="
    stop_local
    stop_remote_all
    rm -f "$DECODE_HOSTS_FILE"
    echo "========== All PD components stopped =========="
fi


if [ "$TYPE" == "dump-logs" ]; then
    echo "========== Proxy Log =========="
    cat "$PROXY_LOG" 2>/dev/null || echo "(no proxy log)"
    echo ""
    echo "========== Producer Log =========="
    cat "$PRODUCER_LOG" 2>/dev/null || echo "(no producer log)"
    idx=0
    while IFS= read -r decode_host; do
        [ -z "$decode_host" ] && continue
        container_name="atom-pd-consumer-${idx}"
        echo ""
        echo "========== Consumer-${idx} Log (${decode_host}) =========="
        # shellcheck disable=SC2086
        ssh $SSH_OPTS "$decode_host" \
            "docker exec ${container_name} cat /tmp/atom_consumer.log 2>/dev/null" || echo "(no log)"
        idx=$((idx + 1))
    done < <(get_decode_hosts)
fi
