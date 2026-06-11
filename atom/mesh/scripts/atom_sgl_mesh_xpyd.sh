#!/usr/bin/env bash
set -euo pipefail

# Launch Atomesh SGLang PD disaggregation on a single node.
#
# Required env:
#   MODEL_PATH      - model path
#
# Optional topology env:
#   NODE_IP=127.0.0.1
#   PREFILL_GPUS=0,1,2,3        DECODE_GPUS=4,5,6,7
#   PREFILL_TP=4                DECODE_TP=4
#   PREFILL_PORT=8010           DECODE_PORT=8020
#   PREFILL_PORTS=<auto>        DECODE_PORTS=<auto>
#   BOOTSTRAP_PORT=8998
#   PREFILL_BOOTSTRAP_PORTS=<auto from BOOTSTRAP_PORT>
#   DECODE_BOOTSTRAP_PORTS=<auto from DECODE_BOOTSTRAP_BASE>
#   DECODE_BOOTSTRAP_BASE=9098
#
# Optional serving env:
#   ROUTER_PORT=8000  POLICY=random  MESH_BIN=/usr/local/bin/atomesh
#   MEM_FRACTION=0.85  KV_CACHE_DTYPE=fp8_e4m3
#   MAX_RUNNING_REQUESTS=128
#   CUDA_GRAPH_BS_START=1  CUDA_GRAPH_BS_END=64
#   IB_DEVICE=<auto from /sys/class/infiniband>  WAIT_TIMEOUT=900
#   LOG_DIR=/workspace/logs
#   CLEANUP_EXISTING=0

: "${MODEL_PATH:?}"

detect_ib_devices() {
    local devices=()

    if [[ -d /sys/class/infiniband ]]; then
        local device
        for device in /sys/class/infiniband/*; do
            [[ -e "${device}" ]] || continue
            devices+=("$(basename "${device}")")
        done
    fi

    if ((${#devices[@]} > 0)); then
        local IFS=,
        echo "${devices[*]}"
    fi
}

NODE_IP="${NODE_IP:-${PREFILL_IP:-${DECODE_IP:-127.0.0.1}}}"
PREFILL_GPUS="${PREFILL_GPUS:-4,5}"
DECODE_GPUS="${DECODE_GPUS:-6,7}"
PREFILL_TP="${PREFILL_TP:-${TP:-2}}"
DECODE_TP="${DECODE_TP:-${TP:-2}}"

PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
DECODE_BOOTSTRAP_BASE="${DECODE_BOOTSTRAP_BASE:-9098}"

MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
CUDA_GRAPH_BS_START="${CUDA_GRAPH_BS_START:-1}"
CUDA_GRAPH_BS_END="${CUDA_GRAPH_BS_END:-64}"
IB_DEVICE="${IB_DEVICE:-$(detect_ib_devices)}"
POLICY="${POLICY:-random}"
MESH_BIN="${MESH_BIN:-/usr/local/bin/atomesh}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
CLEANUP_EXISTING="${CLEANUP_EXISTING:-0}"

PIDS=()

strip_spaces() {
    echo "${1//[[:space:]]/}"
}

csv_to_array() {
    local csv
    csv="$(strip_spaces "$2")"
    local -n out="$1"
    IFS=',' read -ra out <<<"${csv}"
}

generate_ports() {
    local base="$1"
    local count="$2"
    local ports=()

    for i in $(seq 0 $((count - 1))); do
        ports+=("$((base + i))")
    done

    local IFS=,
    echo "${ports[*]}"
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

wait_for_server() {
    local name="$1"
    local port="$2"
    local pid="$3"
    local log_file="$4"
    local timeout_seconds="$5"
    local start_time
    start_time="$(date +%s)"

    echo "[${name}] waiting on port ${port} (timeout=${timeout_seconds}s)..."
    while true; do
        if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            echo "[${name}] ready on port ${port}"
            return 0
        fi

        if ! kill -0 "${pid}" >/dev/null 2>&1; then
            echo "[${name}] process ${pid} exited before port ${port} became ready"
            if [[ -f "${log_file}" ]]; then
                echo "[${name}] last 80 log lines from ${log_file}:"
                tail -n 80 "${log_file}" || true
            fi
            return 1
        fi

        local now
        now="$(date +%s)"
        if ((now - start_time >= timeout_seconds)); then
            echo "[${name}] timeout waiting for port ${port}"
            if [[ -f "${log_file}" ]]; then
                echo "[${name}] last 80 log lines from ${log_file}:"
                tail -n 80 "${log_file}" || true
            fi
            return 1
        fi

        sleep 5
    done
}

validate_port_list() {
    local name="$1"
    local expected="$2"
    shift 2
    local values=("$@")

    if ((${#values[@]} < expected)); then
        echo "Error: ${name} needs ${expected} entries, got ${#values[@]}"
        exit 1
    fi
}

cleanup() {
    if ((${#PIDS[@]} > 0)); then
        echo "[cleanup] stopping ${#PIDS[@]} child process(es)"
        kill "${PIDS[@]}" >/dev/null 2>&1 || true
    fi
}

launch_prefill() {
    local idx="$1"
    local gpu_list="$2"
    local port="$3"
    local bootstrap_port="$4"
    local ib_args=()

    echo "[prefill:${idx}] GPUs=${gpu_list} TP=${PREFILL_TP} port=${port} bootstrap=${bootstrap_port}"

    mkdir -p "${LOG_DIR}"

    if [[ -n "${IB_DEVICE}" ]]; then
        ib_args+=(--disaggregation-ib-device "${IB_DEVICE}")
    fi

    HIP_VISIBLE_DEVICES="${gpu_list}" \
    SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
    ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0 \
    SGLANG_HOST_IP="${NODE_IP}" \
    SGLANG_MOONCAKE_SEND_AUX_TCP=1 \
    MC_TCP_ENABLE_CONNECTION_POOL=true \
    LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}" \
    python3 -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 0.0.0.0 --port "${port}" \
        --trust-remote-code \
        --tp-size "${PREFILL_TP}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        --mem-fraction-static "${MEM_FRACTION}" \
        --page-size 1 \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --disable-radix-cache \
        --log-level info \
        --watchdog-timeout 3600 \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port "${bootstrap_port}" \
        "${ib_args[@]}" \
        >"${LOG_DIR}/prefill_${idx}.log" 2>&1 &
    PIDS+=("$!")
}

launch_decode() {
    local idx="$1"
    local gpu_list="$2"
    local port="$3"
    local bootstrap_port="$4"
    local ib_args=()

    echo "[decode:${idx}] GPUs=${gpu_list} TP=${DECODE_TP} port=${port} bootstrap=${bootstrap_port}"

    if [[ -n "${IB_DEVICE}" ]]; then
        ib_args+=(--disaggregation-ib-device "${IB_DEVICE}")
    fi

    HIP_VISIBLE_DEVICES="${gpu_list}" \
    SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
    ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0 \
    SGLANG_HOST_IP="${NODE_IP}" \
    SGLANG_MOONCAKE_SEND_AUX_TCP=1 \
    MC_TCP_ENABLE_CONNECTION_POOL=true \
    LD_LIBRARY_PATH="$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}" \
    TORCHINDUCTOR_COMPILE_THREADS=128 \
    python3 -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 0.0.0.0 --port "${port}" \
        --trust-remote-code \
        --tp-size "${DECODE_TP}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        --mem-fraction-static "${MEM_FRACTION}" \
        --page-size 1 \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --cuda-graph-bs $(seq "${CUDA_GRAPH_BS_START}" "${CUDA_GRAPH_BS_END}") \
        --disable-radix-cache \
        --log-level info \
        --watchdog-timeout 3600 \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-bootstrap-port "${bootstrap_port}" \
        "${ib_args[@]}" \
        >"${LOG_DIR}/decode_${idx}.log" 2>&1 &
    PIDS+=("$!")
}

launch_router() {
    echo "[router] port=${ROUTER_PORT} policy=${POLICY}"

    if curl -sf "http://127.0.0.1:${ROUTER_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[router] already running"
        return 0
    fi

    "${MESH_BIN}" launch \
        --host 0.0.0.0 --port "${ROUTER_PORT}" \
        --pd-disaggregation \
        "$@" \
        --policy "${POLICY}" \
        --backend sglang \
        --log-dir "${LOG_DIR}" \
        --log-level info \
        --disable-health-check \
        --prometheus-port 29100 \
        >"${LOG_DIR}/router.log" 2>&1 &
    PIDS+=("$!")
}

main() {
    local prefill_gpus=()
    local decode_gpus=()
    local prefill_ports=()
    local decode_ports=()
    local prefill_bootstrap_ports=()
    local decode_bootstrap_ports=()
    local prefill_pids=()
    local decode_pids=()

    csv_to_array prefill_gpus "${PREFILL_GPUS}"
    csv_to_array decode_gpus "${DECODE_GPUS}"

    if ((${#prefill_gpus[@]} % PREFILL_TP != 0)); then
        echo "Error: PREFILL_GPUS count (${#prefill_gpus[@]}) must be divisible by PREFILL_TP (${PREFILL_TP})"
        exit 1
    fi
    if ((${#decode_gpus[@]} % DECODE_TP != 0)); then
        echo "Error: DECODE_GPUS count (${#decode_gpus[@]}) must be divisible by DECODE_TP (${DECODE_TP})"
        exit 1
    fi

    local num_prefill_instances=$(( ${#prefill_gpus[@]} / PREFILL_TP ))
    local num_decode_instances=$(( ${#decode_gpus[@]} / DECODE_TP ))

    PREFILL_PORTS="${PREFILL_PORTS:-$(generate_ports "${PREFILL_PORT}" "${num_prefill_instances}")}"
    DECODE_PORTS="${DECODE_PORTS:-$(generate_ports "${DECODE_PORT}" "${num_decode_instances}")}"
    PREFILL_BOOTSTRAP_PORTS="${PREFILL_BOOTSTRAP_PORTS:-$(generate_ports "${BOOTSTRAP_PORT}" "${num_prefill_instances}")}"
    DECODE_BOOTSTRAP_PORTS="${DECODE_BOOTSTRAP_PORTS:-$(generate_ports "${DECODE_BOOTSTRAP_BASE}" "${num_decode_instances}")}"

    csv_to_array prefill_ports "${PREFILL_PORTS}"
    csv_to_array decode_ports "${DECODE_PORTS}"
    csv_to_array prefill_bootstrap_ports "${PREFILL_BOOTSTRAP_PORTS}"
    csv_to_array decode_bootstrap_ports "${DECODE_BOOTSTRAP_PORTS}"

    validate_port_list PREFILL_PORTS "${num_prefill_instances}" "${prefill_ports[@]}"
    validate_port_list DECODE_PORTS "${num_decode_instances}" "${decode_ports[@]}"
    validate_port_list PREFILL_BOOTSTRAP_PORTS "${num_prefill_instances}" "${prefill_bootstrap_ports[@]}"
    validate_port_list DECODE_BOOTSTRAP_PORTS "${num_decode_instances}" "${decode_bootstrap_ports[@]}"

    echo "ATOM SGLang Mesh xPyD single-node topology:"
    echo "  Node IP: ${NODE_IP}"
    echo "  Model: ${MODEL_PATH}"
    echo "  Prefill: ${num_prefill_instances} instance(s), TP=${PREFILL_TP}, GPUs=${PREFILL_GPUS}, ports=${PREFILL_PORTS}, bootstrap=${PREFILL_BOOTSTRAP_PORTS}"
    echo "  Decode:  ${num_decode_instances} instance(s), TP=${DECODE_TP}, GPUs=${DECODE_GPUS}, ports=${DECODE_PORTS}, bootstrap=${DECODE_BOOTSTRAP_PORTS}"
    echo "  IB devices: ${IB_DEVICE:-<not specified>}"
    echo "  Router:  port=${ROUTER_PORT}, policy=${POLICY}"

    mkdir -p "${LOG_DIR}"

    if [[ "${CLEANUP_EXISTING}" == "1" ]]; then
        echo "[cleanup] stopping existing sglang and atomesh processes"
        pkill -f 'python3 -m sglang.launch_server' || true
        pkill -f "${MESH_BIN} launch" || true
        sleep 3
    fi

    local router_args=()
    for i in $(seq 0 $((num_prefill_instances - 1))); do
        local start_idx=$((i * PREFILL_TP))
        local gpu_ids=()
        for j in $(seq 0 $((PREFILL_TP - 1))); do
            gpu_ids+=("${prefill_gpus[$((start_idx + j))]}")
        done

        local gpu_list
        gpu_list="$(join_by_comma "${gpu_ids[@]}")"
        launch_prefill "$((i + 1))" "${gpu_list}" "${prefill_ports[$i]}" "${prefill_bootstrap_ports[$i]}"
        prefill_pids+=("${PIDS[-1]}")
        router_args+=(--prefill "http://${NODE_IP}:${prefill_ports[$i]}" "${prefill_bootstrap_ports[$i]}")
    done

    for i in $(seq 0 $((num_decode_instances - 1))); do
        local start_idx=$((i * DECODE_TP))
        local gpu_ids=()
        for j in $(seq 0 $((DECODE_TP - 1))); do
            gpu_ids+=("${decode_gpus[$((start_idx + j))]}")
        done

        local gpu_list
        gpu_list="$(join_by_comma "${gpu_ids[@]}")"
        launch_decode "$((i + 1))" "${gpu_list}" "${decode_ports[$i]}" "${decode_bootstrap_ports[$i]}"
        decode_pids+=("${PIDS[-1]}")
        router_args+=(--decode "http://${NODE_IP}:${decode_ports[$i]}")
    done

    for i in $(seq 0 $((num_prefill_instances - 1))); do
        wait_for_server "prefill:$((i + 1))" "${prefill_ports[$i]}" "${prefill_pids[$i]}" "${LOG_DIR}/prefill_$((i + 1)).log" "${WAIT_TIMEOUT}"
    done
    for i in $(seq 0 $((num_decode_instances - 1))); do
        wait_for_server "decode:$((i + 1))" "${decode_ports[$i]}" "${decode_pids[$i]}" "${LOG_DIR}/decode_$((i + 1)).log" "${WAIT_TIMEOUT}"
    done

    launch_router "${router_args[@]}"
    wait_for_server router "${ROUTER_PORT}" "${PIDS[-1]}" "${LOG_DIR}/router.log" "${WAIT_TIMEOUT}"

    echo "ATOM SGLang Mesh xPyD is ready: http://127.0.0.1:${ROUTER_PORT}"
    echo "Logs: ${LOG_DIR}"
    wait
}

trap cleanup INT TERM
main "$@"
