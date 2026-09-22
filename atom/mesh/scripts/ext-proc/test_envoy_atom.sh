#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Start Envoy, Atomesh and ATOM Engine on a single host for manual testing.
Containers remain running after startup; the script prints the cleanup command.

  MODEL_PATH=/models/Qwen3-0.6B \
    bash atom/mesh/scripts/ext-proc/test_envoy_atom.sh

Requires a local model directory, Docker, Python 3, curl and two AMD GPUs.
Both Atomesh and Engine use ATOM_IMAGE, which must include the current ext-proc code.
Existing images are used directly; this script does not build or compile them.

Options:
  --dry-run  Render Envoy config and print container commands without starting them.
  --help     Show this help.

Environment:
  ATOM_IMAGE          ATOM image containing /usr/local/bin/atomesh (default: atom-extproc:test).
  ENVOY_IMAGE         Envoy image (default: envoyproxy/envoy:v1.37.0).
  MODEL_PATH          Required local model directory; recommended Qwen3-0.6B.
  GPU_DEVICES         ROCr device indices/UUIDs (default: 0,1).
  DP_SIZE             Data parallel replicas, TP is fixed at 1 (default: 2).
  GPU_MEMORY_UTIL     Engine GPU memory fraction (default: 0.85).
  ATOM_DP_LM_HEAD_MODE Engine LM head mode (default: default, replicated per DP rank).
                      Set all2all/allgather only with a compatible Engine image.
  ENGINE_PORT         Engine HTTP port (default: 10010).
  ENGINE_INTERNAL_PORT Engine internal port (default: 10011).
  DP_MASTER_PORT      DP rendezvous port (default: 10012).
  MESH_PORT           Mesh HTTP management port (default: 10013).
  EXT_PROC_PORT       Mesh ext-proc gRPC port (default: 10014).
  METRICS_PORT        Mesh Prometheus port (default: 10015).
  ENVOY_PORT          Client-facing HTTP port (default: 10016).
  WAIT_TIMEOUT        Service startup deadline, seconds (default: 900).
  REQUEST_TIMEOUT     Inference wait budget for Mesh/Envoy, seconds (default: 900).
                      Configure the timeout of your test client separately.
EOF
}

dry_run=0
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=1; shift ;;
    '') ;;
    *) usage >&2; exit 2 ;;
esac
[[ $# == 0 ]] || { usage >&2; exit 2; }
ATOM_IMAGE=${ATOM_IMAGE:-atom-extproc:test}
ENVOY_IMAGE=${ENVOY_IMAGE:-envoyproxy/envoy:v1.37.0}
: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-0.6B model directory}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GPU_DEVICES=${GPU_DEVICES:-0,1}
DP_SIZE=${DP_SIZE:-2}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}
ATOM_DP_LM_HEAD_MODE=${ATOM_DP_LM_HEAD_MODE:-default}
ENGINE_PORT=${ENGINE_PORT:-10010}
ENGINE_INTERNAL_PORT=${ENGINE_INTERNAL_PORT:-10011}
DP_MASTER_PORT=${DP_MASTER_PORT:-10012}
MESH_PORT=${MESH_PORT:-10013}
EXT_PROC_PORT=${EXT_PROC_PORT:-10014}
METRICS_PORT=${METRICS_PORT:-10015}
ENVOY_PORT=${ENVOY_PORT:-10016}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-900}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-900}
export DP_SIZE ENGINE_PORT ENGINE_INTERNAL_PORT DP_MASTER_PORT GPU_MEMORY_UTIL GPU_DEVICES
export MESH_PORT EXT_PROC_PORT METRICS_PORT ENVOY_PORT REQUEST_TIMEOUT

python3 - "$dry_run" <<'PY'
import os
import socket
import sys
ports = {key: int(os.environ[key]) for key in (
    "ENGINE_PORT", "ENGINE_INTERNAL_PORT", "DP_MASTER_PORT", "MESH_PORT",
    "EXT_PROC_PORT", "METRICS_PORT", "ENVOY_PORT",
)}
if len(set(ports.values())) != len(ports) or not all(1 <= p <= 65535 for p in ports.values()):
    raise SystemExit("Ports must be distinct integers in 1..65535")
if int(os.environ["DP_SIZE"]) < 1:
    raise SystemExit("DP_SIZE must be positive")
devices = os.environ["GPU_DEVICES"].split(",")
if len(set(devices)) != len(devices) or len(devices) < int(os.environ["DP_SIZE"]):
    raise SystemExit("GPU_DEVICES must contain at least DP_SIZE distinct devices")
if not 0 < float(os.environ["GPU_MEMORY_UTIL"]) < 1:
    raise SystemExit("GPU_MEMORY_UTIL must be between 0 and 1")
if sys.argv[1] == "0":
    listeners = []
    errors = []
    try:
        for name, port in ports.items():
            listener = socket.socket()
            listeners.append(listener)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as error:
                errors.append(f"  {name}={port}: cannot bind 127.0.0.1:{port}: {error}")
    finally:
        for listener in listeners:
            listener.close()
    if errors:
        raise SystemExit(
            "Port preflight failed; no containers were started:\n"
            + "\n".join(errors)
            + "\nInspect listeners with: ss -ltnp"
            + "\nSet the listed environment variables to unused ports and rerun."
        )
PY
[[ $WAIT_TIMEOUT =~ ^[1-9][0-9]*$ && $REQUEST_TIMEOUT =~ ^[1-9][0-9]*$ ]] || {
    echo 'WAIT_TIMEOUT and REQUEST_TIMEOUT must be positive integers' >&2; exit 2;
}
# Leave room for clients using REQUEST_TIMEOUT as their inference timeout.
proxy_idle_timeout=$((REQUEST_TIMEOUT + 30))
(( proxy_idle_timeout >= 300 )) || proxy_idle_timeout=300
proxy_route_timeout=$proxy_idle_timeout
(( proxy_route_timeout >= 1800 )) || proxy_route_timeout=1800
MODEL_PATH=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$MODEL_PATH")
if (( ! dry_run )); then
    for command in docker curl; do command -v "$command" >/dev/null; done
    [[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing $MODEL_PATH/config.json" >&2; exit 2; }
    [[ -e /dev/kfd && -d /dev/dri ]] || { echo 'ROCm devices /dev/kfd and /dev/dri are required' >&2; exit 2; }
fi

run_dir=$(mktemp -d "${TMPDIR:-/tmp}/atomesh-envoy-smoke.XXXXXX")
run_name="atomesh-smoke-$(basename "$run_dir" | cut -d. -f2)"
echo "Config and startup logs: $run_dir"
echo "Engine progress: docker logs -f $run_name-engine"
containers=()
cleanup() {
    local status=$?
    trap - EXIT
    for container in "${containers[@]}"; do
        docker logs "$container" >"$run_dir/$container.log" 2>&1 || true
    done
    if [[ $status == 0 && ${#containers[@]} -gt 0 ]]; then
        printf 'Containers retained. Remove with: docker rm -f'
        printf ' %q' "${containers[@]}"
        printf '\n'
    else
        for container in "${containers[@]}"; do docker rm -f "$container" >/dev/null 2>&1 || true; done
    fi
    echo "Config and startup logs: $run_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sed -e "s/port_value: 8080/port_value: $ENVOY_PORT/" \
    -e "s/port_value: 9002/port_value: $EXT_PROC_PORT/" \
    -e "s/stream_idle_timeout: 300s/stream_idle_timeout: ${proxy_idle_timeout}s/" \
    -e "s/timeout: 1800s/timeout: ${proxy_route_timeout}s/" \
    "$script_dir/../../tests/fixtures/ext-proc/envoy.yaml" >"$run_dir/envoy.yaml"
chmod 755 "$run_dir"
chmod 644 "$run_dir/envoy.yaml"

start_container() {
    local name=$1
    shift
    if (( dry_run )); then
        printf 'docker run -d --name %q' "$name"
        printf ' %q' "$@"
        printf '\n'
    else
        docker run -d --name "$name" "$@" >"$run_dir/$name.cid"
        containers+=("$name")
    fi
}

wait_http() {
    local name=$1 url=$2 expected_status=${3:-200} deadline=$((SECONDS + WAIT_TIMEOUT))
    (( dry_run )) && return 0
    until [[ $(curl --noproxy '*' -s --max-time 2 -o /dev/null -w '%{http_code}' "$url") == "$expected_status" ]]; do
        if [[ $(docker inspect --format '{{.State.Running}}' "$name") != true ]]; then
            docker logs --tail 60 "$name" >&2
            echo "$name exited before readiness" >&2; return 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for $url" >&2; return 1
        fi
        sleep 2
    done
}

if (( ! dry_run )); then
    # Fail before loading the model when the image predates ext-proc support.
    docker run --rm --entrypoint /usr/local/bin/atomesh "$ATOM_IMAGE" launch --help \
        >"$run_dir/atomesh-help.txt"
    grep -q -- '--ext-proc' "$run_dir/atomesh-help.txt" || {
        echo "Image $ATOM_IMAGE lacks ext-proc support; set ATOM_IMAGE to an image that supports --ext-proc" >&2; exit 1;
    }
fi

start_container "$run_name-engine" \
    --network host --shm-size 2g \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit nofile=65536:65536 \
    --env "ROCR_VISIBLE_DEVICES=$GPU_DEVICES" --env USE_ATOMESH_ENTRYPOINTS=0 \
    --env "ATOM_DP_LM_HEAD_MODE=$ATOM_DP_LM_HEAD_MODE" \
    --mount "type=bind,src=$MODEL_PATH,dst=/model,readonly" \
    --entrypoint python "$ATOM_IMAGE" -m atom.entrypoints.openai_server \
    --model /model --served-model-name smoke-model --host 127.0.0.1 \
    --server-port "$ENGINE_PORT" --port "$ENGINE_INTERNAL_PORT" \
    --tensor-parallel-size 1 --data-parallel-size "$DP_SIZE" \
    --data-parallel-master-port "$DP_MASTER_PORT" \
    --enforce-eager --kv_cache_dtype bf16 --max-model-len 1024 \
    --max-num-batched-tokens 1024 --max-num-seqs 4 \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
wait_http "$run_name-engine" "http://127.0.0.1:$ENGINE_PORT/health"

start_container "$run_name-mesh" \
    --network host --entrypoint /usr/local/bin/atomesh "$ATOM_IMAGE" launch \
    --host 127.0.0.1 --port "$MESH_PORT" --prometheus-port "$METRICS_PORT" \
    --backend atom --worker-urls "http://127.0.0.1:$ENGINE_PORT" \
    --dp-aware --policy round_robin \
    --ext-proc --ext-proc-listen "127.0.0.1:$EXT_PROC_PORT" \
    --ext-proc-idle-timeout-secs "$proxy_idle_timeout"
wait_http "$run_name-mesh" "http://127.0.0.1:$MESH_PORT/readiness"

start_container "$run_name-envoy" \
    --network host --user 0 \
    --mount "type=bind,src=$run_dir/envoy.yaml,dst=/etc/envoy/envoy.yaml,readonly" \
    --entrypoint envoy "$ENVOY_IMAGE" \
    -c /etc/envoy/envoy.yaml --disable-hot-restart --concurrency 2 --log-level warn
# Readiness only: GET returns 405 without running model inference.
wait_http "$run_name-envoy" "http://127.0.0.1:$ENVOY_PORT/v1/completions" 405
if (( ! dry_run )); then
    echo "Services ready for manual testing: http://127.0.0.1:$ENVOY_PORT"
fi
echo "Manual test commands: $script_dir/README.md"
