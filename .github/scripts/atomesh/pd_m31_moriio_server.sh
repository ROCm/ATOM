# shellcheck shell=bash
# MiniMax-M3.1-Flash vLLM 1P1D/1P2D/2P2D over MoRIIOConnector (WRITE, or READ with
# ATOMESH_M31_MORIIO_READ_MODE=true) RDMA, cross-node or within one node (single_node), sourced by
# pd_server_vllm.sh when ATOMESH_M31_MORIIO=1. Server env/argv follow the local
# test_scripts/pd/launch_pd_main.sh CONC --offload; only PYTHONPATH, hosts, MoRIIO ports, the rdma
# backend, the offload size and the per-D max-num-seqs are adapted to the two-node layout. Workers
# register with the rank-0 vllm-router sidecar (pd_slurm_job.sh) through MoRIIO discovery.

M31_ROOT="/tmp/atomesh-m31-${SLURM_JOB_ID:-local}-${NODE_RANK}"
M31_HANDSHAKE_PORT=$((6301 + ATOMESH_SERVICE_PORT_OFFSET))
M31_NOTIFY_PORT=$((61005 + ATOMESH_SERVICE_PORT_OFFSET))

m31_conc="${CONC_LIST//[[:space:]]/}"
[[ "${m31_conc}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: M3.1 MoRIIO cases take exactly one concurrency, got '${CONC_LIST}'" >&2
  exit 2
}

M31_ENV=(
  VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
  VLLM_USE_V2_MODEL_RUNNER=1 DEBUG_HIP_DYNAMIC_QUEUES=0 PYTHONNOUSERSITE=1
  AITER_LOG_LEVEL=WARNING PYTHONPATH="${M31_ROOT}/site" PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
  VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
  HIP_FORCE_DEV_KERNARG=1 NCCL_DMABUF_ENABLE=0 HSA_ENABLE_IPC_MODE_LEGACY=1
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600
)
# KEY=VAL words for every server, e.g. an image's opt-in ROCm switches.
read -r -a m31_extra_env <<< "${ATOMESH_M31_EXTRA_ENV:-}"
M31_ENV+=("${m31_extra_env[@]}")
M31_VLLM_ARGS=(
  vllm serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}" --host 0.0.0.0
  --trust-remote-code --language-model-only --disable-uvicorn-access-log
  --moe-backend aiter --block-size 128 --disable-hybrid-kv-cache-manager --enable-prefix-caching
  --kv-cache-dtype nvfp4 --attention-config '{"indexer_kv_dtype":"nvfp4","indexer_decode_cp":true}'
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":4.04}'
  --max-model-len 1048576
  --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice
)
# vllm serve words for every server, e.g. an image's serving switches.
read -r -a m31_extra_args <<< "${ATOMESH_M31_EXTRA_ARGS:-}"
M31_VLLM_ARGS+=("${m31_extra_args[@]}")

quote() {
  local a out=()
  for a; do [[ $a =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]] && out+=("$a") || out+=("'$a'"); done
  echo "${out[*]}"
}

# The image's vllm package with the fork's diff against the image commit applied (M/A copied, D removed).
m31_overlay() {
  local src="${M31_ROOT}/vllm-src" image_vllm
  rm -rf "${M31_ROOT}" && mkdir -p "${M31_ROOT}/site" || return 2
  git init -q "${src}" &&
    git -C "${src}" fetch -q --depth 1 "${ATOMESH_M31_VLLM_REPO}" "${ATOMESH_M31_VLLM_SHA}" "${ATOMESH_M31_VLLM_BASE_SHA}" &&
    git -C "${src}" checkout -q --detach "${ATOMESH_M31_VLLM_SHA}" &&
    git -C "${src}" diff --name-status --no-renames "${ATOMESH_M31_VLLM_BASE_SHA}" HEAD -- vllm \
      > "${M31_ROOT}/overlay.txt" || return 2
  image_vllm="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')" || return 2
  cp -a "${image_vllm}" "${M31_ROOT}/site/vllm" || return 2
  env "${M31_ENV[@]}" python3 - "${src}" "${M31_ROOT}/site" "${M31_ROOT}/overlay.txt" \
    "${ATOMESH_M31_VLLM_BASE_SHA}" "${RUN_DIR}/m31-stack-rank-${NODE_RANK}.json" <<'PY' || return 2
import importlib.metadata, json, pathlib, re, shutil, subprocess, sys

src, site, listing = map(pathlib.Path, sys.argv[1:4])
base, out = sys.argv[4], pathlib.Path(sys.argv[5])
changes = [line.split("\t") for line in listing.read_text().splitlines()]
# An image that patched a file on top of the base commit would silently lose that patch.
for status, rel in changes:
    if status in ("M", "D"):
        base_blob = subprocess.run(["git", "-C", str(src), "show", f"{base}:{rel}"],
                                   check=True, capture_output=True).stdout
        if (site / rel).read_bytes() != base_blob:
            raise SystemExit(f"image {rel} differs from {base}; the overlay would drop the image's change")
    elif status == "A" and (site / rel).exists():
        raise SystemExit(f"image already has {rel}; the overlay would replace it")
for status, rel in changes:
    if status == "D":
        (site / rel).unlink()
    elif status in ("M", "A"):
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, site / rel)
    else:
        raise SystemExit(f"unexpected diff status {status} {rel}")

import vllm
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import MoRIIOConnector  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import SimpleCPUOffloadConnector  # noqa: F401

assert vllm.__file__.startswith(str(site)), vllm.__file__
# The diff is only exact if the image was built from the diff base.
image_commit = re.search(r"\+g([0-9a-f]{7,40})", vllm.__version__)
assert image_commit and base.startswith(image_commit.group(1)), (vllm.__version__, base)
versions = {}
for name in ("vllm", "torch", "triton", "amd-aiter", "mori"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
stack = {"vllm_file": vllm.__file__, "vllm_version": vllm.__version__, "distributions": versions,
         "overlay_counts": {s: sum(1 for c, _ in changes if c == s) for s in sorted({c for c, _ in changes})},
         "overlay": [f"{s}\t{r}" for s, r in changes]}
out.write_text(json.dumps(stack, indent=2) + "\n")
print(f"[m31] overlay {stack['overlay_counts']} on {vllm.__version__}; stack {out}")
PY
}

m31_overlay || exit $?

# m31_moriio ROLE HTTP_PORT HANDSHAKE_PORT NOTIFY_PORT: TP rank r listens on handshake/notify base+r.
# One QP per transfer overflows its 16384-WR send queue on a ~230K-token prompt at C64 (SQ full).
m31_moriio() {
  echo "{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"$1\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${NODE0_ADDR}\",\"proxy_ping_port\":${VLLM_DISCOVERY_PORT},\"http_port\":$2,\"handshake_port\":$3,\"notify_port\":$4,\"backend\":\"rdma\",\"num_workers\":4,\"qp_per_transfer\":4,\"read_mode\":${ATOMESH_M31_MORIIO_READ_MODE:-false}}}"
}

start_vllm_server() {
  local role="$1" log_name="$2" server_port="$3" gpus idx xfer
  local -a cache_env=() role_env=() role_args=() pin=() p_extra_args=()
  build_server_cache_env "${role}" "${server_port}" cache_env
  if [[ "${role}" == prefill ]]; then
    # prefill_single_node: worker idx serves PREFILL_PORT+idx on the GPUs pd_server_atom.sh exported;
    # its handshake/notify bases sit 1000 apart, clear of the decode bases (+100 per D).
    gpus="${HIP_VISIBLE_DEVICES}" idx=$((server_port - PREFILL_PORT))
    xfer="$(m31_moriio kv_producer "${server_port}" $((M31_HANDSHAKE_PORT + 1000 * idx)) $((M31_NOTIFY_PORT + 1000 * idx)))"
    xfer="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"connectors\":[${xfer},{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${ATOMESH_M31_P_CPU_OFFLOAD_BYTES},\"lazy_offload\":${ATOMESH_M31_P_LAZY_OFFLOAD:-false}}}]}}"
    role_args=(--tensor-parallel-size "${PREFILL_TP_SIZE}" --max-num-seqs $((2 * m31_conc)) --port "${server_port}" --gpu-memory-utilization 0.85
      --max-num-batched-tokens 32768)
    [[ "${ATOMESH_M31_P_DBO:-1}" == 1 ]] &&
      role_args+=(--enable-dbo --dbo-prefill-token-threshold 16384 --dbo-decode-token-threshold 32768)
    read -r -a p_extra_args <<< "${ATOMESH_M31_P_EXTRA_ARGS:-}"
    role_args+=("${p_extra_args[@]}" --kv-transfer-config "${xfer}")
  else
    # KEY=VAL words for the decode servers only.
    read -r -a role_env <<< "${ATOMESH_M31_D_EXTRA_ENV:-}"
    # decode_single_node: worker idx serves DECODE_PORT+idx on the GPUs pd_server_atom.sh exported, so
    # each D on the node needs its own handshake/notify bases (4 TP ranks each).
    gpus="${HIP_VISIBLE_DEVICES}" idx=$((server_port - DECODE_PORT))
    # single_node: prefill holds GPUs 0-3 of this node.
    [[ "${SINGLE_NODE_PD}" == "1" ]] && gpus=4,5,6,7
    xfer="$(m31_moriio kv_consumer "${server_port}" $((M31_HANDSHAKE_PORT + 100 * (idx + 1))) $((M31_NOTIFY_PORT + 100 * (idx + 1))))"
    # Every concurrency can land on one D under consistent_hash: FULL graphs for 1..max-num-seqs requests x 8
    # tokens. ATOMESH_M31_D_MAX_SEQS caps each D's decode batch (default CONC: no cap).
    local d_seqs="${ATOMESH_M31_D_MAX_SEQS:-${m31_conc}}"
    # ATOMESH_M31_D_COMPILATION_CONFIG replaces these graph sizes.
    local d_graphs="{\"cudagraph_capture_sizes\":[$(seq -s, 8 8 $((8 * d_seqs)))],\"max_cudagraph_capture_size\":$((8 * d_seqs))}"
    role_args=(--tensor-parallel-size "${DECODE_TP_SIZE}" --max-num-seqs "${d_seqs}" --port "${server_port}" --gpu-memory-utilization 0.85
      --max-num-batched-tokens 8192
      --compilation-config "${ATOMESH_M31_D_COMPILATION_CONFIG:-${d_graphs}}"
      --kv-transfer-config "${xfer}")
  fi
  # GPUs 0-3 sit on NUMA node 0, GPUs 4-7 on node 1; a server spanning both (TP8) runs unpinned.
  if (( ${gpus%%,*} / 4 == ${gpus##*,} / 4 )); then
    read -r -a pin <<< "taskset -c $(< "/sys/devices/system/node/node$(( ${gpus%%,*} / 4 ))/cpulist")"
    if ! "${pin[@]}" true 2>/dev/null; then
      echo "[m31] WARNING: cannot pin ${role} to the NUMA node of GPU ${gpus%%,*}; running unpinned" >&2
      pin=()
    fi
  fi
  local -a cmd=(env "${cache_env[@]}" "${M31_ENV[@]}" "${role_env[@]}" HIP_VISIBLE_DEVICES="${gpus}" "${pin[@]}"
    "${M31_VLLM_ARGS[@]}" "${role_args[@]}")
  echo "[${role}] rank=${NODE_RANK} host=${host_name} ip=${host_ip} port=${server_port} discovery=${NODE0_ADDR}:${VLLM_DISCOVERY_PORT} $(quote "${cmd[@]}")"
  python3 -c 'import json, sys; open(sys.argv[1], "w").write(json.dumps({"role": sys.argv[2], "argv": sys.argv[3:]}, indent=2) + "\n")' \
    "${RUNTIME_LOG_DIR}/${log_name}.launch.json" "${role}" "${cmd[@]}"
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" "${cmd[@]}"
}

m31_scrape() {  # OUT_DIR TAG: /metrics of every server
  local i
  for i in "${!prefill_ips[@]}"; do  # P0 keeps the single-P file name
    curl -sf "http://${prefill_ips[i]}:${prefill_ports[i]}/metrics" > "$1/$2_metrics_prefill${i/#0/}.prom" || echo "WARNING: prefill${i} /metrics scrape failed" >&2
  done
  for i in "${!decode_ips[@]}"; do
    curl -sf "http://${decode_ips[i]}:${decode_ports[i]}/metrics" > "$1/$2_metrics_decode${i}.prom" || echo "WARNING: decode${i} /metrics scrape failed" >&2
  done
}

# AgentX client as the M3.1 Mooncake cases (pd_m31_mc_server.sh) run it, against the router sidecar.
run_aiperf_agentic_benchmark() {
  local venv="/tmp/atomesh-agentx-venv-${ATOMESH_M31_AGENTX_VERSION}"
  local out="${RUN_DIR}/benchmark_results/m31-pd-c${m31_conc}" rc aiperf_json i
  local safe_model="${MODEL_NAME//\//-}"
  local -a metrics=()
  for i in "${!prefill_ips[@]}"; do metrics+=("http://${prefill_ips[i]}:${prefill_ports[i]}/metrics"); done
  for i in "${!decode_ips[@]}"; do metrics+=("http://${decode_ips[i]}:${decode_ports[i]}/metrics"); done
  if ! python3 -m venv "${venv}"; then
    rm -rf "${venv}"
    uv venv --seed --python "$(command -v python3)" "${venv}" || return $?
  fi
  "${venv}/bin/python" -m pip install --quiet "agentx-harness==${ATOMESH_M31_AGENTX_VERSION}" || return $?
  [[ "$("${venv}/bin/agentx" --version)" == "${ATOMESH_M31_AGENTX_VERSION}" ]] || return 2
  mkdir -p "${out}"
  "${venv}/bin/python" -m pip freeze > "${out}/agentx-venv-freeze.txt"

  local -a argv=("${venv}/bin/agentx" profile --url "http://127.0.0.1:${ROUTER_PORT}"
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}"
    --concurrency "${m31_conc}" --benchmark-duration "${AIPERF_BENCHMARK_DURATION}" --tokenizer-trust-remote-code
    --server-metrics "${metrics[@]}"
    --output-artifact-dir "${out}/aiperf_artifacts")
  (( AIPERF_BENCHMARK_DURATION < 900 )) && argv+=(--unsafe-override)
  argv+=(--public-dataset semianalysis_cc_traces_weka_062126)

  quote "${argv[@]}" > "${out}/benchmark_command.txt"
  m31_scrape "${out}" before
  rc=0
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 "${argv[@]}" 2>&1 | tee "${out}/benchmark.log" || rc=$?
  m31_scrape "${out}" final

  aiperf_json="$(find "${out}/aiperf_artifacts" -maxdepth 2 -name profile_export_aiperf.json -print -quit)"
  [[ -n "${aiperf_json}" ]] || { echo "[aiperf][FAIL] profile_export_aiperf.json was not produced" >&2; return 1; }
  write_aiperf_dashboard_json "${aiperf_json}" \
    "${RUN_DIR}/benchmark_results/pd-${BACKEND}-${safe_model}-${TOPOLOGY}-isl${AIPERF_MAX_CONTEXT_LENGTH}-osl1024-conc${m31_conc}-${RANDOM_RANGE_RATIO}.json" \
    "${m31_conc}"
  return "${rc}"
}
