# shellcheck shell=bash
# MiniMax-M3.1-Flash vLLM 1P1D/1P2D over MooncakeConnector RDMA, sourced by pd_server_vllm.sh when
# ATOMESH_M31_MC=1. Server env/argv, proxy and AgentX client follow the local recipe copied
# into .github/benchmark/m31-mooncake-rdma; only the model path, PYTHONPATH, hosts, log
# locations and the decode-node GPU split are adapted to the two-node Slurm layout.

M31_DIR="${ATOMESH_SCRIPT_DIR}/../../benchmark/m31-mooncake-rdma"
M31_ROOT="/tmp/atomesh-m31-${SLURM_JOB_ID:-local}-${NODE_RANK}"
M31_BOOTSTRAP_PORT=$((8998 + ATOMESH_SERVICE_PORT_OFFSET))
ROUTER_READY_PATH="/health"
ROUTER_ALIVE_PATH="/health"

m31_conc="${CONC_LIST//[[:space:]]/}"
[[ "${m31_conc}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: M3.1 Mooncake cases take exactly one concurrency, got '${CONC_LIST}'" >&2
  exit 2
}
# COMMON_ENV, MOONCAKE_ENV, VLLM_ARGS, cpunode and quote exactly as the local common.sh defines them.
eval "$(
  m31_model="${MODEL_PATH}"
  source "${M31_DIR}/common.sh" >/dev/null
  MODEL_PATH="${m31_model}"
  vllm_args "${m31_conc}"
  declare -p COMMON_ENV MOONCAKE_ENV VLLM_ARGS _PLAIN
  declare -f cpunode quote
)"

m31_checkout() {  # URL SHA DIR [--blobless]: exact commit; blobless keeps the tags setuptools_scm reads.
  if [[ "${4:-}" == --blobless ]]; then
    git clone --quiet --filter=blob:none --no-checkout "$1" "$3" || return $?
  else
    git init --quiet "$3" && git -C "$3" fetch --quiet --depth 1 "$1" "$2" || return $?
  fi
  git -C "$3" checkout --quiet --detach "$2" || return $?
  [[ "$(git -C "$3" rev-parse HEAD)" == "$2" ]]
}

m31_prepare() {
  local wheel="${ATOMESH_SCRIPT_DIR}/mooncake-dist/${ATOMESH_M31_MOONCAKE_WHEEL}"
  local digest image_vllm egg build_log="${RUNTIME_LOG_DIR}/m31-aiter-build-rank-${NODE_RANK}.log"
  [[ -f "${MODEL_PATH}/config.json" && -f "${MODEL_PATH}/dspark/config.json" ]] || {
    echo "ERROR: ${MODEL_PATH} is not a complete MiniMax-M3.1-Flash snapshot" >&2
    return 2
  }
  rm -rf "${M31_ROOT}" && mkdir -p "${M31_ROOT}" || return 2

  # The image's Mooncake 0.3.13.post1 lacks multi-protocol support and sends every GPU segment to HipTransport.
  digest="$(sha256sum "${wheel}")" || return 2
  [[ "${digest%% *}" == "${ATOMESH_M31_MOONCAKE_WHEEL_SHA256}" ]] || {
    echo "ERROR: ${wheel} is not ${ATOMESH_M31_MOONCAKE_WHEEL_SHA256}" >&2
    return 2
  }
  python3 -m pip install --quiet --no-cache-dir --no-deps --target "${M31_ROOT}/mooncake" "${wheel}" || return $?

  # Fork source tree plus the image's build products, as PYTHONPATH=/app/vllm-src locally.
  m31_checkout "${ATOMESH_M31_VLLM_REPO}" "${ATOMESH_M31_VLLM_SHA}" "${M31_ROOT}/vllm" || return 2
  image_vllm="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')" || return 2
  cp -a "${image_vllm}"/*.so "${image_vllm}/_version.py" "${image_vllm}/vllm-rs" "${M31_ROOT}/vllm/vllm/" || return 2
  cp -a "${image_vllm}/third_party/triton_kernels" "${M31_ROOT}/vllm/vllm/third_party/" || return 2

  # User recipe; the Slurm uid cannot uninstall the image's aiter, so the egg is shadowed via PYTHONPATH.
  m31_checkout "${ATOMESH_M31_AITER_REPO}" "${ATOMESH_M31_AITER_SHA}" "${M31_ROOT}/aiter-src" --blobless || return 2
  echo "[m31] building aiter ${ATOMESH_M31_AITER_SHA}; log ${build_log}"
  # Explicit && chain: set -e does not apply inside the `m31_prepare || exit` call.
  (
    cd "${M31_ROOT}/aiter-src" &&
      git submodule sync &&
      { git submodule update --init --recursive --depth 1 || git submodule update --init --recursive; } &&
      rm -rf ./aiter/jit/*.so aiter/install_mode ./dist ./build &&
      PYTHONUSERBASE="${M31_ROOT}/aiter" GPU_ARCHS=gfx950 python3 setup.py install --user
  ) > "${build_log}" 2>&1 || { tail -n 80 "${build_log}" >&2; return 2; }
  egg="$(PYTHONUSERBASE="${M31_ROOT}/aiter" python3 -c 'import site; print(site.getusersitepackages())')"
  egg=("${egg}"/amd_aiter-*.egg)
  [[ "${#egg[@]}" -eq 1 && -d "${egg[0]}" ]] || { echo "ERROR: aiter egg not found: ${egg[*]}" >&2; return 2; }

  M31_PYTHONPATH="${M31_ROOT}/vllm:${egg[0]}:${M31_ROOT}/mooncake"
  COMMON_ENV=("${COMMON_ENV[@]/#PYTHONPATH=*/PYTHONPATH=${M31_PYTHONPATH}}")
  PYTHONPATH="${M31_PYTHONPATH}" python3 - "${M31_ROOT}" "${ATOMESH_M31_AITER_VERSION}" \
    "${RUN_DIR}/m31-stack-rank-${NODE_RANK}.json" <<'PY' || return 2
import hashlib, json, pathlib, subprocess, sys
import aiter, mooncake.engine as mc, triton, torch, vllm
root, aiter_version, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
paths = {"vllm": vllm.__file__, "aiter": aiter.__file__, "mooncake": mc.__file__}
for name, prefix in (("vllm", root / "vllm"), ("aiter", root / "aiter"), ("mooncake", root / "mooncake")):
    assert paths[name].startswith(str(prefix)), (name, paths[name])
egg = pathlib.Path(aiter.__file__).parents[1]
assert egg.name.startswith(f"amd_aiter-{aiter_version}-"), egg.name
engine = pathlib.Path(mc.__file__).read_bytes()
assert b"MC_DISABLE_HIP" in engine and mc.SUPPORT_HIP, "Mooncake wheel is not the multi-protocol build"
untracked = subprocess.run(["git", "-C", str(root / "vllm"), "status", "--porcelain", "--ignored", "--untracked-files=normal"],
                           check=True, capture_output=True, text=True).stdout.split("\n")
untracked = sorted(l for l in untracked if l and "__pycache__" not in l)
so = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "vllm/vllm").glob("*.so"))}
stack = {
    "paths": paths, "aiter_egg": egg.name, "vllm_version": vllm.__version__, "torch": torch.__version__,
    "triton": triton.__version__, "mooncake_engine_sha256": hashlib.sha256(engine).hexdigest(),
    "vllm_overlay_untracked": untracked, "vllm_so_sha256": so,
    "image_versions_txt_sha256": hashlib.sha256(pathlib.Path("/app/versions.txt").read_bytes()).hexdigest()
    if pathlib.Path("/app/versions.txt").exists() else None,
}
out.write_text(json.dumps(stack, indent=2) + "\n")
print("[m31] stack " + json.dumps(stack))
PY
}

m31_prepare || exit $?

start_vllm_server() {
  local role="$1" log_name="$2" server_port="$3" node
  local rest='"kv_load_failure_policy":"fail","kv_connector_extra_config":{"mooncake_protocol":"rdma","num_workers":10}'
  local gpus=4,5,6,7
  local -a cache_env=() role_env=() role_args=() pin=()
  build_server_cache_env "${role}" "${server_port}" cache_env
  if [[ "${role}" == prefill ]]; then
    node=0
    role_env=(HIP_VISIBLE_DEVICES=0,1,2,3 VLLM_MOONCAKE_BOOTSTRAP_PORT="${M31_BOOTSTRAP_PORT}"
      MC_MAX_MR_SIZE=1073741824 GLIBC_TUNABLES=glibc.malloc.hugetlb=1)
    role_args=(--port "${server_port}"
      --max-num-batched-tokens 32768 --long-prefill-token-threshold 4096 --prefill-schedule-interval 1
      --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",${rest}}")
  else
    # decode_single_node: pd_server_atom.sh hands each decode worker GPUs 0-3 or 4-7; GPUs 0-3 sit on NUMA 0.
    [[ "${DECODE_SINGLE_NODE_PD}" == 1 ]] && gpus="${HIP_VISIBLE_DEVICES}"
    node=$(( ${gpus%%,*} / 4 ))
    role_env=(HIP_VISIBLE_DEVICES="${gpus}" MC_MAX_MR_SIZE=268435456)
    role_args=(--port "${server_port}"
      --max-num-batched-tokens 8192 --watermark 0.05
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
      --max-cudagraph-capture-size $((2 * m31_conc * 8))
      --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",${rest}}")
  fi
  read -r -a pin <<< "$(cpunode "${node}" 2>/dev/null)"
  if ! "${pin[@]}" true 2>/dev/null; then
    echo "[m31] WARNING: cannot pin ${role} to NUMA node ${node}; running unpinned" >&2
    pin=()
  fi
  local -a cmd=(env "${cache_env[@]}" "${COMMON_ENV[@]}" "${MOONCAKE_ENV[@]}" "${role_env[@]}"
    "${pin[@]}" "${VLLM_ARGS[@]}" "${role_args[@]}")
  echo "[${role}] rank=${NODE_RANK} host=${host_name} ip=${host_ip} port=${server_port} $(quote "${cmd[@]}")"
  python3 -c 'import json, sys; open(sys.argv[1], "w").write(json.dumps({"role": sys.argv[2], "argv": sys.argv[3:]}, indent=2) + "\n")' \
    "${RUNTIME_LOG_DIR}/${log_name}.launch.json" "${role}" "${cmd[@]}"
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" "${cmd[@]}"
}

start_router() {
  local -a cmd=(python3 "${M31_DIR}/mooncake_connector_proxy.py" --host 0.0.0.0 --port "${ROUTER_PORT}"
    --prefill "http://${prefill_ips[0]}:${prefill_ports[0]}" "${M31_BOOTSTRAP_PORT}"
    "${decode_args[@]}")
  echo "[router] $(quote "${cmd[@]}")"
  start_logged_process router_pid "${RUNTIME_LOG_DIR}/proxy.log" "${cmd[@]}"
}

m31_scrape() {  # OUT_DIR TAG: /metrics of every server
  local i
  curl -sf "http://${prefill_ips[0]}:${prefill_ports[0]}/metrics" > "$1/$2_metrics_prefill.prom" || echo "WARNING: prefill /metrics scrape failed" >&2
  for i in "${!decode_ips[@]}"; do
    curl -sf "http://${decode_ips[i]}:${decode_ports[i]}/metrics" > "$1/$2_metrics_decode${i}.prom" || echo "WARNING: decode${i} /metrics scrape failed" >&2
  done
}

run_aiperf_agentic_benchmark() {
  local venv="/tmp/atomesh-agentx-venv-${ATOMESH_M31_AGENTX_VERSION}"
  local out="${RUN_DIR}/benchmark_results/m31-pd-c${m31_conc}" rc aiperf_json i
  local safe_model="${MODEL_NAME//\//-}"
  local -a metrics=("http://${prefill_ips[0]}:${prefill_ports[0]}/metrics")
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
