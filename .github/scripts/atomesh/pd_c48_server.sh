# shellcheck shell=bash
# Isolated C48 model-setting experiment; reuse topology/lifecycle only.

# Prebuilt LMCache ROCm wheels target torch 2.11 and fail to load cuda_ops on
# this image's torch 2.13, so the pinned source is built against the image.
c48_start_lmcache() {
  local config="$1" build="/tmp/atomesh-c48-lmcache-build"
  eval "$(python3 - "${config}" <<'PY'
import json
import shlex
import sys

c = json.load(open(sys.argv[1]))["lmcache"]
env = {
    "C48_LMCACHE_BASE_PORT": int(c["port"]), "C48_LMCACHE_BASE_HTTP_PORT": int(c["http_port"]),
    "C48_LMCACHE_REPO": c["repo"], "C48_LMCACHE_COMMIT": c["commit"],
    "C48_LMCACHE_VERSION": c["version"], "C48_LMCACHE_ARCH": c["rocm_arch"],
    "C48_LMCACHE_BUILD_TOOLS": " ".join(c["build_tools"]),
    "ATOMESH_VLLM_LMCACHE_PACKAGES": " ".join(c["packages"]),
    "ATOMESH_VLLM_LMCACHE_NATIVE_DEBS": " ".join(c["native_debs"]),
    "ATOMESH_VLLM_LMCACHE_PATCHES": " ".join(c["patches"]),
    "ATOMESH_VLLM_LMCACHE_L1_SIZE_GB": c["l1_size_gb"],
    "ATOMESH_VLLM_LMCACHE_SERVER_ARGS": c["server_args"],
}
for key, value in env.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"
  local -a tools=()
  read -r -a tools <<< "${C48_LMCACHE_BUILD_TOOLS}"
  rm -rf "${build}"
  python3 -m pip install --quiet --no-cache-dir --target "${build}/tools" "${tools[@]}"
  git init -q "${build}/src"
  git -C "${build}/src" fetch -q --depth 1 "${C48_LMCACHE_REPO}" "${C48_LMCACHE_COMMIT}"
  git -C "${build}/src" checkout -q FETCH_HEAD
  echo "[lmcache] building ${C48_LMCACHE_REPO}@$(git -C "${build}/src" rev-parse HEAD) for ${C48_LMCACHE_ARCH}"
  (cd "${build}/src" && env PYTHONPATH="${build}/tools" PATH="${build}/tools/bin:${PATH}" \
    BUILD_WITH_HIP=1 CXX="${ROCM_PATH:-/opt/rocm}/bin/hipcc" PYTORCH_ROCM_ARCH="${C48_LMCACHE_ARCH}" \
    MAX_JOBS="$(nproc)" SETUPTOOLS_SCM_PRETEND_VERSION="${C48_LMCACHE_VERSION}" \
    python3 setup.py bdist_wheel --dist-dir "${build}/dist") > "${RUNTIME_LOG_DIR}/lmcache-build.log" 2>&1
  local -a wheels=("${build}"/dist/*.whl)
  [[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || return 2
  export ATOMESH_VLLM_LMCACHE_WHEEL="${wheels[0]}"
  C48_LMCACHE_PORT=$((C48_LMCACHE_BASE_PORT + ATOMESH_SERVICE_PORT_OFFSET))
  start_lmcache "${C48_LMCACHE_PORT}" $((C48_LMCACHE_BASE_HTTP_PORT + ATOMESH_SERVICE_PORT_OFFSET))
  env "${lmcache_env[@]}" python3 -c 'import lmcache.cuda_ops'
}

start_vllm_server() {
  local role="$1" log_name="$2" port="$3"
  local config="${ATOMESH_SCRIPT_DIR}/../../benchmark/c48-model-gap/${ATOMESH_C48_CASE}.json"
  local -a cache_env=() lmcache_args=()
  build_server_cache_env "${role}" "${port}" cache_env
  if [[ "${role}" == "prefill" ]] && python3 -c 'import json, sys; sys.exit("lmcache" not in json.load(open(sys.argv[1])))' "${config}"; then
    c48_start_lmcache "${config}"
    lmcache_args=(--lmcache-port "${C48_LMCACHE_PORT}")
  fi
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" \
    env "${cache_env[@]}" "${lmcache_env[@]}" python3 "${ATOMESH_SCRIPT_DIR}/pd_c48_launch.py" \
    --config "${config}" \
    --role "${role}" --model "${MODEL_PATH}" --host "${host_ip}" \
    --proxy "${NODE0_ADDR}" --port "${port}" --discovery "${VLLM_DISCOVERY_PORT}" \
    --output "${RUNTIME_LOG_DIR}/${log_name}.launch.json" "${lmcache_args[@]}" --execute
}

run_aiperf_agentic_benchmark() {
  ensure_aiperf
  "${AIPERF_VENV}/bin/python" -m pip install 'datasets>=4.7.0'
  local out_dir="${RUN_DIR}/benchmark_results/c48-${ATOMESH_C48_CASE}" conc
  conc="$(python3 -c 'import json, sys; (c,) = json.load(open(sys.argv[1]))["benchmark"]["concurrencies"]; print(int(c))' \
    "${ATOMESH_SCRIPT_DIR}/../../benchmark/c48-model-gap/${ATOMESH_C48_CASE}.json")"
  mkdir -p "${out_dir}"
  git -C "${AIPERF_DIR}" rev-parse HEAD > "${out_dir}/aiperf-commit.txt"
  "${AIPERF_VENV}/bin/python" -m pip freeze > "${out_dir}/client-packages.txt"
  snapshot_eval_metrics "${out_dir}" before
  local -a profile_wrapper=() profile_target=()
  if [[ "${ATOMESH_C48_PREFILL_PROFILE:-0}" == "1" ]]; then
    profile_target=(prefill "http://${prefill_ips[0]}:${prefill_ports[0]}")
  elif [[ "${ATOMESH_C48_DECODE_PROFILE:-0}" == "1" ]]; then
    profile_target=(decode "http://${decode_ips[0]}:${decode_ports[0]}")
  fi
  if [[ ${#profile_target[@]} -eq 2 ]]; then
    profile_wrapper=("${AIPERF_VENV}/bin/python" "${ATOMESH_SCRIPT_DIR}/pd_c48_profile.py"
      "--${profile_target[0]}" "${profile_target[1]}"
      --output "${out_dir}/${profile_target[0]}-profile" --traces "${RUN_DIR}/traces"
      --timeline "${RUNTIME_LOG_DIR}/${profile_target[0]}-timeline"
      --aiperf-log "${out_dir}/logs/aiperf.log"
      --seconds 10 --offsets 600 1500 3000 --ranks 8
      --phase-timeout 7200 --finish-timeout 5400 --)
  fi
  # Match the historical InferenceX agentx invocation and pinned scenario defaults.
  # Paths and addresses follow this allocation; no template or context override.
  local benchmark_rc=0
  "${profile_wrapper[@]}" "${AIPERF_VENV}/bin/aiperf" profile \
    --scenario agentx --url "http://127.0.0.1:${ROUTER_PORT}" \
    --model "${SERVED_MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --concurrency "${conc}" --benchmark-duration 3600 --tokenizer-trust-remote-code \
    --server-metrics "http://${prefill_ips[0]}:${prefill_ports[0]}/metrics" \
                     "http://${decode_ips[0]}:${decode_ports[0]}/metrics" \
    --output-artifact-dir "${out_dir}" \
    --public-dataset semianalysis_cc_traces_weka_062126 \
    2>&1 | tee "${out_dir}/aiperf.log" || benchmark_rc=$?
  snapshot_eval_metrics "${out_dir}" after
  test -s "${out_dir}/profile_export_aiperf.json"
  write_aiperf_dashboard_json "${out_dir}/profile_export_aiperf.json" \
    "${RUN_DIR}/benchmark_results/pd-vllm-c48-${ATOMESH_C48_CASE}.json" "${conc}"
  return "${benchmark_rc}"
}
