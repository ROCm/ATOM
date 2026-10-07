# shellcheck shell=bash
# vLLM backend for the ATOMesh P/D harness, sourced by pd_server_atom.sh when
# BACKEND=vllm. It replaces the worker/router launchers and reuses the shared
# topology, readiness, benchmark and eval flow. Workers register with the
# rank-0 vllm-router sidecar (see pd_slurm_job.sh) through MoRIIO discovery.

# The router answers /health only once a prefill and a decode have registered.
ROUTER_READY_PATH="/health"
ROUTER_ALIVE_PATH="/liveness"
if [[ "${ATOMESH_VLLM_CONNECTOR:-moriio}" == "nixl" ]]; then
  ROUTER_READY_PATH="/healthcheck"
  ROUTER_ALIVE_PATH="/healthcheck"
fi
SERVED_MODEL_NAME="${ATOMESH_VLLM_SERVED_MODEL_NAME:-${MODEL_PATH}}"
VLLM_DISCOVERY_PORT=$((${ATOMESH_VLLM_ROUTER_DISCOVERY_PORT:?vllm.router.discovery_port is required} + ATOMESH_SERVICE_PORT_OFFSET))
VLLM_SITE_DIR="/tmp/atomesh-vllm-site"
LMCACHE_ROOT="/tmp/atomesh-lmcache"

export VLLM_HOST_IP="${host_ip}"
if [[ -n "${MORI_SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MORI_SOCKET_IFNAME}}"
fi

# Server processes only; aiperf and the harness keep the image's Python path.
server_pythonpath="${PYTHONPATH:-}"
lmcache_env=()

install_native_vllm() {
  local repo="${ATOMESH_VLLM_SOURCE_REPO:?vllm.source.repo is required}"
  local sha="${ATOMESH_VLLM_SOURCE_SHA:?vllm.source.sha is required}"
  [[ "${sha}" =~ ^[0-9a-f]{40}$ ]] || return 2
  if [[ "${ATOMESH_VLLM_CLEAN_MAIN:-0}" == "1" ]]; then
    [[ "${repo}" == "https://github.com/vllm-project/vllm" &&
       "${sha}" == "b22494cc0cb4bd9db4a62fb107d92429a4a3249d" &&
       -z "${ATOMESH_VLLM_FORK_REPO:-}${ATOMESH_VLLM_FORK_SHA:-}${ATOMESH_VLLM_LMCACHE_WHEEL:-}${ATOMESH_VLLM_LMCACHE_PATCHES:-}" ]] || return 2
    export VLLM_PLUGINS="" PYTHONNOUSERSITE=1 VLLM_USE_PRECOMPILED=0 VLLM_USE_PRECOMPILED_RUST=0
    unset VLLM_PRECOMPILED_WHEEL_LOCATION
  fi
  if [[ "${ATOMESH_VLLM_CLEAN_MAIN:-0}" == "1" ]]; then
    local -a identity_args=()
    if [[ "${ATOMESH_VLLM_NIXL_MODEL_PROFILE:-m3}" == "v4" && -z "${ATOMESH_VLLM_CHECKPOINT_MANIFEST:-}" ]]; then
      echo "V4 requires a checkpoint identity manifest on every node" >&2
      return 2
    fi
    if [[ -n "${ATOMESH_VLLM_CHECKPOINT_MANIFEST:-}" ]]; then
      identity_args=(--manifest "${ATOMESH_SCRIPT_DIR}/${ATOMESH_VLLM_CHECKPOINT_MANIFEST}")
    fi
    python3 "${ATOMESH_SCRIPT_DIR}/pd_survey_preflight.py" "${MODEL_PATH}" \
      "${RUNTIME_LOG_DIR}/weights-preflight-rank-${NODE_RANK}.json" "${identity_args[@]}" || return $?
  fi
  local src="/tmp/atomesh-native-vllm" venv="/tmp/atomesh-native-venv"
  git init -q "${src}"
  git -C "${src}" fetch -q --depth 1 "${repo}" "${sha}"
  git -C "${src}" checkout -q FETCH_HEAD
  [[ "$(git -C "${src}" rev-parse HEAD)" == "${sha}" ]] || return 2
  uv venv --system-site-packages "${venv}"
  uv pip install --python "${venv}/bin/python" \
    setuptools-scm setuptools-rust wheel ninja cmake
  local log="${RUNTIME_LOG_DIR}/native-build-rank-${NODE_RANK}.log"
  env CCACHE_DIR="${venv}/ccache" CCACHE_TEMPDIR="${venv}/ccache/tmp" \
    PYTHONPATH= VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx950 MAX_JOBS=32 \
    uv pip install --python "${venv}/bin/python" --no-deps \
      --no-build-isolation "${src}" > "${log}" 2>&1 || {
        tail -100 "${log}"
        return 1
      }
  export PATH="${venv}/bin:${PATH}"
  server_pythonpath=""
  env PYTHONPATH= "${venv}/bin/python" - "${venv}" "${sha}" \
    "${RUNTIME_LOG_DIR}/native-manifest-rank-${NODE_RANK}.json" <<'PY'
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch
import vllm
if os.environ.get("ATOMESH_VLLM_CONNECTOR") == "nixl":
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnector
else:
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import MoRIIOConnector

assert Path(vllm.__file__).is_relative_to(sys.argv[1]), vllm.__file__
import vllm._C
assert Path(vllm._C.__file__).is_relative_to(sys.argv[1]), vllm._C.__file__
manifest = {
    "native_extension": vllm._C.__file__,
    "source_repo": os.environ["ATOMESH_VLLM_SOURCE_REPO"],
    "plugins": os.environ.get("VLLM_PLUGINS"),
    "source_sha": sys.argv[2], "source_path": vllm.__file__,
    "torch": torch.__version__, "hip": torch.version.hip,
    "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
}
if os.environ.get("ATOMESH_VLLM_CONNECTOR") == "nixl":
    # Initializes only a transport agent; no GPU allocation, registration or transfer.
    from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
    try:
        assert NixlWrapper is not None and nixl_agent_config is not None
        agent = NixlWrapper("survey-preflight", nixl_agent_config(backends=["UCX"]))
        manifest["nixl_plugins"] = agent.get_plugin_list()
        assert "UCX" in manifest["nixl_plugins"], "UCX plugin unavailable"
        manifest["nixl_ucx_preflight"] = "AVAILABLE_NOT_RDMA_VALIDATED"
        del agent
    except Exception as exc:
        manifest["nixl_ucx_preflight"] = "BLOCKED_ENV"
        manifest["dependency_error"] = repr(exc)
        Path(sys.argv[3]).write_text(json.dumps(manifest, indent=2) + "\n")
        raise
if os.environ.get("ATOMESH_VLLM_CLEAN_MAIN") == "1":
    import triton.language.target_info as target_info
    from vllm.model_executor.models import ModelRegistry

    config = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
    manifest["model_architectures"] = config.get("architectures", [])
    manifest["checkpoint_quantization"] = config.get("text_config", config).get("quantization_config")
    manifest["triton_has_is_hip_gfx1250"] = hasattr(target_info, "is_hip_gfx1250")
    if os.environ.get("ATOMESH_VLLM_HYBRID") == "1":
        import flydsl
        from aiter import ActivationType
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.ops.shuffle import shuffle_weight, shuffle_scale
        from aiter.fused_moe import fused_moe
        assert hasattr(ActivationType, "Situv2") and hasattr(GateMode, "SEPARATED")
        assert all(callable(api) for api in (shuffle_weight, shuffle_scale, fused_moe))
        manifest["aiter_situv2_gate_shuffle_flydsl_imports"] = True
    manifest["registered_model"] = any(
        arch in ModelRegistry.get_supported_archs()
        for arch in manifest["model_architectures"]
    )
Path(sys.argv[3]).write_text(json.dumps(manifest, indent=2) + "\n")
if os.environ.get("ATOMESH_VLLM_CLEAN_MAIN") == "1":
    assert manifest["registered_model"], "BLOCKED_ENV: checkpoint architecture not registered"
    if not manifest["triton_has_is_hip_gfx1250"]:
        print("[dependency] Triton lacks is_hip_gfx1250; review actual selected backend, not a universal model blocker")
print(f"[vllm] native source installation OK: {sys.argv[2]} {vllm.__file__}")
PY
}

join_path() {
  local IFS=":"
  local -a parts=()
  local item
  for item in "$@"; do
    [[ -n "${item}" ]] && parts+=("${item}")
  done
  echo "${parts[*]}"
}

# Split a shell-quoted argument string (e.g. JSON flags) into array $1.
split_args() {
  mapfile -d '' -t "$1" < <(
    python3 -c 'import shlex, sys; sys.stdout.write("".join(a + "\0" for a in shlex.split(sys.argv[1])))' "$2"
  )
}

# Copy the image's vllm package to a writable dir and overlay fork sources on
# it, keeping the image's compiled extensions.
apply_vllm_fork_overlay() {
  local repo="${ATOMESH_VLLM_FORK_REPO:-}"
  [[ -n "${repo}" ]] || return 0
  local sha="${ATOMESH_VLLM_FORK_SHA:-}"
  local files="${ATOMESH_VLLM_FORK_FILES:?vllm.fork.files is required}"
  if [[ ! "${sha}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "[vllm][FAIL] vllm.fork.sha must be a full commit sha, got '${sha}'" >&2
    exit 2
  fi
  local src="/tmp/atomesh-vllm-fork" base_pkg patch_file
  base_pkg="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')"
  echo "[vllm] overlay ${repo}@${sha} on ${base_pkg}"
  rm -rf "${src}" "${VLLM_SITE_DIR}"
  git init -q "${src}"
  git -C "${src}" fetch -q --depth 1 "${repo}" "${sha}"
  git -C "${src}" checkout -q FETCH_HEAD
  mkdir -p "${VLLM_SITE_DIR}"
  cp -a "${base_pkg}" "${VLLM_SITE_DIR}/vllm"
  python3 - "${src}/vllm" "${VLLM_SITE_DIR}/vllm" "${files}" <<'PY'
import os
import shutil
import sys

src_root, dst_root, spec = sys.argv[1:]
if spec == "all":
    files = [
        os.path.relpath(os.path.join(root, name), src_root)
        for root, _, names in os.walk(src_root)
        for name in names
        if name.endswith(".py")
    ]
else:
    files = spec.split()
for rel in files:
    src = os.path.join(src_root, rel)
    if not os.path.isfile(src):
        raise SystemExit(f"missing overlay source: {src}")
    dst = os.path.join(dst_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
print(f"[vllm] overlaid {len(files)} Python files")
PY
  for patch_file in vllm-k3-read-source-lease.patch vllm-k3-read-failure.patch \
    vllm-k3-sync-read-init.patch vllm-k3-full-read-context.patch \
    vllm-k3-read-step-completion.patch vllm-k3-stream-flush.patch \
    vllm-k3-latent-down-online.patch \
    vllm-k3-discovery-heartbeat.patch; do
    patch_file="${ATOMESH_SCRIPT_DIR}/patches/${patch_file}"
    echo "[vllm] applying patch $(sha256sum "${patch_file}")"
    git -C "${VLLM_SITE_DIR}" apply --check "${patch_file}"
    git -C "${VLLM_SITE_DIR}" apply "${patch_file}"
  done
  server_pythonpath="$(join_path "${VLLM_SITE_DIR}" "${server_pythonpath}")"
  env PYTHONPATH="${server_pythonpath}" python3 - "${VLLM_SITE_DIR}" <<'PY'
import sys

import vllm
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

assert vllm.__file__.startswith(sys.argv[1]), vllm.__file__
print(f"[vllm] overlay import OK: {vllm.__file__}")
PY
}

# LMCache MP server backing the prefill MultiConnector. Spur containers run as
# the Slurm user, so packages and native libs are unpacked under /tmp.
install_lmcache() {
  local site="${LMCACHE_ROOT}/site" libs="${LMCACHE_ROOT}/libs" url deb patch_file
  local -a packages=() debs=() patches=()
  read -r -a packages <<< "${ATOMESH_VLLM_LMCACHE_PACKAGES:-}"
  read -r -a debs <<< "${ATOMESH_VLLM_LMCACHE_NATIVE_DEBS:-}"
  rm -rf "${LMCACHE_ROOT}"
  mkdir -p "${site}" "${libs}"
  python3 -m pip install --quiet --no-cache-dir --no-deps --target "${site}" \
    "${packages[@]}" "${ATOMESH_VLLM_LMCACHE_WHEEL}"
  read -r -a patches <<< "${ATOMESH_VLLM_LMCACHE_PATCHES:-}"
  for patch_file in "${patches[@]}"; do
    patch_file="${ATOMESH_SCRIPT_DIR}/${patch_file}"
    echo "[lmcache] applying patch $(sha256sum "${patch_file}")"
    git -C "${site}" apply --check "${patch_file}"
    git -C "${site}" apply "${patch_file}"
  done
  for url in "${debs[@]}"; do
    deb="${LMCACHE_ROOT}/$(basename "${url}")"
    curl -fsSL --retry 3 -o "${deb}" "${url}"
    dpkg-deb -x "${deb}" "${libs}"
  done
  lmcache_env=(
    "PYTHONPATH=$(join_path "${site}" "${server_pythonpath}")"
    "PATH=${site}/bin:${PATH}"
    "LD_LIBRARY_PATH=$(join_path "${libs}/usr/lib/x86_64-linux-gnu" "${LD_LIBRARY_PATH:-}")"
  )
  env "${lmcache_env[@]}" python3 - <<'PY'
import importlib.metadata
import inspect

from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector,
    get_dcp_decorated_model_name,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA

assert issubclass(LMCacheMPConnector, SupportsHMA)
assert "kv_cache_config" in inspect.signature(get_dcp_decorated_model_name).parameters
print(f"[lmcache] {importlib.metadata.version('lmcache')} import OK")
PY
}

start_lmcache() {
  local port="$1"
  local http_port="$2"
  local l1_gb="${ATOMESH_VLLM_LMCACHE_L1_SIZE_GB:?vllm.lmcache.l1_size_gb is required}"
  local mem_kb
  mem_kb="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
  echo "[lmcache] MemTotal=$((mem_kb / 1024 / 1024))GiB l1=${l1_gb}GiB cgroup_memory_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
  if (( l1_gb * 1024 * 1024 * 10 > mem_kb * 9 )); then
    echo "[lmcache][FAIL] l1_size_gb=${l1_gb} exceeds 90% of host memory" >&2
    exit 2
  fi
  install_lmcache
  local -a extra=()
  split_args extra "${ATOMESH_VLLM_LMCACHE_SERVER_ARGS:-}"
  local -a cmd=(
    lmcache server
    --host 127.0.0.1 --port "${port}"
    --http-host 127.0.0.1 --http-port "${http_port}"
    --l1-size-gb "${l1_gb}"
    "${extra[@]}"
  )
  dump_launch_info "LMCACHE" "${cmd[@]}"
  start_logged_process lmcache_pid "${RUNTIME_LOG_DIR}/lmcache-rank-${NODE_RANK}.log" \
    env "${lmcache_env[@]}" "${cmd[@]}"
  wait_http "http://127.0.0.1:${http_port}/healthcheck" "lmcache" 600 "${lmcache_pid}"
}

kv_transfer_config() {
  local role="$1"
  local http_port="$2"
  local lmcache_port="${3:-}"
  python3 - "${role}" "${NODE0_ADDR}" "${VLLM_DISCOVERY_PORT}" "${http_port}" \
    "${lmcache_port}" "${ATOMESH_VLLM_LMCACHE_MQ_TIMEOUT:-6000}" "${ATOMESH_VLLM_CLEAN_MAIN:-0}" <<'PY'
import json
import os
import sys

role, proxy_ip, ping_port, http_port, lmcache_port, mq_timeout, clean_main = sys.argv[1:]
if os.environ.get("ATOMESH_VLLM_CONNECTOR") == "nixl":
    assert clean_main == "1" and not lmcache_port
    print(json.dumps({
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"backends": ["UCX"]},
    }))
    raise SystemExit(0)
moriio = {
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
    "kv_connector_extra_config": {
        "proxy_ip": proxy_ip,
        "proxy_ping_port": ping_port,
        "http_port": http_port,
        "backend": "rdma",
        "read_mode": True,
    },
}
if lmcache_port:
    config = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "connectors": [
                moriio,
                {
                    "kv_connector": "LMCacheMPConnector",
                    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "lmcache.mp.port": int(lmcache_port),
                        "lmcache.mp.mq_timeout": float(mq_timeout),
                    },
                },
            ]
        },
    }
else:
    config = {**moriio, "kv_load_failure_policy": "fail" if clean_main == "1" else "recompute"}
print(json.dumps(config))
PY
}

start_vllm_server() {
  local role="$1"
  local log_name="$2"
  local server_port="$3"
  local prefix="${role^^}"
  local tp_var="${prefix}_TP_SIZE" dcp_var="${prefix}_DCP_SIZE" args_var="${prefix}_SERVER_ARGS"
  local resolved_args
  resolved_args="$(python3 "${ATOMESH_SCRIPT_DIR}/pd_vllm_spec.py" \
    "${role}" "${ATOMESH_EXECUTION_PHASE}" "${SPEC_DECODE_ACCEPTANCE_LENGTH:-}" \
    "${!args_var}")" || return $?
  apply_role_env "ATOMESH_${prefix}_ENV_" "${host_ip}"
  if [[ "${SERVED_MODEL_NAME}" == "Kimi-K3" && "${AITER_SITUV2_A4W4:-}" == "1" ]]; then
    python3 "${ATOMESH_SCRIPT_DIR}/../k3-a4w4/check_aiter_paths.py"
  fi

  local lmcache_port=""
  local -a server_env=("PYTHONPATH=${server_pythonpath}")
  if [[ "${ATOMESH_VLLM_CONNECTOR:-moriio}" == "nixl" ]]; then
    server_env+=("VLLM_NIXL_SIDE_CHANNEL_HOST=${host_ip}"
      "VLLM_NIXL_SIDE_CHANNEL_PORT=$((15559 + NODE_RANK * 100 + ATOMESH_SERVICE_PORT_OFFSET))")
  fi
  if [[ "${role}" == "prefill" && -n "${ATOMESH_VLLM_LMCACHE_WHEEL:-}" ]]; then
    lmcache_port=$((ATOMESH_VLLM_LMCACHE_PORT + ATOMESH_SERVICE_PORT_OFFSET))
    start_lmcache "${lmcache_port}" $((ATOMESH_VLLM_LMCACHE_HTTP_PORT + ATOMESH_SERVICE_PORT_OFFSET))
    server_env=("${lmcache_env[@]}")
  fi

  local -a cache_env=() role_args=()
  build_server_cache_env "${role}" "${server_port}" cache_env
  split_args role_args "${resolved_args}"
  local -a cmd=(
    vllm serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --port "${server_port}"
    --trust-remote-code
    --kv-transfer-config "$(kv_transfer_config "${role}" "${server_port}" "${lmcache_port}")"
    --tensor-parallel-size "${!tp_var}"
  )
  if (( ${!dcp_var} > 1 )); then
    cmd+=(--decode-context-parallel-size "${!dcp_var}")
  fi
  cmd+=("${role_args[@]}")
  if [[ "${ATOMESH_VLLM_DIAGNOSTIC:-0}" == "1" && "${ATOMESH_VLLM_DIAGNOSTIC_MODE:-profile}" == "profile" ]]; then
    cmd+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${RUN_DIR}/traces/${role}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true}")
  fi
  echo "[${role}] rank=${NODE_RANK} host=${host_name} ip=${host_ip} gpu=${HIP_VISIBLE_DEVICES} port=${server_port} discovery=${NODE0_ADDR}:${VLLM_DISCOVERY_PORT}"
  dump_launch_info "${prefix}" "${cmd[@]}"
  python3 - "${RUNTIME_LOG_DIR}/${log_name}.launch.json" "${role}" \
    "${ATOMESH_EXECUTION_PHASE}" "${cmd[@]}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "role": sys.argv[2], "phase": sys.argv[3], "argv": sys.argv[4:]
}, indent=2) + "\n")
PY
  start_logged_process server_pid "${RUNTIME_LOG_DIR}/${log_name}.log" \
    env "${cache_env[@]}" "${server_env[@]}" "${cmd[@]}"
}

start_prefill() {
  start_vllm_server prefill "$1" "${2:-${PREFILL_PORT}}"
}

start_decode() {
  start_vllm_server decode "${1:-decode-rank-${NODE_RANK}}" "${2:-${DECODE_PORT}}"
}

start_router() {
  router_pid=""
  if [[ "${ATOMESH_VLLM_CONNECTOR:-moriio}" == "nixl" ]]; then
    start_logged_process router_pid "${RUNTIME_LOG_DIR}/nixl-proxy.log" \
      python3 /tmp/atomesh-native-vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
      --host 0.0.0.0 --port "${ROUTER_PORT}" \
      --prefiller-hosts "${NODE0_ADDR}" --prefiller-ports "${PREFILL_PORT}" \
      --decoder-hosts "${IP_ARRAY[1]}" --decoder-ports "${DECODE_PORT}"
    return
  fi
  echo "[router] vllm-router sidecar on :${ROUTER_PORT}, discovery :${VLLM_DISCOVERY_PORT}"
}

if [[ "${ATOMESH_VLLM_CLEAN_MAIN:-0}" == "1" && -z "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then
  echo "[vllm][FAIL] clean-main requires a pinned source installation" >&2
  exit 2
fi
if [[ -n "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then
  # Transport-only diagnostics never build vLLM or launch a model. Keep this
  # inside the dispatch block so tests can safely source the function definitions.
  if [[ "${ATOMESH_TRANSPORT_ONLY:-0}" == "1" ]]; then
    write_metadata
    exec python3 "${ATOMESH_SCRIPT_DIR}/pd_transport_probe.py"
  fi
  install_native_vllm
else
  apply_vllm_fork_overlay
fi
