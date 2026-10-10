# shellcheck shell=bash

install_mooncake() {
  local wheel_name=mooncake_transfer_engine_rocm-0.3.13-cp312-cp312-manylinux_2_35_x86_64.whl
  local wheel="${ATOMESH_SCRIPT_DIR}/mooncake-dist/${wheel_name}"
  printf '%s  %s\n' 7dfe1f9acec16843868dde28dcb1d3903aafeaf8b92a657239055581a8c44df6 "${wheel}" | sha256sum -c -
  uv pip install --python /tmp/atomesh-native-venv/bin/python --no-deps --reinstall "${wheel}"
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  python3 "${ATOMESH_SCRIPT_DIR}/mooncake-dist/verify_rocm_wheel.py" \
    > "${RUNTIME_LOG_DIR}/mooncake-package-rank-${NODE_RANK}.log" 2>&1
  python3 - <<'PY'
import mooncake.engine
import openai_harmony
import xgrammar
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import MooncakeConnector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
assert mooncake.engine.SUPPORT_HIP, "Mooncake was not compiled for HIP"
print(f"[mooncake] connector imports OK: {MooncakeConnector.__name__}, {MooncakeStoreConnector.__name__}")
PY
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=$((18998 + ATOMESH_SERVICE_PORT_OFFSET))
  export MOONCAKE_MASTER_PORT=$((19051 + ATOMESH_SERVICE_PORT_OFFSET))
}

prepare_mooncake_store() {
  [[ "$1" == "prefill" ]] || return 0
  export MOONCAKE_CONFIG_PATH="${RUNTIME_LOG_DIR}/mooncake-store-rank-${NODE_RANK}.json"
  python3 "${ATOMESH_SCRIPT_DIR}/pd_mooncake_config.py" store > "${MOONCAKE_CONFIG_PATH}"
  local master
  master="$(python3 -c 'import pathlib, mooncake; print(pathlib.Path(mooncake.__file__).parent / "mooncake_master")')"
  start_logged_process mooncake_master_pid "${RUNTIME_LOG_DIR}/mooncake-master.log" \
    "${master}" --port "${MOONCAKE_MASTER_PORT}"
  python3 - "${MOONCAKE_MASTER_PORT}" <<'PY'
import socket
import sys
import time
for attempt in range(60):
    try:
        with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("Mooncake master did not listen within 60 seconds")
PY
}
