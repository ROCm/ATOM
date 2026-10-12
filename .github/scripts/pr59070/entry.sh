#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ATOMESH_SCRIPT_DIR="${scripts}/../atomesh"
export RUNTIME_LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${RUNTIME_LOG_DIR}"
while IFS= read -r name; do
  export "${name#ATOMESH_ENV_}=${!name}"
done < <(compgen -A variable ATOMESH_ENV_)

export host_ip="127.0.0.1"
export ATOMESH_SERVICE_PORT_OFFSET=0
# Reuse the tested source-build installer; neither server nor connector starts here.
source "${ATOMESH_SCRIPT_DIR}/pd_server_vllm.sh"
eval_python="/tmp/atomesh-native-venv/bin/python"
uv pip install --python "${eval_python}" \
  'lm-eval[api]==0.4.13' 'transformers==5.17.0' 'tokenizers>=0.23.2' \
  > "${RUNTIME_LOG_DIR}/eval-install.log" 2>&1
export PYTHONPATH="${scripts}"
exec "${eval_python}" -u "${scripts}/run.py"
