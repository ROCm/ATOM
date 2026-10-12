#!/usr/bin/env bash
# Called by the C48 Spur workload before Docker starts the ordinary-user service.
c48_append_port_mount() {
  local c48_source="$1/.github/benchmark/c48-model-gap/moriio_common.py"
  local c48_target="/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"
  local c48_digest
  [[ -f "${c48_source}" && -r "${c48_source}" ]] || return 2
  c48_digest="$(sha256sum "${c48_source}")" || return $?
  if [[ "${c48_digest%% *}" != "23cd517ed12a0102bbedd0aeb755fae7a865f8d5f56bb2bb1f59e02d0d804268" ]]; then
    echo "ERROR: C48 startup source mount has an unexpected SHA256" >&2
    return 2
  fi
  docker_args+=(--mount "type=bind,source=${c48_source},target=${c48_target},readonly")
}
