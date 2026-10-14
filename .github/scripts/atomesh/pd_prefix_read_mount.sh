#!/usr/bin/env bash
# Product connector mount: distinct from pd_c48_port_mount.sh's startup fix.
prefix_read_append_mount() {
  local repo="$1"
  python3 "${repo}/.github/scripts/atomesh/pd_prefix_read.py" verify-bundle --repo "${repo}" >/dev/null || return $?
  docker_args+=(--mount "type=bind,source=${repo}/.github/benchmark/prefix-read-fixed/moriio_connector.py,target=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py,readonly")
  docker_args+=(--mount "type=bind,source=${repo}/.github/benchmark/prefix-read-fixed/runtime/factory.py,target=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/factory.py,readonly")
  docker_args+=(--mount "type=bind,source=${repo}/.github/benchmark/prefix-read-fixed/runtime/core.py,target=/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py,readonly")
  docker_args+=(--mount "type=bind,source=${repo}/.github/benchmark/prefix-read-fixed/runtime/routers.py,target=/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/launchers/api_server/routers.py,readonly")
}
