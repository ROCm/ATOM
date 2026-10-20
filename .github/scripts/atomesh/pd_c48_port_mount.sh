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
  # The loading fix is fixed to this experiment branch and mounted read-only.
  local c48_fix_source c48_fix_target c48_fix_digest
  c48_fix_source="$1/.github/benchmark/c48-model-gap/online-quant-fix/kda.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/amd/kda.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "8318855c6eede76b8072bc7ba744523a331c6ebe2bd3d145b6e6e3c1722518ed" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  c48_fix_source="$1/.github/benchmark/c48-model-gap/online-quant-fix/meta.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/reload/meta.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "3bb991f43b5b53fab7383ec89c0ef5add4d035dc7312bbd55a4bebe7291ab56d" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  [[ "${ATOMESH_ENV_ATOMESH_C48_CASE:-}" == "V-QD" ]] || return 0
  c48_fix_source="$1/.github/benchmark/c48-model-gap/cpu-state-refresh/manager.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/v1/simple_kv_offload/manager.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "d3e3dd53a9301438e11e549a371c28c21cc72674994f0123de7d9ec1e937760d" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  # Identical FQ backend to Slurm 9070; only Prefill enables the FP8 path.
  c48_fix_source="$1/.github/benchmark/c48-model-gap/mla-prefill-fp8/aiter_flash_attn.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "5012ac98ac714a9d28743345250bb587925b6e411b8756eb99f514d9706f26ae" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  c48_fix_source="$1/.github/benchmark/c48-model-gap/prefill-fairness/scheduler.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "625c3c17e331463847d594d17087d8ff25c459b96b558a8602b884cc7f27c3d2" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
}
