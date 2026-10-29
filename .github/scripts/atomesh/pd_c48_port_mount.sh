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
  if [[ "${ATOMESH_ENV_ATOMESH_C48_CASE:-}" == *-MC ]]; then
    # MooncakeStoreConnector reports stats without a Prometheus exporter; upstream MultiConnector asserts on it.
    c48_fix_source="$1/.github/benchmark/c48-model-gap/mooncake-multi-connector/multi_connector.py"
    c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py"
    [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
    c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
    [[ "${c48_fix_digest%% *}" == "397e20f11593dfa652d7338a848945db0ab90c4f5424c9b129068278518c5646" ]] || return 2
    docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  fi
  local c48_kda_dir=online-quant-fix c48_kda_sha=8318855c6eede76b8072bc7ba744523a331c6ebe2bd3d145b6e6e3c1722518ed
  if [[ "${ATOMESH_ENV_ATOMESH_C48_CASE:-}" == *-INPAD || "${ATOMESH_ENV_ATOMESH_C48_CASE:-}" == *-INPAD-* ]]; then
    c48_kda_dir=inproj-pad128
    c48_kda_sha=26b42764a2f4a248cccaa360435659d5f6bfa9e784dc91c28a3d6a1ea48273da
  fi
  c48_fix_source="$1/.github/benchmark/c48-model-gap/${c48_kda_dir}/kda.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/amd/kda.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "${c48_kda_sha}" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  c48_fix_source="$1/.github/benchmark/c48-model-gap/online-quant-fix/meta.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/reload/meta.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "3bb991f43b5b53fab7383ec89c0ef5add4d035dc7312bbd55a4bebe7291ab56d" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  local c48_fix_dir c48_fix_sha
  case "${ATOMESH_ENV_ATOMESH_C48_CASE:-}" in
    V-QD-CPUREF)
      c48_fix_dir=cpu-lru-refresh
      c48_fix_sha=7bcfccfdb1cf57c70300848abc861985007627f87ad1d7cb7d12530349b97dea ;;
    V-QD-CPUREF2|V-QD-CPUREF2-LPT3072|V-QD-CPUREF2-LPT6144|V-QD-CPUREF2-FP8ATTN|V-QD-CPUREF2-FP8ATTN-PROFILE|V-QD-CPUREF2-FP8ATTN-FQ|V-QD-CPUREF2-FP8ATTN-FQ2|V-QD-CPUREF2-FP8ATTN-FQ2-PQ|V-QD-CPUREF2-FP8ATTN-FQ2-PQ-LDFP4|V-QD-CPUREF2-FP8ATTN-FQ2-PQ-LDFP4-INPAD|V-QD-CPUREF2-FP8ATTN-FQ2-PQ-LDFP4-INPAD-WARM|V-QD-CPUREF2-FP8ATTN-FQ3|V-QD-CPUREF2-FP8ATTN-FQ2-WARM|V-QD-CPUREF2-FP8ATTN-FQ2-WARM1|V-QD-CPUREF2-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM)
      c48_fix_dir=cpu-lru-refresh-v2
      c48_fix_sha=76e038dc1bcfdf0fbe95e6a87844e8409ac45dd7f85f115267bc85d6e84f739c ;;
    V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM|V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5|V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-PNT|V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT|V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-SW-C*|V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT-SWB-C*)
      c48_fix_dir=cpu-lru-refresh-v3
      c48_fix_sha=18da014b1eed073f6b227b92fc0163baa1e7a6a5fbcbcb31364a4696fa1cbd3f ;;
    *) return 0 ;;
  esac
  c48_fix_source="$1/.github/benchmark/c48-model-gap/${c48_fix_dir}/manager.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/v1/simple_kv_offload/manager.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "${c48_fix_sha}" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  [[ "${ATOMESH_ENV_ATOMESH_C48_CASE}" == V-QD-CPUREF[23]-FP8ATTN* ]] || return 0
  # Inert unless VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8=1, which only the Prefill role sets.
  c48_fix_dir=mla-prefill-fp8 c48_fix_sha=5012ac98ac714a9d28743345250bb587925b6e411b8756eb99f514d9706f26ae
  if [[ "${ATOMESH_ENV_ATOMESH_C48_CASE}" == V-QD-CPUREF[23]-FP8ATTN-FQ3* ]]; then
    c48_fix_dir=mla-prefill-fp8-fusedq
    c48_fix_sha=d0b8caee05f50a132c7f6915fe0dd3d594f7eef8c5e8f95af6fe39ea24e230c3
  fi
  c48_fix_source="$1/.github/benchmark/c48-model-gap/${c48_fix_dir}/aiter_flash_attn.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "${c48_fix_sha}" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  [[ "${ATOMESH_ENV_ATOMESH_C48_CASE}" == V-QD-CPUREF[23]-FP8ATTN-FQ[23]-PQ-LDFP4* ]] || return 0
  # Inert unless the role's online config selects routed_expert_down_proj (Prefill only).
  c48_fix_dir=latent-down-mxfp4
  c48_fix_sha=aa52cf11f1cd1914c2940515b1b3cfe8cc2e06df899811d38090af6581f3ffaa
  if [[ "${ATOMESH_ENV_ATOMESH_C48_CASE}" == *-PP8-MC ]]; then
    c48_fix_dir=pp-aux-hidden
    c48_fix_sha=5620f0ecf2c82a522f65f348238e08d28125382b82416ae5afc68ffb7be15bd4
  fi
  c48_fix_source="$1/.github/benchmark/c48-model-gap/${c48_fix_dir}/linear.py"
  c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/amd/linear.py"
  [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
  c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
  [[ "${c48_fix_digest%% *}" == "${c48_fix_sha}" ]] || return 2
  docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
  if [[ "${ATOMESH_ENV_ATOMESH_C48_CASE}" == *-PP8-MC ]]; then
    local c48_pp_name c48_pp_sha
    while read -r c48_pp_name c48_pp_sha; do
      c48_fix_source="$1/.github/benchmark/c48-model-gap/pp-mooncake/${c48_pp_name}"
      c48_fix_target="/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/${c48_pp_name}"
      [[ -f "${c48_fix_source}" && -r "${c48_fix_source}" ]] || return 2
      c48_fix_digest="$(sha256sum "${c48_fix_source}")" || return $?
      [[ "${c48_fix_digest%% *}" == "${c48_pp_sha}" ]] || return 2
      docker_args+=(--mount "type=bind,source=${c48_fix_source},target=${c48_fix_target},readonly")
    done <<'C48_PP_MOONCAKE_FILES'
mooncake_connector.py 5b702ae788d982993a93fa6b9fbde30d3403fdd72630850814f9c214a8227fe2
dcp_transfer.py 706b4ebc441a2a0f3ae2eaf9de432af0738b129033bc8278da55886dc300992e
dcp_staging.py 253741ba6c6b8537a0678152d7dfd57e6da8139392f73148c8fcbe84727a49a5
C48_PP_MOONCAKE_FILES
  fi
}
