#!/usr/bin/env bash
# ATOMesh benchmark CI entrypoint; submit one bounded, model-free TP4 job.
set -euo pipefail
export PATH="/usr/local/slurm-24.05.5.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

if [[ -z "${REPRO_IMAGE}" ]]; then
  echo "::error::No ATOM image was resolved for minimax_prefill_repro verify."
  exit 1
fi

current_user="$(id -un 2>/dev/null || id -u)"
log_root="${ATOMESH_LOG_ROOT//\$\{USER\}/${current_user}}"
log_root="${log_root//\$USER/${current_user}}"
if [[ "$(hostname -s)" == "pit2-vm-amd-xl-02" ]]; then
  log_root="/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG"
fi
run_dir="${log_root%/}/minimax-prefill-repro-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
mkdir -p "${run_dir}"
chmod 0777 "${run_dir}"
echo "run_dir=${run_dir}" >> "${GITHUB_OUTPUT}"
workspace_out="${GITHUB_WORKSPACE}/minimax-prefill-repro-out"
mkdir -p "${workspace_out}"
echo "workspace_out=${workspace_out}" >> "${GITHUB_OUTPUT}"

CURRENT_USER="${current_user}"
ATOMESH_CELL_ID="minimax-prefill-repro"
RESULT_DIR="${run_dir}"
SLURM_JOB_NAME="minimax-prefill-repro-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
SLURM_OUTPUT="${run_dir}/slurm-%j.out"
SLURM_ERROR="${run_dir}/slurm-%j.err"
SLURM_CANCEL_HELPER="${run_dir}/slurm-cancel.sh"
SLURM_LOG_POLL_INTERVAL="${SLURM_LOG_POLL_INTERVAL:-15}"
SLURM_ACCOUNT="${ATOMESH_SLURM_ACCOUNT:-}"
SLURM_PARTITION="${ATOMESH_SLURM_PARTITION:-}"
SLURM_QOS=""
USES_SPUR_CONTROLLER=0
SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-}"
if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
  USES_SPUR_CONTROLLER=1
  SPUR_CONTROLLER_ADDR="http://134.199.196.72:6817"
elif [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi355-crusoe" ]]; then
  USES_SPUR_CONTROLLER=1
  SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817"
  SLURM_ACCOUNT="amd-aifw-dev"
  SLURM_PARTITION=""
  SLURM_QOS="amd-aifw-dev-qos"
fi
if [[ "$(hostname -s)" == "pit2-vm-amd-xl-02" ]]; then
  SLURM_PARTITION="amd-spur"
  SLURM_QOS="amd-frameworks-qos"
fi
if [[ "${SLURM_ACCOUNT}" == "amd-atom" ]]; then
  SLURM_QOS="amd-atom-qos"
fi
source "${GITHUB_WORKSPACE}/.github/scripts/slurm_submit_helpers.sh"
detect_slurm_backend
install_slurm_cancel_traps

if [[ "${1:-}" == "--preflight" ]]; then
  source "${GITHUB_WORKSPACE}/.github/scripts/atomesh/cleanup_minimax_prefill_repro.sh"
  exec > >(tee "${workspace_out}/scheduler-preflight.log") 2>&1
  echo "account=${SLURM_ACCOUNT} partition=${SLURM_PARTITION} qos=${SLURM_QOS}"
  cleanup_previous_minimax_repro "${REPRO_CLEANUP_RUN_ID:-}"
  echo 'GPU allocation limit: 20 minutes after scheduling. Actions/queueing ceiling: 24 hours.'
  exit 0
fi
if [[ "$#" != 0 ]]; then
  echo 'ERROR: expected no arguments or --preflight' >&2
  exit 1
fi

# GPU nodes must see the checkout; mirror onto NFS (same pattern as
# atomesh-cicd-mi350 shared workspace), even on TW where the Actions
# workspace may not be node-visible.
shared_repo="${run_dir}/repo"
rm -rf "${shared_repo}"
mkdir -p "${shared_repo}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${GITHUB_WORKSPACE}/" "${shared_repo}/"
else
  tar -C "${GITHUB_WORKSPACE}" -cf - . | tar -C "${shared_repo}" -xf -
fi
chmod -R a+rX "${shared_repo}"
chmod +x "${shared_repo}/.github/scripts/atomesh/"*.sh \
  "${shared_repo}/.github/scripts/"*.sh 2>/dev/null || true

export REPRO_REPO_ROOT="${shared_repo}"
export REPRO_OUT_DIR="${run_dir}/out"
mkdir -p "${REPRO_OUT_DIR}"
chmod 0777 "${REPRO_OUT_DIR}"


rank_script="${run_dir}/run-minimax-prefill-repro-rank.sh"
cat > "${rank_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export REPRO_IMAGE="${REPRO_IMAGE}"
export REPRO_REPO_ROOT="${REPRO_REPO_ROOT}"
export REPRO_OUT_DIR="${REPRO_OUT_DIR}"
exec bash "${REPRO_REPO_ROOT}/.github/scripts/atomesh/run_minimax_prefill_repro.sh"
EOF
chmod +x "${rank_script}"

batch_script="${run_dir}/minimax-prefill-repro.sbatch.sh"
cat > "${batch_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
set +e
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 "${rank_script}"
srun_rc=\$?
set -e
printf '%s\n' "\${srun_rc}" > "${run_dir}/slurm-job.rc.tmp"
mv "${run_dir}/slurm-job.rc.tmp" "${run_dir}/slurm-job.rc"
exit "\${srun_rc}"
EOF
chmod +x "${batch_script}"

node_count=1
nodes="${ATOMESH_SINGLE_NODE:-}"
if [[ "${nodes}" == "auto" ]]; then
  nodes=""
fi
sbatch_cmd=(
  sbatch
  --parsable
  --exclusive
  --export=ALL
  --job-name "${SLURM_JOB_NAME}"
  --nodes 1
  --ntasks 1
  --ntasks-per-node 1
  --time 00:20:00
  --output "${SLURM_OUTPUT}"
  --error "${SLURM_ERROR}"
)
if [[ -n "${nodes}" ]]; then
  slurm_node_selection_args "${nodes}" "${node_count}"
  sbatch_cmd+=("${SLURM_NODE_SELECTION_ARGS[@]}")
fi
if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
  sbatch_cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
else
  [[ -z "${SLURM_ACCOUNT}" ]] || sbatch_cmd+=(--account "${SLURM_ACCOUNT}")
  [[ -z "${SLURM_PARTITION}" ]] || sbatch_cmd+=(--partition "${SLURM_PARTITION}")
  [[ -z "${SLURM_QOS}" ]] || sbatch_cmd+=(--qos "${SLURM_QOS}")
  sbatch_cmd+=(--gres gpu:4)
  if [[ "${USES_SPUR_CONTROLLER}" == "1" && -n "${SPUR_CONTROLLER_ADDR}" ]]; then
    sbatch_cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi
fi
sbatch_cmd+=("${batch_script}")

printf ' %q' "${sbatch_cmd[@]}"
echo
write_slurm_cancel_helper ""
set +e
sbatch_output="$("${sbatch_cmd[@]}")"
sbatch_rc=$?
set -e
echo "${sbatch_output}"
if [[ "${sbatch_rc}" -ne 0 ]]; then
  echo "sbatch submit exit code: ${sbatch_rc}"
  exit "${sbatch_rc}"
fi

JOB_ID="$(parse_sbatch_job_id "${sbatch_output}")"
SLURM_JOB_ACTIVE=1
echo "${JOB_ID}" | tee "${run_dir}/slurm-job-id"
write_slurm_cancel_helper "${JOB_ID}"
set_slurm_job_log_paths "${JOB_ID}"
SLURM_STATUS_DIR="${run_dir}"
SLURM_STATUS_RANKS=1
monitor_slurm_job "${JOB_ID}"

# Same fail-closed pattern as rdma-smoke: monitor_slurm_job only waits
# for a terminal state; read_slurm_exit_code turns FAILED into a
# non-zero Actions step exit (run 38520975962 stayed green otherwise).
SLURM_ACCOUNTING_TIMEOUT="${SLURM_ACCOUNTING_TIMEOUT:-180}"
read_slurm_exit_code "${JOB_ID}"
read_slurm_status_files "${run_dir}" 1
SLURM_JOB_ACTIVE=0
echo "slurm_state=${SLURM_STATE}"
echo "slurm_exit_code=${SLURM_EXIT_CODE}"
echo "slurm job exit code: ${SLURM_JOB_RC}"

echo "=== minimax_prefill_repro logs ==="
tail -n 80 "${REPRO_OUT_DIR}/rank.log" 2>/dev/null || true
if [[ -f "${REPRO_OUT_DIR}/rank-rc" ]]; then
  rank_rc="$(cat "${REPRO_OUT_DIR}/rank-rc")"
  echo "rank_rc=${rank_rc}"
  if [[ "${rank_rc}" != "0" ]]; then
    echo "::error::minimax_prefill_repro rank exited ${rank_rc}"
    exit "${rank_rc}"
  fi
else
  echo "::error::minimax_prefill_repro rank-rc missing"
  exit 1
fi
if [[ "${SLURM_JOB_RC}" != "0" ]]; then
  echo "::error::minimax_prefill_repro Slurm job ${JOB_ID} failed with rc=${SLURM_JOB_RC} state=${SLURM_STATE}"
fi
exit "${SLURM_JOB_RC}"
