#!/usr/bin/env bash
set -euo pipefail

# Self-contained local SWE-bench Lite pipeline:
#   1. mini-swe-agent generates patches through the live OpenAI-compatible API.
#   2. The official SWE-bench harness evaluates those patches with local Docker.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MINI_SWE_AGENT_VERSION="2.4.5"
HARNESS_VERSION="4.1.0"
DATASET_NAME="princeton-nlp/SWE-bench_Lite"
TASK_NAME="swebench_lite"

OUTPUT_DIR=""
MODEL_NAME_ARG=""
API_MODEL_ARG=""
API_BASE=""
RUN_ID=""
LIMIT="${EVAL_LIMIT:-}"
VENV="${SWEBENCH_VENV:-/tmp/atomesh-swebench-venv-${SLURM_JOB_ID:-local}}"
AGENT_WORKERS="${SWEBENCH_AGENT_WORKERS:-32}"
AGENT_STEP_LIMIT="${SWEBENCH_AGENT_STEP_LIMIT:-150}"
CASE_TIMEOUT="${SWEBENCH_CASE_TIMEOUT:-3600}"
AGENT_TIMEOUT="${SWEBENCH_AGENT_TIMEOUT:-21600}"
AGENT_EXIT_GRACE="${SWEBENCH_AGENT_EXIT_GRACE:-300}"
AGENT_CMD_TIMEOUT="${SWEBENCH_AGENT_CMD_TIMEOUT:-300}"
AGENT_RUNTIME_TIMEOUT="${SWEBENCH_AGENT_RUNTIME_TIMEOUT:-3600}"
AGENT_PULL_TIMEOUT="${SWEBENCH_AGENT_PULL_TIMEOUT:-900}"
WATCHDOG_POLL="${SWEBENCH_WATCHDOG_POLL:-30}"
SCORE_TIMEOUT="${SWEBENCH_SCORE_TIMEOUT:-7200}"
SCORE_WORKERS="${SWEBENCH_MAX_WORKERS:-4}"
INSTANCE_TIMEOUT="${SWEBENCH_EVAL_TIMEOUT:-900}"
NAMESPACE="${SWEBENCH_NAMESPACE:-swebench}"
DOCKER_EXECUTABLE="${SWEBENCH_DOCKER_EXECUTABLE:-${MSWEA_DOCKER_EXECUTABLE:-docker}}"
# A full 300-instance run pulls ~120 GiB of SWE-bench images onto the *host*
# daemon, which outlives this job. Slurm allocations are --exclusive, so the
# contention is sequential rather than concurrent: whatever this run leaves
# behind is inherited by the next job scheduled onto the same node. Refuse to
# start without headroom, and give the images back on exit. Set MIN_DISK_GB=0
# to skip the preflight check, PRUNE_IMAGES=false to keep the images warm for
# the next run on a dedicated node.
# NOTE: every instance gets its own `sweb.eval.*` image; the `sweb.env.*` and
# `sweb.base.*` layers underneath are shared by all instances of a project.
# shared base layer
#   └─ shared project env layer
#        ├─ instance A eval layer
#        ├─ instance B eval layer
#        └─ instance C eval layer
MIN_DISK_GB="${SWEBENCH_MIN_DISK_GB:-150}"
PRUNE_IMAGES="${SWEBENCH_PRUNE_IMAGES:-true}"

usage() {
  cat <<'EOF'
Usage:
  run_swebench_lite.sh \
    --output-dir DIR \
    --model-name NAME \
    [--api-model SERVED_NAME] \
    --api-base URL \
    --run-id ID \
    [--limit N|full] [--venv DIR] [--agent-workers N]
EOF
}

require_value() {
  if [[ "$#" -lt 2 || -z "${2:-}" ]]; then
    echo "ERROR: $1 requires a value" >&2
    usage >&2
    exit 2
  fi
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-dir)
      require_value "$@"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-name)
      require_value "$@"
      MODEL_NAME_ARG="$2"
      shift 2
      ;;
    --api-model)
      require_value "$@"
      API_MODEL_ARG="$2"
      shift 2
      ;;
    --api-base)
      require_value "$@"
      API_BASE="$2"
      shift 2
      ;;
    --run-id)
      require_value "$@"
      RUN_ID="$2"
      shift 2
      ;;
    --limit)
      require_value "$@"
      LIMIT="$2"
      shift 2
      ;;
    --venv)
      require_value "$@"
      VENV="$2"
      shift 2
      ;;
    --agent-workers)
      require_value "$@"
      AGENT_WORKERS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" || -z "${MODEL_NAME_ARG}" || -z "${API_BASE}" ]]; then
  echo "ERROR: --output-dir, --model-name, and --api-base are required" >&2
  usage >&2
  exit 2
fi
API_MODEL_ARG="${API_MODEL_ARG:-${MODEL_NAME_ARG}}"
RUN_ID="${RUN_ID:-atomesh-swebench-$(date +%Y%m%d%H%M%S)-$$}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "ERROR: --run-id may contain only letters, numbers, '.', '_', and '-'" >&2
  exit 2
fi

positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${name} must be a positive integer, got '${value}'" >&2
    exit 2
  fi
}

positive_integer "SWEBENCH_AGENT_WORKERS" "${AGENT_WORKERS}"
positive_integer "SWEBENCH_AGENT_STEP_LIMIT" "${AGENT_STEP_LIMIT}"
positive_integer "SWEBENCH_CASE_TIMEOUT" "${CASE_TIMEOUT}"
positive_integer "SWEBENCH_AGENT_TIMEOUT" "${AGENT_TIMEOUT}"
positive_integer "SWEBENCH_AGENT_EXIT_GRACE" "${AGENT_EXIT_GRACE}"
positive_integer "SWEBENCH_AGENT_CMD_TIMEOUT" "${AGENT_CMD_TIMEOUT}"
positive_integer "SWEBENCH_AGENT_RUNTIME_TIMEOUT" "${AGENT_RUNTIME_TIMEOUT}"
positive_integer "SWEBENCH_AGENT_PULL_TIMEOUT" "${AGENT_PULL_TIMEOUT}"
positive_integer "SWEBENCH_WATCHDOG_POLL" "${WATCHDOG_POLL}"
positive_integer "SWEBENCH_SCORE_TIMEOUT" "${SCORE_TIMEOUT}"
positive_integer "SWEBENCH_MAX_WORKERS" "${SCORE_WORKERS}"
positive_integer "SWEBENCH_EVAL_TIMEOUT" "${INSTANCE_TIMEOUT}"

if [[ ! "${MIN_DISK_GB}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "ERROR: SWEBENCH_MIN_DISK_GB must be a non-negative integer," \
    "got '${MIN_DISK_GB}'" >&2
  exit 2
fi
if [[ "${PRUNE_IMAGES}" != "true" && "${PRUNE_IMAGES}" != "false" ]]; then
  echo "ERROR: SWEBENCH_PRUNE_IMAGES must be 'true' or 'false'," \
    "got '${PRUNE_IMAGES}'" >&2
  exit 2
fi

case "${LIMIT}" in
  ""|full|FULL|0)
    LIMIT=""
    EXPECTED_INSTANCES=300
    ;;
  *)
    positive_integer "EVAL_LIMIT" "${LIMIT}"
    if (( LIMIT > 300 )); then
      echo "ERROR: EVAL_LIMIT cannot exceed SWE-bench Lite's 300 instances" >&2
      exit 2
    fi
    EXPECTED_INSTANCES="${LIMIT}"
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
GENERATION_DIR="${OUTPUT_DIR}/generation"
AGENT_OUTPUT_DIR="${GENERATION_DIR}/agent_out"
CONFIG_PATH="${GENERATION_DIR}/mini_swe_agent.yaml"
rm -rf "${GENERATION_DIR}"
mkdir -p "${AGENT_OUTPUT_DIR}"

if ! DOCKER_EXECUTABLE="$(command -v "${DOCKER_EXECUTABLE}")"; then
  echo "ERROR: Docker CLI '${DOCKER_EXECUTABLE}' is unavailable" >&2
  echo "       Mount the host Docker socket and CLI into the rank-0 container." >&2
  exit 2
fi
export MSWEA_DOCKER_EXECUTABLE="${DOCKER_EXECUTABLE}"

if ! "${DOCKER_EXECUTABLE}" version >/dev/null 2>&1; then
  echo "ERROR: cannot reach the local Docker daemon with ${DOCKER_EXECUTABLE}" >&2
  echo "       Check /var/run/docker.sock and its group permissions." >&2
  exit 2
fi

echo "[swebench] Docker daemon is available"
"${DOCKER_EXECUTABLE}" version --format \
  '[swebench] Docker client={{.Client.Version}} server={{.Server.Version}}'
"${DOCKER_EXECUTABLE}" system df || true

# Free space on the daemon's storage, not this container's filesystem. The
# daemon reports its root as a path in the *host* namespace, so this only works
# when pd_slurm_job.sh bind-mounted that path in at the same location. Returns
# non-zero when it did not, in which case the caller degrades to a warning.
docker_root_free_gb() {
  local root avail root_fs self_fs
  root="$("${DOCKER_EXECUTABLE}" info --format '{{.DockerRootDir}}' 2>/dev/null)" \
    || return 1
  [[ -n "${root}" && -d "${root}" ]] || return 1
  # A same-named empty directory in this container's image would make df report
  # the container rootfs -- a different disk entirely, wrong in both directions:
  # too high lets the run fill the daemon anyway, too low aborts a node that had
  # room. The bind mount always shows up as its own filesystem, so a root that
  # reports the same one as `/` is the image directory, not the daemon storage.
  # Compared by filesystem rather than by looking for `image/` and `containers/`
  # inside: the daemon root is commonly 0710 root:root, and the Spur path runs
  # the container as a normal user, which can df the path but not descend it.
  root_fs="$(df --output=source "${root}" 2>/dev/null | tail -n 1)"
  self_fs="$(df --output=source / 2>/dev/null | tail -n 1)"
  [[ -n "${root_fs}" && "${root_fs}" != "${self_fs}" ]] || return 1
  avail="$(df -BG --output=avail "${root}" 2>/dev/null | tail -n 1 | tr -dc '0-9')"
  [[ -n "${avail}" ]] || return 1
  printf '%s\n' "${avail}"
}

# Tags of the SWE-bench images the daemon holds, sorted for comm(1). Covers the
# prebuilt eval images plus the env/base layers the harness builds on a miss.
# Returns non-zero if the daemon could not be queried at all. That has to stay
# distinguishable from "queried fine, holds no sweb images": both produce no
# output, but the second is a legitimate empty set and the first is no answer,
# and every caller here diffs the output against a baseline. Only grep is
# allowed to fail quietly, because no matches is exactly the empty set.
swebench_image_refs() {
  local listing
  listing="$(
    "${DOCKER_EXECUTABLE}" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null
  )" || return 1
  printf '%s\n' "${listing}" \
    | grep -E '(^|/)sweb\.(eval|env|base)\.' \
    | sort -u \
    || true
}

if (( MIN_DISK_GB > 0 )); then
  if free_gb="$(docker_root_free_gb)"; then
    echo "[swebench] Docker root has ${free_gb} GiB free"
    if (( free_gb < MIN_DISK_GB )); then
      echo "ERROR: Docker root has ${free_gb} GiB free but this run needs about" \
        "${MIN_DISK_GB} GiB for SWE-bench images" >&2
      echo "       Reclaim space on the node, or lower/disable the check with" \
        "SWEBENCH_MIN_DISK_GB." >&2
      exit 2
    fi
  else
    echo "WARN: the Docker root is not visible from this container, so free" \
      "space cannot be measured; running without the preflight disk check." \
      "The run needs about ${MIN_DISK_GB} GiB and will fail late if the node" \
      "does not have it." >&2
  fi
fi

# Baseline for the prune on exit. A failed query here would write an empty file,
# which reads as "the node had no images" and would make the cleanup delete
# every sweb image on the node, including ones this run never pulled. Drop the
# file instead: the cleanup skips the prune when it is missing.
PRE_EXISTING_IMAGES_FILE="${GENERATION_DIR}/pre_existing_images.txt"
if ! swebench_image_refs > "${PRE_EXISTING_IMAGES_FILE}"; then
  rm -f "${PRE_EXISTING_IMAGES_FILE}"
  echo "WARN: could not list Docker images, so the pre-run image baseline is" \
    "unknown; images pulled by this run will be left on the node instead of" \
    "risking the removal of images that were already there." >&2
fi

if [[ ! -x "${VENV}/bin/python3" ]]; then
  rm -rf "${VENV}"
  python3 -m venv "${VENV}"
fi
PYTHON="${VENV}/bin/python3"
MINI_EXTRA="${VENV}/bin/mini-extra"

runtime_ready=0
if "${PYTHON}" - "${MINI_SWE_AGENT_VERSION}" "${HARNESS_VERSION}" <<'PY'
import importlib.metadata
import sys

expected = {
    "mini-swe-agent": sys.argv[1],
    "swebench": sys.argv[2],
}
for package, version in expected.items():
    if importlib.metadata.version(package) != version:
        raise SystemExit(1)
PY
then
  runtime_ready=1
fi

if [[ "${runtime_ready}" != "1" ]]; then
  echo "[swebench] installing pinned local evaluation runtime"
  "${PYTHON}" -m pip install --upgrade pip
  "${PYTHON}" -m pip install --no-cache-dir \
    "mini-swe-agent==${MINI_SWE_AGENT_VERSION}" \
    "swebench==${HARNESS_VERSION}"
fi

if [[ ! -x "${MINI_EXTRA}" ]]; then
  echo "ERROR: mini-swe-agent entrypoint was not installed at ${MINI_EXTRA}" >&2
  exit 1
fi
"${PYTHON}" "${SCRIPT_DIR}/patch_mini_swe_agent.py"

"${PYTHON}" - \
  "${CONFIG_PATH}" \
  "${API_BASE}" \
  "${API_MODEL_ARG}" \
  "${AGENT_STEP_LIMIT}" \
  "${CASE_TIMEOUT}" \
  "${AGENT_CMD_TIMEOUT}" \
  "${AGENT_RUNTIME_TIMEOUT}" \
  "${AGENT_PULL_TIMEOUT}" \
  "${DOCKER_EXECUTABLE}" \
  "${RUN_ID}" <<'PY'
import sys
from pathlib import Path

import yaml
from minisweagent.config import builtin_config_dir

(
    output_path,
    api_base,
    model_name,
    step_limit,
    case_timeout,
    command_timeout,
    runtime_timeout,
    pull_timeout,
    docker_executable,
    run_id,
) = sys.argv[1:]

default_path = builtin_config_dir / "benchmarks" / "swebench.yaml"
config = yaml.safe_load(default_path.read_text(encoding="utf-8"))

step_limit_int = int(step_limit)
guidance = f"""

<additional_critical_guidance>
- You have a hard budget of {step_limit_int} commands. Reproduce, fix, verify,
  and submit before that budget is exhausted. A fix that is not submitted
  receives no credit.
- Before submitting, run the focused test(s) covering the issue whenever the
  repository environment permits it.
- If the package cannot build or import after a few attempts, make the source
  fix and submit it instead of spending the whole budget repairing the image.
- `git diff` does not submit. Follow the exact submission command from the
  task instructions.
</additional_critical_guidance>
"""
config["agent"]["instance_template"] = (
    config["agent"]["instance_template"].rstrip() + guidance + "\n"
)
config["agent"]["step_limit"] = step_limit_int
config["agent"]["cost_limit"] = 0.0
config["agent"]["wall_time_limit_seconds"] = int(case_timeout)

environment = config["environment"]
environment.update(
    {
        "environment_class": "docker",
        "timeout": int(command_timeout),
        "pull_timeout": int(pull_timeout),
        "container_timeout": f"{int(runtime_timeout)}s",
        "executable": docker_executable,
        "run_args": [
            "--rm",
            "--label",
            f"atomesh.swebench.run={run_id}",
        ],
    }
)

provider_model = (
    model_name if model_name.startswith("openai/") else f"openai/{model_name}"
)
config["model"] = {
    "model_name": provider_model,
    "cost_tracking": "ignore_errors",
    "observation_template": config["model"]["observation_template"],
    "format_error_template": config["model"]["format_error_template"],
    "model_kwargs": {
        "api_base": api_base.rstrip("/"),
        "api_key": "dummy",
        "custom_llm_provider": "openai",
        "drop_params": True,
        "parallel_tool_calls": True,
        "temperature": 0.0,
    },
}

Path(output_path).write_text(
    yaml.safe_dump(config, default_flow_style=False, sort_keys=False),
    encoding="utf-8",
)
PY

cleanup_nested_docker_resources() {
  local rc=$?
  set +e
  local -a container_ids=()
  local -a new_images=()
  local -a stuck_images=()
  local current_images_file="${PRE_EXISTING_IMAGES_FILE}.now"
  if [[ -n "${mini_pid:-}" ]] && kill -0 "${mini_pid}" 2>/dev/null; then
    kill "${mini_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${mini_pid}" 2>/dev/null || true
  fi
  if [[ "${SWEBENCH_KEEP_TRAJECTORIES:-false}" != "true" ]]; then
    find "${AGENT_OUTPUT_DIR}" -type f -name '*.traj*' -delete \
      2>/dev/null || true
  fi
  # Two sources, because only the agent containers are ours to label: the label
  # is injected into the mini-swe-agent environment config, while the official
  # harness starts its own `sweb.eval.<instance>.<run_id>` scoring containers.
  # A scoring run cut short by SWEBENCH_SCORE_TIMEOUT leaves those behind, and
  # a leftover container pins the image the prune below tries to reclaim.
  mapfile -t container_ids < <(
    {
      "${DOCKER_EXECUTABLE}" ps -aq \
        --filter "label=atomesh.swebench.run=${RUN_ID}"
      "${DOCKER_EXECUTABLE}" ps -aq \
        --filter "name=sweb\.eval\..*${RUN_ID}"
    } 2>/dev/null | sort -u
  )
  if [[ "${#container_ids[@]}" -gt 0 ]]; then
    "${DOCKER_EXECUTABLE}" rm -f "${container_ids[@]}" >/dev/null 2>&1 || true
  fi
  if [[ "${PRUNE_IMAGES}" == "true" && -f "${PRE_EXISTING_IMAGES_FILE}" ]]; then
    # Into a file rather than <(swebench_image_refs): a process substitution
    # hides the exit status, so a failed query would look like an empty image
    # list and turn the whole baseline into "new" images to delete.
    if ! swebench_image_refs > "${current_images_file}"; then
      echo "WARN: could not list Docker images; skipping the prune. Images" \
        "pulled by this run stay on the Docker root." >&2
    else
      mapfile -t new_images < <(
        comm -13 "${PRE_EXISTING_IMAGES_FILE}" "${current_images_file}"
      )
      if [[ "${#new_images[@]}" -gt 0 ]]; then
        echo "[swebench] reclaiming ${#new_images[@]} image(s) pulled by this run"
        # Plain rmi, never rmi -f. If a container still references an image, rmi
        # refuses and says so on stderr; rmi -f would strip the tag and leave the
        # layers behind as a dangling image that nothing can reuse and only
        # `docker image prune` can find. Only stdout (one "Untagged:" line per
        # image) is dropped -- failures have to stay visible.
        "${DOCKER_EXECUTABLE}" rmi "${new_images[@]}" >/dev/null || true
        # Re-diff instead of trusting the rmi exit code: report what actually
        # went away, so a fully failed prune cannot read as a successful one.
        # A failed re-query is reported as unverified, never as success.
        if ! swebench_image_refs > "${current_images_file}"; then
          echo "WARN: could not list Docker images to confirm the prune;" \
            "${#new_images[@]} image(s) may still occupy the Docker root" >&2
        else
          mapfile -t stuck_images < <(
            comm -13 "${PRE_EXISTING_IMAGES_FILE}" "${current_images_file}"
          )
          if [[ "${#stuck_images[@]}" -gt 0 ]]; then
            echo "WARN: ${#stuck_images[@]} of ${#new_images[@]} image(s) could" \
              "not be reclaimed and still occupy the Docker root" >&2
          else
            echo "[swebench] reclaimed all ${#new_images[@]} image(s)"
          fi
        fi
        "${DOCKER_EXECUTABLE}" system df || true
      fi
    fi
    rm -f "${current_images_file}"
  fi
  return "${rc}"
}
trap cleanup_nested_docker_resources EXIT

slice_args=()
if [[ -n "${LIMIT}" ]]; then
  slice_args=(--slice "0:${LIMIT}")
fi

export MSWEA_COST_TRACKING=ignore_errors
echo "[swebench] generating ${EXPECTED_INSTANCES} SWE-bench Lite predictions"
echo "[swebench] agent workers=${AGENT_WORKERS} step_limit=${AGENT_STEP_LIMIT}"
echo "[swebench] case_timeout=${CASE_TIMEOUT}s generation_timeout=${AGENT_TIMEOUT}s"

"${MINI_EXTRA}" swebench \
  -c "${CONFIG_PATH}" \
  --subset lite \
  --split test \
  --environment-class docker \
  "${slice_args[@]}" \
  --workers "${AGENT_WORKERS}" \
  --output "${AGENT_OUTPUT_DIR}" \
  > >(tee "${GENERATION_DIR}/console.log") 2>&1 &
mini_pid=$!

prediction_count() {
  "${PYTHON}" - "${AGENT_OUTPUT_DIR}/preds.json" <<'PY' 2>/dev/null || echo 0
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(len(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else 0)
PY
}

deadline=$(( $(date +%s) + AGENT_TIMEOUT ))
grace_deadline=0
killed_after_complete=0
timed_out=0
while kill -0 "${mini_pid}" 2>/dev/null; do
  now="$(date +%s)"
  if (( now >= deadline )); then
    echo "ERROR: generation exceeded ${AGENT_TIMEOUT}s; stopping mini-swe-agent" >&2
    kill "${mini_pid}" 2>/dev/null || true
    sleep 5
    kill -9 "${mini_pid}" 2>/dev/null || true
    timed_out=1
    break
  fi

  done_count="$(prediction_count)"
  if (( done_count >= EXPECTED_INSTANCES )); then
    if (( grace_deadline == 0 )); then
      grace_deadline=$(( now + AGENT_EXIT_GRACE ))
      echo "[swebench] all predictions written; waiting for clean agent exit"
    elif (( now >= grace_deadline )); then
      echo "WARN: mini-swe-agent hung after writing all predictions; stopping it" >&2
      kill "${mini_pid}" 2>/dev/null || true
      sleep 5
      kill -9 "${mini_pid}" 2>/dev/null || true
      killed_after_complete=1
      break
    fi
  fi
  sleep "${WATCHDOG_POLL}"
done

mini_rc=0
wait "${mini_pid}" || mini_rc=$?
done_count="$(prediction_count)"
if (( done_count != EXPECTED_INSTANCES )); then
  echo "ERROR: generation produced ${done_count}/${EXPECTED_INSTANCES} predictions" >&2
  exit 1
fi
if (( timed_out == 1 )); then
  echo "ERROR: generation timed out even though all predictions were written" >&2
  exit 1
fi
if (( mini_rc != 0 && killed_after_complete == 0 )); then
  echo "WARN: mini-swe-agent exited ${mini_rc} after producing all predictions" >&2
fi

cp "${AGENT_OUTPUT_DIR}/preds.json" "${OUTPUT_DIR}/agent_preds.json"
echo "[swebench] scoring ${done_count} predictions with local Docker"

score_rc=0
timeout --signal=TERM --kill-after=60 "${SCORE_TIMEOUT}" \
  "${PYTHON}" "${SCRIPT_DIR}/swebench_score.py" \
  --predictions-file "${AGENT_OUTPUT_DIR}/preds.json" \
  --out-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_NAME_ARG}" \
  --run-id "${RUN_ID}" \
  --dataset-name "${DATASET_NAME}" \
  --task-name "${TASK_NAME}" \
  --max-workers "${SCORE_WORKERS}" \
  --instance-timeout "${INSTANCE_TIMEOUT}" \
  --namespace "${NAMESPACE}" \
  --expected-instances "${EXPECTED_INSTANCES}" \
  --harness-version "${HARNESS_VERSION}" \
  > >(tee "${OUTPUT_DIR}/swebench_score.log") 2>&1 || score_rc=$?

if (( score_rc != 0 )); then
  echo "ERROR: official SWE-bench scoring failed with exit code ${score_rc}" >&2
  exit "${score_rc}"
fi

if [[ "${SWEBENCH_KEEP_TRAJECTORIES:-false}" != "true" ]]; then
  find "${AGENT_OUTPUT_DIR}" -type f -name '*.traj*' -delete 2>/dev/null || true
fi

echo "[swebench] local SWE-bench Lite evaluation completed: ${OUTPUT_DIR}"
