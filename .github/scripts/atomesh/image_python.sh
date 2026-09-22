#!/usr/bin/env bash
# Source this in CI steps that must use Python/dependencies from the ATOM image.
# Host commands only orchestrate Docker/Slurm; no environment is installed.
if [[ -z "${ATOMESH_DOCKER_USE_SUDO:-}" ]]; then
  if command docker info >/dev/null 2>&1; then
    export ATOMESH_DOCKER_USE_SUDO=0
  elif command -v sudo >/dev/null 2>&1 && command sudo -n docker info >/dev/null 2>&1; then
    export ATOMESH_DOCKER_USE_SUDO=1
  else
    echo "ERROR: this CI runner cannot access Docker directly or through existing passwordless sudo; no environment or permissions were changed" >&2
    return 1
  fi
fi

docker() {
  if [[ "${ATOMESH_DOCKER_USE_SUDO}" == "1" ]]; then
    command sudo -n docker "$@"
  else
    command docker "$@"
  fi
}
# The setup script uses the same selected Docker access for its build container.
export -f docker

python3() {
  local image="${ATOMESH_PYTHON_IMAGE:?ATOMESH_PYTHON_IMAGE is required}"
  local name directory
  local -A mounted=()
  local -a args=(run --rm -i --user "$(id -u):$(id -g)"
    -v "${PWD}:${PWD}" -w "${PWD}"
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="${PWD}:${PWD}/tests"
    -e USER="$(id -un 2>/dev/null || id -u)"
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/atomesh-ci-inductor
    -e XDG_CACHE_HOME=/tmp/atomesh-ci-cache
    -e HIP_VISIBLE_DEVICES=)
  for name in $(compgen -e); do
    case "${name}" in
      ATOMESH_*|RESULT_DIR|MODEL_NAME|CASE_NAME|SUITE|INPUT_ATOMESH_IMAGE)
        args+=(-e "${name}=${!name}") ;;
    esac
  done
  for name in GITHUB_OUTPUT GITHUB_STEP_SUMMARY; do
    if [[ -n "${!name:-}" ]]; then
      args+=(-e "${name}=${!name}")
      directory="$(dirname "${!name}")"
      if [[ -z "${mounted[$directory]:-}" ]]; then
        args+=(-v "${directory}:${directory}")
        mounted["${directory}"]=1
      fi
    fi
  done
  docker "${args[@]}" --entrypoint python3 "${image}" "$@"
}
