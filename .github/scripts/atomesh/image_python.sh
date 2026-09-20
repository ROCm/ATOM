#!/usr/bin/env bash
# Source this in CI steps that must use Python/dependencies from the ATOM image.
# Host commands only orchestrate Docker/Slurm; no environment is installed.
python3() {
  local image="${ATOMESH_PYTHON_IMAGE:?ATOMESH_PYTHON_IMAGE is required}"
  local name directory
  local -A mounted=()
  local -a args=(run --rm -i --user "$(id -u):$(id -g)"
    -v "${PWD}:${PWD}" -w "${PWD}"
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="${PWD}:${PWD}/tests"
    -e HIP_VISIBLE_DEVICES=)
  for name in $(compgen -e); do
    case "${name}" in
      ATOMESH_*|RESULT_DIR|MODEL_NAME|CASE_NAME|SUITE|INPUT_ATOMESH_IMAGE|USER)
        args+=(-e "${name}") ;;
    esac
  done
  for name in GITHUB_OUTPUT GITHUB_STEP_SUMMARY; do
    if [[ -n "${!name:-}" ]]; then
      args+=(-e "${name}")
      directory="$(dirname "${!name}")"
      if [[ -z "${mounted[$directory]:-}" ]]; then
        args+=(-v "${directory}:${directory}")
        mounted["${directory}"]=1
      fi
    fi
  done
  command docker "${args[@]}" --entrypoint python3 "${image}" "$@"
}
