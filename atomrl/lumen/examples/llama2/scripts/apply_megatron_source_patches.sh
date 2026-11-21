#!/usr/bin/env bash
# Apply Lumen SOURCE patches to a Megatron-LM checkout (torch-less entry).
#
# Usage (source this file, do not execute directly):
#   source examples/llama2/scripts/apply_megatron_source_patches.sh
#   apply_lumen_megatron_source_patches /workspace/megatron_lm llama
#   apply_lumen_megatron_source_patches /workspace/megatron_lm llama,lora
#
# Tags use comma-separated OR semantics (see examples/dsv4/patch_megatron_source.py).

apply_lumen_megatron_source_patches() {
    local megatron_root="${1:?megatron root required}"
    local tags="${2:-llama}"
    local lumen_root="${LUMEN_ROOT:-${LUMEN_DIR:-/workspace/Lumen}}"
    local patch_script="${lumen_root}/examples/dsv4/patch_megatron_source.py"

    if [[ ! -f "${patch_script}" ]]; then
        echo "[patch] ERROR: missing ${patch_script}" >&2
        return 1
    fi
    if [[ ! -d "${megatron_root}" ]]; then
        echo "[patch] ERROR: Megatron checkout not found: ${megatron_root}" >&2
        return 1
    fi

    echo "[patch] applying SOURCE tags=${tags} to ${megatron_root}"
    PYTHONPATH="${lumen_root}" python3 "${patch_script}" "${megatron_root}" --tag "${tags}"
}

resolve_megatron_root() {
    local candidate
    for candidate in \
        "${MEGATRON_ROOT:-}" \
        "${MEGATRON_PATH:-}" \
        /workspace/megatron_lm \
        /workspace/Megatron-LM; do
        [[ -n "${candidate}" && -f "${candidate}/megatron/core/__init__.py" ]] || continue
        printf '%s\n' "${candidate}"
        return 0
    done
    return 1
}
