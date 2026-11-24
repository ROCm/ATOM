#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Source this file, then call
#
#   example_tokenizer <llama2|llama31|qwen3> <dir>
#
# Tokenizer files are not distributed with this repository. When a user-supplied
# <dir> already holds tokenizer_config.json, it is left untouched. Otherwise the
# files are downloaded from a pinned public Hugging Face repository.

ensure_tokenizer() {  # <dir> <hf-repo> <revision> <file>...
    local dir="$1" repo="$2" revision="$3" f complete=1
    local expected="${repo}@${revision}"
    local marker="${dir}/.atomrl-tokenizer-revision"
    shift 3
    for f in "$@"; do
        [ -s "${dir}/${f}" ] || complete=0
    done
    if [ "${complete}" -eq 1 ]; then
        # A complete user-supplied tokenizer has no marker and remains untouched.
        [ ! -f "${marker}" ] && return 0
        [ "$(cat "${marker}")" = "${expected}" ] && return 0
    fi

    local auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    local parent stage backup=""
    parent="$(dirname "${dir}")"
    mkdir -p "${parent}"
    stage="$(mktemp -d "${dir}.tmp.XXXXXX")" || return 1
    for f in "$@"; do
        echo "Fetching ${repo}/${f} into ${stage}"
        if ! curl -fsSL "${auth[@]}" -o "${stage}/${f}.part" \
                "https://huggingface.co/${repo}/resolve/${revision}/${f}"; then
            rm -rf "${stage}"
            echo "ERROR: could not download ${f} from ${repo}." >&2
            echo "  If the repo is gated, export HF_TOKEN; otherwise point the" >&2
            echo "  tokenizer path at a local directory holding: $*" >&2
            return 1
        fi
        mv "${stage}/${f}.part" "${stage}/${f}"
    done
    printf '%s\n' "${expected}" > "${stage}/.atomrl-tokenizer-revision"

    if [ -e "${dir}" ]; then
        backup="${dir}.backup.$$"
        if ! mv "${dir}" "${backup}"; then
            rm -rf "${stage}"
            return 1
        fi
    fi
    if ! mv "${stage}" "${dir}"; then
        [ -n "${backup}" ] && mv "${backup}" "${dir}"
        rm -rf "${stage}"
        return 1
    fi
    if [ -n "${backup}" ]; then
        rm -rf "${backup}"
    fi
    return 0
}

example_tokenizer() {  # <llama2|llama31|qwen3> <dir>
    case "$1" in
        llama2)  ensure_tokenizer "$2" NousResearch/Llama-2-7b-hf \
                     8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
                     config.json tokenizer.json tokenizer.model tokenizer_config.json ;;
        llama31) ensure_tokenizer "$2" NousResearch/Meta-Llama-3.1-8B \
                     1f47e50cdbe801ad8a5174156ec3a0655108fb9f \
                     config.json special_tokens_map.json tokenizer.json tokenizer_config.json ;;
        qwen3)   ensure_tokenizer "$2" Qwen/Qwen3-8B-Base \
                     49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
                     config.json generation_config.json merges.txt tokenizer.json \
                     vocab.json tokenizer_config.json ;;
        *) echo "ERROR: no example tokenizer named '$1'" >&2; return 1 ;;
    esac
}
