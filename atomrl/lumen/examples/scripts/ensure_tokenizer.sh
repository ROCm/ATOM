#!/usr/bin/env bash
# Source this file, then call
#
#   example_tokenizer <llama2|llama31|qwen3> <dir>
#
# Tokenizer files are not distributed with this repository. When a user-supplied
# <dir> already holds tokenizer_config.json, it is left untouched. Otherwise the
# files are downloaded from the model's Hugging Face repo. The meta-llama repos
# are gated: export HF_TOKEN when a local tokenizer is absent.

ensure_tokenizer() {  # <dir> <hf-repo> <revision> <file>...
    local dir="$1" repo="$2" revision="$3" f complete=1
    shift 3
    for f in "$@"; do
        [ -s "${dir}/${f}" ] || complete=0
    done
    [ "${complete}" -eq 1 ] && return 0
    local auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    mkdir -p "${dir}"
    for f in "$@"; do
        [ -s "${dir}/${f}" ] && continue
        echo "Fetching ${repo}/${f} into ${dir}"
        if ! curl -fsSL "${auth[@]}" -o "${dir}/${f}.part" \
                "https://huggingface.co/${repo}/resolve/${revision}/${f}"; then
            rm -f "${dir}/${f}.part"
            echo "ERROR: could not download ${f} from ${repo}." >&2
            echo "  If the repo is gated, export HF_TOKEN; otherwise point the" >&2
            echo "  tokenizer path at a local directory holding: $*" >&2
            return 1
        fi
        mv "${dir}/${f}.part" "${dir}/${f}"
    done
}

example_tokenizer() {  # <llama2|llama31|qwen3> <dir>
    case "$1" in
        llama2)  ensure_tokenizer "$2" meta-llama/Llama-2-7b-hf \
                     01c7f73d771dfac7d292323805ebc428287df4f9 \
                     config.json tokenizer.json tokenizer.model tokenizer_config.json ;;
        llama31) ensure_tokenizer "$2" meta-llama/Llama-3.1-8B \
                     d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
                     config.json special_tokens_map.json tokenizer.json tokenizer_config.json ;;
        qwen3)   ensure_tokenizer "$2" Qwen/Qwen3-8B-Base \
                     49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
                     config.json generation_config.json merges.txt tokenizer.json \
                     vocab.json tokenizer_config.json ;;
        *) echo "ERROR: no example tokenizer named '$1'" >&2; return 1 ;;
    esac
}
