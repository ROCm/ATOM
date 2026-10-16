#!/usr/bin/env bash
set -euo pipefail
gen_root=${GEN_ROOT:-/home/yifehuan/data/qwen38_mono_repro}
gen_scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gen_atom=$(cd -- "$gen_scripts/../.." && pwd)
gen_aiter=${GEN_AITER:-/home/yifehuan/glm5_tileRT/aiter}
export GEN_TP=$1 GEN_MODE=$2 GEN_RUN="$gen_root/tp${1}_${2}" GEN_PROBE=1
if [[ "$GEN_TP" != 8 || ( "$GEN_MODE" != native && "$GEN_MODE" != mono ) ]]; then
  echo 'This reproduction harness accepts only: 8 native|mono' >&2
  exit 2
fi
if [[ "${GEN_RESOURCE_AUDIT:-0}" == 1 ]]; then
  echo 'Resource-audit bypass is not allowed in performance acceptance' >&2
  exit 2
fi
if [[ -n "${GEN_SUFFIX:-}" ]]; then export GEN_RUN="${GEN_RUN}_${GEN_SUFFIX}"; fi
gen_max_seqs=${GEN_MAX_SEQS:-8}
gen_capture_sizes=${GEN_CAPTURE_SIZES:-'[1,2,4,8]'}
mkdir -p "$GEN_RUN"
export PROMETHEUS_MULTIPROC_DIR="$GEN_RUN/prometheus"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
export PYTHONPATH="$gen_scripts/bootstrap:$gen_aiter:$gen_atom"
export ATOM_MONO_ENABLE=1 ATOM_QWEN_MONO_ENABLE=1
if [[ "$GEN_MODE" == native ]]; then export ATOM_QWEN_MONO_ENABLE=0; fi
export HIP_VISIBLE_DEVICES=0
if [[ "$GEN_TP" == 8 ]]; then export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; fi
export AITER_JIT_DIR=${AITER_JIT_DIR:-$gen_root/cache/jit}
export PYTHONUNBUFFERED=1 FLYDSL_DUMP_IR=0
echo $$ > "$GEN_RUN/server.pid"
extra_args=()
if [[ "$GEN_TP" == 8 ]]; then extra_args+=(--enable-expert-parallel); fi
gen_level=3
if [[ "${GEN_EAGER:-0}" == 1 ]]; then
  extra_args+=(--enforce-eager)
  gen_level=0
fi
exec python3 -m atom.entrypoints.openai_server \
  --model /shared_nfs/huggingface_models/Qwen/Qwen3.8-Flash-Next-FP8 \
  -tp "$GEN_TP" "${extra_args[@]}" --max-num-seqs "$gen_max_seqs" --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --no-enable_prefix_caching --kv-cache-dtype bf16 --gpu-memory-utilization 0.92 \
  --level "$gen_level" --cudagraph-mode FULL --cudagraph-capture-sizes "$gen_capture_sizes" \
  --torch-profiler-dir "$GEN_RUN/profiles" \
  --port 8123 > "$GEN_RUN/server.log" 2>&1
