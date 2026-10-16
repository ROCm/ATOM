#!/usr/bin/env bash
set -euo pipefail
qax_root=/home/yifehuan/data/qwen38_agentx_native_20261016
qax_tp=${1:?TP required}
qax_k=${2:?MTP depth required}
qax_tag=${3:?unique run tag required}
qax_run="$qax_root/$qax_tag"
mkdir -p "$qax_run/prometheus"
export PYTHONPATH=/home/yifehuan/glm5_tileRT/atom:/home/yifehuan/glm5_tileRT/aiter
export ATOM_MONO_ENABLE=0 ATOM_QWEN_MONO_ENABLE=0
export HCC_AMDGPU_TARGET=gfx950 PYTORCH_ROCM_ARCH=gfx950
export HIP_VISIBLE_DEVICES=$(seq -s, 0 "$((qax_tp-1))")
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PROMETHEUS_MULTIPROC_DIR="$qax_run/prometheus"
export AITER_JIT_DIR="$qax_root/cache/aiter-ea3d22a3-gfx950"
qax_args=()
if (( qax_k > 0 )); then qax_args+=(--method mtp --num-speculative-tokens "$qax_k"); fi
printf '%s\n' "$$" > "$qax_run/server.pid"
exec python3 -m atom.entrypoints.openai_server \
 --model /shared_nfs/huggingface_models/Qwen/Qwen3.8-Flash-Next-FP8 \
 --trust-remote-code -tp "$qax_tp" --enable-expert-parallel \
 --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 8192 \
 --kv-cache-dtype bf16 --gpu-memory-utilization 0.90 \
 --no-enable_prefix_caching --level 3 --cudagraph-mode FULL \
 --cudagraph-capture-sizes '[1,2,4,8,16,32]' \
 "${qax_args[@]}" --torch-profiler-dir "$qax_run/profiles" \
 > "$qax_run/server.log" 2>&1
