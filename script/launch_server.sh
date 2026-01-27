#!/bin/bash
# export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0
# export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0

# export AITER_QUICK_REDUCE_QUANTIZATION=INT4
# export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export HIP_VISIBLE_DEVICES=0

# Prefer local aiter checkout (for FlyDSL GEMM integration).
# If you want to use the container/site-packages aiter instead, comment this block.
LOCAL_AITER="/home/gyu/zxe/aiter"
if [ -d "${LOCAL_AITER}" ]; then
  export PYTHONPATH="${LOCAL_AITER}:${PYTHONPATH}"
  echo "[launch_server] Using local aiter from ${LOCAL_AITER}"
fi

# Optional FlyDSL GEMM backend (requires FlyDSL checkout).
export AITER_FLYDSL_DEBUG=0
export AITER_USE_FLYDSL_GEMM=0
export DSL2_ROOT=/home/gyu/zxe/FlyDSL

# Reduce warmup / peak memory pressure (helps avoid OOM during per-token quant).
# Tune these up once stable.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL=/data/pretrained-models/deepseek-ai/DeepSeek-V3/
# MODEL=/data/pretrained-models/amd/DeepSeek-R1-0528-MXFP4-ASQ
# MODEL=/mnt/raid0/changcui/hf/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d
# MODEL=/mnt/raid0/changcui/models/Qwen3-235B-A22B
# MODEL=/mnt/raid0/guanbao/Qwen/Qwen3-235B-A22B-FP8
# MODEL=/mnt/raid0/guanbao/Qwen/Qwen3-235B-A22B-Instruct-2507
# MODEL=/mnt/raid0/ygan/Qwen3-235B-A22B-Instruct-2507-FP8
# MODEL=/mnt/raid0/feiyue/hf/Meta-Llama-3-8B
# MODEL=/mnt/raid0/pretrained_model/Kimi-K2-Thinking
# MODEL=/mnt/raid0/pretrained_model/models--RedHatAI--Qwen3-235B-A22B-FP8-dynamic/snapshots/627b08fd72c21c3e0565d87764adce2d87427afc
MODEL=/mnt/raid0/pretrained_model/Qwen3-14B-FP8-dynamic

rm -rf /root/.cache/atom/

# OpenAI-compatible server for lm_eval (default: http://0.0.0.0:8000/v1/completions).
python -m atom.entrypoints.openai_server \
  --model ${MODEL} \
  -tp 1 \
  --kv_cache_dtype fp8 \
  --enforce-eager \
  --level 0 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --server-port 8001 \
  2>&1 | tee openai_server.log

# Other launch modes (examples):
# python -m atom.entrypoints.openai_server --model ${MODEL} -tp 8 --block-size 1 --kv_cache_dtype fp8
# python -m atom.entrypoints.openai_server --model ${MODEL} -tp 4 --kv_cache_dtype fp8 --enable-expert-parallel --torch-profiler-dir profile_traces 2>&1 | tee profile_server_log.log
# python -m atom.examples.simple_inference --model ${MODEL} -tp 1 --kv_cache_dtype fp8 --enforce-eager --level 0 --temperature 0 2>&1 | tee simple_inf_out.log
