# Agentic-K3 recipe

## Environment
### Docker image
```bash
docker pull rocm/atom-dev:vllm-kimi-k3-20260807
```
Make sure old torch is used to avoid perf regression:
```bash
root@mi355-gpu-56:/app# pip list | grep torch
torch                                    2.10.0+rocm7.2.4.lw.git3d3aa833
torch_c_dlpack_ext                       0.1.5
torchaudio                               2.10.0+rocm7.2.4.git5047768f
torchvision                              0.25.0+rocm7.2.4.git82df5f59
```

### Install ATOM/AITER/aiperf
- ATOM: https://github.com/ROCm/ATOM/tree/k3-dev
- AITER: https://github.com/ROCm/aiter/tree/k3-dev
- aiperf: https://github.com/SemiAnalysisAI/InferenceX/tree/main/utils, aiperf @ 754356e

## server
```bash
#!/usr/bin/env bash
set -euo pipefail
 
MODEL="${K3_MODEL:-/workspace/shared/data/amd_int/models/Kimi-K3}"
model_name=$(basename ${MODEL})

export ATOM_TORCH_PROFILER_DIR=./${model_name}_traces
TORCH_PROFILER_DIR=./${model_name}_traces
export ATOM_PROFILER_MORE=${ATOM_PROFILER_MORE:-0}
 
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"
export AITER_SITUV2_A4W4=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_FLYDSL_STAGE2_FP8=1
export ATOM_STATE_CHECKPOINT_DEMAND=0
export ATOM_MLA_MAX_SPLIT_PER_BATCH=256
 
ENABLE_LMCACHE="${ENABLE_LMCACHE:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
PORT="${PORT:-8000}"
SERVER_LOG="${SERVER_LOG:-k3_mi355_tp8_lmcache_${ENABLE_LMCACHE}.log}"
KV_TRANSFER_ARGS=()
if [ "$ENABLE_LMCACHE" == 1 ]; then
    export PYTHONHASHSEED=0
    export LMCACHE_LOCAL_CPU=True            # enable CPU (L2) tier
    export LMCACHE_MAX_LOCAL_CPU_SIZE=200    # CPU tier size, GiB (raise/lower to fit host DRAM)
    export LMCACHE_CHUNK_SIZE=256            # MUST be a multiple of --block-size (32 -> 256/32=8 OK)
    export OFFLOAD_PROFILE=1                 # log [OFFLOAD-LOAD-PROF]/[OFFLOAD-SAVE-PROF] to confirm reuse
    # Optional NVMe (L3) tier:
    # export LMCACHE_LOCAL_DISK=/nvme/lmcache
    # export LMCACHE_MAX_LOCAL_DISK_SIZE=2000
    KV_TRANSFER_ARGS=(--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}')
fi
 
rm -rf ~/.cache/atom/*
rm -rf ~/.cache/vllm/*
rm -rf ~/.triton/*
rm -rf /root/.cache/inductor/*
rm -rf /root/.cache/vllm/*
rm -rf /root/.cache/atom/*
rm -rf /root/.triton/*
 
python -m atom.entrypoints.openai_server \
  --model "$MODEL" \
  --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.88 \
  --enable_prefix_caching \
  --state-checkpoint-interval-tokens -1 \
  --server-port "$PORT" \
  --block-size 128 \
  --method dspark --draft-model Inferact/Kimi-K3-DSpark --num-speculative-tokens 2 --spec-decode-acceptance-rate 0.77 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}' \
  "${KV_TRANSFER_ARGS[@]}" \
  --torch-profiler-dir ${TORCH_PROFILER_DIR} 2>&1 | tee "$SERVER_LOG"

```

## Aiperf benchmark
```bash
#!/usr/bin/env bash
model=/workspace/shared/data/amd_int/models/Kimi-K3
model_name=$(basename ${model})

PORT=${PORT:-8000}
CON_LIST=(${CON_LIST:-1 4 8})
RESULTS_DIR=${RESULTS_DIR:-./results}
ARTIFACT_DIR=${ARTIFACT_DIR:-${RESULTS_DIR}/aiperf_artifacts}
mkdir -p ${RESULTS_DIR}

export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

echo "CON_LIST: ${CON_LIST[@]}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "ARTIFACT_DIR: ${ARTIFACT_DIR}"

for CON in ${CON_LIST[@]}; do
    echo "Running aiperf profile for ${model_name} with concurrency ${CON}"
    ARTIFACT_DIR=${ARTIFACT_DIR}/con${CON}
    # Deterministic warmup: --warmup-requests-per-lane 10 (fixed request count per lane)
    # instead of time-based --agentic-cache-warmup-duration; adds
    # --trace-idle-gap-cap-seconds 300 + --stats-interval 30.
    aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:${PORT} --endpoint /v1/chat/completions --endpoint-type chat --streaming --model ${model} --concurrency ${CON} --benchmark-duration 3600 --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 --output-artifact-dir ${ARTIFACT_DIR} --public-dataset semianalysis_cc_traces_weka_062126 2>&1 | tee ${RESULTS_DIR}/aiperf_${model_name}_${CON}.log
    echo "Sleeping for 5 seconds"
    sleep 5
done
```