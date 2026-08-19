# Agentic-K3 recipe

## Environment

### Docker image

```bash
docker pull rocm/atom-dev:ubuntu24.04_py3.12_pytorch_release_2.10.0_kimi_k3_agentic_0818
```

### Install ATOM/AITER/aiperf

- ATOM: [https://github.com/ROCm/ATOM/tree/k3-dev](https://github.com/ROCm/ATOM/tree/k3-dev)
- AITER: [https://github.com/ROCm/aiter/tree/k3-dev](https://github.com/ROCm/aiter/tree/k3-dev)
- aiperf: [https://github.com/SemiAnalysisAI/InferenceX/tree/main/utils](https://github.com/SemiAnalysisAI/InferenceX/tree/main/utils), [aiperf @ 754356e](https://github.com/SemiAnalysisAI/aiperf/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75)

## server

```bash
#!/usr/bin/env bash
set -euo pipefail
 
MODEL="${K3_MODEL:-moonshotai/Kimi-K3}"

export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"
export AITER_SITUV2_A4W4=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_FLYDSL_STAGE2_FP8=1
export ATOM_STATE_CHECKPOINT_DEMAND=0
export ATOM_MLA_MAX_SPLIT_PER_BATCH=256

CONC="${CONC:-1}"
PORT="${PORT:-8000}"

if [[ ! "$CONC" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONC must be a positive integer; got: $CONC" >&2
  exit 2
fi

case "$CONC" in
  1|4)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
    ATOM_ENABLE_REPLAYSSM="${ATOM_ENABLE_REPLAYSSM:-0}"
    STATE_CHECKPOINT_SLOTS="${STATE_CHECKPOINT_SLOTS:-}"
    ENABLE_LMCACHE="${ENABLE_LMCACHE:-0}"
    ENABLE_STATE_OFFLOAD="${ENABLE_STATE_OFFLOAD:-0}"
    ;;
  8)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
    ATOM_ENABLE_REPLAYSSM="${ATOM_ENABLE_REPLAYSSM:-1}"
    STATE_CHECKPOINT_SLOTS="${STATE_CHECKPOINT_SLOTS:-96}"
    ENABLE_LMCACHE="${ENABLE_LMCACHE:-0}"
    ENABLE_STATE_OFFLOAD="${ENABLE_STATE_OFFLOAD:-0}"
    ;;
  10)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
    ATOM_ENABLE_REPLAYSSM="${ATOM_ENABLE_REPLAYSSM:-0}"
    STATE_CHECKPOINT_SLOTS="${STATE_CHECKPOINT_SLOTS:-16}"
    ENABLE_LMCACHE="${ENABLE_LMCACHE:-1}"
    LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-32}"
    ENABLE_STATE_OFFLOAD="${ENABLE_STATE_OFFLOAD:-1}"
    ;;
  *)
    # No explicit profile — default knobs + threshold-based ReplaySSM/slots.
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
    ENABLE_LMCACHE="${ENABLE_LMCACHE:-0}"
    ENABLE_STATE_OFFLOAD="${ENABLE_STATE_OFFLOAD:-0}"
    REPLAYSSM_CONC_THRESHOLD="${REPLAYSSM_CONC_THRESHOLD:-8}"
    if [ -z "${ATOM_ENABLE_REPLAYSSM:-}" ]; then
      if ((10#$CONC >= 10#$REPLAYSSM_CONC_THRESHOLD)); then
        ATOM_ENABLE_REPLAYSSM=1
      else
        ATOM_ENABLE_REPLAYSSM=0
      fi
    fi
    if [ "$ATOM_ENABLE_REPLAYSSM" == 1 ]; then
      STATE_CHECKPOINT_SLOTS="${STATE_CHECKPOINT_SLOTS:-96}"
    else
      STATE_CHECKPOINT_SLOTS="${STATE_CHECKPOINT_SLOTS:-}"
    fi
    ;;
esac
export ATOM_ENABLE_REPLAYSSM

SERVER_LOG="${SERVER_LOG:-k3_mi355_tp8_server_conc${CONC}.log}"

# Extra in-GPU state checkpoint slots (empty → engine default 0, flag omitted).
STATE_CKPT_ARGS=()
if [ -n "$STATE_CHECKPOINT_SLOTS" ]; then
  STATE_CKPT_ARGS=(--state-checkpoint-slots "$STATE_CHECKPOINT_SLOTS")
fi

# CPU state-offload tier — spills recurrent SSM state off-GPU (c10 profile).
if [ "$ENABLE_STATE_OFFLOAD" == 1 ]; then
    export OFFLOAD_STATE=1
    export OFFLOAD_STATE_STAGING_GROUPS="${OFFLOAD_STATE_STAGING_GROUPS:-8}"
    export OFFLOAD_STATE_MIN_LOAD_TOKENS="${OFFLOAD_STATE_MIN_LOAD_TOKENS:-0}"
    export OFFLOAD_GPU_STAGING_CHUNKS="${OFFLOAD_GPU_STAGING_CHUNKS:-16}"
    # Left at 0 (rightmost rung wins, paid or free). Raise it if the
    # `joint kv:` line shows boundaries dominated by `tier=` -- each of those
    # paid an entry-sized H2D plus a park for whatever prefix it reached past
    # the nearest free checkpoint, and this is how many tokens that has to be
    # worth. `demoted=` then counts the trades it declined.
    # export OFFLOAD_STATE_TIER_MARGIN_TOKENS=4096
fi

# LMCache paged-KV offload (L2 CPU tier).
KV_TRANSFER_ARGS=()
if [ "$ENABLE_LMCACHE" == 1 ]; then
    export PYTHONHASHSEED=0
    export LMCACHE_LOCAL_CPU=True
    export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-200}"
    # One chunk == one hash block (--block-size 128), so the KV grid and the
    # state-checkpoint grid coincide. The joint load then aims both legs at the
    # same number instead of rounding the KV leg up to the chunk that covers
    # the state boundary, which removes the overshoot and the
    # `hbm_off_chunk_grid` refusal. Set 256 to A/B against the old grid.
    export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-128}"
    # Paged KV for a hybrid (K3 keeps a KDA recurrent state) is saved and
    # loaded by default; set 0 to run state-checkpoints-only, which is what
    # every measurement before the joint load was taken under.
    export OFFLOAD_KV_FOR_HYBRID="${OFFLOAD_KV_FOR_HYBRID:-1}"
    export OFFLOAD_PROFILE="${OFFLOAD_PROFILE:-1}"
    KV_TRANSFER_ARGS=(--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}')
fi

# clear cache
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
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable_prefix_caching \
  --state-checkpoint-interval-tokens -1 \
  "${STATE_CKPT_ARGS[@]}" \
  --server-port "$PORT" \
  --block-size 128 \
  --method dspark --draft-model Inferact/Kimi-K3-DSpark --num-speculative-tokens 2 --spec-decode-acceptance-rate 0.77 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}' \
  "${KV_TRANSFER_ARGS[@]}" \
  2>&1 | tee "$SERVER_LOG"
```

## Aiperf benchmark
Benchmark each conc with a fresh server instance, which is started by the server script above.

```bash
#!/usr/bin/env bash
model=moonshotai/Kimi-K3
model_name=$(basename ${model})

export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

PORT=${PORT:-8000}
CON_LIST=(${CON_LIST:-1})
RESULTS_DIR=${RESULTS_DIR:-./results}
ARTIFACT_DIR=${ARTIFACT_DIR:-${RESULTS_DIR}/aiperf_artifacts}
mkdir -p ${RESULTS_DIR}

for CON in ${CON_LIST[@]}; do
    echo "Running aiperf profile for ${model_name} with concurrency ${CON}"
    CON_ARTIFACT_DIR=${ARTIFACT_DIR}/con${CON}
    aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:${PORT} --endpoint /v1/chat/completions --endpoint-type chat --streaming --model ${model} --concurrency ${CON} --benchmark-duration 3600 --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 --output-artifact-dir ${CON_ARTIFACT_DIR} --public-dataset semianalysis_cc_traces_weka_062126 2>&1 | tee ${RESULTS_DIR}/aiperf_${model_name}_${CON}.log
    echo "Sleeping for 5 seconds"
    sleep 5
done
```
