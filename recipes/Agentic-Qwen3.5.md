# Qwen3.5 AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against
Qwen3.5-397B-A17B (MXFP4 experts, FP8 attention) with ATOM on AMD MI355X GPUs.
It is the configuration ATOM submits to InferenceX for this model.

The workload is the AIPerf scenario `inferencex-agentx-mvp` on the public dataset
`semianalysis_cc_traces_weka_062126_256k`: long-context, multi-turn coding
traces with subagent fan-out. Start a **fresh server for each point**.

Related guide: [`Qwen3.5.md`](Qwen3.5.md) (generic serving, MTP, GSM8K).

## Configuration

| Item | Value |
|---|---|
| Hardware | MI355X (`gfx950`): TP4 on GPUs 0-3, TP2 on GPUs 0-1 |
| Image | `rocm/atom-dev` nightly, 2026-11-12 or newer |
| Model | `amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2` |
| KV cache | FP8, prefix caching, GDN state checkpoints every 16384 prompt tokens |
| Speculative decoding | MTP, 3 draft tokens; synthetic acceptance length 3.39 (the InferenceX golden value for this model) |
| Paged decode attention | FlyDSL |
| Client | AIPerf `inferencex-agentx-mvp`, 3600 s per point |

## Points

| TP | CONC | Prefix cache | PrefillDelayer |
|---|---|---|---|
| 4 | 1, 4, 8, 16, 24 | GPU | defaults (stall 10 / TTFT cap 200 ticks) |
| 4 | 32, 40 | GPU | stall and TTFT cap 15 ticks |
| 4 | 48, 56, 64, 80 | GPU + LMCache CPU offload | stall and TTFT cap 15 ticks |
| 2 | 12 | GPU | defaults |
| 2 | 16, 20, 24 | GPU + LMCache CPU offload | defaults |
| 2 | 32, 40 | GPU + LMCache CPU offload | stall and TTFT cap 15 ticks |

TP2 covers the high-throughput end of the curve. LMCache starts at CONC 16 at
TP2 and at CONC 48 at TP4, because two GPUs hold half the device KV of four.
CUDA-graph capture sizes are `1..max(8, 2 * CONC)` on every point.

## 1. Start the ATOM Server

```bash
export CONC="${CONC:-16}"
export TP="${TP:-4}"   # 4 or 2, see Points
export MODEL_PATH="${MODEL_PATH:-/data/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2}"
if [ "${TP}" = 2 ]; then
  export HIP_VISIBLE_DEVICES=0,1
else
  export HIP_VISIBLE_DEVICES=0,1,2,3
fi
export AITER_LOG_LEVEL=WARNING
export PYTHONHASHSEED=0

# Kernels and fusions.
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_PA_FLYDSL=1

# GDN state checkpoints.
export ATOM_STATE_CHECKPOINT_DEMAND=0
export ATOM_STATE_CHECKPOINT_END_MARGIN=5
export ATOM_GDN_STATE_MIDSTEP=1

# PrefillDelayer: protect 4 decode passes after each prefill; coalesce longer
# from CONC 32 up.
export ATOM_PREFILL_DECODE_INTERVAL=4
export ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000
if [ "${CONC}" -ge 32 ]; then
  export ATOM_PREFILL_DELAYER_STALL_TICKS=15
  export ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS=15
fi

# LMCache CPU offload: from CONC 16 up at TP2, from CONC 48 up at TP4.
if [ "${TP}" = 2 ]; then LMCACHE_FROM=16; else LMCACHE_FROM=48; fi
if [ "${CONC}" -ge "${LMCACHE_FROM}" ]; then
  export ATOM_KV_OFFLOAD=lmcache
  export LMCACHE_LOCAL_CPU=True
  export LMCACHE_CHUNK_SIZE=256
  export LMCACHE_MAX_LOCAL_CPU_SIZE=253
  export LMCACHE_NUMA_MODE=auto
  export OFFLOAD_MIN_LOAD_TOKENS=8192
  export OFFLOAD_STATE_CPU_SIZE=110
fi

TOP=$(( 2 * CONC > 8 ? 2 * CONC : 8 ))
python -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --kv_cache_dtype fp8 -tp "${TP}" \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.39 \
  --max-num-seqs 1024 \
  --cudagraph-capture-sizes "[$(seq -s, 1 ${TOP})]" \
  --state-checkpoint-interval-tokens 16384 \
  --timeout-keep-alive 1800 \
  --server-port 8000
```

Notes:

- Leave `AITER_QUICK_REDUCE_QUANTIZATION` unset: InferenceX does not accept a
  quantized all-reduce.
- `--timeout-keep-alive 1800`: uvicorn's default of 5 s closes idle connections
  that AIPerf's pool is about to reuse, which shows up as `Connection reset by
  peer` failures and can abort a run in the warm-up phase.
- `ATOM_STATE_CHECKPOINT_END_MARGIN=5`: Qwen3.5's chat template re-renders the
  last few tokens of a prompt when the next turn is appended, so the prompt-end
  GDN checkpoint is kept clear of them and the next turn can reuse it.
- `--state-checkpoint-interval-tokens 16384` adds a GDN checkpoint at each 16k
  chunk boundary, so a request that shares a long prefix with another resumes
  mid-prompt instead of from zero. With midstep these checkpoints cost no extra
  forward.
- `OFFLOAD_STATE_CPU_SIZE=110` (GiB per rank) keeps about 4680 GDN state
  images per rank at TP4 (24 MiB each) and about 2340 at TP2 (48 MiB each:
  a TP2 rank holds twice the heads); the 32 GiB default drops checkpoints the
  engine still plans to load. The TP2 LMCache points use the same per-rank
  sizes as TP4 (`LMCACHE_MAX_LOCAL_CPU_SIZE=253`, `OFFLOAD_STATE_CPU_SIZE=110`).
- Keep `ATOM_PREFILL_DECODE_INTERVAL=4`: shorter protection costs p90
  interactivity and throughput, longer protection gains nothing.

## 2. Run the AgentX Profile

```bash
export CONC="${CONC:-16}"
export PORT="${PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-/data/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/qwen35-agentx-tp${TP:-4}-c${CONC}}"

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

mkdir -p "${OUTPUT_DIR}"

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "http://127.0.0.1:${PORT}" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --concurrency "${CONC}" \
  --benchmark-duration 3600 \
  --stats-interval 30 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 \
  --warmup-grace-period 1800 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --tokenizer-trust-remote-code \
  --max-context-length 262144 \
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --apply-chat-template \
  --output-artifact-dir "${OUTPUT_DIR}" \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --server-metrics "http://127.0.0.1:${PORT}/metrics" \
  2>&1 | tee "${OUTPUT_DIR}/aiperf.log"
```

p90 interactivity is 1000 / p90 ITL; throughput is input plus output tokens
per second divided by the GPUs serving (4 or 2).

## Accuracy

Synthetic acceptance is performance-only. For GSM8K, start the server without
`--spec-decode-acceptance-length` and run 3-shot on the completions endpoint:

```bash
lm_eval --model local-completions --tasks gsm8k --num_fewshot 3 \
  --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False"
```

Avoid 20-shot: Qwen3.5 then often answers with an immediate end of sequence.
