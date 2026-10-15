# Qwen3.5 AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against
Qwen3.5-397B-A17B-FP8 with ATOM on 4×AMD MI355X GPUs. The configuration is the
one measured in the 2026-10-14 sweep (one hour per concurrency) against the
public SGLang numbers of InferenceX PR #3611:

- `Qwen/Qwen3.5-397B-A17B-FP8` (GDN linear attention + full attention, MoE)
- TP4, FP8 KV cache, MTP with 3 draft tokens
- native GPU prefix caching with GDN state checkpoints on every band; LMCache
  CPU offload from CONC 48 up
- the PrefillDelayer on every band, with longer coalescing from CONC 32 up
- FlyDSL paged decode attention

The workload is the AIPerf scenario `inferencex-agentx-mvp` on the public dataset
`semianalysis_cc_traces_weka_062126_256k`: long-context, multi-turn coding
traces with subagent fan-out. Start a **fresh server for each concurrency**.

Related guide: [`Qwen3.5.md`](Qwen3.5.md) (generic serving, MTP, GSM8K).

| Band | CONC | Serving defaults |
|---|---|---|
| Low | 1, 4, 8, 16 | GPU prefix cache only, delayer defaults (stall 10 / TTFT cap 200 ticks) |
| Mid | 32 | as Low, delayer stall and TTFT cap 15 ticks |
| High | 48, 64, 80 | as Mid, plus LMCache CPU offload (253 GiB/rank KV, 110 GiB/rank GDN state) |

CUDA-graph capture sizes are `1..max(8, 2 * CONC)` on every band.

## Validated Configuration

| Item | Value |
|---|---|
| Hardware | 4×MI355X (`gfx950`), GPUs 0-3 |
| Image | `rocm/atom-dev:nightly_202610131505` |
| ATOM | main plus #2549, #2588, #2589, #2590, #2591 (see [Required changes](#required-changes)) |
| Target | [`Qwen/Qwen3.5-397B-A17B-FP8`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) |
| Speculative decoding | MTP, 3 draft tokens, synthetic acceptance length 3.39 (InferenceX golden value for this model, the same as the SGLang arm of #3611) |
| Client | AIPerf `inferencex-agentx-mvp`, 3600 s per concurrency |

### Required changes

The measured configuration relies on these PRs; the flags they add do nothing
on a tree without them.

| PR | What it does | Flag |
|---|---|---|
| #2588 | Hashes each prompt once, in the API server, and walks the prefix cache once per admission | on by default |
| #2589 | Keeps the prompt-end GDN checkpoint clear of the tokens Qwen3.5's chat template rewrites on the next turn | `ATOM_STATE_CHECKPOINT_END_MARGIN=5` |
| #2590 | Publishes MTP accepted-token counts in one non-blocking copy | always on |
| #2591 (needs #2549) | Writes GDN checkpoints inside the prefill forward (midstep); fixes the MTP draft's KV write slot | `ATOM_GDN_STATE_MIDSTEP=1` |

## 1. Start the ATOM Server

```bash
export CONC="${CONC:-32}"
export MODEL_PATH="${MODEL_PATH:-/data/Qwen3.5-397B-A17B-FP8}"
export HIP_VISIBLE_DEVICES=0,1,2,3
export AITER_LOG_LEVEL=WARNING
export PYTHONHASHSEED=0

# Kernels and fusions.
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_PA_FLYDSL=1
export ATOM_PA_FLYDSL_PLAN_MAX_PARTITIONS=64
export ATOM_PA_FLYDSL_PLAN_WG_BUDGET=256
# INT4 quick all-reduce on large prefill all-reduces (>= 2048 tokens at TP4).
# It costs about 2.4 GSM8K points on this image; see Accuracy.
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

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

# LMCache CPU offload from CONC 48 up. The GDN state tier defaults to 32 GiB,
# about 1360 state images per rank, which at CONC 64 holds only ~400 s of
# checkpoints: older images are dropped while the engine still plans loads to
# them, and those loads fall back to recompute. 110 GiB holds about 4680.
if [ "${CONC}" -ge 48 ]; then
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
  --kv_cache_dtype fp8 -tp 4 \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.39 \
  --max-num-seqs 1024 \
  --cudagraph-capture-sizes "[$(seq -s, 1 ${TOP})]" \
  --state-checkpoint-interval-tokens 16384 \
  --timeout-keep-alive 1800 \
  --server-port 8000
```

Notes:

- `--timeout-keep-alive 1800` matters for AgentX. uvicorn's default of 5 s
  closes idle connections that AIPerf's pool (300 s, oldest first) is about to
  reuse; at CONC 32 that turned into about 10 `Connection reset by peer`
  failures per hour, and a failure in the warm-up phase can abort the run.
  SGLang's #3611 recipe uses 1800 too.
- `--state-checkpoint-interval-tokens 16384` adds a GDN checkpoint at each 16k
  chunk boundary, so a request that shares a long prefix with another resumes
  mid-prompt instead of from zero. With midstep these checkpoints cost no
  extra forward.
- Shorter decode protection (`ATOM_PREFILL_DECODE_INTERVAL` 2 or 0) lowers TTFT
  at CONC 64 but costs p90 interactivity and throughput (CONC 64, 900 s:
  interval 0 gave e2e interactivity +28%, p90 interactivity -32%, throughput
  -6%); keep 4.

## 2. Run the AgentX Profile

```bash
export CONC="${CONC:-32}"
export PORT="${PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-/data/Qwen3.5-397B-A17B-FP8}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/qwen35-agentx-c${CONC}}"

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

## 3. Results

2026-10-14 sweep on MI355X, one hour per concurrency, against the public
MI355X SGLang points of InferenceX #3611. p90 interactivity is 1000 / p90 ITL
(decode only); e2e interactivity is 1 / p90(request latency / output tokens),
so it includes TTFT; throughput is input plus output tokens per second per GPU.
This sweep ran the LMCache state tier at its 32 GiB default; at CONC 64 the
110 GiB setting above removed all state-load failures, cut new prompt tokens
per request from 5353 to 4449 and raised p90 interactivity 5% (900 s A/B).

| CONC | p90 interactivity (x SGLang) | e2e interactivity (x SGLang) | tput/GPU (x SGLang) |
|---|---|---|---|
| 1 | 300.8 (0.96) | 158.5 (1.27) | 4,058 (1.03) |
| 4 | 254.5 (1.07) | 143.7 (1.41) | 6,310 (1.02) |
| 8 | 216.1 (0.94) | 123.4 (1.34) | 10,210 (1.00) |
| 16 | 152.6 (0.97) | 87.1 (1.28) | 22,467 (1.02) |
| 32 | 91.8 (1.02) | 53.5 (1.12) | 37,725 (1.01) |
| 48 | 71.8 (1.06) | 36.0 (0.99) | 47,591 (1.03) |
| 64 | 42.1 (0.93) | 21.5 (0.79) | 54,413 (0.96) |
| 80 | 30.1 (1.03) | 14.1 (0.82) | 55,520 (0.94) |

The remaining gap at CONC 64 / 80 is TTFT: requests queue behind the decode
protection after each prefill.

## Accuracy

Synthetic acceptance is performance-only. For GSM8K, start the server without
`--spec-decode-acceptance-length` and run 3-shot on the completions endpoint:

```bash
lm_eval --model local-completions --tasks gsm8k --num_fewshot 3 \
  --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False"
```

Full GSM8K (1319 questions) on this image with the environment above:
flexible-extract 0.871 without `AITER_QUICK_REDUCE_QUANTIZATION`, 0.848 with
`INT4`. Drop the INT4 setting when accuracy matters more than prefill time.
Avoid 20-shot: Qwen3.5 then often answers with an immediate end of sequence.
