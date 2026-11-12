# DeepSeek-V4.1-Flash agentic on ATOM — MI355X single node

- Hardware: MI355X, one node; each server uses 2 or 4 GPUs
- Model: `DeepSeek-V4.1-Flash`, FP8 (MXFP8) KV, FP4 index cache
- Execution: level 3, FULL graphs, DSpark with 5 draft tokens and fixed acceptance length 3.51
- Workload: `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`
- Profiling: 3,600 seconds per concurrency point, 5 warmup requests per lane

## Layout per concurrency

Pick the layout by the client concurrency. Start a fresh server for each point
and use the same `CONC` in the server and client shells.

| CONC | Layout | GPUs | Prefill delayer decode interval | Notes |
|---:|---|---:|---:|---|
| 1 | TP2 or TP4 | 2 / 4 | 1000000 | `--max-num-seqs 2`, capture `[1,2]` |
| 2 | TP2 or TP4 | 2 / 4 | 192 | |
| 4, 8 | TP4 | 4 | 192 | |
| 16 | TP2 | 2 | 96 | |
| 32 | TP2 | 2 | 40 | |
| 64 | DTP2 (DP attention, MoE over TP) | 2 | 10 | session affinity |
| 128, 256 | DEP2 (DP attention + expert parallel) | 2 | 10 | session affinity; c256 also captures 256 |

Best practices:

- Leave `--max-num-seqs` at ATOM's default except at concurrency 1. A cap below
  the client concurrency queues requests and inflates TTFT.
- At concurrency 1 the decode interval is never reached, so a new turn's
  prefill waits for the running decode instead of interleaving with it.
  DSpark's drafter warmup needs two sequences, hence `--max-num-seqs 2`.
- From concurrency 2 the delayer interval shrinks as the load grows, so a new
  prefill does not stall the running decodes for long, at a small TTFT cost.
- Capture every decode graph size from 1 through 32, plus 48, 64 and 128 (and
  256 for c256).
- The DP layouts run without TBO. Keep each agent session on one DP rank
  (`ATOM_DP_SESSION_AFFINITY=1` on the server, a session header from the
  client) so its prefix cache stays on that rank.
- Set `--timeout-keep-alive 900`. uvicorn's 5 s default resets AIPerf's pooled
  connections at high concurrency.

## Server

```bash
export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4.1-Flash"
export CONC=32                       # 1, 2, 4, 8, 16, 32, 64, 128, 256
export TP=2                          # 2, or 4 for CONC 4 and 8 (1 and 2 run either)
export OMP_NUM_THREADS=4
export ATOM_NUMA_BIND=0
export ATOM_DISABLE_MMAP=true
export AITER_LOG_LEVEL=WARNING

CAPTURE_SIZES="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]"
EXTRA_ARGS=()
export ATOM_ENABLE_PREFILL_DELAYER=1
case "$CONC" in
  1)        export ATOM_PREFILL_DECODE_INTERVAL=1000000
            CAPTURE_SIZES="[1,2]"; EXTRA_ARGS+=(--max-num-seqs 2) ;;
  2|4|8)    export ATOM_PREFILL_DECODE_INTERVAL=192 ;;
  16)       export ATOM_PREFILL_DECODE_INTERVAL=96 ;;
  32)       export ATOM_PREFILL_DECODE_INTERVAL=40 ;;
  64|128|256)
            TP=2
            export ATOM_PREFILL_DECODE_INTERVAL=10
            export GPU_MAX_HW_QUEUES=5
            export ATOM_DP_SESSION_AFFINITY=1
            export ATOM_DP_LB_REQ_EQUIV=512
            EXTRA_ARGS+=(--enable-dp-attention --dp-load-balance least_tokens)
            [ "$CONC" -ge 128 ] && EXTRA_ARGS+=(--enable-expert-parallel)
            [ "$CONC" -eq 256 ] && CAPTURE_SIZES="${CAPTURE_SIZES%]},256]" ;;
esac
if [ "$TP" -eq 4 ]; then export HIP_VISIBLE_DEVICES=0,1,2,3; else export HIP_VISIBLE_DEVICES=0,1; fi

python3 -u -m atom.entrypoints.openai_server   --model "$MODEL_PATH" --trust-remote-code   --host 0.0.0.0 --server-port 8000   --tensor-parallel-size "$TP"   --kv_cache_dtype fp8 --index-cache-dtype fp4   --gpu-memory-utilization 0.9   --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384   --enable_prefix_caching --block-size 16   --state-checkpoint-interval-tokens 8192   --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$CAPTURE_SIZES"   --method dspark --num-speculative-tokens 5   --spec-decode-acceptance-length 3.51   --tool-call-parser dsml_v41   --timeout-keep-alive 900   "${EXTRA_ARGS[@]}"
```

`--spec-decode-acceptance-length` fixes the acceptance length for throughput
runs and makes the output text synthetic. Drop it for accuracy checks.

## Client

Run in another shell after the server is ready. `MODEL_PATH` must match the
server's model name and contain the matching tokenizer files.

```bash
export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4.1-Flash"
export TP=2                          # Matching the server
export CONC=32                       # Matching the server

export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
# DP layouts (CONC 64 and up): send the session header for session affinity.
[ "$CONC" -ge 64 ] && export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1

aiperf profile --scenario inferencex-agentx-mvp   --url http://localhost:8000 --endpoint /v1/chat/completions   --endpoint-type chat --streaming   --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --tokenizer-trust-remote-code   --concurrency "$CONC" --benchmark-duration 3600   --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10   --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75   --warmup-requests-per-lane 5 --trace-idle-gap-cap-seconds 300   --agentic-warmup-grace-period 1800   --use-server-token-count --no-gpu-telemetry   --num-dataset-entries 393 --slice-duration 1.0   --server-metrics http://localhost:8000/metrics   --public-dataset semianalysis_cc_traces_weka_062126   --output-artifact-dir "./aiperf-tp$TP-c$CONC"
```

## Performance

Measured with an earlier setup (BF16 KV pool, `--max-num-seqs 128`, no prefill
delayer, no DP attention); not yet re-measured with the settings above.

- **X = 1000 / P90 ITL(ms)**, output tokens/s/user.
- **Y = (total input + total output tokens) / request span / GPU count**,
  tokens/s/GPU. Input includes cached tokens.
- Request span runs from the first successful profiling request start to the
  last successful profiling request end. GPU count is 2 for TP2 and 4 for TP4.

### TP2

| Concurrency | X (output tokens/s/user) | Y (tokens/s/GPU) | Successful requests |
|---:|---:|---:|---:|
| 1 | 287.23 | 10,133.87 | 270 |
| 2 | 255.57 | 10,743.47 | 417 |
| 8 | 166.86 | 27,095.30 | 1,288 |
| 16 | 104.87 | 46,969.42 | 2,366 |
| 32 | 57.33 | 83,359.20 | 3,828 |
| 64 | 24.83 | 103,358.43 | 5,950 |

### TP4

| Concurrency | X (output tokens/s/user) | Y (tokens/s/GPU) | Successful requests |
|---:|---:|---:|---:|
| 2 | 284.58 | 5,691.77 | 425 |
| 8 | 221.80 | 14,553.90 | 1,345 |
| 16 | 141.46 | 26,473.64 | 2,523 |
| 32 | 82.72 | 47,932.15 | 4,212 |
| 64 | 40.45 | 67,353.94 | 7,371 |
