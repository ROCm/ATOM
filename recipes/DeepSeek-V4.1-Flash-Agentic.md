# DeepSeek-V4.1-Flash agentic on ATOM — MI355X single node

- Hardware: MI355X, one node; each server uses 2 or 4 GPUs
- Image: `rocm/atom-dev:nightly_202611101510`
- Model: `DeepSeek-V4.1-Flash`, FP8 (MXFP8) KV, FP4 index cache
- Execution: level 3, FULL graphs, DSpark with 5 draft tokens and fixed acceptance length 3.51
- Workload: AgentX, dataset `semianalysis_cc_traces_weka_062126`
- Client: `agentx-harness==1.0.13`, 3,600 seconds per concurrency point

## Layout per concurrency

These are the points and settings of the InferenceX config
`dsv41flash-fp4-mi355x-atom-agentic-dspark`. Start a fresh server for each
point and use the same `TP` and `CONC` in the server and client shells.

| TP | CONC | Layout | Prefill delayer decode interval | Notes |
|---:|---|---|---:|---|
| 2 | 1 | TP2 | 1000000 | `--max-num-seqs 2`, capture `[1,2]` |
| 2 | 2 | TP2 | 192 | |
| 2 | 16 | TP2 | 96 | |
| 2 | 32 | TP2 | 40 | |
| 2 | 64 | DTP2 (DP attention, MoE over TP) | 10 | session affinity |
| 2 | 128, 256 | DEP2 (DP attention + expert parallel) | 10 | session affinity; c256 also captures 256 |
| 4 | 1 | TP4 | 1000000 | `--max-num-seqs 2`, capture `[1,2]` |
| 4 | 2, 4, 8 | TP4 | 192 | |
| 4 | 32 | TP4 | 16 | |

Best practices:

- Leave `--max-num-seqs` at ATOM's default except at concurrency 1. A cap below
  the client concurrency queues requests and inflates TTFT.
- At concurrency 1 the decode interval is never reached, so a new turn's
  prefill waits for the running decode instead of interleaving with it.
  DSpark's block-drafter warmup needs two sequences, hence `--max-num-seqs 2`.
- From concurrency 2 the delayer interval shrinks as the load grows, so a new
  prefill does not stall the running decodes for long, at a small TTFT cost.
- Capture every decode graph size from 1 through 32, plus 48, 64 and 128 (and
  256 for c256).
- The DP layouts run without TBO; it does not support SWA bounded replay. Keep
  each agent session on one DP rank (`ATOM_DP_SESSION_AFFINITY=1` on the
  server, a session header from the client) so its prefix cache stays on that
  rank. `ATOM_DP_LB_REQ_EQUIV` weights in-flight requests under `least_tokens`.
- Set `--timeout-keep-alive 900`. uvicorn's 5 s default resets AIPerf's pooled
  connections at high concurrency.

## Server

```bash
export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4.1-Flash"
export TP=2                          # 2 for CONC 1, 2, 16, 32, 64, 128, 256; 4 for CONC 1, 2, 4, 8, 32
export CONC=32
export OMP_NUM_THREADS=4
export ATOM_NUMA_BIND=0
export ATOM_DISABLE_MMAP=true
export AITER_LOG_LEVEL=WARNING

CAPTURE_SIZES="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]"
EXTRA_ARGS=()
export ATOM_ENABLE_PREFILL_DELAYER=1
case "$TP/$CONC" in
  2/1|4/1)  export ATOM_PREFILL_DECODE_INTERVAL=1000000
            CAPTURE_SIZES="[1,2]"; EXTRA_ARGS+=(--max-num-seqs 2) ;;
  2/2|4/2|4/4|4/8)
            export ATOM_PREFILL_DECODE_INTERVAL=192 ;;
  2/16)     export ATOM_PREFILL_DECODE_INTERVAL=96 ;;
  2/32)     export ATOM_PREFILL_DECODE_INTERVAL=40 ;;
  4/32)     export ATOM_PREFILL_DECODE_INTERVAL=16 ;;
  2/64|2/128|2/256)
            export ATOM_PREFILL_DECODE_INTERVAL=10
            export GPU_MAX_HW_QUEUES=5
            export ATOM_DP_SESSION_AFFINITY=1
            export ATOM_DP_LB_REQ_EQUIV=512
            EXTRA_ARGS+=(--enable-dp-attention --dp-load-balance least_tokens)
            [ "$CONC" -ge 128 ] && EXTRA_ARGS+=(--enable-expert-parallel)
            [ "$CONC" -eq 256 ] && CAPTURE_SIZES="${CAPTURE_SIZES%]},256]" ;;
  *)        echo "TP=$TP CONC=$CONC is not an InferenceX point" >&2; exit 1 ;;
esac

python3 -u -m atom.entrypoints.openai_server   --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4.1-Flash   --trust-remote-code   --host 0.0.0.0 --server-port 8000   --tensor-parallel-size "$TP"   --kv_cache_dtype fp8 --index-cache-dtype fp4   --gpu-memory-utilization 0.9   --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384   --enable_prefix_caching --block-size 16   --state-checkpoint-interval-tokens 8192   --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$CAPTURE_SIZES"   --method dspark --num-speculative-tokens 5   --spec-decode-acceptance-length 3.51   --tool-call-parser dsml_v41   --timeout-keep-alive 900   "${EXTRA_ARGS[@]}"
```

`--spec-decode-acceptance-length` fixes the acceptance length for throughput
runs and makes the output text synthetic. Drop it for accuracy checks.

## Client

Run in another shell after the server is ready. The client is the AgentX
harness that InferenceX pins; install it in its own Python 3.11 environment.
`MODEL_PATH` must contain the matching tokenizer files.

```bash
pip install agentx-harness==1.0.13 "datasets>=4.7.0"

export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4.1-Flash"
export TP=2                          # Matching the server
export CONC=32                       # Matching the server

# Long AgentX stalls exceed AIPerf's 30 s TCP_USER_TIMEOUT default.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX='atom:'
# DP layouts (CONC 64 and up): session affinity keys on the AIPerf session id.
[ "$CONC" -ge 64 ] && export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1

agentx profile   --url http://localhost:8000   --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer "$MODEL_PATH"   --concurrency "$CONC" --benchmark-duration 3600   --tokenizer-trust-remote-code   --server-metrics http://localhost:8000/metrics   --output-artifact-dir "./aiperf-tp$TP-c$CONC"   --public-dataset semianalysis_cc_traces_weka_062126
```
