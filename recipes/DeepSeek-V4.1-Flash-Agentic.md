# DeepSeek-V4.1-Flash agentic on ATOM — TP2 / TP4

- Hardware: MI355X, single-node TP2 or TP4, no expert parallelism
- Model: `DeepSeek-V4.1-Flash`, BF16 KV, FP8 index cache
- Execution: level 3, FULL graphs, DSpark with 5 draft tokens and fixed acceptance length 3.51
- Workload: `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`
- Profiling: 3,600 seconds per concurrency point, 5 warmup requests per lane

Choose one server command below. Set the model path and use the same `CONC`
in the server and client shells. Start a fresh server for each point.

Both configurations use `--max-num-seqs 128`. At c32, capture every size from
1 through 32; other points use the sparse capture list shown below.

## TP2 server

```bash
export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4.1-Flash"
export CONC=16                       # 1, 2, 8, 16, 32, 64
export HIP_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export ATOM_NUMA_BIND=0
export ATOM_DISABLE_MMAP=true
export AITER_LOG_LEVEL=WARNING

CAPTURE_SIZES="[1,2,3,4,5,6,7,8,16,32,48,64,128]"
if [ "$CONC" -eq 32 ]; then
  CAPTURE_SIZES="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]"
fi

python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --trust-remote-code \
  --host 0.0.0.0 --server-port 8000 \
  --tensor-parallel-size 2 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 --max-num-seqs 128 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --enable_prefix_caching --block-size 16 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$CAPTURE_SIZES" \
  --method dspark --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 3.51 \
  --tool-call-parser dsml_v41
```

## TP4 server

```bash
export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4.1-Flash"
export CONC=16                       # 2, 8, 16, 32, 64
export HIP_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export ATOM_NUMA_BIND=0
export ATOM_DISABLE_MMAP=true
export AITER_LOG_LEVEL=WARNING

CAPTURE_SIZES="[1,2,3,4,5,6,7,8,16,32,48,64,128]"
if [ "$CONC" -eq 32 ]; then
  CAPTURE_SIZES="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]"
fi

python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --trust-remote-code \
  --host 0.0.0.0 --server-port 8000 \
  --tensor-parallel-size 4 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 --max-num-seqs 128 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --enable_prefix_caching --block-size 16 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$CAPTURE_SIZES" \
  --method dspark --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 3.51 \
  --tool-call-parser dsml_v41
```

## Client

Run in another shell after the server is ready. `MODEL_PATH` must match the
server's model name and contain the matching tokenizer files.

```bash
export MODEL_PATH="/models/deepseek-ai/DeepSeek-V4.1-Flash"
export TP=2                          # 2 or 4, matching the server
export CONC=16                       # Matching server CONC

export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8000 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --tokenizer-trust-remote-code \
  --concurrency "$CONC" --benchmark-duration 3600 \
  --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 5 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics http://localhost:8000/metrics \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir "./aiperf-tp$TP-c$CONC"
```

## Performance

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
