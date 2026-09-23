# DeepSeek-V4.1-Flash AgentX recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay on one TP2 or TP4 ATOM
instance, with no expert parallelism, BF16 KV, FP8 index cache, level 3 FULL
graphs and DSpark with five draft tokens. Start with **TP2 c16** for a balanced
point, **TP2 c32** for higher throughput, or **TP4 c8** for interactivity above
200 output tokens/s/user in these measurements.

The measurements use **forced acceptance length 3.51**. They measure execution
performance under a synthetic acceptance schedule, not generation quality or
the model's natural acceptance rate. The explicit
`ATOM_DSV41_BENCHMARK_SYNTHETIC=1` opt-in is required for this benchmark;
ordinary DSpark keeps real target verification. See the
[model recipe](DeepSeek-V4.1-Flash.md) for quality and runtime limitations.

## Measured operating points

Each point has a 3,600-second profiling setting; startup, dataset preparation
and warmup are separate. AgentX concurrency counts session trees with subagent
fan-out, not a fixed batch of decode requests. Different points finish different
request mixes, so this is an operating-point comparison, not a controlled TP
scaling experiment or an ISL-normalized comparison.

The axes use the local `ix_metrics.py` definition:

- **X = 1000 / P90 ITL(ms)**, output tokens/s/user. This is ordinary interactivity,
  without TTFT or E2E normalization.
- **Y = (sum input tokens + sum output tokens) / request span / GPU count**,
  tokens/s/GPU. Input includes cached tokens.
- Request span is `(last successful profiling request end - first successful
  profiling request start)` in seconds; warmup and error records are excluded.
  GPU count is 2 for TP2 and 4 for TP4.

| TP | Concurrency | X | Y (tokens/s/GPU) | Successful requests | Selection |
|---:|---:|---:|---:|---:|---|
| 2 | 1 | 287.23 | 10,133.87 | 270 | Lowest-load/highest-X reference |
| 2 | 2 | 255.57 | 10,743.47 | 417 | Optional low-load point |
| 4 | 2 | 284.58 | 5,691.77 | 425 | TP2 c1 has higher X and Y |
| 4 | 8 | 221.80 | 14,553.90 | 1,345 | Recommended high-interactivity point |
| 2 | 8 | 166.86 | 27,095.30 | 1,288 | Recommended intermediate point |
| 4 | 16 | 141.46 | 26,473.64 | 2,523 | TP2 c8 has higher X and Y |
| 2 | 16 | 104.87 | 46,969.42 | 2,366 | Recommended balanced point |
| 4 | 32 | 82.72 | 47,932.15 | 4,212 | Optional narrow throughput step |
| 2 | 32 | 57.33 | 83,359.20 | 3,828 | Recommended throughput point |
| 4 | 64 | 40.45 | 67,353.94 | 7,371 | TP2 c32 has higher X and Y |
| 2 | 64 | 24.83 | 103,358.43 | 5,950 | Maximum measured Y; low interactivity |

A compact sweep is **TP2 c1 → TP4 c8 → TP2 c8 → TP2 c16 → TP2 c32**.
Add TP2 c64 only if approximately 25 output tokens/s/user is acceptable.
TP2 c2 and TP4 c32 also remain on the measured X/Y frontier, but offer small
throughput increments relative to their faster neighbors.

Useful comparisons:

| Candidate | Reference | X change | Y change |
|---|---|---:|---:|
| TP2 c8 | TP4 c16 | +17.96% | +2.35% |
| TP2 c16 | TP4 c32 | +26.79% | -2.01% |
| TP2 c32 | TP4 c64 | +41.73% | +23.76% |
| TP2 c64 | TP2 c32 | -56.69% | +23.99% |

TP2 c16 is the close **per-GPU throughput** match to TP4 c32. TP2 c32 and
TP4 c16 are not equivalent: their (X, Y) coordinates are (57.33, 83,359.20)
and (141.46, 26,473.64). There is no universal concurrency conversion between
TP2 and TP4. Dividing Y by GPU count also does not measure multiple co-located
instances; these runs used one instance per benchmark job.

### Evidence and metric reconstruction

- TP2: [Action 35834179304](https://github.com/ROCm/ATOM/actions/runs/35834179304),
  commit `f3f112e482aebfb0efde44afc7e599a2630b962f`.
- TP4: [Action 35823705134](https://github.com/ROCm/ATOM/actions/runs/35823705134),
  commit `016f4ee9bee58447fb83be5f2481f1623bfac972`.
  c2/8/16/32/64 completed; c1 failed and has no point in this table. The overall
  TP4 workflow conclusion is therefore failure.
- Both: `rocm/atom-dev:nightly_202609221542`, MI355X 288 GB, reported ROCm 7.2.4,
  image-provided AIPerf 0.12.0, no AITER reinstall. The workflow checked the
  AIPerf version, not its exact installed commit.
- [CSV with source fields](data/deepseek-v41-agentic-20260923.csv).

These historical coordinates are **reconstructed from uploaded aggregate JSON**:
`p90_itl_ms`, `total_input_tokens`, `total_output_tokens`,
`benchmark_duration_s` and `tensor_parallel_size`. Token totals also agree with
mean sequence length times successful request count. The artifacts do not
include `profile_export.jsonl`, so the historical points have not been independently
revalidated record by record. c32 and c64 on TP2 report request error rates of
0.03% and 0.07%; a successful CI job does not imply zero request errors.

Do not substitute `total_token_throughput / TP` for Y. AIPerf can use an explicit
observation window for that field. For example, TP2 c64 has 750,280,742 total
tokens and a 3,629.509147904-second request span:

```text
X = 1000 / 40.27481646885673 = 24.82941172
Y = 750280742 / 3629.509147904 / 2 = 103358.43105
```

The exported total throughput instead implies a 3,930.003136760-second window,
giving 95,455.49 tokens/s/GPU. Preserve both definitions if comparing dashboards.

## Prepare one point

Run from an otherwise idle MI355X host with the checkpoint already available at
`/models/deepseek-ai/DeepSeek-V4.1-Flash`. Change `MODEL_HOST_ROOT` if the host's
model cache is elsewhere. Use one fresh container and output directory per point,
and finish/stop the previous point before launching another on the same GPUs or
port.

For historical reproduction, the commands below select the source commit that
produced each table. The TP2 admission fix and synthetic benchmark opt-in are
also included with this recipe in ATOM; performance on newer source revisions
must be measured separately.

```bash
set -euo pipefail
TP=2
CONC=16                         # TP2: 1/2/8/16/32/64; TP4: 2/8/16/32/64
IMAGE=rocm/atom-dev:nightly_202609221542
MODEL_HOST_ROOT=/models
case "$TP" in
  2)
    ATOM_REV=f3f112e482aebfb0efde44afc7e599a2630b962f
    GPU_IDS=0,1
    ;;
  4)
    ATOM_REV=016f4ee9bee58447fb83be5f2481f1623bfac972
    GPU_IDS=0,1,2,3
    ;;
  *) echo "This recipe measures TP2 and TP4" >&2; exit 1 ;;
esac

CASE="v41-tp$TP-c$CONC-$(date -u +%Y%m%dT%H%M%SZ)"
git clone https://github.com/ROCm/ATOM.git "$CASE-source"
cd "$CASE-source"
git fetch origin "$ATOM_REV"
git checkout --detach "$ATOM_REV"
SOURCE_DIR="$PWD"
ART="$PWD/results"
mkdir -p "$ART"
CTR="$CASE"

docker run -dt --name "$CTR" --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$SOURCE_DIR:/workspace" -v "$MODEL_HOST_ROOT:/models:ro" \
  -v "$ART:/results" -w /workspace \
  -e HIP_VISIBLE_DEVICES="$GPU_IDS" \
  -e TP="$TP" -e CONC="$CONC" \
  -e OMP_NUM_THREADS=4 -e ATOM_NUMA_BIND=0 -e AITER_LOG_LEVEL=WARNING \
  -e ATOM_DISABLE_MMAP=true -e ATOM_DSV41_BENCHMARK_SYNTHETIC=1 \
  "$IMAGE" bash

docker exec "$CTR" aiperf --version
docker image inspect "$IMAGE" --format '{{.Id}} {{json .RepoDigests}}' \
  > "$ART/image.txt"
git rev-parse HEAD > "$ART/atom-commit.txt"
```

Keep the image's AITER and AIPerf installations for this reproduction.

## Start the server

This uses the recorded fixed `--max-num-seqs 128` for every point. Only c32
uses all capture sizes 1 through 32; c64 uses the ordinary sparse list.
Changing c32 capture or sizing the scheduler as `2 * CONC` changes the recipe.

```bash
docker exec -d "$CTR" bash -lc '
set -euo pipefail
CAPTURE="[1,2,3,4,5,6,7,8,16,32,48,64,128]"
if [ "$CONC" -eq 32 ]; then
  CAPTURE="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]"
fi
exec python -u -m atom.entrypoints.openai_server \
  --model /models/deepseek-ai/DeepSeek-V4.1-Flash \
  --server-port 8000 --host 0.0.0.0 --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --max-num-seqs 128 --level 3 --cudagraph-mode FULL \
  --cudagraph-capture-sizes "$CAPTURE" \
  --method dspark --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 3.51 \
  --enable_prefix_caching --block-size 16 \
  --state-checkpoint-interval-tokens 8192 --tool-call-parser dsml_v41 \
  > /results/server.log 2>&1
'

# Allow up to 45 minutes for loading, compilation and capture.
for attempt in $(seq 1 270); do
  if curl -fsS http://localhost:8000/health -o /dev/null; then
    break
  fi
  sleep 10
done
curl -fsS http://localhost:8000/health -o /dev/null
curl -fsS --max-time 120 http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/deepseek-ai/DeepSeek-V4.1-Flash","prompt":"hi","max_tokens":1}' \
  -o /dev/null
```

Run the client only after both readiness checks succeed. There is no
`--enable-expert-parallel`, DPA, DCP, CPU offload or dynamic DSpark calibration
in these measured launches.

## Run AIPerf

The dataset is `semianalysis_cc_traces_weka_062126`, with five warmup requests
per lane and a 3,600-second profiling setting. The historical job name
`256k1k` is a label: this command does **not** enforce fixed 262,144/1,024
request lengths or pass `--max-context-length`. The trace supplies the lengths.

```bash
docker exec "$CTR" bash -lc '
set -euo pipefail
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url http://localhost:8000 \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model /models/deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer /models/deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC" --benchmark-duration 3600 --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 5 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics http://localhost:8000/metrics \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir /results/aiperf \
  2>&1 | tee /results/client.log
'
```

Keep `server.log`, `client.log`, the image/source identifiers, and the entire
`aiperf` directory, including `profile_export.jsonl`. Check request errors and
cancellations as well as the process exit status.

## Compute X/Y from the new run's individual records

Run on the host, in the same shell where `ART` and `TP` were set. This implements
the local ordinary-interactivity definition and excludes warmup and error
records. It fails on incomplete successful records instead of silently computing
a partial token total or request span.

```bash
python3 - "$ART/aiperf/profile_export.jsonl" "$TP" <<'PY'
import json
import math
import sys

path, gpu_count = sys.argv[1], int(sys.argv[2])
assert gpu_count > 0

def value(record, name):
    metric = record["metrics"][name]
    result = metric["value"] if isinstance(metric, dict) else metric
    assert isinstance(result, (int, float)) and not isinstance(result, bool)
    assert math.isfinite(result)
    return result

def p90(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])

records = []
errors = warmup = 0
with open(path) as stream:
    for line in stream:
        if not line.strip():
            continue
        record = json.loads(line)
        phase = record.get("metadata", {}).get("benchmark_phase")
        if phase not in (None, "profiling"):
            warmup += 1
            continue
        if record.get("error"):
            errors += 1
            continue
        records.append(record)

assert records, "No successful profiling records"
itl = [value(r, "inter_token_latency") for r in records]
itl = [v for v in itl if v > 0]
assert itl, "No positive ITL values"
input_tokens = sum(int(value(r, "input_sequence_length")) for r in records)
output_tokens = sum(int(value(r, "output_sequence_length")) for r in records)
start = min(int(r["metadata"]["request_start_ns"]) for r in records)
end = max(int(r["metadata"]["request_end_ns"]) for r in records)
duration = (end - start) / 1e9
assert duration > 0
print(json.dumps({
    "gpu_count": gpu_count,
    "successful_requests": len(records),
    "profiling_errors": errors,
    "excluded_warmup_records": warmup,
    "request_span_s": duration,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "x_output_tok_s_user": 1000 / p90(itl),
    "y_total_tok_s_gpu": (input_tokens + output_tokens) / duration / gpu_count,
}, indent=2))
PY
```

After collecting the result, stop only this recipe's container:

```bash
docker stop "$CTR"
```

Create a new source/output directory and container for the next point. Reusing a
warm server across concurrency points changes the initial prefix-cache state.
