# Qwen3.8-Flash-Next AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against
Qwen3.8-Flash-Next with ATOM on a single AMD MI355X GPU. The validated
configuration uses:

- `amd/Qwen3.8-Flash-Next-Quark-MXFP4` (MXFP4 routed experts, BF16 elsewhere)
- TP1 on one GPU
- BF16 KV cache (the only KV format this model supports)
- native GPU prefix caching, no CPU offload
- native MTP with three draft tokens and synthetic acceptance length 2.32
- the SemiAnalysis Weka AgentX workload, 256K-context trace set

The workload uses the AIPerf scenario `inferencex-agentx-mvp` and public
dataset `semianalysis_cc_traces_weka_062126_256k`. It replays long-context,
multi-turn coding traces with subagent fan-out rather than a fixed ISL/OSL
workload. The 256K trace set and every client flag below match the InferenceX
Qwen3.8-Flash-Next AgentX runs.

Related guide: [`Qwen3.8-Flash-Next.md`](Qwen3.8-Flash-Next.md) (generic
serving, MTP, GSM8K).

Only **CONC 16** has been validated. Start a fresh server for each run.

## Validated Configuration

| Item | Value |
|---|---|
| Hardware | 1×MI355X (`gfx950`) |
| Model | `amd/Qwen3.8-Flash-Next-Quark-MXFP4` (with the MTP exclusion fix below) |
| ATOM | `e6182e08b` (`rocm/atom-dev:nightly_202610151507`) |
| Parallelism | TP1 |
| KV cache | BF16, default 16-token blocks |
| Prefix cache | Enabled (required for AgentX multi-turn prefix hits) |
| `max-num-seqs` | 160 (sizes the GDN state pool; see below) |
| Speculative decoding | Native MTP, `--num-speculative-tokens 3` |
| Synthetic acceptance | `--spec-decode-acceptance-length 2.32` (performance-only) |
| GPU memory utilization | 0.95 |
| Profiling duration | 3,600 seconds |
| Warmup | 10 additional one-token requests per lane |
| AIPerf | `0.12.0` |

## 1. Start the ATOM Server

```bash
export MODEL_PATH=${MODEL_PATH:-/data/amd/Qwen3.8-Flash-Next-Quark-MXFP4-mtpfix}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-amd/Qwen3.8-Flash-Next-Quark-MXFP4}
export PORT=${PORT:-8000}
export CONC=${CONC:-16}

if [[ "${CONC}" != "16" ]]; then
  echo "Only CONC=16 has been validated for Qwen3.8-Flash-Next AgentX." >&2
  exit 2
fi

export AITER_LOG_LEVEL=WARNING

python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --server-port "${PORT}" \
  --trust-remote-code \
  -tp 1 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 160 \
  --max-num-batched-tokens 16384 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32]' \
  --enable_prefix_caching \
  --method mtp \
  --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 2.32 \
  2>&1 | tee "server-qwen38-agentx-c${CONC}.log"
```

### Why `max-num-seqs` is 160

Qwen3.8-Flash-Next has 36 GDN (linear-attention) layers. To reuse a prefix,
ATOM needs the KV blocks **and** a GDN state snapshot at the end of that
prefix. For this model the snapshot lives in a slot of the GDN state pool
(`StateTransfer.fork`, not the PAGE path), and that pool has exactly
`max-num-seqs` slots (one per request under ReplaySSM, which MTP enables by
default). Retained snapshots can only use slots that in-flight requests leave
free, so a pool sized to `2 * CONC = 32` caps prefix reuse.

Each slot costs about 132 MB (FP32 SSM state over 36 layers plus the ReplaySSM
record buffers), and it comes out of the same budget as the KV cache. The
validated server log reports:

```text
sub-pool kv: entries=127255, entry_bytes=497664, reserved=58.98GB
sub-pool state: entries=160, entry_bytes=132182056, reserved=19.70GB
```

That is about 2.04M KV tokens (127,255 blocks × 16) next to 160 state slots.
160 mirrors the 160-slot Mamba cache the InferenceX SGLang B200 recipe uses at
this concurrency. Raising `max-num-seqs` buys more snapshots at the cost of KV
tokens; at 512 the state pool would leave too little KV for 16 concurrent
~100K-token AgentX sessions.

Most of this GPU's memory goes to weights: the checkpoint is 169.7 GiB, of
which 95.4 GiB is the BF16 PLE n-gram table (`ple_embedding.ngram_embedding`).

### Synthetic acceptance semantics

`--spec-decode-acceptance-length 2.32` is the InferenceX golden acceptance
length for Qwen3.8-Flash-Next at three speculative tokens with thinking on
(`golden_al_distribution/qwen3.8next_mtp.yaml`, `thinking_on[3]`). The draft
layer still runs; the flag fixes how many draft tokens are committed, so
engine-to-engine comparisons do not depend on draft quality. The validated run
measured `1 + mtp_accepted_tokens / mtp_decode_steps = 2.320` from the server
metrics.

With this flag set, the server returns placeholder text (`synthetic` per
token) because the committed tokens are not the model's own output. AgentX
replays recorded assistant turns
(`AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0`), so the placeholder text
does not affect later prompts.

This mode is **performance-only**. Do not pass
`--spec-decode-acceptance-length` for GSM8K or any correctness evaluation.

## 2. Run the AgentX Profile

Run this once against a newly started server. `--model` must match
`--served-model-name`.

```bash
export CONC=${CONC:-16}
export PORT=${PORT:-8000}
export MODEL_PATH=${MODEL_PATH:-/data/amd/Qwen3.8-Flash-Next-Quark-MXFP4-mtpfix}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-amd/Qwen3.8-Flash-Next-Quark-MXFP4}
export OUTPUT_DIR=${OUTPUT_DIR:-results/qwen38-agentx-c${CONC}}

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true

mkdir -p "${OUTPUT_DIR}"

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "http://127.0.0.1:${PORT}" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model "${SERVED_MODEL_NAME}" \
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
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --server-metrics "http://127.0.0.1:${PORT}/metrics" \
  --output-artifact-dir "${OUTPUT_DIR}" \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  2>&1 | tee "${OUTPUT_DIR}/aiperf.log"
```

## Validated Result (CONC 16)

From `profile_export_aiperf.json` of the validated run (3,615 s profiling
window, 0 errors, 0 preemptions):

| Metric | Value |
|---|---|
| Total token throughput per GPU (input + output) | 70,021 tok/s |
| Output token throughput per GPU | 623 tok/s |
| TTFT p50 / p90 | 0.42 s / 2.85 s |
| Inter-token latency p50 / p90 | 9.00 ms / 18.92 ms |
| Request latency mean | 10.7 s |
| Prefix cache hit (realized / theoretical) | 88.84% / 96.14% |
| Completed requests | 2,482 |
| Prompt / generated tokens | 251.9M / 2.26M |
| Effective concurrency | 7.31 |
| Measured acceptance length | 2.320 |

The realized hit rate is the client-side `overall_usage_prompt_cache_read_pct`
(usage cached tokens / prompt tokens). KV cache usage peaked at 99.2% (mean
41.9%) without preemption.

## Accuracy

Synthetic acceptance is performance-only. For accuracy, start the server
without `--spec-decode-acceptance-length` (keep `--method mtp` to exercise real
draft verification) and run the GSM8K evaluation from
[`Qwen3.8-Flash-Next.md`](Qwen3.8-Flash-Next.md#gsm8k-evaluation) with
`model=${SERVED_MODEL_NAME}`.

GSM8K, 1,319 questions, MXFP4 checkpoint with the MTP head loaded unquantized:

| Configuration | strict-match | flexible-extract |
|---|---|---|
| MTP3, real verification | 0.9757 ± 0.0042 | 0.9757 ± 0.0042 |
| No MTP | 0.9765 ± 0.0042 | 0.9757 ± 0.0042 |

Real MTP3 averaged 3.19 tokens per forward on this run (72.9% draft
acceptance).
