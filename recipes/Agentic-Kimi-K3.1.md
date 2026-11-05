# Kimi-K3.1 AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against
Kimi-K3.1 Pro with ATOM on 8×AMD MI355X GPUs. The per-concurrency settings
follow the Kimi-K3 ATOM AgentX recipe in
[SemiAnalysisAI/InferenceX#4307](https://github.com/SemiAnalysisAI/InferenceX/pull/4307)
(`kimik3/atom/mi355x-fp4-mtp/agentic.yaml`). The deviations are listed in
[Deviations from InferenceX#4307](#deviations-from-inferencex4307).

- `moonshotai/Kimi-K3.1-Pro` (MXFP4 routed experts, KDA + MLA with a DSA
  indexer on all 24 MLA layers), served by ATOM through the Kimi-K3 class
  (`ARCH_ALIASES`, see [`Kimi-K3.1.md`](Kimi-K3.1.md))
- TP8; DCP=1 at CONC 1/4, DCP=8 from CONC 14 up
- FP8 KV cache, native GPU prefix caching, LMCache CPU offload from CONC 14 up
- DSpark speculative decoding on every band, with the InferenceX synthetic
  acceptance length
- mono decode at CONC 1/4
- no `AITER_QUICK_REDUCE_QUANTIZATION`, no `--online_quant_config`: every
  non-MoE linear stays BF16

The workload is the AIPerf scenario `inferencex-agentx-mvp` on the public
dataset `semianalysis_cc_traces_weka_062126`. It replays long multi-turn coding
traces with subagent fan-out rather than a fixed ISL/OSL. Start a **fresh
server for each concurrency**.

Related guides: [`Kimi-K3.1.md`](Kimi-K3.1.md) (generic serving, precision,
GSM8K), [`Agentic-Kimi-K3.md`](Agentic-Kimi-K3.md) (the Kimi-K3 version of this
recipe), [`DSpark.md`](DSpark.md).

| Band | CONC | Serving defaults |
|---|---|---|
| Interactive | 1, 4 | TP8, **DCP=1**, DSpark **7**, synthetic AL **3.84**, **mono decode on**, KDA state FP32, GPU prefix cache only |
| Mid | 14, 16 | TP8, **DCP=8**, DSpark **4**, synthetic AL **3.36**, LMCache **128 GiB**, ReplaySSM, KDA state FP16 |
| Throughput | 48, 56, 72 | TP8, **DCP=8**, DSpark **4**, synthetic AL **3.36**, LMCache **224 GiB** with a **128 GiB** state cap, on-demand state checkpoints, ReplaySSM |

## 0. Container

Use `rocm/atom-dev:kimi-k31-pro`. On top of ATOM `main` (083891e93), it carries
the following changes, none of them merged upstream yet:

| Change | Why AgentX needs it |
|---|---|
| State offload gathers the DSA index-cache planes of a PAGE unit | Required on every LMCache band (CONC ≥ 14). Without it, the first CPU-tier store of a K3.1 state entry fails on a size mismatch, because the unit views miss the 24 index-cache layers. |
| Router gate + latent down projection packed into one 4480×7168 BF16 GEMM | Performance, CONC 1/4. Controlled by `ATOM_KIMI_FUSE_GATE_DOWN`: `auto` (default) packs only when DCP=1, `1` always, `0` never. DCP=8 keeps the two GEMMs, because the packed build has hung CUDA-graph capture there. |
| AITER `model_configs/kimik31pro_bf16_tuned_gemm.csv` (296 rows, gfx950, 256 CUs) | Performance. Without the online quant config, K3.1 serves BF16 GEMMs that the stock tables do not cover: 2272×7168 (MLA `fused_qkv_a` + indexer), 6284×7168 (KDA fused `in_proj`), 18432×1536 (`q_b_proj` under DCP8), 4096×1536 (indexer `wq_b`) and 4480×7168 (the packed gate/down). Those shapes otherwise fall back to torch. |

It also carries the K3.1 architecture alias and the DSpark draft RoPE fix. Both
are now on `main` as #3003.

## 1. Start the ATOM Server

Set `CONC` to one of 1, 4, 14, 16, 48, 56 or 72, and set `DRAFT_MODEL_PATH` to a
local directory holding the Kimi-K3.1 DSpark draft. Then run the launcher below.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/Kimi-K3.1-Pro}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3.1-Pro}"
: "${DRAFT_MODEL_PATH:?set to a local Kimi-K3.1 DSpark draft directory}"
export PORT="${PORT:-8000}"
export CONC="${CONC:-14}"

# The image bakes in a build-host proxy that nothing serves on a node.
export HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy=
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1

unset AITER_QUICK_REDUCE_QUANTIZATION
export PYTHONNOUSERSITE=1
export AITER_LOG_LEVEL=WARNING
export AITER_SITUV2_A4W4=1
export AITER_FLYDSL_STAGE2_FP8=1
export AITER_FLYDSL_USE_HERD=1
export AITER_FLYDSL_HERD_MIN_M=64
export ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1

# Defaults shared by every band; the case below overrides what differs.
DCP=1
NUM_SPECULATIVE_TOKENS=7
SPEC_DECODE_ACCEPTANCE_LENGTH=3.84
MAX_NUM_SEQS=32
GRAPH_MAX=
ENABLE_LMCACHE=0
export ATOM_GDN_SSM_DTYPE=fp32          # KDA recurrent state; mono requires fp32
export ATOM_MONO_ENABLE=0
export ATOM_ENABLE_REPLAYSSM=0
export ATOM_STATE_CHECKPOINT_DEMAND=0   # anchor-only state checkpoints
export AITER_REUSE_IDENTICAL_COMM_GROUPS=0

case "${CONC}" in
  1)
    # max-num-seqs 1 fails the DSpark drafter's warmup, which runs two sequences.
    MAX_NUM_SEQS=2; GRAPH_MAX=2
    export ATOM_MONO_ENABLE=1
    # Holds new prefill while any request decodes.
    export ATOM_PREFILL_DECODE_INTERVAL=1000000
    ;;
  4)
    # Mono serves up to 64 rows: 8 sequences x 8 rows (DSpark k=7) fit.
    MAX_NUM_SEQS=8; GRAPH_MAX=64
    export ATOM_MONO_ENABLE=1
    ;;
  14|16)
    DCP=8; NUM_SPECULATIVE_TOKENS=4; SPEC_DECODE_ACCEPTANCE_LENGTH=3.36
    GRAPH_MAX=$((8 * CONC))
    ENABLE_LMCACHE=1; export LMCACHE_MAX_LOCAL_CPU_SIZE=128
    export ATOM_ENABLE_REPLAYSSM=1
    export ATOM_GDN_SSM_DTYPE=fp16      # deviation, see below
    if [[ "${CONC}" == "16" ]]; then
      export ATOM_PREFILL_DECODE_INTERVAL=4 ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000
    fi
    ;;
  48|56|72)
    DCP=8; NUM_SPECULATIVE_TOKENS=4; SPEC_DECODE_ACCEPTANCE_LENGTH=3.36
    MAX_NUM_SEQS=$((2 * CONC)); GRAPH_MAX=$((2 * CONC))
    ENABLE_LMCACHE=1
    # On-demand state checkpoints at shared-prefix targets; the state cap is a
    # limit inside the shared LMCache pool, not a second pool.
    export ATOM_STATE_CHECKPOINT_DEMAND=1
    export LMCACHE_MAX_LOCAL_CPU_SIZE=224 OFFLOAD_STATE_CPU_SIZE=128
    export ATOM_ENABLE_REPLAYSSM=1
    export ATOM_PREFILL_DECODE_INTERVAL=4 ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000
    if [[ "${CONC}" -ge 56 ]]; then
      export AITER_REUSE_IDENTICAL_COMM_GROUPS=1
    fi
    ;;
  *)
    echo "Unsupported CONC=${CONC}; this recipe covers 1, 4, 14, 16, 48, 56, 72." >&2
    exit 2
    ;;
esac

if [[ "${DCP}" == "8" ]]; then
  # Deviation: long-context DCP8 prefill otherwise dies on
  # HSA_STATUS_ERROR_OUT_OF_RESOURCES from caching-allocator fragmentation.
  export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

ATOM_CMD=(
  python3 -m atom.entrypoints.openai_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 127.0.0.1
  --server-port "${PORT}"
  --trust-remote-code
  --tool-call-parser kimi_k3
  --tensor-parallel-size 8
  --decode-context-parallel-size "${DCP}"
  --kv_cache_dtype fp8
  --block-size 128
  --enable_prefix_caching
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.9
  --state-checkpoint-interval-tokens -1
  --level 3
  --cudagraph-mode FULL
  --cudagraph-capture-sizes "[$(seq -s, 1 "${GRAPH_MAX}")]"
  --method dspark
  --draft-model "${DRAFT_MODEL_PATH}"
  --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}"
  --spec-decode-acceptance-length "${SPEC_DECODE_ACCEPTANCE_LENGTH}"
)

if [[ "${ENABLE_LMCACHE}" == "1" ]]; then
  export PYTHONHASHSEED=0
  export LMCACHE_LOCAL_CPU=True
  # The offload hash block is block-size 128 x DCP 8, so the KV and the
  # state-checkpoint grids coincide.
  export LMCACHE_CHUNK_SIZE=1024
  # K3 is a hybrid; without this the paged KV never reaches the CPU tier.
  export OFFLOAD_KV_FOR_HYBRID=1
  export LMCACHE_NUMA_MODE=auto ATOM_NUMA_BIND=1 ATOM_NUMA_NODE=0,0,0,0,1,1,1,1 ATOM_AUTO_NUMA_BIND=0
  export OFFLOAD_PROFILE=1
  # A K3 state entry does not fit the default 8 MiB staging buffer.
  export OFFLOAD_GPU_STAGING_CHUNKS=32
  ATOM_CMD+=(--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}')
fi

rm -rf /root/.cache/atom/*
"${ATOM_CMD[@]}" 2>&1 | tee "server-kimik31-agentx-c${CONC}.log"
```

### Per-concurrency server table

| CONC | DCP | spec | AL | mono | KDA state | LMCache (state cap) | ReplaySSM | demand ckpt | `max-num-seqs` | graph sizes | delayer | reuse comm groups |
|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|---|---:|
| 1 | 1 | 7 | 3.84 | on | fp32 | off | 0 | 0 | 2 | 1–2 | interval 1000000 | 0 |
| 4 | 1 | 7 | 3.84 | on | fp32 | off | 0 | 0 | 8 | 1–64 | off | 0 |
| 14 | 8 | 4 | 3.36 | off | fp16 | 128 GiB | 1 | 0 | 32 | 1–112 | off | 0 |
| 16 | 8 | 4 | 3.36 | off | fp16 | 128 GiB | 1 | 0 | 32 | 1–128 | 4 / 5000 ms | 0 |
| 48 | 8 | 4 | 3.36 | off | fp32 | 224 GiB (128) | 1 | 1 | 96 | 1–96 | 4 / 5000 ms | 0 |
| 56 | 8 | 4 | 3.36 | off | fp32 | 224 GiB (128) | 1 | 1 | 112 | 1–112 | 4 / 5000 ms | 1 |
| 72 | 8 | 4 | 3.36 | off | fp32 | 224 GiB (128) | 1 | 1 | 144 | 1–144 | 4 / 5000 ms | 1 |

All bands use FP8 KV, block size 128, `--max-num-batched-tokens 8192`,
`--gpu-memory-utilization 0.9`, `--state-checkpoint-interval-tokens -1`,
`--level 3` and FULL CUDA graphs.

With `--kv_cache_dtype fp8` and no `--index-cache-dtype`, the DSA index cache
follows the KV cache and is FP8, which is what the NoPE index writer needs.
[`Kimi-K3.1.md`](Kimi-K3.1.md) validates BF16 KV with an explicit
`--index-cache-dtype fp8`. FP8 KV is the AgentX setting.

### Synthetic acceptance

`--spec-decode-acceptance-length` is the InferenceX synthetic acceptance. The
DSpark draft still runs, but the rejection sampler commits draft tokens so that
the mean accepted length per forward converges to the given value, whatever the
draft actually predicts. The values are the InferenceX golden lengths for
Kimi-K3 DSpark: **3.84** at `--num-speculative-tokens 7` and **3.36** at 4. The
InferenceX harness injects them into every AgentX performance job, and they are
written here so that a local run measures the same thing.

The server log shows whether the flag took effect: `[MTP Stats]` should report
`Average toks/fwd` at the target value (3.36 or 3.84).

This mode is **performance-only**. Do not pass the flag for GSM8K or any other
correctness run.

### Deviations from InferenceX#4307

| Item | #4307 | This recipe | Reason |
|---|---|---|---|
| Model / draft | `moonshotai/Kimi-K3`, `Inferact/Kimi-K3-DSpark` | `moonshotai/Kimi-K3.1-Pro`, Kimi-K3.1 DSpark draft | K3.1 |
| `ATOM_GDN_SSM_DTYPE` at CONC 14/16 | fp32 | fp16 | fp32 is a mono requirement and mono is off on these bands; these are the settings the CONC 14/16 runs were made with |
| `PYTORCH_{HIP,CUDA}_ALLOC_CONF=expandable_segments:True` on DCP8 bands | unset | set | Without it, DCP8 prefill of long contexts (several hundred thousand tokens) failed with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (free memory 0 MB) or an NCCL watchdog timeout, at CONC 14 and CONC 72. With it set, a CONC 72 run completed prefills past 700K tokens without error. |
| `--host` | `0.0.0.0` (behind the atomesh frontend) | `127.0.0.1` | Direct ATOM serving; do not expose the port |

## 2. Run the AgentX Profile

Run this once per concurrency against a freshly started server. `--model` must
match `--served-model-name`.

```bash
export CONC="${CONC:-14}"
export PORT="${PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-/data/Kimi-K3.1-Pro}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3.1-Pro}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/kimik31-agentx-c${CONC}}"

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
  --model "${SERVED_MODEL_NAME}" \
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
  --tokenizer "${MODEL_PATH}" \
  --tokenizer-trust-remote-code \
  --apply-chat-template \
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --output-artifact-dir "${OUTPUT_DIR}" \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --server-metrics "http://127.0.0.1:${PORT}/metrics" \
  2>&1 | tee "${OUTPUT_DIR}/aiperf.log"
```

The InferenceX `agentx profile` client resolves to the same profiling phase:
duration, warmup, trajectory window, idle-gap cap and grace period all match.
A warmup phase of roughly 30 minutes at CONC 72 runs before the 3600-second
window.

### Reported metrics

These are the InferenceX definitions, computed over profiling-phase requests
only:

| Metric | Definition |
|---|---|
| Throughput per GPU | `effective_total_throughput / 8` (input + output tokens) from `profile_export_aiperf.json` |
| P90 interactivity | `1 / p90(ITL)` over requests |
| P90 normalized interactivity | `1 / p90(ITL + TTFT / OSL)` over requests, from `profile_export.jsonl` (it cannot be derived from the aggregate export) |
| GPU / CPU prefix hit | `[Cache Stats]` cached / reusable and `[Cache Stats offload]` CPU-tier tokens / reusable in the server log |

## 3. Accuracy

Synthetic acceptance is performance-only. For GSM8K, launch the same band
**without** `--spec-decode-acceptance-length`, then run the lm-eval command in
[`Agentic-Kimi-K3.md`](Agentic-Kimi-K3.md#accuracy) with the K3.1 served name.

## Validation status

| CONC | Status |
|---|---|
| 1, 4 | 3600-second runs with these settings (mono on, KDA state fp32, gate/down packed, AITER rows for the packed shape), with **real** DSpark acceptance. Not yet re-run at the synthetic AL. |
| 14, 16 | 3600-second runs with these settings, with **real** DSpark acceptance and without the 2272/6284/18432 BF16 tuned rows. Not yet re-run at the synthetic AL. |
| 48, 56, 72 | 3600-second runs at the synthetic AL 3.36 with LMCache 128/192 GiB, `ATOM_STATE_CHECKPOINT_DEMAND=0`, KDA state fp16 and HERD off (the pre-#4307 settings). The #4307 settings above (224 GiB, demand checkpoints, fp32 state, HERD) are not yet validated on K3.1. |

Throughput and interactivity numbers for this exact configuration will be added
once every band has been re-run from this recipe.
