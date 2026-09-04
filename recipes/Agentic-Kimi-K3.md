# Kimi-K3 AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against Kimi-K3
with ATOM on 8×AMD MI355X GPUs. The validated configuration uses:

- `moonshotai/Kimi-K3` (MXFP4 MoE, KDA + MLA)
- TP8; DCP=1 on CONC 1/2/4, DCP=8 on CONC ≥ 8
- FP8 KV cache
- native GPU prefix caching; LMCache CPU offload from CONC 8 up
- DSpark speculative decoding on CONC ≤ 16, with synthetic acceptance
- the SemiAnalysis Weka AgentX workload

The workload uses the AIPerf scenario `inferencex-agentx-mvp` and public
dataset `semianalysis_cc_traces_weka_062126`. It replays long-context, multi-turn
coding traces with subagent fan-out rather than a fixed ISL/OSL workload.

Kimi-K3 AgentX uses **three serving bands**. Speculative decoding, DCP, LMCache
size, and `max-num-seqs` all change with concurrency. Start a **fresh server for
each concurrency**.

Related guides: [`Kimi-K3.md`](Kimi-K3.md) (generic serving / GSM8K),
[`DSpark.md`](DSpark.md) (DSpark draft).

| Band | CONC | Serving defaults |
|---|---|---|
| Interactive | 1, 2, 4 | TP8, **DCP=1**, DSpark **7**, synthetic AL **3.84**, GPU prefix cache only (no LMCache). CUDA-graph capture grows with `2 * CONC * 8`. |
| Mid | 8, 12, 16 | TP8, **DCP=8**, DSpark **3**, synthetic AL **3.00**, LMCache **128 GiB**. C8/C12 use `max-num-batched-tokens 4096`; C16 uses 8192. |
| Throughput | 32, 40, 56, 64 | TP8, **DCP=8**, **no spec**, LMCache on. C32/C40 use 128 GiB; C56/C64 use **192 GiB** and a larger `max-num-seqs` so the in-flight window can track `2 * CONC`. |

The AgentX **client** is the same at every concurrency. Only `--concurrency`
changes.

## Validated Configuration

| Item | Value |
|---|---|
| Hardware | 8×MI355X (`gfx950`) |
| Target | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) (MXFP4 MoE, KDA + MLA) |
| Draft | [`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark) (used only in the interactive and mid bands) |
| Parallelism | TP8; DCP=1 (CONC 1/2/4) or DCP=8 (CONC ≥ 8) |
| KV cache | FP8, `--block-size 128` |
| Prefix cache | Enabled (required for AgentX multi-turn prefix hits) |
| Speculative decoding | DSpark (`--method dspark`), not native MTP |
| Synthetic acceptance | `--spec-decode-acceptance-length` (performance-only; see below) |
| Profiling duration | 3,600 seconds |
| Warmup | 10 additional one-token requests per lane |
| AIPerf | `0.12.0` (`agentx-v1.0.4`) |

### Per-concurrency server table

`graph_max = 2 * CONC * (1 + spec)`. Capture sizes are the dense range
`[2, 3, …, graph_max]`. `max-num-seqs` is **not** always `2 * CONC`.

| CONC | DCP | spec | AL | LMCache | ReplaySSM | `max-num-seqs` | batched tokens | GPU util | `graph_max` |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 1 | 7 | 3.84 | off | 0 | 32 | 8192 | 0.88 | 16 |
| 2 | 1 | 7 | 3.84 | off | 0 | 32 | 8192 | 0.88 | 32 |
| 4 | 1 | 7 | 3.84 | off | 0 | 32 | 8192 | 0.88 | 64 |
| 8 | 8 | 3 | 3.00 | 128 GiB | 1 | 32 | 4096 | 0.88 | 64 |
| 12 | 8 | 3 | 3.00 | 128 GiB | 1 | 24 | 4096 | 0.88 | 96 |
| 16 | 8 | 3 | 3.00 | 128 GiB | 1 | 32 | 8192 | 0.86 | 128 |
| 32 | 8 | 0 | — | 128 GiB | 0 | 64 | 8192 | 0.86 | 64 |
| 40 | 8 | 0 | — | 128 GiB | 0 | 80 | 8192 | 0.86 | 80 |
| 56 | 8 | 0 | — | **192 GiB** | 0 | **112** | 8192 | 0.86 | 112 |
| 64 | 8 | 0 | — | **192 GiB** | 0 | **128** | 8192 | 0.86 | 128 |

C8–C40 use 128 GiB LMCache; C56/C64 use **192 GiB**.
`AITER_REUSE_IDENTICAL_COMM_GROUPS=1` only on C56/C64; every other CONC leaves
it off. LMCache CPU size is `LMCACHE_MAX_LOCAL_CPU_SIZE`; chunk size is 1024
tokens.

## 1. Start the ATOM Server

Start a fresh server for each concurrency point. The block below is one
launcher: set `CONC`, then run it.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/Kimi-K3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-Inferact/Kimi-K3-DSpark}"
export TP="${TP:-8}"
export PORT="${PORT:-8000}"
export CONC="${CONC:-8}"

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_SITUV2_A4W4=1
export AITER_FLYDSL_STAGE2_FP8=1

ONLINE_QUANT_CONFIG='{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'

case "${CONC}" in
  1|2|4)
    DCP=1
    MAX_NUM_SEQS=32
    MAX_NUM_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.88
    ENABLE_LMCACHE=0
    ATOM_ENABLE_REPLAYSSM=0
    NUM_SPECULATIVE_TOKENS=7
    SPEC_DECODE_ACCEPTANCE_LENGTH=3.84
    ;;
  8)
    DCP=8
    MAX_NUM_SEQS=32
    MAX_NUM_BATCHED_TOKENS=4096
    GPU_MEMORY_UTILIZATION=0.88
    ENABLE_LMCACHE=1
    LMCACHE_MAX_LOCAL_CPU_SIZE=128
    ATOM_ENABLE_REPLAYSSM=1
    NUM_SPECULATIVE_TOKENS=3
    SPEC_DECODE_ACCEPTANCE_LENGTH=3.00
    ;;
  12)
    DCP=8
    MAX_NUM_SEQS=24
    MAX_NUM_BATCHED_TOKENS=4096
    GPU_MEMORY_UTILIZATION=0.88
    ENABLE_LMCACHE=1
    LMCACHE_MAX_LOCAL_CPU_SIZE=128
    ATOM_ENABLE_REPLAYSSM=1
    NUM_SPECULATIVE_TOKENS=3
    SPEC_DECODE_ACCEPTANCE_LENGTH=3.00
    ;;
  16)
    DCP=8
    MAX_NUM_SEQS=32
    MAX_NUM_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.86
    ENABLE_LMCACHE=1
    LMCACHE_MAX_LOCAL_CPU_SIZE=128
    ATOM_ENABLE_REPLAYSSM=1
    NUM_SPECULATIVE_TOKENS=3
    SPEC_DECODE_ACCEPTANCE_LENGTH=3.00
    ;;
  32)
    DCP=8
    MAX_NUM_SEQS=64
    MAX_NUM_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.86
    ENABLE_LMCACHE=1
    LMCACHE_MAX_LOCAL_CPU_SIZE=128
    ATOM_ENABLE_REPLAYSSM=0
    NUM_SPECULATIVE_TOKENS=0
    SPEC_DECODE_ACCEPTANCE_LENGTH=""
    ;;
  40)
    DCP=8
    MAX_NUM_SEQS=80
    MAX_NUM_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.86
    ENABLE_LMCACHE=1
    LMCACHE_MAX_LOCAL_CPU_SIZE=128
    ATOM_ENABLE_REPLAYSSM=0
    NUM_SPECULATIVE_TOKENS=0
    SPEC_DECODE_ACCEPTANCE_LENGTH=""
    ;;
  56)
    DCP=8
    MAX_NUM_SEQS=112
    MAX_NUM_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.86
    ENABLE_LMCACHE=1
    LMCACHE_MAX_LOCAL_CPU_SIZE=192
    ATOM_ENABLE_REPLAYSSM=0
    AITER_REUSE_IDENTICAL_COMM_GROUPS=1
    NUM_SPECULATIVE_TOKENS=0
    SPEC_DECODE_ACCEPTANCE_LENGTH=""
    ;;
  64)
    DCP=8
    MAX_NUM_SEQS=128
    MAX_NUM_BATCHED_TOKENS=8192
    GPU_MEMORY_UTILIZATION=0.86
    ENABLE_LMCACHE=1
    LMCACHE_MAX_LOCAL_CPU_SIZE=192
    ATOM_ENABLE_REPLAYSSM=0
    AITER_REUSE_IDENTICAL_COMM_GROUPS=1
    NUM_SPECULATIVE_TOKENS=0
    SPEC_DECODE_ACCEPTANCE_LENGTH=""
    ;;
  *)
    echo "Unsupported CONC=${CONC}; AgentX Kimi-K3 covers 1,2,4,8,12,16,32,40,56,64." >&2
    exit 2
    ;;
esac

AITER_REUSE_IDENTICAL_COMM_GROUPS="${AITER_REUSE_IDENTICAL_COMM_GROUPS:-0}"
export AITER_REUSE_IDENTICAL_COMM_GROUPS
export ATOM_ENABLE_REPLAYSSM
SPEC_TOKENS_FOR_GRAPH=0
if [[ "${NUM_SPECULATIVE_TOKENS}" != "0" ]]; then
  SPEC_TOKENS_FOR_GRAPH="${NUM_SPECULATIVE_TOKENS}"
fi
GRAPH_MAX=$((2 * CONC * (1 + SPEC_TOKENS_FOR_GRAPH)))
CUDAGRAPH_CAPTURE_SIZES="[$(seq -s, 2 "${GRAPH_MAX}")]"

ATOM_CMD=(
  python3 -m atom.entrypoints.openai_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --server-port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size "${TP}"
  --decode-context-parallel-size "${DCP}"
  --kv_cache_dtype fp8
  --block-size 128
  --enable_prefix_caching
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --level 3
  --cudagraph-mode FULL
  --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}"
  --online_quant_config "${ONLINE_QUANT_CONFIG}"
)

if [[ "${NUM_SPECULATIVE_TOKENS}" != "0" ]]; then
  ATOM_CMD+=(
    --method dspark
    --draft-model "${DRAFT_MODEL_PATH}"
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}"
    --spec-decode-acceptance-length "${SPEC_DECODE_ACCEPTANCE_LENGTH}"
  )
fi

if [[ "${ENABLE_LMCACHE}" == "1" ]]; then
  export PYTHONHASHSEED=0
  export LMCACHE_LOCAL_CPU=True
  export LMCACHE_MAX_LOCAL_CPU_SIZE
  export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
  export LMCACHE_NUMA_MODE="${LMCACHE_NUMA_MODE:-auto}"
  export ATOM_NUMA_BIND="${ATOM_NUMA_BIND:-1}"
  export ATOM_NUMA_NODE="${ATOM_NUMA_NODE:-0,0,0,0,1,1,1,1}"
  export OFFLOAD_PROFILE="${OFFLOAD_PROFILE:-1}"
  export OFFLOAD_GPU_STAGING_CHUNKS="${OFFLOAD_GPU_STAGING_CHUNKS:-32}"
  ATOM_CMD+=(--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}')
fi

"${ATOM_CMD[@]}" 2>&1 | tee "server-kimik3-agentx-c${CONC}.log"
```

### Synthetic acceptance semantics

`--spec-decode-acceptance-length` is the InferenceX synthetic acceptance
override. The DSpark draft still runs; this flag controls how many draft tokens
are committed so engine-to-engine comparisons do not depend on draft-head
quality.

| Band | Flag | Meaning |
|---|---|---|
| Interactive (spec=7) | `--spec-decode-acceptance-length 3.84` | Target committed tokens / forward ≈ 3.84 |
| Mid (spec=3) | `--spec-decode-acceptance-length 3.00` | Target committed tokens / forward ≈ 3.00 |
| Throughput | omit | No speculative decoding |

This mode is **performance-only**. Do not pass
`--spec-decode-acceptance-length` for GSM8K, SWE-bench, or any correctness
evaluation.

Do not combine DSpark `confidence_schedule` / ragged verify with these forced
acceptance flags; ATOM rejects that pair at startup. See [`DSpark.md`](DSpark.md).

### ReplaySSM

Kimi-K3 KDA decode can rebuild SSM state from a checkpoint ring
(`ATOM_ENABLE_REPLAYSSM=1`). The table above is the AgentX default: off for
CONC 1/2/4, on for CONC 8/12/16, off from CONC 32 up. Override with
`ATOM_ENABLE_REPLAYSSM=0` or `1` when comparing the other setting.

### Use GPU prefix caching without LMCache

The interactive band already omits LMCache. For a mid/throughput point, unset
the LMCache variables and drop `--kv-transfer-config`:

```bash
unset PYTHONHASHSEED
unset LMCACHE_LOCAL_CPU
unset LMCACHE_MAX_LOCAL_CPU_SIZE
unset LMCACHE_CHUNK_SIZE
```

## 2. Run the AgentX Profile

Run this once per concurrency against a newly started server. `--model` must
match `--served-model-name` from the server (not a filesystem path).

```bash
export CONC="${CONC:-8}"
export PORT="${PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-/data/Kimi-K3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/kimik3-agentx-c${CONC}}"

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

## Accuracy

Synthetic acceptance is performance-only. For GSM8K, start the interactive
band **without** `--spec-decode-acceptance-length` (keep `--method dspark` if
you want DSpark correctness, or drop DSpark entirely). Then:

```bash
export MODEL_PATH="${MODEL_PATH:-/data/Kimi-K3}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"
export PORT="${PORT:-8000}"

python3 -m lm_eval \
  --model local-chat-completions \
  --apply_chat_template \
  --tasks gsm8k \
  --output_path ./eval_out_kimik3 \
  --log_samples \
  --num_fewshot 5 \
  --model_args \
    "model=${SERVED_MODEL_NAME},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=4,timeout=1800,tokenized_requests=False,tokenizer=${MODEL_PATH},trust_remote_code=True" \
  --gen_kwargs max_tokens=512,temperature=0,top_p=1
```

`--model` in `model_args` must be the served name. Passing the local checkpoint
path here returns HTTP 400 from ATOM.
