# DeepSeek-V4.1-Flash: two TP2 replicas on four MI355X GPUs

For 8192-input/1024-output requests at global concurrency 256, two TP2 replicas
(each concurrency 128) improved total token throughput per GPU over one TP4
server at concurrency 256. Both deployments disable speculation and use normal
level-3 compilation/FULL graphs, BF16 KV and FP4 index cache. The existing DSpark
and TP4 configurations remain available.

Use the official `rocm/atom-dev:latest` image. Run these commands in an environment
with the model at `/models/DeepSeek-V4.1-Flash` and all four GPUs visible; the
client or gateway must distribute requests between the two endpoints and limit
each replica to 128 concurrent requests.

```bash
ROCR_VISIBLE_DEVICES=0,1 python -m atom.entrypoints.openai_server \
  --model /models/DeepSeek-V4.1-Flash --served-model-name dsv41-replica \
  --host 127.0.0.1 --server-port 18971 --port 8016 \
  --data-parallel-master-port 29530 \
  -tp 2 --kv_cache_dtype bf16 --index_cache_dtype fp4
```

In a separate shell:

```bash
ROCR_VISIBLE_DEVICES=2,3 python -m atom.entrypoints.openai_server \
  --model /models/DeepSeek-V4.1-Flash --served-model-name dsv41-replica \
  --host 127.0.0.1 --server-port 18972 --port 8026 \
  --data-parallel-master-port 29540 \
  -tp 2 --kv_cache_dtype bf16 --index_cache_dtype fp4
```

These are independent complete models, with separate TP groups and caches. No
DP-attention or expert-parallel switch is involved. Each TP2 replica has a larger
weight footprint per GPU than TP4. The measured run retained default prefix
caching and default graph configuration; adding eager mode, changing the cache
policy, or enabling speculation creates a different comparison.

## Whole-model validation

Measured on 2026-10-22, job 234185 / crsuse2-m2m-174, using official
`rocm/atom-dev:latest` via udocker P1, frozen ATOM `443832207` and AITER
`eb379ca87`. A/B/A/B uses the same four GPUs sequentially. Each formal arm has
2560 completed requests, 18,893,887 input tokens and 2,353,757 generated tokens,
random range ratio 0.8, seed 2026102456 and the chat template. Each arm has a
separate 512-request warmup. No auxiliary GPU experiment ran during formal
measurement. Aggregate throughput is all completed input plus output tokens,
divided by the single global wall time and four GPUs.

| Arm | Deployment | Total tokens/s/GPU | TTFT ms | TPOT ms |
|---|---|---:|---:|---:|
| A1 | TP4, C256 | 10840.83 | 1414.06 | 51.16 |
| B1 | TP2 + TP2, C128 each | 12244.75 | 1363.23 | 43.38 |
| A2 | TP4, C256 | 12066.48 | 1349.96 | 45.79 |
| B2 | TP2 + TP2, C128 each | 12715.13 | 1170.99 | 42.19 |

Paired throughput gains are **12.95% and 5.38%**. Pooling both arms' token totals
and wall times gives **11420.87 → 12475.51 tokens/s/GPU, +9.23%**. The arithmetic
mean comparison is +8.96%. TP4 A2 itself is 11.31% faster than A1; this drift
prevents treating the first pair as a stable 13% guarantee. Prefix-cache reuse is
183168 tokens per TP4 arm versus 99840 across both TP2 replicas, so the observed
TP2 gain does not come from a greater number of cached tokens. Request and
output-token counts agree across all arms.

Full GSM8K test, 1319 unique questions, 3-shot, `local-completions`, client
concurrency 65, default task generation settings (maximum 256 output tokens),
`exact_match,flexible-extract`: **0.9249431387414708**, exceeding the dashboard
threshold **0.91**. Per-sample recomputation agrees. This score was obtained on
one TP2 replica with normal graph configuration, no speculation and profiling
off. A separate saved GSM8K request confirms native prefill and actual target
mono decode on both ranks, including graph launches. It does not trace every
scoring step or establish accuracy for unrelated tasks or new kernel prototypes.

Workspace evidence, kept outside production source:
`validation/wide_decode_20261022/allocations/234185/replica_abab/RESULTS.json`,
its four detailed formal JSON files and manifest;
`validation/wide_decode_20261022/allocations/234185/tp2_accuracy/manifest.json`,
full evaluator results/samples and `route_audit.json`.

This is a serving parallelism change. It has not been validated at lower
concurrencies, with DSpark, on other GPU models, or as a 2× optimization. It is a
separate candidate configuration rather than a replacement for every dashboard
scenario. A gateway's own latency and scheduling overhead must be included in
any deployment benchmark using that gateway.
