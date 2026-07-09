# CPP — GLM-5.2 Pipeline-Parallel Long-Context (60k) Prefill: bugs, fixes, reproduction

Date: 2026-07-09 · Hardware: 8×MI355X (gfx950), single node · Container: `cpp_dev`
Model: `/mnt/models/GLM-5.2-MXFP4` (GlmMoeDsaForCausalLM, 78 layers, MLA 576-dim, 64 attn heads, DSA sparse indexer `index_topk=2048`).

## TL;DR

pp8/pp4 long-context (>2048 token) prefill was blocked by **two independent bugs**, both invisible to short-prompt (GSM8K ≈1k tok) runs because those stay under `index_topk` and take the dense flash path:

- **Bug A** — chunk-1 head-count abort. bf16 sparse prefill → AITER `mla_prefill_asm_fwd`, which only supports `num_q_heads/num_kv_heads ∈ {16,128}`. pp→tp1→64 heads → SIGABRT.
- **Bug B** — chunk-2+ GPU memory access fault (PP-specific). GLM-5.2 IndexShare `shared` layers reuse the previous `full` layer's top-k indices; default PP splits put `shared` layers at stage starts whose governing `full` layer is on the previous stage (different GPU) → stale/garbage `sparse_kv_indices` → OOB read.

**Working config (no code fix, launch flags only):** `--kv_cache_dtype fp8` (clears Bug A) + `VLLM_PP_LAYER_PARTITION` aligned so every stage starts on a `full` layer (clears Bug B).

## Root-cause evidence

- `full` layers = `[0,1,2,6,10,14,18,…,74]` (index 0-2, then every 4th). Repeating unit after the first 3: `[full, shared, shared, shared]`. A PP stage boundary is safe iff it lands on a `full` layer.
- Kernel gate (asm_mla.cu `mla_prefill_asm_fwd`): `gqa_ratio == 16 || 128` and `num_kv_heads == 1` ⇒ MLA `num_heads ∈ {16,128}`. gfx950 bf16 **prefill** kernels exist only for Gqa 16 and 128.
- Dispatch: `use_prefill_mla = is_sparse_mla and max_seqlen_k > topk_tokens(2048)` (`attention_mla.py:1133`). bf16 sparse prefill calls `mla_prefill_fwd` (`attention_mla.py:914`); fp8 calls the head-flexible `mla_decode_fwd` (`:881`) because with fp8 KV `dtype_q=dtype_kv=fp8` → kernel key `(fp8,fp8)`, which has a Gqa=64 kernel.
- Isolation: tp8/pp1 runs 60k end-to-end (all 4 chunks) → Bug B is PP-specific.

Operator-level unit tests (isolated, one head count per subprocess):
`.claude/cpp_logs/test_mla_prefill_heads.py` (16 ✓, 64 ✗ SIGABRT, 128 ✓),
`.claude/cpp_logs/test_mla_decode_heads.py` (16/32/64/128 all ✓ bf16),
`.claude/cpp_logs/test_mla_parity.py` (prefill vs decode op, cosine 0.999997).

## Reproduction

All commands run inside the `cpp_dev` container. Kill leftovers first (renamed `ATOM::` workers survive `pkill -f openai_server`; kill by PID and wait for VRAM to drain):
```bash
docker exec cpp_dev bash -c 'for p in $(ps aux | grep -E "atom.entrypoints|ATOM::" | grep -v grep | grep -v defunct | awk "{print \$2}"); do kill -9 $p; done'
```

Aligned partitions (each stage starts on a `full` layer):
- pp8: `VLLM_PP_LAYER_PARTITION="10,8,8,12,8,12,8,12"` (starts 0,10,18,26,38,46,58,66)
- pp4: `VLLM_PP_LAYER_PARTITION="18,20,20,20"` (starts 0,18,38,58)

### Server launch (pp8 example; pp4/tp4 differ only in `-tp/-pp` and the partition)
```bash
AITER_LOG_LEVEL=WARNING \
VLLM_PP_LAYER_PARTITION="10,8,8,12,8,12,8,12" \
ATOM_PROFILER_TIMEOUT=900 \
python -m atom.entrypoints.openai_server \
  --model /mnt/models/GLM-5.2-MXFP4 -tp 1 -pp 8 \
  --level 0 --enforce-eager --kv_cache_dtype fp8 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 --enable_chunked_prefill \
  --gpu-memory-utilization 0.8 \
  --torch-profiler-dir /it-share/yajizhan/code/ATOM/.claude/cpp_logs/trace_pp8_60k \
  --port 8006 --server-port 8000
```
- pp4: `-tp 1 -pp 4` + `VLLM_PP_LAYER_PARTITION="18,20,20,20"`
- tp4 (TP baseline): `-tp 4` (no `-pp`, no `VLLM_PP_LAYER_PARTITION`)

Debugging tip: a GPU memory fault surfaces as a silent `exitcode=-6`. Add `HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3` to make kernels synchronous and print the real fault location.

### Throughput + trace benchmark
```bash
python -m atom.benchmarks.benchmark_serving --backend vllm \
  --model /mnt/models/GLM-5.2-MXFP4 --tokenizer /mnt/models/GLM-5.2-MXFP4 \
  --host 127.0.0.1 --port 8000 --endpoint /v1/completions \
  --dataset-name random --random-input-len 60000 --random-output-len 1 --random-range-ratio 1.0 \
  --num-prompts 32 --max-concurrency 16 --profile \
  --save-result --result-dir .../.claude/cpp_logs --result-filename <tag>_bench_result.json --trust-remote-code
```
`--profile` wraps `/start_profile … /stop_profile`; traces land per stage in the profiler dir as `pp{stage}_rank_0/*.pt.trace.json.gz` (the per-stage `pp{}` prefix is the one code change, below).

### Accuracy (GSM8K, 5-shot, 100 samples)
Use `local-completions` (NOT `local-chat-completions` → drops to 0.28) and **no `--batch_size`** (ATOM `/v1/completions` rejects list prompts with 422; rely on `num_concurrent`):
```bash
lm_eval --model local-completions \
  --model_args "model=/mnt/models/GLM-5.2-MXFP4,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 5 --limit 100
```

## Results (ISL=60000, OSL=1, conc=16, num_prompts=32, fp8 KV, 16k chunk, level-0 eager)

| config | GPUs | successful | duration (s) | throughput (tok/s) | Mean TTFT (ms) |
|---|---|---|---|---|---|
| pp8 (tp1×pp8, aligned) | 8 | 32/32 | 26.68 | 71,968 | 10,446 |
| pp4 (tp1×pp4, aligned) | 4 | 32/32 | 42.58 | 45,096 | 16,372 |
| tp4 (tp4×pp1) | 4 | 32/32 | 105.95 | 18,123 | 40,508 |

**pp4 = 2.49× tp4 prefill throughput at equal GPU count** (2.47× lower TTFT). Traces: `.claude/cpp_logs/trace_{pp8,pp4,tp4}_60k/`.

Accuracy: **pp4 GSM8K = 0.95 flexible / 0.94 strict** — matches the pp1×tp4 baseline (0.95/0.94); no regression. (Short-context only — validates PP plumbing + fp8, not the long-context sparse path.)

## Code change committed

`atom/model_engine/model_runner.py` — profiler output dir gets a `pp{stage}_` prefix when `pipeline_parallel_size > 1`, so each PP stage's trace lands in its own subdir instead of colliding in `rank_0/`. No behavior change at pp=1.

## Open items (proper fixes, not yet implemented)

1. **Bug A general fix**: route bf16 sparse prefill to `mla_decode_fwd` (like the fp8 branch) so bf16 KV also works at 64 heads — numerically identical for the `max_q_len=1` sparse case (validated, cosine 0.999997). Removes the fp8 requirement.
2. **Bug B general fix**: transmit the `full` layer's sparse indices across PP stage boundaries (add to inter-stage `IntermediateTensors`), or force boundary `shared` layers to recompute. Removes the `VLLM_PP_LAYER_PARTITION` alignment constraint and lets the balanced default splits (`[9,10,…]` for pp8, range 1) work.
3. **Long-context accuracy**: compare pp4/pp8 logits or output vs tp8 (trusted reference) on a >2048 prompt — the sparse path is exercised for the first time on PP.
