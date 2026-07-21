# DeepSeek-V4 Usage Guide

[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) is a million-token-context Mixture-of-Experts (MoE) large language model from DeepSeek. It builds on the V3.2 architecture with hash-based expert routing (3 hash layers + sigmoid + bias), a Compressed Sparse Attention (CSA) indexer that selects top-1024 prior tokens per query, and Multi-Latent Attention (MLA) with LoRA-compressed QKV projections. Weights are stored natively in FP8 (E4M3) with UE8M0 block-scaled scales. ATOM ships built-in support via the `DeepseekV4ForCausalLM` architecture — no `--trust-remote-code` is needed.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### FP8 on 8xMI355X GPUs (TP8 + FP8 KV Cache)

```bash
AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --kv_cache_dtype fp8 -tp 8
```

Tips on server configuration:
- **MoE backend**: V4-Pro routes 6 experts out of 384 with hash-based selection. The default fused MoE path with `AITER_BF16_FP8_MOE_BOUND=0` + `ATOM_MOE_GU_ITLV=1` handles the FP4 e2m1 microscaling weights correctly — measured GSM8K (1319 samples, 3-shot flexible-extract) = 0.9522 on MI355X/gfx950.
- Use `--kv_cache_dtype fp8` for memory efficiency. The CSA indexer's compressed K cache is stored separately in FP8 regardless.
- Set `AITER_LOG_LEVEL=WARNING` before starting to suppress aiter kernel log noise.
- Clear compile cache before restarting after code changes: `rm -rf /root/.cache/atom/*`
- V4-Pro reuses the DeepSeek-V3 config schema; V4-specific fields (compress ratios, hash layers, index head dims) are read from the HF config automatically.

### FP8 on MI308 / gfx942 (V4-Flash-Base, FP8 per-block routed experts)

[DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) ships the same V4 architecture (mHC + CSA + HCA + sparse attn + MTP) as V4-Pro, but **routed experts are FP8 e4m3 per-block 128×128** (instead of V4-Pro's FP4 e2m1 microscaling). This trades a small expert-memory increase for end-to-end ROCm `gfx942` (MI308) compatibility — `aiter`'s FP8 grouped GEMM has been tuned for `gfx942`, while the FP4 path was authored for `gfx950` (MI355X).

```bash
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Flash-Base \
  --kv_cache_dtype fp8 -tp 8
```

**The routed-expert quant scheme is auto-detected** from the HF `quantization_config` dict:

| Field | V4-Pro (FP4) | V4-Flash-Base (FP8) |
|---|---|---|
| `quant_method` | `quark` (with FP4 layer pattern) | `fp8` |
| `fmt` | `e2m1` | `e4m3` |
| `weight_block_size` | (per_1x32, microscaling) | `[128, 128]` |
| `scale_fmt` | `ue8m0` | `ue8m0` |

Override knobs (escape hatches, normally not needed):

- **`ATOM_USE_TRITON_MOE=1`** — `gfx942` defaults to Triton MoE automatically (no need to set), but it doesn't hurt to set explicitly. On `gfx950`, V4-Pro uses the fused MoE path by default (see V4-Pro section above); Triton MoE remains available as an alternative backend.

#### Auto-detection logic

The routed-expert quant spec is resolved in this priority order (see [`_detect_v4_routed_quant_spec`](../atom/models/deepseek_v4.py)):

1. **HF config `expert_dtype`** — if `config.json` declares `expert_dtype` (e.g. `"fp8"` / `"fp4"`), use it directly.
2. **Parser-derived layer spec** — if the ckpt's `quantization_config.layer_quant_config` (Quark) or global config (compressed-tensors / generic) directly produces a per-layer spec for `ffn.experts.*.w*`, that wins.
3. **Heuristic from `quant_method` / `fmt`** — strings containing `fp8` → FP8 block; `fp4` / `mxfp4` → FP4.
4. **V4-Pro fallback** — historical default.

For V4-Flash-Base's HF `quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "scale_fmt": "ue8m0"}`, the GenericParser (regex `block|1x128`) extracts `(per_1x128, fp8)` global spec, and step 2 hits → routed expert spec is `(QuantType.per_1x128, dtypes.fp8)`. `dtypes.fp8` from `aiter` resolves to `float8_e4m3fnuz` on `gfx942` and `float8_e4m3fn` on `gfx950` — picked correctly per platform without code changes.

#### MI308 specifics

- KV pool slot sizes are identical to V4-Pro (584 B per token, FP8 NoPE 448 B + BF16 RoPE 128 B + 8 B UE8M0 scales).
- The CSA indexer's K cache stays FP8 (132 B / token) regardless of routed-expert dtype.
- Compressor / Indexer Triton kernels (`fused_compress_attn`, `sparse_attn_v4_paged_decode`) are SKU-agnostic.
- Three-stream concurrency (main / alt / compress) works identically.
- TP / EP sharding follows V4-Pro layout — `n_routed_experts=256, top-k=6` matches the standard FusedMoE expert-shard math.

### PD Disaggregation

For PD-disaggregated serving (1P+1D, 2P+1D DPA, with/without MTP), see
[recipes/mesh/DeepSeek-V4.md](mesh/DeepSeek-V4.md).

### EPLB (Expert Parallel Load Balancing)

With `--enable-expert-parallel`, each GPU (EP rank) holds a fixed subset of
physical experts. Real traffic routes unevenly across V4-Pro's 384 routed
experts, so some EP ranks end up doing far more MoE work than others every
step — the imbalanced ranks become the straggler the rest of the group waits
on at the dispatch/combine barrier. EPLB periodically re-measures per-expert
load and re-places experts (optionally replicating hot ones onto spare
physical slots) to even that out.

**Requires `--enable-expert-parallel`** — EPLB rebalances *physical* expert
placement across EP ranks, so it's a no-op without EP (plain TP shards every
expert's weights across all ranks; there's nothing to rebalance).

```bash
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --kv_cache_dtype fp8 -tp 8 --enable-expert-parallel --enable-dp-attention \
  --eplb-enable \
  --eplb-config '{"num_redundant_experts": 64, "placement_policy": "biased", "rebalance_interval": 200}'
```

- **`--eplb-enable`** (alias `--enable-eplb`) is the master switch. `--eplb-config` only tunes it once enabled.
- **`--eplb-config`** is a single JSON dict (no per-field flags), parsed into an `EPLBConfig`. Unknown keys raise at startup (fail fast on typos). Supported keys:

  | Key | Default | Meaning |
  |---|---|---|
  | `num_redundant_experts` | `0` | Extra physical expert slots per MoE layer set aside for replicas. `0` = pure rearrangement only (no extra memory; still rebalances via re-placement). Must be a multiple of the GPU count. |
  | `placement_policy` | `"naive"` | How the redundant-expert budget is spent. `"naive"`: greedy-replicate + `balanced_packing` spreads replicas thinly (DeepSeek-reference algorithm) — works with `num_redundant_experts=0` (pure rearrangement) or `>0`. `"biased"`: fully replicates the top-K hottest experts to **every** GPU (K = `num_redundant_experts // num_gpus`), trading memory for eliminating cross-GPU traffic on the hottest experts — **requires `num_redundant_experts > 0` (K ≥ 1); with `num_redundant_experts=0` it silently computes K=0 and falls back to identical behavior as `"naive"` ** If you set `placement_policy="biased"` and don't see the expected throughput gain, check `num_redundant_experts` first. |
  | `rebalance_interval` | `3000` | Forward-pass steps between rebalance attempts. **Tune to the workload**: prefill-heavy/short runs accumulate steps slowly — use a small interval (e.g. `200`) or a short eval simply never triggers a rebalance. Decode-heavy runs accumulate steps fast — too small an interval (e.g. `200` for a long decode run) fires rebalances every few seconds and the migration overhead itself becomes the bottleneck; use a larger interval (e.g. `3000`) so rebalances land roughly every 8–12× the load-window size. |
  | `load_window_size` | `1000` | Non-dummy (real) forward passes accumulated into the load histogram before a rebalance decision uses it. Must be `<= rebalance_interval`. |
  | `rebalance_min_balancedness` | `2.0` | Gate: skip a rebalance if the measured per-GPU balancedness is already `>=` this value. Per-GPU balancedness is bounded by ~1.0 in practice, so the `2.0` default is **inert** (always rebalances every interval) — lower it toward `~0.85–0.9` to let already-balanced layers skip rebalancing (saves plan/migration cost when biased placement is already flat). |
  | `rebalance_balancedness_agg` | `"min"` | `"min"` or `"mean"` — how per-layer balancedness scores are aggregated for the gate. |
  | `rebalance_layers_per_chunk` | `64` | MoE layers migrated per rebalance chunk (chunking bounds per-rebalance P2P burst size). |
  | `p2p_batch_chunk_size` | `32` | P2P batch chunk size used while migrating expert weights. |

- **Memory**: `num_redundant_experts=0` (pure rearrangement) costs nothing extra. Each redundant slot costs one extra physical expert's weights on the GPU(s) holding it — `biased` concentrates that cost onto every GPU for the top-K experts, `naive` spreads it thinly.
- EPLB is safe to enable with `num_redundant_experts=0` any time EP is on — it just periodically re-places experts across the existing physical slots to flatten per-GPU load, with no placement/memory tradeoffs to reason about.

**Measured effect** (8×MI355X, EP+DPA, 8k-in/1-out prefill, `mnbt=8192`, conc=128; relative to EP with EPLB disabled):

| Config | `eplb-config` | Throughput (req/s) | Δ vs no-EPLB |
|---|---|---|---|
| No EPLB | *(omit `--eplb-enable`)* | 6.48 | — |
| Pure rearrange | `{"num_redundant_experts": 0, "placement_policy": "naive"}` | 7.56 | **+16.7%** |
| Naive, 64 redundant | `{"num_redundant_experts": 64, "placement_policy": "naive"}` | 8.06 | **+24.4%** |
| Biased, 64 redundant | `{"num_redundant_experts": 64, "placement_policy": "biased"}` | 8.68 | **+34.0%** |

Pure rearrangement flattens the per-layer MoE straggler that the rest of the group waits on at the combine barrier; `biased` additionally removes cross-GPU dispatch/combine traffic for the hottest (top-8) experts. Effect size is workload-dependent — regimes with a pronounced per-layer straggler benefit most.

**Accuracy** (GSM8K 5-shot, `local-completions`, same three configs, measured separately from the throughput run above after the `num_redundant_experts>0` startup-bug fixes; see the `DeepSeek-V4-Pro EPLB r0` / `r64 naive` / `r64 biased` cases in [`.github/benchmark/models_accuracy.json`](../.github/benchmark/models_accuracy.json) for the exact CI configs):

| Config | `eplb-config` | GSM8K exact_match (flexible / strict) | Rebalances during eval | Crashes |
|---|---|---|---|---|
| Pure rearrange | `{"num_redundant_experts": 0, "placement_policy": "naive"}` | 0.9560 / 0.9568 | 4 | 0 |
| Naive, 64 redundant | `{"num_redundant_experts": 64, "placement_policy": "naive"}` | 0.956 | 4 | 0 |
| Biased, 64 redundant | `{"num_redundant_experts": 64, "placement_policy": "biased"}` | 0.9553 | 4 | 0 |

All three match the no-EPLB baseline (≈0.95–0.96).

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-V4-Pro --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=1.0 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

Performance on 8xMI355X GPUs with the following environment:
- Date measured: 2026-05-23.
- Docker image: rocm/atom:latest.
- ATOM: `feat/v4-swa-write-tok-n-guard-opus-default` branch (commit bf9b133e).
- `ATOM_USE_TRITON_MOE=1`, `--kv_cache_dtype fp8`.

The numbers below are a snapshot. For the latest data tracked across commits, see [rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/).

### FP8 (TP8, FP8 KV Cache) — no MTP

| ISL  | OSL  | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TPOT (ms) |
| ---- | ---- | ----------- | ----------- | ------------------------- | ------------------------ | -------------- |
| 1024 | 1024 | 4           | 40          | 195.31                    | 392.53                   | 19.66          |
| 1024 | 1024 | 8           | 80          | 367.43                    | 732.14                   | 21.09          |
| 1024 | 1024 | 16          | 160         | 668.02                    | 1343.15                  | 23.19          |
| 1024 | 1024 | 32          | 320         | 1145.71                   | 2287.81                  | 26.90          |
| 1024 | 1024 | 64          | 640         | 1808.69                   | 3618.19                  | 33.96          |
| 1024 | 1024 | 128         | 1280        | 2847.24                   | 5700.73                  | 43.26          |
| 1024 | 1024 | 256         | 2560        | 4289.93                   | 8575.71                  | 57.55          |

### FP8 (TP8, FP8 KV Cache) — MTP-3

Add `--method mtp --num-speculative-tokens 3` to the server launch. MTP-3
trades a small amount of memory for ~1.5–2× lower TPOT and ~1.3–1.5×
higher total throughput at the same concurrency.

| ISL  | OSL  | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TPOT (ms) |
| ---- | ---- | ----------- | ----------- | ------------------------- | ------------------------ | -------------- |
| 1024 | 1024 | 4           | 40          | 528.46                    | 1061.26                  | 7.25           |
| 1024 | 1024 | 8           | 80          | 907.09                    | 1806.44                  | 8.17           |
| 1024 | 1024 | 16          | 160         | 1391.13                   | 2795.02                  | 10.95          |
| 1024 | 1024 | 32          | 320         | 2159.04                   | 4308.13                  | 14.01          |
| 1024 | 1024 | 64          | 640         | 3222.33                   | 6441.40                  | 18.75          |
| 1024 | 1024 | 128         | 1280        | 4376.29                   | 8755.90                  | 27.77          |
| 1024 | 1024 | 256         | 2560        | 5701.20                   | 11388.96                 | 43.06          |

Here are the steps to reinstall ATOM/AITER in the docker, if you are trying to verify with other specific commits:
```bash
# uninstall existing ATOM/AITER
pip uninstall -y atom amd-aiter

cd PATH_TO_ATOM
# normally ATOM is already installed in develop mode
# you may just do checkout without reinstall
git checkout specific_branch_or_commit
pip install -e .

cd PATH_TO_AITER
rm -rf aiter/jit/build aiter/jit/*.so
git checkout specific_branch_or_commit
git submodule sync && git submodule update --init --recursive
python setup.py develop
```

### Accuracy test

We verified the lm_eval accuracy on gsm8k dataset with command:
```bash
lm_eval \
  --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4-Pro,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
```

Reference accuracy on 8xMI355X GPUs (FP8, FP8 KV Cache, `ATOM_USE_TRITON_MOE=1`,
measured 2026-05-23 at commit bf9b133e):

**no MTP**:
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9553|±  |0.0057|
|     |       |strict-match    |     5|exact_match|↑  |0.9560|±  |0.0056|
```

**MTP-3** (`--method mtp --num-speculative-tokens 3`, average acceptance ≈ 64.5%):
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9530|±  |0.0058|
|     |       |strict-match    |     5|exact_match|↑  |0.9538|±  |0.0058|
```
