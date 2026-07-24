---
name: trace-kernel-breakdown
description: Generate framework-aware trace kernel breakdowns from PyTorch/Chrome trace files. Currently supports SGLang and ATOM and is designed to extend to other serving/runtime platforms. Use when analyzing .trace.json.gz files, profiler output, prefill/decode kernel names, kernel time, kernel shape, module classification, or optional baseline-vs-candidate performance comparisons.
---

# Trace Kernel Breakdown

Use this skill to turn framework trace files into kernel breakdown CSVs with:

- `phase`: `prefill` or `decode`
- `module`: model-structure-aware module name
- `kernel_name`: normalized kernel name
- `kernel_time_us`: GPU kernel duration
- `m,n,k`: shape fields when reliable
- `variant`: optional execution variant such as `index` or `shared_index`
- `tflops`, `bandwidth_gb_s`: derived metrics when shape metadata is reliable

It also supports optional baseline-vs-candidate comparison after breakdown CSVs are generated.

## One-command usage

Run the helper script:

```bash
python3 .claude/skills/trace-kernel-breakdown/scripts/generate_trace_breakdown.py \
  --trace /path/to/trace.json.gz \
  --out /path/to/prefill_decode_kernel_names_simple.csv
```

Useful options:

```bash
# Prefer a prefill layer with M close to 7000.
--target-prefill-m 7000

# Force framework if auto-detection is ambiguous.
--framework atom
--framework sglang

# Select decode annotations by token count, e.g. decode[bs=32 tok=32 d=32].
--target-decode-tokens 32

# Generate an analysis summary with per-kernel ratios, variant counts,
# weighted total time, TFLOPS, and memory bandwidth estimates.
--summary-out /path/to/summary.csv

# Read model config for GLM-style IndexShare counts (full/index vs shared).
--model-config /path/to/config.json

# Generate an optional comparison report from two generated CSVs.
--compare-baseline baseline_breakdown.csv --compare-candidate candidate_breakdown.csv --compare-out compare_report.md
```

## Framework detection

Use the trace content and paths:

- ATOM traces often contain `prefill[...]` / `decode[...]` user annotations and live under `trace/rank_*/...pt.trace.json.gz`.
- SGLang traces may not have phase annotations. They often need inference from CPU shape events and kernel order.
- SGLang profiler may also emit stage files like `*-EXTEND.trace.json.gz` and `*-DECODE.trace.json.gz`; treat `EXTEND` as prefill and `DECODE` as decode.

## Model-agnostic adaptation

Do not assume the model is GLM. This workflow should also work for Kimi, MiniMax, Qwen, dense models, MoE models, MLA models, and other transformer variants.

When the model changes:

1. Inspect the trace first. Detect whether it has explicit `prefill[...]` / `decode[...]` annotations, stage-specific filenames, or only CPU/GPU event patterns.
2. Inspect the serving container or source tree when available. Look for attention backend, MoE implementation, KV cache update, RoPE/positional embedding, and communication code. Use code structure to name modules, not just kernel strings.
3. Infer the model layer structure from trace markers:
   - Dense transformer: norm -> attention projections -> RoPE/KV/cache -> attention core -> output projection -> communication/norm -> MLP -> communication.
   - MoE transformer: same attention side, followed by gate/topk/sort/expert kernels.
   - MLA attention: q/kv projections, qk norm/rope/cache, MLA logits/topk/reduce/apply, output projection.
   - Non-MLA attention: use names like `attn_qkv_*`, `attn_rope_*`, `attn_core`, `attn_out_*` rather than forcing `mla_*`.
4. If a model has no MoE, omit MoE modules. If it has shared experts, include `moe_shared_expert` or `moe_gate_setup` where appropriate.
5. If the model has unfamiliar kernels, include them in order with a conservative module such as `attn_misc`, `mlp_misc`, or `misc`, and state that source inspection is needed to refine them.
6. Preserve a single full layer. Do not mix multiple layers just because the trace window is long.

## Critical rules

1. Do not list only CPU ops. CPU ops are reference metadata for `Input Dims` and `External id`; the CSV should list GPU kernels.
2. Do not rely only on `External id`. Some important attention kernels have no CPU shape event or no `External id`; include the full GPU time window for the selected layer.
3. Do not truncate attention. For old SGLang traces, the attention section includes many GPU-only kernels such as GEMM, `main_kernel`, `CatArray*`, `set_mla*`, `topk_transform_*`, and related copy/index kernels.
4. Avoid extra layers. Select one full transformer layer:
   - attention norm/projection
   - q/k norm and RoPE/KV cache work
   - attention core
   - output projection
   - communication
   - post-attention norm
   - MoE gate/topk/sort/expert
   - communication
5. Preserve kernel order. Use GPU timestamp order for the final CSV. Use CPU order only to infer boundaries and shapes.
6. Normalize generic PyTorch kernel names:
   - `vectorized_layer_norm_kernel` -> `native_layer_norm`
   - `indexSelectSmallIndex` / `vectorized_gather_kernel` -> `index_select`
   - `CatArrayBatchedCopy*` -> `cat_copy`
   - `bfloat16tofloat32_copy_kernel` -> `copy_bf16_to_fp32`
   - `bfloat16_copy_kernel` -> `copy_fp32_to_bf16`
   - `float8_copy_kernel` -> `copy_fp8`
   - `direct_copy_kernel` -> `copy_`
   - `FillFunctor` -> `fill_`
   - `CUDAFunctor_add` -> `add`
   - `MulFunctor` -> `mul`
7. For IndexShare models, output both representative variants when present:
   - `variant=index`: full/indexer layer that computes sparse top-k.
   - `variant=shared_index`: layer that reuses a prior index and skips logits/top-k.
   - This applies to both prefill and decode. Prefill `index` commonly includes `fp8_mqa_logits`; decode `index` commonly includes `deepgemm_fp8_paged_mqa_logits`.
   - Weighted totals must use model config counts, not a simple average. For GLM-5.2, `indexer_types` commonly gives 21 `full` layers and 57 `shared` layers.

## Module naming guidance

Prefer explicit module labels over broad `attn`:

ATOM/MLA decode example:

- `attn_input_norm`
- `attn_qkv_quant`
- `attn_qkv_gemm`
- `attn_qk_norm_quant`
- `attn_qk_gemm`
- `attn_v_quant`
- `attn_v_gemm`
- `attn_rope_norm`
- `attn_kv_cache_load`
- `attn_rope_quant`
- `attn_rope_gemm`
- `attn_sparse_indexer`
- `attn_k_cache_update`
- `mla_logits`
- `mla_topk`
- `mla_score_gemm`
- `mla_value_gemm`
- `attn_rope_cache_update`
- `mla_index_convert`
- `mla_value_apply`
- `mla_reduce`
- `attn_out_copy`
- `attn_out_gemm`
- `attn_out_quant`
- `attn_out_proj`
- `all_reduce`
- `post_attn_norm`
- `moe_gate_gemm`
- `moe_topk`
- `moe_sort`
- `moe_quant`
- `moe_expert`

ATOM/MLA prefill example adds:

- `fmha_prefill`
- `attn_context_gemm`
- `attn_context_cat_copy`
- `attn_kv_cache_update`
- `moe_sort_clear`
- `moe_sort_p0`
- `moe_sort_p1`
- `moe_sort_p23`

SGLang names can differ, but keep equivalent structure:

- `q_norm_quant`, `q_gemm`
- `kv_norm_quant`, `k_gemm`, `kv_quant`, `v_gemm`, `kb_gemm`
- `attn_qk_norm_copy`, `attn_qk_norm`
- `attn_rope_prepare`, `attn_hadamard`, `attn_quant`
- `attn_kv_cache_update`, `attn_kv_cache_gemm`
- `mla_cache_setup`, `mla_logits`, `mla_topk`
- `mla_value_gemm`, `mla_cache_update`, `mla_reduce`
- `attn_out_quant`, `attn_out_proj_gemm`
- `moe_gate_setup`, `moe_gate_gemm`, `moe_expert`

## Summary CSV Guidance

When a user asks for an Excel/summary-style breakdown:

1. Generate the detailed breakdown first.
2. Generate a summary CSV with `--summary-out` and, when available, `--model-config`.
3. The summary rows should include:
   - `variant_count`: number of model layers represented by this variant.
   - `ratio`: kernel time divided by that variant's single-layer time.
   - `weighted_time_ms`: `kernel_time_us * variant_count / 1000`.
   - `layer_time`: one single-layer total per `(phase, variant)`.
   - `phase_weighted_total`: sum of weighted totals for all variants in a phase.
4. For GLM IndexShare, check that totals match the model config:
   - `prefill_total_ms = prefill_index_layer_us * full_count / 1000 + prefill_shared_layer_us * shared_count / 1000`.
   - `decode_total_ms = decode_index_layer_us * full_count / 1000 + decode_shared_layer_us * shared_count / 1000`.
5. A wide spreadsheet layout is acceptable for presentation, but the generated summary should stay long-form so it can be filtered and pivoted.

## Derived Metrics

The helper estimates performance metrics only when shape metadata is reliable:

- GEMM-like kernels (`gemm`, `Cijk`, `batched_gemm`, `mfma_moe`) use:
  - `flops = 2 * m * n * k`
  - `tflops = flops / kernel_time_us / 1e6`
  - `bandwidth_gb_s = estimated_bytes / kernel_time_us / 1e3`
- GEMM byte estimates use `A + B + C` traffic with dtype inferred from the kernel name:
  - BF16/HGEMM/Cijk: 2 bytes
  - FP8/A8W8: 1 byte
  - FP4/MXFP4 weights: 0.5 bytes for weights, BF16 output when indicated
- Non-GEMM attention/MoE kernels with reliable 2D shape use a conservative read+write bandwidth estimate.
- If shape metadata is missing or ambiguous, leave `tflops` / `bandwidth_gb_s` blank and explain in `metric_note`. Do not guess decode shapes.

## Optional comparison

Baseline-vs-candidate comparison is optional. When requested:

1. Generate breakdown CSVs for baseline and candidate first.
2. Aggregate by high-level module groups:
   - `attention_core`
   - `moe`
   - `projection_gemm`
   - `rope_kv_cache`
   - `communication`
   - `norm`
   - `quantization`
3. Compute:

```text
saved_us = baseline_module_time - candidate_module_time
share_of_phase_saving = saved_us / (baseline_phase_total - candidate_phase_total)
speedup = baseline_phase_total / candidate_phase_total
```

4. Sort optimization items by `saved_us` descending.
5. Report regressions where `saved_us < 0`.
6. If prefill shapes differ, normalize before comparing:

```text
normalized_baseline_time = baseline_time * (candidate_prefill_m / baseline_prefill_m)
```

Mention both raw and shape-normalized speedups.

## Known pitfalls

- Excel preview may cache `.xlsx` content. If changes do not appear, close and reopen the tab.
- Do not write xlsx internals by hand. Use `openpyxl` when editing spreadsheets.
- ATOM and SGLang may have asynchronous GPU timing relative to CPU shape events. Use CPU shape events for shape/module inference, but include GPU kernels by the selected GPU time window.
- Some decode traces have no shape metadata. Leave `m,n,k` blank rather than guessing.

