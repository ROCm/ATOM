# Online Quantization Guide

ATOM can quantize or re-quantize model weights while loading them by passing
`--online_quant_config` to the engine. The source checkpoint stays on disk
unchanged; quantization happens in memory inside `process_weights_after_loading`
right after the loader finishes copying tensors.

This guide covers when to use online quantization, the full configuration
syntax, ready-to-run recipes for the most common model families, how to verify
the result, and troubleshooting tips. For the dataclass-level field reference,
see [`configuration_guide.md` § 3.7](./configuration_guide.md#37-online-quantization-at-load-time).

---

## 1. When to use online quantization

Use online quantization when one of the following holds:

- The model only ships an unquantized (BF16/FP16) or FP8-block checkpoint, and
  you want to evaluate a different runtime format (e.g. MXFP4 experts) without
  rebuilding the checkpoint offline.
- You want to sweep mixed-precision recipes (different formats for attention vs.
  MoE experts vs. shared experts) on the same source weights.
- You need a quick A/B between FP8 and MXFP4 on the same model without
  downloading two separate Hugging Face repos.

Prefer an offline pre-quantized checkpoint (e.g. `amd/DeepSeek-R1-0528-MXFP4`)
when one already exists for your target format — it has lower load time,
deterministic per-layer assignment, and no online quantization overhead on
every restart.

### Supported source-checkpoint formats

Online quantization is only activated when the source model's `quant_method`
is one of:

| Source `quant_method` | Behavior |
|---|---|
| _(none, i.e. BF16/FP16 model)_ | Quantized directly from float weights. |
| `fp8` (block FP8, `QuantType.per_1x128`) | FP8 block weights are dequantized to BF16 first, then re-quantized. |
| `mxfp4` | **Not re-quantized.** Source MXFP4 weights are currently passed through unchanged — there is no dequant path for `per_1x32`, so the requested target format does not take effect on these layers. |

---

## 2. Configuration syntax

The flag accepts a single JSON object with three optional fields:

```bash
--online_quant_config '{
  "global_quant_config": "ptpc_fp8",
  "layer_quant_config": {"*expert*": "mxfp4"},
  "exclude_layer": ["lm_head", "*.gate.*"]
}'
```

| Field | Type | Description |
|---|---|---|
| `global_quant_config` | `str` | Default target format applied to every Linear / MoE layer. Omit (or pass `""`) to leave non-matching layers at their source precision. |
| `layer_quant_config` | `dict[str, str]` | Per-layer target overrides. Keys are fnmatch-style globs such as `"*expert*"`, `"*.mlp.gate_proj"`. Matched layers override `global_quant_config`. |
| `exclude_layer` | `str` \| `list[str]` | Layer name patterns to leave at source precision. Supports exact match and glob (`*`). Prefer a JSON list when excluding more than one pattern. |

Resolution order for a given layer name:

1. If it matches `exclude_layer` → not quantized.
2. Otherwise, first matching `layer_quant_config` pattern (in dict order).
3. Otherwise, fall back to `global_quant_config`.
4. If `global_quant_config` is also empty, the layer keeps its source format.

### 2.1 Target formats

Only two target formats are currently supported. Any other string (for example
`ptpc_i8`, `mxi4`, `mxfp8`) will either be rejected by the JSON parser or
trigger an assertion in the loader when the layer's weight is quantized.

| Format string | Underlying `QuantType` | Weight dtype |
|---|---|---|
| `ptpc_fp8` | `QuantType.per_Token` | `torch.float8_e4m3fn` |
| `mxfp4` | `QuantType.per_1x32` | packed FP4 (`torch.float4_e2m1fn_x2`, group size 32) |

### 2.2 Picking the right pattern

ATOM's resolver runs against the **fully-qualified layer name** as reported by
`model.named_modules()`. Useful patterns:

| Pattern | Matches | Why |
|---|---|---|
| `"*expert*"` | MoE expert weights (e.g. `model.layers.3.mlp.experts`) | Substring match on the fused expert module. |
| `"*.gate.*"` | MoE router / gate Linear | Always exclude — quantizing the router destroys top-k accuracy. |
| `"lm_head"` | Output projection | Always exclude — kept at source precision avoids logit-distribution shift. |
| `"*shared_expert*"` | Shared experts in DeepSeek / Qwen3 MoE | Keep at higher precision if you see accuracy regressions. |

---

## 3. Recipes

The four recipes below are the configurations validated in
[ROCm/ATOM#653](https://github.com/ROCm/ATOM/pull/653). Each has been A/B
tested against its offline-quantized equivalent on gsm8k accuracy and
ISL=1024 / OSL=1024 / concurrency=128 throughput.

All commands assume you are inside the standard ATOM container
(`docker pull rocm/atom:latest`).

### 3.1 Qwen3-30B-A3B-Thinking-2507 — full per-token FP8

BF16 source → every Linear and the fused expert module quantized to
`ptpc_fp8`. The matching offline checkpoint is
`amd/Qwen3-30B-A3B-Thinking-2507-ptpc`.

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  -tp 4 \
  --online_quant_config '{
    "global_quant_config": "ptpc_fp8",
    "exclude_layer": ["lm_head", "*.gate.*"]
  }'
```

### 3.2 Qwen3-235B-A22B-Instruct-2507 — full MXFP4

BF16 source → every Linear (including experts) quantized to `mxfp4`, served
with expert parallel.

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  -tp 2 --enable-expert-parallel \
  --online_quant_config '{
    "global_quant_config": "mxfp4",
    "exclude_layer": ["lm_head", "*.gate.*"]
  }'
```

### 3.3 DeepSeek-R1-0528 — FP8 attention + MXFP4 experts

FP8 source → non-expert Linear stays at `ptpc_fp8`, fused MoE experts are
downgraded to `mxfp4`. The matching offline checkpoint layout is
`amd/DeepSeek-R1-0528-MXFP4`.

```bash
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --enforce-eager -tp 8 \
  --online_quant_config '{
    "global_quant_config": "ptpc_fp8",
    "layer_quant_config": {"*expert*": "mxfp4"},
    "exclude_layer": ["lm_head", "*.gate.*"]
  }'
```

`--enforce-eager` mirrors the configuration used by the PR's accuracy
reproduction. Drop it to get full CUDA-graph throughput; it does not affect
the online quantization output.

### 3.4 DeepSeek-R1-0528 + MTP-3 — FP8 attention + MXFP4 experts

Same online quantization recipe as § 3.3, layered with MTP-3 speculative
decoding for ~2.5× lower TPOT.

```bash
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --enforce-eager -tp 8 \
  --method mtp --num-speculative-tokens 3 \
  --online_quant_config '{
    "global_quant_config": "ptpc_fp8",
    "layer_quant_config": {"*expert*": "mxfp4"},
    "exclude_layer": ["lm_head", "*.gate.*"]
  }'
```

`--method mtp --num-speculative-tokens 3` is independent of online
quantization — it can be added to any of the recipes above without changing
the `--online_quant_config` JSON.

---

## 3.5 Plugin mode (`vllm serve`)

In the [vLLM out-of-tree plugin backend](vllm_plugin_backend_guide.md) you launch
with `vllm serve`, whose CLI does not understand ATOM's `--online_quant_config`
flag. Instead, pass the **same JSON object** through vLLM's official plugin
escape hatch `--additional-config`, under the `online_quant_config` key. ATOM
reads it during the vLLM→ATOM config translation and routes it through the
identical load-time quantization path (`process_weights_after_loading`),
including the `online_quant_info_*.json` dump described in § 4.

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --additional-config '{"online_quant_config": {
    "global_quant_config": "ptpc_fp8",
    "layer_quant_config": {"*expert*": "mxfp4"},
    "exclude_layer": ["lm_head", "*.gate.*"]
  }}'
```

The schema, target formats, pattern semantics, and resolution order are
identical to the `--online_quant_config` flag documented in § 2. Omitting it
leaves weights at their source precision. As with the standalone flag, online
quantization only activates when the source checkpoint's `quant_method` is
unquantized or per-block FP8 (see § 1).

---

## 4. Verifying the result

When online quantization runs, rank 0 writes
`online_quant_info_<timestamp>_<ns>.json` to:

1. `$ATOM_TORCH_PROFILER_DIR` if the env var is set, otherwise
2. the current working directory.

A representative payload:

```json
{
  "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
  "online_quant_config": {
    "global_quant_config": "ptpc_fp8",
    "exclude_layer": ["lm_head", "*.gate.*"]
  },
  "elapsed_seconds": 2.343,
  "num_layers": 144,
  "layers": [
    {
      "layer": "model.layers.0.self_attn.qkv_proj",
      "quant_type": "per_Token",
      "quant_dtype": "torch.float8_e4m3fn"
    },
    {
      "layer": "model.layers.0.mlp.experts",
      "quant_type": "per_Token",
      "quant_dtype": "torch.float8_e4m3fn"
    }
  ]
}
```

Things to check:

- `num_layers` matches your expectation. For a Qwen3 MoE with 48 transformer
  blocks you should see `48 × 3 = 144` entries (qkv_proj + o_proj + experts).
  A drastically smaller count usually means a typo in the pattern made
  everything fall into `exclude_layer`.
- Per-layer `quant_type` / `quant_dtype` reflect the format you intended for
  that pattern. The mapping is:

  | Format string | `quant_type` | `quant_dtype` |
  |---|---|---|
  | `ptpc_fp8` | `per_Token` | `torch.float8_e4m3fn` |
  | `mxfp4` | `per_1x32` | `torch.uint8` (packed FP4x2) |

- `elapsed_seconds` indicates the post-loading processing time on rank 0. A
  large jump from one restart to another with the same config usually points
  to a TP gather being triggered (see § 5.2).

The runtime also logs a one-line summary in the server log:

```
Weight post-processing done: 2.34 seconds, 144 layers online-quantized
Online quantization info saved to /root/online_quant_info_20260525_033839_112444436.json
```

---

## 5. Notes and gotchas

### 5.1 When online quantization activates

`--online_quant_config` is only applied when the source checkpoint's
`quant_method` is unquantized or per-block FP8 (see § 1).

### 5.2 Tensor-parallel behavior

Tensor-parallel weights are gathered onto a single rank before quantization
**only** when local quantization would produce different scales than quantizing
the full unpartitioned weight. Concretely:

- `ptpc_fp8` (`per_Token`): scales are per output channel and the channel
  dimension is exactly what TP shards on, so quantization is done locally with
  no gather.
- `mxfp4` (`per_1x32`): scales are within 32-element blocks along the input
  dimension; for `RowParallelLinear` this requires a gather on the input dim
  before quantization, then re-sharding. This is the most expensive case.

If load time grows linearly with TP size, your recipe is hitting the gather
path.

### 5.3 Only Linear and fused MoE modules are quantized

Modules whose weights are not loaded through ATOM's `LinearMethodBase` or
`FusedMoEMethodBase` paths are skipped silently. In practice this means
embeddings, layernorms, attention bias, and any custom op kept in BF16 will not
appear in `online_quant_info_*.json` — that is expected.

### 5.4 Compile cache

The compile cache (`/root/.cache/atom/*`) is keyed on the full quantization
config hash. Switching `--online_quant_config` between runs will trigger a
recompile on first startup. If you are iterating rapidly:

```bash
rm -rf /root/.cache/atom/*
```

### 5.5 Always exclude the MoE gate

The MoE router (`*.gate.*`) is a tiny Linear that produces top-k routing
logits. Quantizing it consistently produces large accuracy drops on every MoE
model we have measured. Keep it in the exclude list unless you have a specific
reason not to.

---
