# Quantization

AtomRL implements a comprehensive quantization stack supporting FP8, MXFP8, and MXFP4. The [FP8-RL](https://arxiv.org/abs/2601.18150) stack provides low-precision rollouts and hybrid FP8 training in Lumen with importance-sampling correction. MXFP8 and MXFP4 online quantization via AITER enables efficient inference for SDDD teacher models.

## FP8 rollout (ATOM path)

### W8A8 linear layers

`FP8RolloutQuantizer` replaces eligible `nn.Linear` modules with a blockwise W8A8 path (`atomrl.quantization.fp8_rollout`). Weights are quantized once per replacement; activations are quantized per forward using dynamic scales derived from 128-wide blocks.

Key configuration fields (YAML → `RolloutQuantConfig`):

| Field | Meaning |
| --- | --- |
| `precision` | `"fp8"` enables rollout quantizers; `"bf16"` disables |
| `use_deep_gemm` | Prefer fused GEMM when `deep_gemm` is installed |
| `num_first_layers_in_bf16` / `num_last_layers_in_bf16` | Keep boundary layers in BF16 for stability |

### KV-cache quantization

`FP8KVCacheQuantizer` maintains per-projection FP8 scales on `q_proj`, `k_proj`, and `v_proj` modules. `recalibrate_scales` should run whenever policy weights change—typically every RL step before ATOM generation—because FP8 scales track the current BF16 weights.

### Weight quantizer

`WeightQuantizer` provides standalone blockwise FP8 packing for `state_dict` tensors (`quantize_state_dict` / `dequantize_state_dict`). Rollout code reuses the same quantizer for consistent numerics between offline transforms and live inference patches.

## FP8 training (Lumen path)

`FP8TrainingManager` reads `QuantizationConfig` (or a full `AtomRLConfig`) and calls `atomrl.lumen.quantize.enable` with a `QuantConfig` derived from:

| YAML field | Effect |
| --- | --- |
| `quantization.training.fp8` | When set (for example `"hybrid"`), overrides rollout precision inside `FP8Config.from_config` |
| `quantization.training.fp8_recipe` | `"blockwise"` (default) or `"tensorwise"` when supported |
| `quantization.training.fp8_weight_cache` | Registers optimizer post-step hooks to avoid redundant re-quantization |

Hybrid E4M3/E5M2 training follows Lumen’s implementation: forward activations/weights use narrower mantissa where safe, while backward paths use wider dynamic range formats.

## Rollout correction (TIS and MIS)

When FP8 rollouts sample actions under `log p_fp8` but training evaluates `log p_bf16`, reweight advantages with closed-form token corrections (`atomrl.quantization.rollout_correction`).

### TIS (truncated importance sampling)

Let `ρ_t = exp(log π_bf16 - log π_fp8)`. TIS clamps `ρ_t` to `[1/c, c]` where `c` is `RolloutCorrectionConfig.clip` (default `1.5`), then multiplies advantages:

```python
from atomrl.quantization.rollout_correction import token_level_tis

adv_corrected = token_level_tis(bf16_logprobs, fp8_logprobs, advantages, clip=1.5)
```

### MIS (mean importance sampling)

MIS keeps raw ratios but normalizes to mean 1 across tokens, reducing bias from global scale drift:

```python
from atomrl.quantization.rollout_correction import token_level_mis

adv_corrected = token_level_mis(bf16_logprobs, fp8_logprobs, advantages)
```

### Configuration

```yaml
quantization:
  rollout_correction:
    enabled: true
    method: tis   # or mis
    clip: 1.5     # used by tis
```

`apply_rollout_correction` clones the incoming `DataProto`, replaces `advantages`, and annotates `meta["rollout_correction"]`.

## Memory and throughput trade-offs

Illustrative expectations when enabling the full FP8 stack on MI300-class GPUs (model and sequence dependent):

| Mode | Activations / KV | Policy memory | Notes |
| --- | --- | --- | --- |
| BF16 baseline | BF16 | Highest | Simplest numerics |
| FP8 rollout only | FP8 | Moderate reduction | Still trains BF16 weights |
| FP8 rollout + training | FP8 | Largest reduction | Requires Lumen FP8 support on backend |
| + `fp8_weight_cache` | n/a | Fewer re-quant spikes | Helps large MoE actors |

```{warning}
Always validate FP8 recipes against a short BF16 reference run (`num_training_steps` 50–100) before scaling out; correction hyperparameters (`clip`, `method`) interact with reward scale.
```

Further reading: {doc}`/examples/fp8_training`, {doc}`/api/quantization`, {doc}`/api/config`.
