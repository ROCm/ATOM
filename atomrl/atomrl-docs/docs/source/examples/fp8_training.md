# FP8 training example

This example mirrors `configs/grpo_dense_fp8.yaml`: FP8 W8A8 rollout in ATOM, hybrid FP8 training in Lumen, and token-level TIS correction so FP8 rollouts remain usable as a behavior policy.

## FP8 rollout configuration

Rollout precision lives under `quantization.rollout`. The reference recipe enables blockwise W8A8 with DeepGEMM hooks when available:

```yaml
quantization:
  rollout:
    precision: fp8
    use_deep_gemm: true
    num_first_layers_in_bf16: 0
    num_last_layers_in_bf16: 0
```

Keeping the first/last layers in BF16 (`num_*_layers_in_bf16`) is useful when numerics at the embedding or LM head are fragile; start with all-FP8, then selectively relax boundary layers if you observe instability.

## FP8 training configuration

Training-time FP8 is controlled by `quantization.training`:

```yaml
quantization:
  training:
    fp8: hybrid
    fp8_recipe: blockwise
    fp8_weight_cache: true
```

`fp8: hybrid` maps to Lumen’s hybrid E4M3/E5M2 style training path via `FP8TrainingManager`. `fp8_weight_cache` registers optimizer post-step hooks so cached quantized weights stay warm across micro-batches.

## Rollout correction

When rollouts run in FP8 but PPO-style losses re-evaluate actions under BF16, enable correction to adjust advantages:

```yaml
quantization:
  rollout_correction:
    enabled: true
    method: tis
    clip: 1.5
```

`apply_rollout_correction` looks for FP8 log-probs (`fp8_logprobs` or `fp8_log_probs`) and BF16-side references (`bf16_logprobs`, `old_log_probs`, or `ref_log_probs`). See {doc}`/advance/fp8_quantization` for the TIS/MIS formulas.

## Comparing FP8 against BF16

Run two short jobs with identical overrides except precision:

```bash
# BF16 baseline
python examples/run_grpo.py --config configs/grpo_dense_bf16.yaml \
  num_training_steps=100 logger.log_interval=1

# FP8 stack
python examples/run_grpo.py --config configs/grpo_dense_fp8.yaml \
  num_training_steps=100 logger.log_interval=1
```

Compare:

| Signal | BF16 baseline | FP8 stack |
| --- | --- | --- |
| Tokens / second | Lower memory bandwidth use | Often higher; depends on DeepGEMM/ATOM path |
| `loss_pg` noise | Lower | Slightly higher; watch correction meta |
| GPU memory headroom | Smaller | Larger KV and activation footprint savings |

```{warning}
If you disable `rollout_correction` while keeping FP8 rollouts on, monitor KL to the reference policy and off-policy metrics—uncorrected FP8 behavior policies can bias GRPO/DAPO advantages.
```

## Related documentation

- Conceptual deep dive: {doc}`/advance/fp8_quantization`
- Python APIs: {doc}`/api/quantization`, {doc}`/api/config`
