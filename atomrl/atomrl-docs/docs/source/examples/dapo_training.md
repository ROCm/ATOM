# DAPO Training

This example walks through DAPO (Decoupled Advantage Policy Optimization) training with asymmetric clipping and dynamic sampling.

## When to Use DAPO

DAPO is preferred over GRPO when:

- You want **asymmetric clipping** (different bounds for ratio increase vs decrease)
- Your reward signal has **high variance** across groups and you want to automatically filter degenerate groups via **dynamic sampling**
- Responses frequently exceed the max length and you want **overlong reward shaping**

## BF16 Baseline

```bash
python examples/run_dapo.py --config configs/dapo_dense_bf16.yaml
```

Example config:

```yaml
algorithm:
  name: dapo
  dapo:
    num_generations: 4
    clip_ratio_low: 0.8
    clip_ratio_high: 1.28
    kl_coeff: 0.02
    dynamic_sampling: true
    overlong_reward_shaping: true
    token_level_pg: true

policy:
  model_name: Qwen/Qwen3-0.6B
  training_backend: fsdp2
  max_total_sequence_length: 512
  train_micro_batch_size: 4
```

## FP8 DAPO

```bash
python examples/run_dapo.py --config configs/dapo_dense_fp8.yaml
```

The FP8 config adds quantization on top:

```yaml
quantization:
  rollout:
    precision: fp8
  training:
    fp8: hybrid
  rollout_correction:
    enabled: true
    method: tis
    clip: 1.5
```

## Key Differences from GRPO

| Feature | GRPO | DAPO |
|---------|------|------|
| Clip bounds | Symmetric `[1-ε, 1+ε]` | Asymmetric `[low, high]` |
| Sampling | All groups | Dynamic (filter zero-variance groups) |
| Length penalty | None | Overlong reward shaping |
| PG level | Token or sequence | Token-level by default |

## Monitoring

Watch for these metrics in the training logs:

- `loss_pg` — Policy gradient loss; should decrease
- `active_frac` — Fraction of groups kept after dynamic sampling; healthy range is 0.7–1.0
- `kl` — KL divergence from reference policy; should stay bounded

## Next Steps

- See {doc}`/advance/algorithms` for the full DAPO algorithm specification
- See {doc}`/advance/fp8_quantization` for FP8 configuration details
