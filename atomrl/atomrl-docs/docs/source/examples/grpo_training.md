# GRPO training example

Group Relative Policy Optimization (GRPO) normalizes rewards within each prompt’s sample group and applies a clipped surrogate policy loss—no critic network is required. This page walks through the shipped BF16 baseline, shows how to swap configs, and describes the log output you should expect.

## BF16 baseline configuration

The reference recipe `configs/grpo_dense_bf16.yaml` trains a small dense Qwen3 checkpoint with ATOM generation and FSDP2 updates:

```yaml
policy:
  model_name: Qwen/Qwen3-0.6B
  training_backend: fsdp2
  generation_backend: atom

algorithm:
  name: grpo
  grpo:
    num_generations: 8
    kl_coeff: 0.0
    clip_ratio: 0.2

quantization:
  rollout:
    precision: bf16
  training:
    fp8: null
```

`num_generations` must divide the rollout batch so advantages can be reshaped into `[num_prompts, num_generations]`.

## Running with different configs

Launch the same driver with another YAML path:

```bash
# Baseline dense BF16
python examples/run_grpo.py --config configs/grpo_dense_bf16.yaml

# Same algorithm family, FP8 stack enabled
python examples/run_grpo.py --config configs/grpo_dense_fp8.yaml

# Larger multi-GPU sweep
python examples/run_grpo.py --config configs/grpo_dense_fp8_8gpu.yaml
```

### CLI overrides

`AtomRLConfig.from_yaml` accepts OmegaConf dotlist overrides after the config path:

```bash
python examples/run_grpo.py --config configs/grpo_dense_bf16.yaml \
  policy.train_global_batch_size=16 \
  algorithm.grpo.num_generations=4 \
  num_training_steps=50
```

Overrides are handy for interactive debugging without duplicating entire YAML files.

## Expected output

While exact metrics depend on the reward function and dataset, healthy runs share these traits:

- **Scheduler logs** show Ray workers placed on visible GPUs without repeated restart loops.
- **Rollout logs** report generation throughput and sequence lengths within `policy.max_total_sequence_length`.
- **Training logs** emit monotonically increasing step indices and finite `loss_pg` scalars from `GRPOAlgorithm.compute_loss`.
- **Checkpoints** appear under `checkpointing.checkpoint_dir` every `save_steps` updates.

```{note}
If `num_generations` does not divide the batch size built by your dataset and rollout settings, GRPO advantage computation raises a clear error—fix the global batch or generations before chasing GPU issues.
```

## Next steps

- FP8 numerics: {doc}`/examples/fp8_training` and {doc}`/advance/fp8_quantization`
- Algorithm details: {doc}`/advance/algorithms` and {doc}`/api/algorithms`
