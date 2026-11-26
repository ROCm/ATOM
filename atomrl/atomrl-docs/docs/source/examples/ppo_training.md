# PPO Training

This example demonstrates Proximal Policy Optimization with GAE-Lambda advantages and a value function.

## When to Use PPO

PPO is the right choice when:

- You have a **trained critic** (value function) and want GAE advantages
- You need **value loss clipping** for stable value estimation
- The task benefits from **multi-epoch** updates on each rollout batch

## Configuration

```yaml
algorithm:
  name: ppo
  ppo:
    clip_ratio: 0.2
    num_ppo_epochs: 4
    num_mini_batches: 4
    discount: 0.99
    gae_lambda: 0.95
    kl_coeff: 0.01

policy:
  model_name: Qwen/Qwen3-0.6B
  training_backend: fsdp2
  max_total_sequence_length: 512
  train_micro_batch_size: 4

cluster:
  num_nodes: 1
  gpus_per_node: 8
```

## Running PPO

```bash
python examples/run_grpo.py \
    --config configs/grpo_dense_bf16.yaml \
    algorithm.name=ppo \
    algorithm.ppo.num_ppo_epochs=4 \
    algorithm.ppo.num_mini_batches=4
```

PPO reuses the same launcher as GRPO — the algorithm is selected via `algorithm.name`.

## PPO with FP8

```bash
python examples/run_grpo.py \
    --config configs/grpo_dense_fp8.yaml \
    algorithm.name=ppo \
    algorithm.ppo.num_ppo_epochs=4
```

## Key Metrics

| Metric | Healthy Range | Indicates |
|--------|---------------|-----------|
| `loss_pg` | Decreasing | Policy improving |
| `loss_vf` | Decreasing | Value function fitting returns |
| `explained_variance` | > 0.5, trending toward 1.0 | Critic predicting returns well |
| `kl` | < 0.1 | Policy not drifting too far from reference |

## PPO vs GRPO vs DAPO

| Feature | PPO | GRPO | DAPO |
|---------|-----|------|------|
| Advantages | GAE-Lambda (critic) | Group-relative (no critic) | Group-relative + dynamic |
| Epochs per batch | Multiple (4 typical) | Single | Single |
| Value loss | Yes (clipped MSE) | No | No |
| Entropy bonus | Optional | No | No |
| Critic required | Yes | No | No |

## Next Steps

- See {doc}`/advance/algorithms` for the full PPO algorithm specification
- See {doc}`/api/algorithms` for the loss function API
- See {doc}`/advance/distributed` for multi-node training
