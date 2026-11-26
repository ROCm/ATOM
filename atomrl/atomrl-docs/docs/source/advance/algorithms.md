# Algorithms

AtomRL supports two categories of post-training algorithms: **RL algorithms** (GRPO, DAPO, PPO) for reinforcement learning from human feedback and reward models, and **SDDD** (Speculative Decoding Draft Distillation) for training speculative decoding draft models.

## RL Algorithms

AtomRL ships three policy-gradient-style trainers—GRPO, DAPO, and PPO—that share the same `DataProto` batching and loss helpers. Pick the algorithm based on whether you need a critic, asymmetric clipping, or group-relative rewards.

## GRPO

Group Relative Policy Optimization normalizes scalar rewards within each prompt’s `num_generations` group and applies a symmetric clipped surrogate on token log-probabilities.

- **Group-relative advantages** — for rewards `r` shaped `[B]`, reshape to `[B // G, G]`, subtract group means, divide by group standard deviation, then flatten.
- **Clipped surrogate** — uses `policy_gradient_loss` with `algorithm.grpo.clip_ratio`.

Configuration lives in `AlgorithmConfig.grpo` (see {doc}`/api/config`). Tensor requirements are documented on `GRPOAlgorithm` in `atomrl/algorithms/grpo.py`.

## DAPO

Decoupled Advantage Policy Optimization (DAPO-style) keeps group-relative advantages but adds training stabilizers:

- **Asymmetric clip** — ratios clamp to `[clip_ratio_low, clip_ratio_high]` via `asymmetric_clip_loss`.
- **Dynamic sampling** — when `dynamic_sampling=true`, entire groups with near-zero reward variance are masked out (`dapo_sample_mask`).
- **Overlong shaping** — optional linear penalty subtracted from rewards when `batch.meta["response_lengths"]` exceeds `policy.max_total_sequence_length`.

Enable token-level policy gradients with `token_level_pg=true` (default).

## PPO

Classic Proximal Policy Optimization uses a critic network and GAE-Lambda advantages:

- **GAE-Lambda** — `_gae_returns` constructs token-level advantages and returns using `algorithm.ppo.gae_lambda` and `discount`.
- **Value loss** — `value_loss` clips value deltas similar to the policy ratio clip.
- **Entropy bonus** — although not always logged separately, KL and entropy-style regularizers use `kl_penalty` / `entropy_bonus` helpers in related code paths.

PPO requires `values`, `attention_mask`, and compatible `rewards` tensors on the batch.

## Loss functions reference

| Function | Role |
| --- | --- |
| `policy_gradient_loss` | Symmetric clipped policy surrogate |
| `asymmetric_clip_loss` | DAPO-style asymmetric clipping |
| `value_loss` | Clipped critic regression |
| `kl_penalty` | Reference KL penalty `mean(ref_logp - logp)` |
| `entropy_bonus` | Sampling entropy surrogate from token log-probs |

## Choosing an RL algorithm

| Scenario | Suggested algorithm |
| --- | --- |
| Preference-style scalar rewards, no critic | GRPO |
| Long responses, asymmetric off-policy noise, token-level updates | DAPO |
| Classic actor–critic with value bootstrapping | PPO |

```{warning}
GRPO and DAPO assume the batch is grouped in contiguous blocks of size `num_generations`. If your sampler interleaves prompts, normalize ordering before calling `compute_advantages`.
```

Python hooks and registry keys: {doc}`/api/algorithms`.

## SDDD (Speculative Decoding Draft Distillation)

SDDD trains lightweight draft models (Eagle3, DFlash) to predict a teacher model's hidden states, enabling speculative decoding at inference time. Unlike RL algorithms that optimize a policy via reward signals, SDDD uses direct distillation from teacher hidden states.

- **Teacher inference**: A large teacher model (e.g., Kimi K2.5, Qwen3-8B) runs on ATOM/vLLM, processing datasets and producing hidden states at each layer.
- **Hidden-state transfer**: Teacher hidden states are transferred to training GPUs via Mooncake (async RDMA/TCP) or MORI (GPU Direct P2P RDMA).
- **Draft model training**: An Eagle3 or DFlash draft model is trained via FSDP2 to predict the teacher's hidden states, learning to approximate the teacher's internal representations.

SDDD supports a two-phase training recipe: Phase 1 for foundation training on a large general dataset, and Phase 2 for domain-specific diversity.

For detailed configuration and examples, see {doc}`/advance/sddd`.
