# MoE Rollout Routing Replay (R3)

MoE reinforcement learning couples two different code paths: high-throughput inference routers (ATOM) and training-time routers inside Megatron/Lumen. Small numerical and implementation differences can yield different expert assignments for identical tokens, which shows up as unstable KL divergences and collapsed expert utilization.

## Problem statement

During on-policy or near on-policy training we expect the training forward to evaluate the same routing distribution as the rollout engine for each token. In practice:

- Inference stacks may fuse softmax, top-k, and dispatch differently than training.
- FP8 rollouts change logits slightly, which alters argmax routes even when BF16 training is “close.”
- Expert parallel sharding reorders reductions, amplifying bitwise drift.

The result is **router inconsistency**: the policy gradient is computed against a different routing process than the one that produced the data, which breaks the stationarity assumptions underlying PPO-style surrogates.

## R3 solution (record / transfer / replay)

R3 implements a three-phase contract:

1. **Record** — During ATOM rollout, `RouterRecorder` installs forward hooks on detected MoE modules and stores CPU float copies of router logits keyed by layer index.
2. **Transfer** — `R3Manager.transfer_distributions` copies recorded tensors into `DataProto` using `add_router_distributions`, preserving them across Ray merges.
3. **Replay** — `RouterReplayer` installs hooks during training forwards to overwrite router outputs with the recorded logits (mode controlled by `replay_mode`).

This aligns expert choice distributions between engines without forcing a single codebase path for inference and training.

## Configuration

```yaml
moe:
  r3:
    enabled: true
    record_router_logits: true
    replay_mode: distribution   # or hard_assignment
```

| Field | Purpose |
| --- | --- |
| `enabled` | Master switch for recorder/replayer contexts |
| `record_router_logits` | When false, record phase becomes a no-op but APIs remain safe |
| `replay_mode` | `distribution` feeds soft logits; `hard_assignment` can lock discrete decisions when supported |

Structured types are `MoEConfig` → `R3Config` in {doc}`/api/config`.

## Monitoring

Use KL and utilization side channels alongside standard RL metrics:

- **Policy–reference KL** — spikes often precede MoE collapse when routers disagree.
- **Expert utilization** — compute from captured `router_logits` with `check_expert_utilization`.
- **Router entropy** — `compute_router_entropy` tracks how decisive routing is over time.

```python
from atomrl.moe.moe_utils import compute_router_entropy, check_expert_utilization

util = check_expert_utilization(router_logits, num_experts=128)
ent = compute_router_entropy(router_logits)
```

## Best practices

- Enable R3 for **all MoE RL runs** unless you are explicitly isolating a baseline.
- Keep `record_router_logits=true` until you verify hooks are lightweight in your deployment.
- When debugging, log histograms of `router_dist_layer_*` norms per layer to spot missing tensors after `DataProto.merge`.
- Pair R3 with FP8 correction ({doc}`/advance/fp8_quantization`) when rollouts are quantized—both address distinct mismatch sources.

API entry points: {doc}`/api/moe` and {doc}`/api/protocol`.
