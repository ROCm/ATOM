# Algorithms API

The `atomrl.algorithms` package implements policy-gradient algorithms on top of `DataProto` batches. All concrete classes inherit from `BaseAlgorithm` and register themselves in `ALGORITHM_REGISTRY`.

## `BaseAlgorithm`

```python
class BaseAlgorithm:
    def __init__(self, config: AtomRLConfig) -> None: ...

    def compute_advantages(self, batch: DataProto) -> DataProto: ...
    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]: ...
    def get_config(self) -> dict[str, Any]: ...
```

- `compute_advantages` mutates or replaces batch tensors such as `advantages`, `returns`, or `dapo_sample_mask`.
- `compute_loss` returns a scalar `loss` for backpropagation plus a `metrics` dict with detached floats for logging.

## `GRPOAlgorithm`

Group Relative Policy Optimization:

- Requires `rewards` shaped `[B]` (or `[B, 1]`).
- Builds `advantages` by normalizing rewards within each contiguous group of `config.algorithm.grpo.num_generations` rows.
- `compute_loss` consumes `log_probs`, `old_log_probs`, optional `ref_log_probs` (when `kl_coeff > 0`), and optional `response_mask` / `attention_mask`.

```python
from atomrl.algorithms import GRPOAlgorithm
from atomrl.core.config import AtomRLConfig

algo = GRPOAlgorithm(AtomRLConfig())
batch = algo.compute_advantages(batch)
loss, metrics = algo.compute_loss(batch)
```

## `DAPOAlgorithm`

DAPO-style updates extend GRPO-style advantages with optional filtering and asymmetric clipping:

- May write `dapo_sample_mask` when `dynamic_sampling` is enabled.
- Applies `overlong_reward_shaping` when configured and `batch.meta["response_lengths"]` is present.
- `compute_loss` respects `token_level_pg` to switch between token-wise and sequence-averaged asymmetric objectives.

## `PPOAlgorithm`

Classic PPO with GAE-Lambda:

- Requires `values` `[B, T]`, `attention_mask`, and `rewards` (scalar per sequence or token-aligned tensor).
- Writes `advantages` and `returns` tensors used by `compute_loss`.
- Adds `0.5 * value_loss` to the policy loss and logs `explained_variance` when masks permit.

## `ALGORITHM_REGISTRY`

`atomrl.core.registry.Registry` mapping `algorithm.name` strings to constructors or callables:

```python
from atomrl.core.registry import ALGORITHM_REGISTRY

cls = ALGORITHM_REGISTRY.get("grpo")
algo = cls(config)
```

Registered keys mirror `AlgorithmName` (`grpo`, `dapo`, `ppo`).

## Loss functions (`atomrl.algorithms.loss_functions`)

### `policy_gradient_loss(logprobs, old_logprobs, advantages, clip_ratio, *, mask=None)`

Symmetric clipped surrogate:

```python
from atomrl.algorithms.loss_functions import policy_gradient_loss

loss_pg = policy_gradient_loss(
    log_probs, old_log_probs, adv_tokens, clip_ratio=0.2, mask=response_mask
)
```

### `asymmetric_clip_loss(logprobs, old_logprobs, advantages, clip_low, clip_high, *, mask=None)`

Same as above but clamps ratios to `[clip_low, clip_high]` (DAPO).

### `value_loss(values, old_values, returns, clip_ratio, *, mask=None)`

Clipped value regression combining clipped/unclipped MSE terms.

### `kl_penalty(logprobs, ref_logprobs, *, mask=None)`

Token-average `ref_logprobs - logprobs`, used as an approximate KL penalty.

### `entropy_bonus(logprobs, *, mask=None)`

Surrogate entropy from sampled log-probs; **subtract** `coeff * entropy_bonus` from the loss to encourage exploration.

```{note}
Import concrete algorithms from `atomrl.algorithms`; import loss helpers from `atomrl.algorithms.loss_functions` directly.
```

See also: {doc}`/api/config`, {doc}`/api/protocol`, {doc}`/advance/algorithms`.
