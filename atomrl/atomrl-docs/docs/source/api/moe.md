# MoE API

`atomrl.moe` contains Rollout Routing Replay utilities, expert-parallel helpers, and router diagnostics.

## `RouterRecorder`

Forward hooks capture MoE router logits during inference.

| Method | Description |
| --- | --- |
| `install_hooks(model)` | Registers hooks on modules detected by `iter_moe_modules` |
| `remove_hooks()` | Detaches all recorder hooks |
| `get_distributions()` | Returns `dict[int, Tensor]` of CPU float logits keyed by layer index |
| `clear()` | Drops recorded tensors without removing hooks |
| `recording(model)` | Context manager that installs/removes hooks around a code block |

```python
from atomrl.moe.router_recorder import RouterRecorder

rec = RouterRecorder()
with rec.recording(model):
    _ = model(tokens)
router_tensors = rec.get_distributions()
```

## `RouterReplayer`

Replaces router outputs during training to match recorded logits.

| Method | Description |
| --- | --- |
| `install_hooks(model, distributions)` | Maps `distributions` to MoE layers and registers forward hooks |
| `remove_hooks()` | Clears hooks and internal caches |

```python
from atomrl.moe.router_replayer import RouterReplayer

rep = RouterReplayer()
rep.install_hooks(train_model, router_tensors)
try:
    loss = train_model(batch)
finally:
    rep.remove_hooks()
```

## `R3Manager`

High-level orchestration tying recorder/replayer together with `DataProto`.

| Member | Description |
| --- | --- |
| `record_phase(model)` | Context manager yielding `RouterRecorder` when R3 is enabled |
| `replay_phase(model, distributions)` | Context manager yielding `RouterReplayer` with hooks installed |
| `transfer_distributions(data, recorded)` | Static helper cloning a `DataProto` and attaching `router_dist_layer_*` tensors |
| `clear()` | Clears recorder buffers and removes stray hooks |

```python
from atomrl.core.config import R3Config
from atomrl.moe.r3_manager import R3Manager

mgr = R3Manager(R3Config(enabled=True))
with mgr.record_phase(atom_model):
    sequences = atom_model.generate(...)
recorded = mgr.recorder.get_distributions()
batch = R3Manager.transfer_distributions(batch, recorded)
with mgr.replay_phase(lumen_model, batch.get_router_distributions()):
    loss, metrics = algo.compute_loss(batch)
```

## `ExpertParallelManager`

Checkpoint-oriented expert parallel utilities.

| Method | Description |
| --- | --- |
| `setup_ep(model, ep_size)` | Stores `_atomrl_expert_parallel_size` on the model |
| `reshard_for_inference(state_dict, train_ep_size, infer_ep_size)` | Concatenates expert shards when narrowing EP width |

```python
from atomrl.moe.expert_parallel import ExpertParallelManager

epm = ExpertParallelManager(config.policy.training.megatron_cfg)
epm.setup_ep(model, ep_size=2)
new_sd = epm.reshard_for_inference(state_dict, train_ep_size=4, infer_ep_size=2)
```

## `moe_utils`

| Symbol | Description |
| --- | --- |
| `iter_moe_modules(model)` | Yields `(layer_index, qualified_name, module)` candidates |
| `compute_load_balance_loss(router_logits, num_experts, top_k)` | Switch-style auxiliary loss encouraging balanced experts |
| `compute_router_entropy(router_logits)` | Mean Shannon entropy of softmax router distributions |
| `check_expert_utilization(router_logits, num_experts)` | Returns summary dict with per-expert mean/std softmax mass |

```python
from atomrl.moe import moe_utils

for idx, name, module in moe_utils.iter_moe_modules(model):
    print(idx, name, type(module).__name__)

diag = moe_utils.check_expert_utilization(router_logits, num_experts=64)
entropy = moe_utils.compute_router_entropy(router_logits)
```

```{note}
`RouterRecorder` / `RouterReplayer` rely on heuristics in `iter_moe_modules`; exotic MoE block names may need additional patterns if hooks fail to attach.
```

See also: {doc}`/advance/moe_r3`, {doc}`/api/protocol`, {doc}`/api/config`.
