# DataProto API

`atomrl.core.protocol.DataProto` is the tensor batch exchanged between Ray workers. Tensors live on CPU for serialization; each worker calls `cuda()` or `to(device)` before math.

## Constructor

```python
from atomrl.core.protocol import DataProto
import torch

batch = DataProto(
    tensors={
        "input_ids": torch.ones(4, 32, dtype=torch.long),
        "attention_mask": torch.ones(4, 32, dtype=torch.bool),
    },
    meta={"response_lengths": [12, 10, 15, 9]},
)
```

- `tensors` — optional `dict[str, torch.Tensor]`; defaults to `{}`.
- `meta` — optional `dict[str, Any]` for small python side metadata (lengths, flags, metrics).

## Item access

`DataProto` implements mapping-style accessors on **tensor** keys:

```python
batch["input_ids"] = batch["input_ids"].clamp(min=0)
assert "attention_mask" in batch
print(len(batch))  # batch size (dim 0 of first tensor)
```

`keys()` lists tensor keys; `batch_size` mirrors `len(batch)`.

## Split and merge

```python
chunks = batch.split(2)          # two near-equal parts along dim 0
merged = DataProto.merge(chunks)
```

`merge` concatenates along dimension 0 and requires identical tensor keys for non-empty chunks. Use this after collecting rollout shards from data-parallel workers.

## Mini-batches

```python
for mini in batch.mini_batches(batch_size=2):
    train_on_gpu(mini.cuda())
```

Yields new `DataProto` views referencing sliced tensors; `meta` is shallow-copied per chunk.

## Device movement

```python
on_gpu = batch.to("cuda:0")
also_gpu = batch.cuda(device=0)
back = on_gpu.cpu()
```

`meta` is copied via `dict.copy()`; tensors are moved with `.to`.

## Router distribution helpers (R3)

```python
batch.add_router_distributions(layer_idx=3, logits=router_logits_cpu)
all_dists = batch.get_router_distributions()  # dict[int, Tensor]
assert batch.has_router_distributions()
```

Recorded tensors are stored as `router_dist_layer_{idx}` on the CPU for deterministic serialization.

## Utility methods

| Method | Description |
| --- | --- |
| `select(keys)` | New `DataProto` with a subset of tensors |
| `update(other)` | In-place merge of tensors and meta from another proto |

## Common tensor keys

| Category | Keys | Required by |
| --- | --- | --- |
| Prompting | `input_ids`, `attention_mask` | generation |
| Policy | `log_probs`, `old_log_probs` | GRPO/DAPO/PPO losses |
| Rewards | `rewards` | GRPO/DAPO advantage code |
| Critic | `values`, `returns`, `old_values` | PPO |
| Reference | `ref_log_probs` | KL penalties |
| Masks | `response_mask` | token masking (optional; falls back to `attention_mask`) |
| FP8 correction | `fp8_logprobs` / `fp8_log_probs`, `bf16_logprobs` | `apply_rollout_correction` |
| MoE R3 | `router_dist_layer_*` | R3 replay path |

```{warning}
Algorithms assume consistent batch axis ordering after `merge`. Do not concatenate unrelated prompts before GRPO/DAPO grouping logic unless you also rebuild `rewards` ordering.
```

See also: {doc}`/api/algorithms`, {doc}`/api/moe`, {doc}`/api/quantization`.
