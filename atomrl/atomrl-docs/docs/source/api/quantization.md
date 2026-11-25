# Quantization API

`atomrl.quantization` groups FP8 rollout modules, KV-cache helpers, training bridges, and rollout correction utilities.

## `FP8Config`

Frozen dataclass summarizing numerics for both rollout and training paths.

| Field | Default | Description |
| --- | --- | --- |
| `precision` | `"bf16"` | Active precision string (`fp8`, `hybrid`, etc.) |
| `recipe` | `"blockwise"` | `"blockwise"` or `"tensorwise"` scaling recipe |
| `use_deep_gemm` | `True` | Prefer DeepGEMM where installed |
| `num_first_layers_in_bf16` / `num_last_layers_in_bf16` | `0` | BF16 boundary layers |
| `use_weight_pow2_scale` / `use_activation_pow2_scale` | `False` | Power-of-two scale rounding toggles |

Factory:

```python
from atomrl.core.config import AtomRLConfig
from atomrl.quantization import FP8Config

cfg = AtomRLConfig()
fp8 = FP8Config.from_config(cfg.quantization)
assert fp8.is_fp8_enabled() == (cfg.quantization.rollout.precision == "fp8" or bool(cfg.quantization.training.fp8))
```

## `WeightQuantizer`

Blockwise FP8 packing for weights:

```python
from atomrl.quantization import FP8Config, WeightQuantizer

wq = WeightQuantizer(FP8Config(precision="fp8"))
q_w, scales = wq.quantize_tensor(weight_matrix.float(), block_size=128)
restored = wq.dequantize_tensor(q_w, scales, block_size=128)
```

`quantize_state_dict` walks a PyTorch `state_dict` and replaces eligible 2-D weights with FP8 tensors plus `_fp8_scales` / `_fp8_meta` sidecars.

## `FP8RolloutQuantizer`

Installs W8A8 linear replacements:

```python
from atomrl.quantization import FP8Config, FP8RolloutQuantizer

quant = FP8RolloutQuantizer(FP8Config.from_config(config.quantization))
quant.quantize_model(actor_model)
# ... rollout ...
quant.restore_model(actor_model)
```

`should_skip_layer` consults `num_first_layers_in_bf16` / `num_last_layers_in_bf16` to keep early/late blocks in BF16.

## `FP8KVCacheQuantizer`

Maintains `_atomrl_fp8_kv_scale_{q,k,v}` buffers on projection layers:

```python
from atomrl.quantization import FP8Config, FP8KVCacheQuantizer

kvq = FP8KVCacheQuantizer(FP8Config(precision="fp8"))
kvq.enable(model)
kvq.recalibrate_scales(model)  # call after each weight swap
```

## `FP8TrainingManager`

Bridges YAML config to `atomrl.lumen.quantize`:

```python
from atomrl.quantization import FP8TrainingManager

mgr = FP8TrainingManager(config)
mgr.enable(model)
mgr.register_optimizer_hooks(optimizer)

# After resharding / weight reload
mgr.reset_fp8_state(model)
```

If `atomrl.lumen` is not installed, `enable` logs a warning and returns without mutating the model.

## Rollout correction

### `token_level_tis(bf16_logprobs, fp8_logprobs, advantages, clip=1.5)`

Truncated importance sampling: builds `exp(log_bf16 - log_fp8)`, clamps to `[1/clip, clip]`, multiplies advantages.

### `token_level_mis(bf16_logprobs, fp8_logprobs, advantages)`

Mean-normalized ratios so the average weight is 1.

### `apply_rollout_correction(batch, config) -> DataProto`

Reads `RolloutCorrectionConfig` nested inside `AtomRLConfig` / `QuantizationConfig`, picks BF16/FP8 log-prob tensors from known keys, and returns a new `DataProto` with updated `advantages`.

```python
from atomrl.quantization import apply_rollout_correction

batch = apply_rollout_correction(batch, config)
```

```{warning}
Correction functions assume token-aligned log-probability tensors. Sequence-level averages must be broadcast before calling these helpers.
```

See also: {doc}`/advance/fp8_quantization`, {doc}`/api/config`, {doc}`/api/protocol`.
