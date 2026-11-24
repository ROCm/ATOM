# atomrl.lumen.quantize

The quantization lifecycle module — entry point for enabling FP8 training on any model.

## quant.enable

```python
atomrl.lumen.quantize.enable(model, config=None, *, format=None, scaling=None)
```

Patches `model` in-place for quantized training. Walks the module tree and replaces eligible linear, attention, and MLP layers with Lumen's FP8 equivalents.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | The model to patch |
| `config` | `QuantConfig \| None` | Full configuration object. If provided, `format` and `scaling` are ignored. |
| `format` | `str \| None` | String shorthand for format: `"fp8_e4m3"`, `"fp8_e5m2"`, `"hybrid"`, `"mxfp8"` |
| `scaling` | `str \| None` | String shorthand for scaling: `"dynamic"`, `"delayed"`, `"blockwise"` |

**Returns:** `None` (model is modified in-place)

**Example:**

```python
import atomrl.lumen.quantize as quant

# Full config
quant.enable(model, config=QuantConfig(
    format=QuantFormat.FP8_E4M3,
    scaling=ScalingType.DELAYED,
))

# String shorthand
quant.enable(model, format="fp8_e4m3", scaling="delayed")
```

## QuantConfig

```python
class atomrl.lumen.quantize.QuantConfig(
    format: QuantFormat,
    scaling: ScalingType,
    amax_algo: AmaxAlgo = AmaxAlgo.MAX,
    history_len: int = 16,
    quantize_activation: bool = True,
    quantize_grad: str | None = None,
)
```

Central configuration object for all quantization settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | `QuantFormat` | — | FP8 numerical format |
| `scaling` | `ScalingType` | — | How scales are computed |
| `amax_algo` | `AmaxAlgo` | `MAX` | Amax history aggregation algorithm |
| `history_len` | `int` | `16` | Steps of amax history (delayed scaling) |
| `quantize_activation` | `bool` | `True` | Quantize forward activations |
| `quantize_grad` | `str \| None` | `None` | Gradient format: `"fp8"`, `"mxfp8"`, `"fp4"`, or `None` |

## QuantFormat

```python
class atomrl.lumen.quantize.QuantFormat(Enum):
    FP8_E4M3   # 4-bit exponent, 3-bit mantissa
    FP8_E5M2   # 5-bit exponent, 2-bit mantissa
    HYBRID     # E4M3 forward + E5M2 backward
    MXFP8      # Microscaling FP8 (per-block shared exponent)
```

## ScalingType

```python
class atomrl.lumen.quantize.ScalingType(Enum):
    DYNAMIC    # Per-step scale from current tensor amax
    DELAYED    # Scale from rolling amax history
    BLOCKWISE  # Per-block independent scales (for MXFP8)
```

## AmaxAlgo

```python
class atomrl.lumen.quantize.AmaxAlgo(Enum):
    MAX          # Maximum amax in history window
    MOST_RECENT  # Most recent amax only
```
