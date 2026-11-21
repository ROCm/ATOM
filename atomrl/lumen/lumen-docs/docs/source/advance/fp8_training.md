# FP8 Quantized Training

This page covers Lumen's FP8 training capabilities in detail — supported formats, scaling strategies, configuration options, and how they map to training performance.

## Supported Formats

| Format | `QuantFormat` enum | Description |
|--------|-------------------|-------------|
| **FP8 E4M3** | `FP8_E4M3` | 4-bit exponent, 3-bit mantissa. Higher precision, preferred for forward pass activations and weights. |
| **FP8 E5M2** | `FP8_E5M2` | 5-bit exponent, 2-bit mantissa. Wider dynamic range, suited for gradients. |
| **Hybrid** | `HYBRID` | E4M3 for forward, E5M2 for backward — the recommended default for pre-training. |
| **MXFP8** | `MXFP8` | Microscaling FP8 with per-block shared exponents. Finer granularity than per-tensor scaling. |

```{note}
FP4 support is on the roadmap but not yet available.
```

## Scaling Strategies

Scaling determines how the FP8 representable range is matched to actual tensor value distributions.

| Strategy | `ScalingType` enum | How it works |
|----------|-------------------|--------------|
| **Dynamic** | `DYNAMIC` | Computes scale from the current tensor's amax at each step. Simple but adds a synchronization point. |
| **Delayed** | `DELAYED` | Maintains a rolling amax history and uses the historical maximum to set scale. Avoids per-step sync. Recommended for large-scale training. |
| **Blockwise** | `BLOCKWISE` | Computes independent scales per block (tile) of the tensor. Used with MXFP8. |

### Amax History (Delayed Scaling)

With delayed scaling, Lumen tracks the absolute-maximum (amax) of each tensor over a configurable window:

```python
config = QuantConfig(
    scaling=ScalingType.DELAYED,
    amax_algo=AmaxAlgo.MAX,       # Use max of history window
    history_len=16,               # Number of steps to track
)
```

| `AmaxAlgo` | Behavior |
|------------|----------|
| `MAX` | Scale from the maximum amax seen in the history window |
| `MOST_RECENT` | Scale from the most recent amax only |

### AMD FNUZ Auto-Detection

AMD Instinct hardware uses the **FNUZ** (Finite, No Unsigned Zero) FP8 variant where negative zero encodes NaN. Lumen auto-detects the hardware and selects the correct FP8 semantics — no user configuration needed.

## Configuration Reference

`QuantConfig` is the single object that controls all precision behavior:

```python
from atomrl.lumen.quantize import AmaxAlgo, QuantConfig, QuantFormat, ScalingType

config = QuantConfig(
    format=QuantFormat.FP8_E4M3,
    scaling=ScalingType.DELAYED,
    amax_algo=AmaxAlgo.MAX,
    history_len=16,
    quantize_activation=True,
    quantize_grad="fp8",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `QuantFormat` | — | FP8 format for weights and compute |
| `scaling` | `ScalingType` | — | Scaling strategy |
| `amax_algo` | `AmaxAlgo` | `MAX` | How to aggregate amax history |
| `history_len` | `int` | `16` | Number of steps in amax history (delayed scaling only) |
| `quantize_activation` | `bool` | `True` | Whether to quantize forward activations. `False` gives weight-only quantization. |
| `quantize_grad` | `str \| None` | `None` | Gradient quantization: `"fp8"`, `"mxfp8"`, `"fp4"`, or `None` (BF16 gradients) |

## FP8 in the Training Loop

### Forward Pass

When `quant.enable(model)` is called, Lumen patches linear layers with a **fused quant → GEMM → dequant** pipeline. The scaling manager updates scales from amax history before each forward call.

### Backward Pass

With `quantize_grad="fp8"`, gradient GEMMs also run in FP8. The E5M2 format is typically used for gradients due to its wider dynamic range.

### Optimizer Step

Lumen stores a **BF16 or FP32 master copy** of weights for the optimizer. After `optimizer.step()`, the updated master weights are re-quantized to FP8 for the next forward pass.

### Communication

With FP8 all-gather enabled, FSDP parameter gathering communicates uint8 tensors instead of BF16, achieving ~48% bandwidth saving per all-gather. See {doc}`/advance/distributed` for details.

## Memory Savings

FP8 reduces memory consumption at multiple levels:

| Component | BF16 | FP8 | Saving |
|-----------|------|-----|--------|
| Weight storage (per element) | 2 bytes | 1 byte + scale overhead | ~48% |
| All-gather payload | 2 bytes/param | 1 byte/param + scale | ~48% |
| Activation storage | 2 bytes | 1 byte (with FP8 activation store) | ~50% |

For a 70B-parameter model, FP8 weight storage alone saves approximately **70 GB** compared to BF16.
