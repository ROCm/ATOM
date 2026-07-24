# ATOM vLLM-Omni Diffusion Model Plugin

Models under this directory run with the **vLLM-Omni plugin** — they cannot run standalone with native ATOM. For native ATOM models, see `atom/models/` instead.

## What the Plugin Does

The ATOM plugin replaces vLLM's linear layers (`vllm.model_executor.layers.linear`) with ATOM's AITER-accelerated equivalents (`atom.model_ops.linear`), enabling ROCm-optimized quantized GEMM kernels for diffusion model inference.

The plugin hooks into vllm-omni at startup via `register_omni_model()` in `atom/plugin/vllm_omni/register.py`. It uses **monkey-patching** rather than registering new pipeline classes: the stock vllm-omni pipelines are left in place, but the transformer class they instantiate is swapped out before any model is loaded.

---

## How to Add a New Model

Follow the pattern used for Wan2.2 in `wan2_2/wan2_2_transformer.py`.

### Step 1: Identify what to replace

Open the stock vllm-omni transformer file for your model (e.g. `vllm_omni/diffusion/models/<model>/`). Look for uses of:

```python
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
```

These are the layers to replace with their `atom.model_ops.linear` equivalents.

### Step 2: Create an ATOM transformer file

Create `atom/plugin/vllm_omni/diffusion/models/<model>/` and add a `<model>_transformer.py`.

**Import pattern:**

```python
from atom.model_ops.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm_omni.diffusion.models.<model>.<model>_transformer import (
    StockSelfAttention,
    StockCrossAttention,
    StockFeedForward,
    StockTransformerBlock,
    StockTransformerModel,
    # any helper functions needed in forward() overrides
)
```

**For each layer class** that uses vLLM linears, create an ATOM subclass:

```python
class ATOMStockSelfAttention(StockSelfAttention):
    def __init__(self, ...):
        super().__init__(...)
        # Replace linear layers after super().__init__() creates the vllm ones
        self.to_qkv = QKVParallelLinear(hidden_size=dim, head_size=head_dim,
                                        total_num_heads=num_heads, bias=True)
        self.num_heads = self.to_qkv.num_heads        # refresh from atom layer
        self.num_kv_heads = self.to_qkv.num_kv_heads
        self.to_out = RowParallelLinear(inner_dim, dim, bias=True)
```

**Check if `forward()` needs an override.** Two cases require it:

| Situation | What to do |
|-----------|-----------|
| Stock `forward()` does `out, _ = self.layer(x)` (tuple unpack) | Override `forward()` — atom layers return a plain tensor, not `(tensor, None)` |
| Stock `forward()` does `out = self.layer(x)` | No override needed — atom and vllm (with `return_bias=False`) both return plain tensors |

The `QKVParallelLinear` case always requires an override because vLLM returns a tuple:

```python
    def forward(self, hidden_states, ...):
        # atom returns plain tensor; vllm returns (tensor, None)
        qkv = self.to_qkv(hidden_states)   # NOT: qkv, _ = self.to_qkv(hidden_states)
        ...
```

**For feedforward layers** that wrap `ColumnParallelLinear` inside a helper (e.g. `ColumnParallelGELU`), replace the inner `.proj` attribute:

```python
class ATOMStockFeedForward(StockFeedForward):
    def __init__(self, dim, inner_dim, dim_out=None, bias=True):
        super().__init__(dim=dim, inner_dim=inner_dim, dim_out=dim_out, bias=bias)
        dim_out = dim_out or dim
        self.net_0.proj = ColumnParallelLinear(dim, inner_dim, bias=bias)
        self.net_2 = RowParallelLinear(inner_dim, dim_out, bias=bias)
        # forward() inherited — helper's forward() calls self.proj(x) → plain tensor ✓
```

**Compose into a block and top-level model:**

```python
class ATOMStockTransformerBlock(StockTransformerBlock):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6, ...):
        super().__init__(...)
        head_dim = dim // num_heads
        self.attn1 = ATOMStockSelfAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, eps=eps)
        self.attn2 = ATOMStockCrossAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, eps=eps)
        self.ffn   = ATOMStockFeedForward(dim=dim, inner_dim=ffn_dim, dim_out=dim)
        # forward() inherited from StockTransformerBlock unchanged

class ATOMStockTransformerModel(StockTransformerModel):
    def __init__(self, ..., num_layers=N, ...):
        super().__init__(...)  # builds rope, embeddings, norm, proj_out
        inner_dim = num_attention_heads * attention_head_dim
        # Replace all blocks after super() creates the stock ones
        self.blocks = nn.ModuleList([
            ATOMStockTransformerBlock(inner_dim, ffn_dim, num_attention_heads, eps, ...)
            for _ in range(num_layers)
        ])
        # forward(), load_weights(), _sp_plan all inherited from StockTransformerModel
```

### Step 3: Register via monkey-patch in `register.py`

Open `atom/plugin/vllm_omni/register.py` and add to the monkey-patch block at the end of `register_omni_model()`:

```python
import vllm_omni.diffusion.models.<model>.pipeline_<model> as _<model>_pipeline
from atom.plugin.vllm_omni.diffusion.models.<model>.<model>_transformer import ATOM<Model>TransformerModel
_<model>_pipeline.<StockTransformerModel> = ATOM<Model>TransformerModel
```

Python resolves module-level names at call time, so patching the name in the pipeline module's namespace causes all subsequent `create_transformer_from_config()` calls to instantiate the ATOM model — no pipeline file copies needed.

**You only need to patch the base pipeline module.** If variant pipelines (e.g. i2v, ti2v) import `create_transformer_from_config` *from* the base pipeline rather than defining their own, they will automatically pick up the patch — patching the same name twice in different modules would be redundant. Check the variant pipeline's imports to confirm:

```python
# If you see this in pipeline_<model>_i2v.py, one patch covers all variants:
from vllm_omni.diffusion.models.<model>.pipeline_<model> import create_transformer_from_config
```

**Do not copy pipeline files.** If the stock pipeline needs no changes beyond the transformer class swap, patching is sufficient. Only create a new pipeline class if you need to change the pipeline's own logic (e.g. different preprocessing, scheduler, or VAE).

### Step 4: Update `__init__.py`

Add your model's ATOM transformer class to `atom/plugin/vllm_omni/diffusion/models/<model>/__init__.py` (if the directory needs one). Re-export stock pipeline helpers from `vllm_omni` directly rather than copying them.

---

## API Compatibility Notes

### `atom.model_ops.linear` vs `vllm.model_executor.layers.linear`

| vLLM class | ATOM equivalent | Notes |
|---|---|---|
| `ColumnParallelLinear(in, out, bias, gather_output=False, return_bias=False)` | `ColumnParallelLinear(in, out, bias)` | Extra kwargs absorbed via `**kwargs`, silently ignored |
| `RowParallelLinear(in, out, bias, input_is_parallel=True, return_bias=False)` | `RowParallelLinear(in, out, bias)` | `reduce_results=True` by default — matches vLLM behavior |
| `QKVParallelLinear(hidden_size, head_size, total_num_heads, bias)` | `QKVParallelLinear(hidden_size, head_size, total_num_heads, bias)` | Same constructor; **different return type** (see below) |

### Critical: `QKVParallelLinear` return type difference

```python
# vLLM: returns (tensor, None) tuple
qkv, _ = self.to_qkv(hidden_states)

# ATOM: returns plain tensor — must NOT unpack
qkv = self.to_qkv(hidden_states)
```

`ColumnParallelLinear` and `RowParallelLinear` forward signatures are compatible — both return a plain tensor when vLLM's `return_bias=False` (the standard config for diffusion models).

### `atom.model_ops.linear` forward signature

```python
def forward(self, x: Tensor, x_scale: Tensor | None = None, otype=bf16) -> Tensor
```

Calling `layer(x)` works as expected; `x_scale` and `otype` are used for quantized inference and default safely to unquantized bfloat16.

---

## Current Models

| Model | Transformer file | Registered via |
|-------|-----------------|----------------|
| Wan2.2 (T2V / I2V / TI2V) | `wan2_2/wan2_2_transformer.py` | monkey-patch in `register.py` |
