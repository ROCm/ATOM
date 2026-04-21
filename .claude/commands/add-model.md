# Adding a New Model

## Steps

### 1. Evaluate Reuse

Current model reuse relationships:
- `deepseek_v2.py` ← shared by DeepSeek V3, V3.2, GLM-5
- `deepseek_mtp.py` ← DeepSeek MTP models
- `qwen3_next_mtp.py` ← Qwen3-Next MTP models
- `qwen3_5_mtp.py` ← Qwen3.5 MTP (hybrid attention: full + GatedDeltaNet)

If the new model architecture is similar to an existing one, prefer reuse over new implementation.

### 2. Implement the Model

Create a model file under `atom/models/`. Minimum required:

```python
class NewModelForCausalLM(nn.Module):
    # Weight loading mappings (see Step 6)
    packed_modules_mapping = { ... }
    weights_mapping = { ... }

    def __init__(self, atom_config: Config, prefix: str = ""):
        ...

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        ...
        return hidden_states

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

Reference: `llama.py` (simple), `qwen3_5.py` (hybrid attention), `deepseek_v2.py` (MoE + MLA).

### 3. Register the Model

Add to `support_model_arch_dict` in `atom/model_engine/model_runner.py`:

```python
support_model_arch_dict = {
    ...
    "NewModelForCausalLM": ("new_model", "NewModelForCausalLM"),
}
```

Optional: register vLLM plugin in `atom/plugin/register.py`.

### 4. Handle Multimodal Wrappers

If the HF model is multimodal (e.g. `Qwen3_5MoeForConditionalGeneration`), create a `TextOnly` wrapper:

```python
class NewModelTextOnly(nn.Module):
    weights_mapping = {
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    }
    skip_weight_prefixes = ["model.visual."]

    def __init__(self, atom_config, prefix=""):
        self.language_model = NewModelForCausalLM(atom_config)
        ...
```

Register the `TextOnly` variant in `support_model_arch_dict`.

### 5. Weight Loading

The loader uses class-level attributes on the model class to handle checkpoint ↔ model name mismatches:

**`packed_modules_mapping`** — Maps checkpoint's separate weights into fused model parameters:
```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),   # checkpoint q_proj → model's fused qkv_proj, shard "q"
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    # MoE shared expert gate (if applicable)
    ".gate.": (".gate.", 0),
    "shared_expert_gate": ("gate", 1),
}
```

**`weights_mapping`** — Bulk substring replacement applied to every weight name:
```python
weights_mapping = {
    "model.language_model.": "language_model.model.",  # multimodal wrapper remap
    "mtp.": "model.",                                   # MTP checkpoint prefix remap
}
```

**`skip_weight_prefixes`** — List of checkpoint name prefixes to skip entirely:
```python
skip_weight_prefixes = ["model.visual."]  # skip vision encoder weights
```

**Quantization exclusion** — Some checkpoint layers are BF16 while the rest are FP8, listed in `quantization_config.modules_to_not_convert`. For the exclusion to work, `ColumnParallelLinear` must receive the correct `prefix` argument so `get_layer_quant_config(prefix)` can match:
```python
# Wrong: prefix="" → get_layer_quant_config("") → global FP8 config → BF16 weight cast to FP8 → zeros
self.fc = ColumnParallelLinear(in_size, out_size, quant_config=quant_config)

# Right: prefix matches modules_to_not_convert entry → BF16 preserved
self.fc = ColumnParallelLinear(in_size, out_size, quant_config=quant_config, prefix=f"{prefix}.fc")
```

### 6. Adding MTP (Speculative Decoding) Support

If the model has native MTP layers in its checkpoint:

**a) Create the MTP model** (`atom/models/new_model_mtp.py`):

```python
class NewModelMTP(nn.Module):
    packed_modules_mapping = { ... }
    weights_mapping = {"mtp.": "model."}  # checkpoint mtp.* → model state_dict model.*

    @staticmethod
    def remap_mtp_weight_name(name):
        """Filter: return name to load, None to skip."""
        shared_weight_names = ["embed_tokens", "lm_head"]
        if name.startswith("mtp."):
            return name
        if any(key in name for key in shared_weight_names):
            return name
        return None  # skip target model weights

    def __init__(self, atom_config, prefix=""):
        self.model = NewModelMultiTokenPredictor(atom_config, ...)
        self.lm_head = ParallelLMHead(...)

    def forward(self, input_ids, positions, hidden_states, ...):
        ...
    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

**b) Register in `eagle.py`**:
```python
support_eagle_model_arch_dict = {
    ...
    "NewModelMTPArch": "atom.models.new_model_mtp.NewModelMTP",
}
```

**c) Register in `config.py`**:
```python
# In SpeculativeConfig:
_MTP_TYPE_MAP = { ..., "new_model_type": "new_model_mtp" }
_MTP_CONFIG = { ..., "new_model_mtp": ("mtp_num_hidden_layers", "NewModelMTPArch") }
```

**d) Register in `loader.py`** — If the model uses `weights_mapping` + `remap_mtp_weight_name` (Qwen-style), no loader changes needed. If it uses layer-index based remap (DeepSeek-style), add the model type to `_remap_mtp_weight` or define `remap_mtp_weight_name` as an instance method on the MTP class.

**e) MHA vs MLA attention in propose loop** — If the model uses paged attention (MHA) instead of MLA, the draft propose loop in `eagle.py` needs MHA-specific metadata updates (`block_tables`, `context_lens` instead of `kv_last_page_lens`). Check the `use_mla` branching.

**f) GDN/hybrid attention** — If the model uses GatedDeltaNet + full attention, ensure `GDNAttentionMetadataBuilder.prepare_mtp_decode` handles the draft step metadata update.

### 7. Validate

```bash
# Quick smoke test (no server needed)
python -m atom.examples.simple_inference --model <model_path> --kv_cache_dtype fp8

# Server + accuracy
python -m atom.entrypoints.openai_server --model <model_path> --kv_cache_dtype fp8 -tp <N>
lm_eval --model local-completions \
  --model_args model=<model>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 3

# MTP validation (if applicable)
python -m atom.entrypoints.openai_server --model <model_path> --kv_cache_dtype fp8 -tp <N> --method mtp
# Check: (1) output is coherent, (2) MTP Stats acceptance rate > 0%, (3) GSM8K score matches non-MTP baseline
```

### 8. Add CI Test Entry

Add to `.github/benchmark/models_accuracy.json`:
```json
{
  "display": "New-Model-Name",
  "path": "org/model-name",
  "prefix": "new-model",
  "args": "--kv_cache_dtype fp8 -tp 4",
  "threshold": "0.XX",
  "baseline": "0.XXXX",
  "runner": "atom-mi355-8gpu.predownload",
  "frequency": "pr"
}
```

## Checklist

- [ ] Model file created under `atom/models/`
- [ ] Registered in `support_model_arch_dict`
- [ ] `packed_modules_mapping` covers all fused weights
- [ ] `weights_mapping` handles checkpoint ↔ model name differences
- [ ] BF16 layers have correct `prefix` for quant exclusion
- [ ] `@support_torch_compile` NOT added unless verified with Dynamo
- [ ] Smoke test passes (`simple_inference`)
- [ ] Accuracy validated (`lm_eval gsm8k`)
- [ ] CI entry added
- [ ] Recipe created under `recipes/`
- [ ] If MTP: acceptance rate > 0%, GSM8K matches baseline

## Key Files

| File | Purpose |
|------|---------|
| `atom/models/` | Model implementations |
| `atom/model_engine/model_runner.py` | Registration (`support_model_arch_dict`), KV cache binding |
| `atom/spec_decode/eagle.py` | MTP registration (`support_eagle_model_arch_dict`), propose loop |
| `atom/config.py` | `SpeculativeConfig` MTP type mapping |
| `atom/model_loader/loader.py` | Weight loading, name remap dispatch |
| `atom/model_ops/` | AITER kernel wrappers (linear, attention, fused_moe) |
| `atom/plugin/register.py` | vLLM plugin registration |
| `atom/model_ops/attentions/gdn_attn.py` | GDN attention backend (hybrid models) |
