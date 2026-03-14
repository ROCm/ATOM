# Adding a New Model

## Steps

### 1. Evaluate Reuse of Existing Implementations

Current model reuse relationships:
- `deepseek_v2.py` ← shared by DeepSeek V3, V3.2, GLM-5
- `deepseek_mtp.py` ← DeepSeek MTP models
- `qwen3_next_mtp.py` ← Qwen3-Next MTP models

If the new model architecture is similar to an existing one, prefer reuse.

### 2. Implement the Model

Create a model file under `atom/models/`:

```python
# atom/models/new_model.py
class NewModelForCausalLM(nn.Module):
    def __init__(self, *, config, ...):
        ...

    def forward(self, input_ids, positions, intermediate_tensors, ...):
        ...
```

Reference existing implementations (e.g., `llama.py`, `qwen3.py`).

### 3. Register the Model

Add to `support_model_arch_dict` in `atom/model_engine/model_runner.py`:

```python
support_model_arch_dict = {
    ...
    "NewModelForCausalLM": ("new_model", "NewModelForCausalLM"),
}
```

Optional: register vLLM plugin in `atom/plugin/register.py`.

### 4. Validate

```bash
# Quick smoke test
python -m atom.examples.simple_inference --model <model_path> --kv_cache_dtype fp8

# Accuracy validation
lm_eval --model local-completions \
  --model_args model=<model>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5
```

### 5. Add CI Test Entry

Add to `matrix.include` in `.github/workflows/atom-test.yaml`:

```yaml
- model_name: "New-Model-Name"
  model_path: "org/model-name"
  extraArgs: "--kv_cache_dtype fp8 -tp 8"
  env_vars: ""
  accuracy_test_threshold: "0.XX"  # set based on actual test results
  runner: atom-mi355-8gpu.predownload
  run_on_pr: true
```

## Important Notes

- **NEVER modify** `@support_torch_compile` decorated model files
- Define custom env vars in `atom/utils/envs.py`
- Quantization format is auto-detected from HF `config.json` `quantization_config`
- Supported quant formats: FP8 (per-tensor, per-token), INT8, MXFP4, Quark

## Key Files

| File | Purpose |
|------|---------|
| `atom/models/` | Model implementations |
| `atom/model_engine/model_runner.py` | Model registration (`support_model_arch_dict`) |
| `atom/plugin/register.py` | vLLM plugin registration |
| `atom/model_ops/` | AITER kernel wrappers (linear, attention, fused_moe) |
| `atom/config.py` | Configuration definitions |
