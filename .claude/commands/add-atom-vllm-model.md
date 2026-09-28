---
name: add-atom-vllm-model
description: Add a new model architecture to ATOM vLLM plugin by reusing or implementing model code, registering architecture mappings, updating plugin model registry, validating with smoke and accuracy tests, and adding CI matrix entries. Use when the user asks to onboard/support a new model in ATOM or ATOM plugin mode.
---

# Add ATOM vLLM Model

Use this skill when the task is to add support for a new model architecture in ATOM, especially for vLLM plugin mode.

## Inputs To Collect First

Confirm these before coding:

- Hugging Face model path (or local checkpoint path)
- `architectures[0]` from model `config.json`
- Whether this is plain CausalLM, MoE, MTP, or recurrent/GDN variant
- Target framework path: ATOM native, vLLM plugin, or both
- Required quantization format and tp/pp/ep settings

If `architectures[0]` is unknown, ask for the model path and inspect `config.json`.

## Workflow

Copy this checklist and update progress:

```text
Model onboarding progress:
- [ ] 1) Reuse analysis completed
- [ ] 2) Model implementation (or reuse wiring) completed
- [ ] 3) ATOM model runner registration completed
- [ ] 4) Plugin registration completed (if needed)
- [ ] 5) Smoke test passed
- [ ] 6) Accuracy eval completed
- [ ] 7) Accuracy result table inserted into recipe
- [ ] 8) `recipes/atom_vllm` recipe added/updated
- [ ] 9) CI matrix entry added
```

### 1) Reuse Analysis (Do This First)

Prefer reusing existing model implementations when architecture is compatible.

Known reuse patterns:

- `deepseek_v2.py` can serve DeepSeek V3, V3.2, GLM-5 style variants
- `deepseek_mtp.py` for DeepSeek MTP variants
- `qwen3_next_mtp.py` for Qwen3-Next MTP variants

Also inspect existing model files in `atom/models/` before creating a new one.

### 2) Implement Or Reuse Model Code

If new implementation is needed, add a file under `atom/models/`:

```python
class NewModelForCausalLM(nn.Module):
    def __init__(self, *, config, ...):
        ...

    def forward(self, input_ids, positions, intermediate_tensors, ...):
        ...
```

Use existing files such as `llama.py`, `qwen3.py`, `qwen3_next.py`, `deepseek_v2.py` as templates.

### 3) Register In ATOM Model Runner

Update `support_model_arch_dict` in `atom/model_engine/model_runner.py`.

Important: follow the **current local style** in that file (for example, string qualname mapping if that is what existing entries use). Do not introduce a different mapping format.

### 4) Register In Plugin Model Registry

For plugin mode support, update `atom/plugin/register.py`:

- Add imports for new model classes
- Add architecture key to `_ATOM_SUPPORTED_MODELS`

If the model is ATOM-only and not used through plugin path, explicitly document why plugin registration is skipped.

### 5) Smoke Test

Run a quick smoke test with `vllm serve`:

```bash
vllm serve <model_path> \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

### 6) Accuracy Validation

In another terminal, run accuracy validation:

```bash
lm_eval --model local-completions \
  --model_args model=<model>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5
```

### 6.5) Insert Accuracy Result Into Recipe

After `lm_eval` finishes, insert the measured result into the corresponding
`recipes/atom_vllm/<Model-Name>.md` section using this table format:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.93  |±  |0.0256|
|     |       |strict-match    |     5|exact_match|↑  |0.93  |±  |0.0256|
```

Requirements:

- Use the actual `n-shot` and measured values/stderr from the run output (do not round aggressively).
- Keep the table adjacent to the `lm_eval` command in the recipe.
- Include the raw result JSON path for traceability.

### 7) Add `recipes/atom_vllm` Entry

For every new model added to ATOM vLLM plugin support, also add or update a usage recipe under:

- `recipes/atom_vllm/<Model-Name>.md`

Minimum recipe content:

1. One-line scope: model name + backend context (`ATOM vLLM plugin backend`)
2. `docker pull` step (`rocm/atom-dev:vllm-latest`)
3. `vllm serve` launch commands (include the exact model path and key args such as TP, KV dtype, async scheduling)
4. Optional benchmark command (`vllm bench serve` or project benchmark script)
5. Accuracy validation command (`lm_eval --model local-completions ...`)
6. Accuracy result table using the standard `|Tasks|Version|...|` format + raw JSON path
7. Any model-specific env vars and caveats (for example, plugin attention toggle if required)

Use existing files in `recipes/atom_vllm/` (such as `DeepSeek-R1.md`, `Qwen3.5.md`, `Kimi-K2.5.md`) as style references.

If this onboarding also updates CI matrix arguments (TP size, env vars, model id), make sure the recipe launch command is aligned with those CI settings.

### 8) Add CI Test Entry

Update `.github/workflows/atom-vllm-oot-test.yaml` and add the model entry to the
**nightly accuracy path only**.

Do **not** add this model to the pull-request OOT matrix (`jobs.atom-vllm-oot.strategy.matrix.include`).

Add it in `jobs.prepare-oot-image -> step "Resolve image source and model matrix"` in
the Python `models = [...]` list:

```python
{
    "toggle_env": "RUN_NEW_MODEL_TP8",
    "model_name": "New-Model-Name TP8",
    "model_path": "org/model-name",
    "extra_args": "--tensor-parallel-size 8",
    "accuracy_test_threshold": 0.XX,
    "env_vars": "",
    "runner": "linux-atom-mi35x-8",
},
```

If this model should be **nightly only** (not selectable in manual workflow_dispatch),
do not add a corresponding input toggle under `on.workflow_dispatch.inputs`.

Set `accuracy_test_threshold` from real eval results, not guesswork.

## Guardrails

- Never modify model files that are decorated with `@support_torch_compile`.
- Define new env vars in `atom/utils/envs.py` only.
- Keep quantization behavior aligned with HF `config.json` `quantization_config`.
- Supported quant formats include FP8, INT8, MXFP4, Quark.
- Do not change unrelated model mappings during onboarding.

## Output Format

When done, report:

1. Reuse decision and rationale
2. Files changed
3. New architecture key(s) registered
4. Smoke and accuracy results
5. CI entry details
6. Recipe file path and key launch command
7. Any follow-up risks or TODOs

