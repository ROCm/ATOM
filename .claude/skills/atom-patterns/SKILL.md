---
name: atom-patterns
description: Coding patterns and architecture index for the ATOM LLM inference engine
version: 1.2.0
scope: ATOM repo on AMD ROCm
last_updated: 2026-05-20
---

# ATOM Patterns

## Code Architecture

```
atom/
├── config.py                  # Config, QuantizationConfig, HF config loading
├── entrypoints/               # Server entry (openai_server.py)
├── examples/                  # simple_inference.py (offline smoke test)
├── model_engine/              # Core engine pipeline
│   ├── llm_engine.py          # Top-level engine
│   ├── engine_core.py         # Per-DP-rank loop
│   ├── scheduler.py           # Batch scheduling
│   └── model_runner.py        # Forward pass, CUDAGraph, KV cache binding
├── model_loader/
│   └── loader.py              # Weight loading (safetensors, FP8/FP4, WeightsMapper)
├── model_ops/                 # AITER kernel wrappers
│   ├── linear.py              # LinearBase, ColumnParallel, RowParallel
│   ├── moe.py                 # FusedMoE, Mxfp4MoEMethod, weight_loader
│   ├── fused_moe_triton.py    # Triton matmul_ogs MoE path
│   ├── attention_mla.py       # MLA attention (DeepSeek)
│   ├── attention_mha.py       # Standard MHA attention
│   └── paged_attention.py     # Paged attention backend
├── models/                    # Model implementations (one file per family;
│                              # `*_mtp.py` for MTP / speculative variants —
│                              # run `ls atom/models/` for the current set)
├── spec_decode/
│   └── eagle.py               # MTP proposer (speculative decoding)
├── plugin/                    # vLLM/SGLang plugin adapters
└── utils/
    ├── envs.py                # All ATOM_* env var definitions
    └── forward_context.py     # Module-level forward context
```

## Model Implementation Pattern

### Adding a New Model

Every model class follows this contract:

```python
class NewModelForCausalLM(nn.Module):
    # Weight loading config (class-level)
    packed_modules_mapping = { ... }
    weights_mapping = { ... }
    
    def __init__(self, config: Config, prefix: str = ""):
        ...
    
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return hidden_states  # or logits
    
    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

Registration in `model_runner.py`:
```python
support_model_arch_dict = {
    "NewModelForCausalLM": ("new_model", "NewModelForCausalLM"),
}
```

### Model Reuse Relationships

One model file often serves multiple HuggingFace `model_type`s when their architecture is identical or a strict subset. The mapping lives in `atom/model_engine/model_runner.py:support_model_arch_dict` — that's the authoritative source. Common patterns:

- A shared base file covers a family of closely-related releases (e.g. DeepSeek V3 / V3.2 / GLM-5 all dispatch to one `deepseek_v2.py`).
- A standalone file when the architecture diverges enough that sharing causes branching everywhere (e.g. DeepSeek V4's hot-cold sparse attention + FP4 MoE).
- An `_mtp.py` companion when the spec-decode head differs from the base model.

### TP Parallel Linear Pattern

- `ColumnParallelLinear`: shards output dim, no all-reduce needed
- `RowParallelLinear`: shards input dim, all-reduce on output (`reduce_results=True`)
- `ReplicatedLinear`: full copy on each rank (gates, small projections)

MoE pattern: FusedMoE + shared_experts both use `reduce_results=False`, parent does one all-reduce.

## Workflows

### Adding a Model (file co-change pattern)

1. `atom/models/new_model.py` — Model implementation
2. `atom/model_engine/model_runner.py` — Register in `support_model_arch_dict`
3. `atom/config.py` — Add to `_CONFIG_REGISTRY` if config schema differs
4. `.github/benchmark/models_accuracy.json` — CI accuracy test entry
5. `recipes/` — Usage recipe

### Bug Fix Workflow

1. Identify bug via activation dump / per-layer comparison
2. Fix in model file
3. `grep` same pattern across codebase (fix-then-sweep)
4. Verify with `simple_inference.py` smoke test
5. Run `lm_eval` for accuracy regression

### FP8/FP4 Weight Loading

- Checkpoint weights: `weight` (FP8/FP4 packed) + `weight.scale` (E8M0 block scale)
- ATOM renames `.scale` → `.weight_scale_inv` → `.weight_scale` (auto-rename in loader)
- `process_weights_after_loading()` hook: shuffle weights for CK kernel layout
- FP4 expert weights: `Mxfp4MoEMethod.create_weights()` + `mxf4_merged_weight_loader()`

### Debug Instrumentation Rules

- **NEVER modify `@support_torch_compile` decorated models** (breaks Dynamo)
- Put debug code in `forward()` (has `@torch.inference_mode()`), NOT in `run_model()`
- Gate debug prints with env vars (e.g., `ATOM_V4_DIAG=1`)
- Use `--level 0 --enforce-eager` to disable both torch.compile and CUDAGraph

## Testing Patterns

- Test location: `tests/` directory at repo root
- Framework: pytest
- No GPU needed: tests mock AITER and `torch.cuda`
- Naming: `test_<module>.py` (e.g., `test_scheduler.py`, `test_block_manager.py`)
- Smoke test: `python -m atom.examples.simple_inference --model <path> --kv_cache_dtype fp8`
- Accuracy: `lm_eval` with gsm8k (CI threshold != actual baseline)

## Environment Variables

Authoritative list: `atom/utils/envs.py` (all `ATOM_*` defined as lazy lambdas). To read what an env var does, grep the file for its name — the lambda and the call site comment describe the behavior. Required per-model env vars are listed in `.github/benchmark/models.json` and `.github/benchmark/models_accuracy.json`.

## CI/CD

- Accuracy tests: `.github/benchmark/models_accuracy.json` (model matrix, thresholds, baselines).
- Benchmark: `.github/benchmark/models.json` (server args, bench args, runner pinning).
- Dashboard: `.github/dashboard/index.html` (gh-pages).
- Workflows + action versions: `.github/workflows/*.yaml` is the source of truth (action versions pinned per workflow).
