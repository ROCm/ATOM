# Megatron Patch Registry

Last updated: 09/01/2026.

Lumen integrates with upstream **Megatron-LM** through a centralized **patch registry**
(`lumen/patches/`). Each patch is registered with a **phase** (when it runs), **tags**
(for filtering), and optional **environment gates**.

Repository docs: [Lumen documentation](https://github.com/ROCm/ATOM/tree/main/atomrl/lumen/lumen-docs)
Source tree: [lumen-docs on GitHub](https://github.com/ROCm/ATOM/tree/main/atomrl/lumen/lumen-docs)

---

## Refactoring framework

This section describes the design behind the `dev_patch` migration: why the registry
exists, how patches are classified, and how legacy code was folded in.

### Problem (before)

Megatron integration changes were spread across:

| Location | Issue |
|----------|-------|
| `examples/*/scripts/patch_*.py` | One-off disk scripts; hard to list, test, or compose |
| `lumen/models/megatron.py` | Large inline monkey-patches (IMPORT, MODEL_BUILD, LoRA, cross-entropy) |
| `lumen/models/megatron_patches.py` | Monolithic ~1.3k-line import-time patch file |
| `lumen/models/dsv4/megatron/spec.py` | Eager side-effect patches at import |
| DSV4 entry scripts | Redundant `install_mmap_checkpoint()` without full IMPORT bootstrap |

Consequences: duplicate apply sites, unclear ordering, no single catalog, fragile CI/host workflows.

### Solution: phase-driven registry

Each patch is a **single named unit** registered with `@register_patch` and applied only in its phase:

```
SOURCE          → modify Megatron-LM checkout on disk (container bootstrap)
IMPORT          → monkey-patch Megatron modules before model classes load
ARGS            → register CLI flags on an ArgumentParser
CONFIG_BUILD    → mutate TransformerConfig after args → config
MODEL_BUILD     → side effects during spec / model construction
TRAINING        → setup_model_and_optimizer / evaluate hooks
```

**Separation of concerns by directory:**

| Directory | Phase(s) | Responsibility |
|-----------|----------|----------------|
| `lumen/patches/source/` | SOURCE | On-disk Megatron file edits (DSV4, ROCm, LLaMA, LoRA) |
| `lumen/patches/runtime/` | IMPORT | Import-time monkey-patches (`megatron_import.py`, `moe_fused_router.py`) |
| `lumen/patches/builders/` | ARGS, CONFIG_BUILD, MODEL_BUILD | Per-workflow builder hooks (`dsv4`, `llama`, generic GPT) |
| `lumen/patches/training/` | TRAINING | Training-loop hooks (FP8 gather, HIP graphs, early stop) |

### Migration map (legacy → registry)

| Legacy location | Registry destination | Status |
|-----------------|---------------------|--------|
| `examples/dsv4/patch_rocm_megatron_dsv4.py` | `source/dsv4.py` + `source/rocm.py` | Deprecated wrapper |
| `examples/llama2/scripts/patch_gpt_layer_specs.py` | `source/llama.py` | Deprecated wrapper |
| `examples/llama2/scripts/patch_*_lora*.py` | `source/llama_lora.py` | Deprecated wrapper |
| `examples/llama2/scripts/patch_mlp_fp8_store.py`, `patch_swiglu_fp8_dtype.py` | `runtime/megatron_import.py` | OBSOLETE headers |
| `lumen/models/megatron_patches.py` (body) | `runtime/megatron_import.py` | Shim re-export only |
| `megatron.py` TRAINING hooks | `training/megatron_hooks.py` | Migrated |
| `megatron.py` shared ARGS | `builders/megatron_args.py` | Migrated |
| `megatron.py` GPT MODEL_BUILD | `builders/megatron_model.py` | Migrated |
| `megatron.py` `_patch_cross_entropy` | IMPORT `cross_entropy` (opt-in) | Migrated; alias kept |
| `megatron.py` `_patch_lora_for_layernorm_linear` | MODEL_BUILD `lora_layernorm_linear` | Migrated; alias kept |
| `megatron.py` FP8 param storage | `lumen/models/fp8_param_storage.py` | Extracted helper |
| `deepseek_v4.py` inline EAV spec hooks | `builders/dsv4.py::build_dsv4_transformer_block_spec` | Migrated |
| `spec.py` eager bootstrap patch | MODEL_BUILD `dsv4_megatron_bootstrap` | Removed eager call |
| DSV4 entries `install_mmap_checkpoint()` | `install_all()` | Unified bootstrap |

**Intentionally out of registry:** `LumenConfig` (`lumen/config.py`),
mbridge convert (`examples/dsv4/tools/lumen_mbridge.py`).

### Apply patterns (how patches get triggered)

Patches are **not** all applied at import time. Each phase has explicit call sites:

| Pattern | API | When |
|---------|-----|------|
| Disk bootstrap | `python3 examples/dsv4/patch_megatron_source.py <megatron_root>` | Host/container before training |
| IMPORT (default set) | `install_all()` or `import atomrl.lumen.models.megatron` | Before Megatron model/checkpoint load |
| IMPORT (opt-in) | `install_cross_entropy()` etc. | CLI flag or explicit call |
| ARGS | `apply_args_patches(parser, tags={...})` | Entry script `add_*_args` |
| CONFIG_BUILD | `apply_config_build(config, args, tags={...})` | After `core_transformer_config_from_args` |
| MODEL_BUILD | `apply_model_build(config, args, tags={...})` or `names={...}` | Spec/builder hooks |
| TRAINING | `apply_training_patches(names={...})` | Entry script before `pretrain()` |

**SOURCE CLI tag filtering:**

- Comma in one `--tag` → **OR** (e.g. `--tag dsv4,rocm`, `--tag llama,lora`)
- Repeat `--tag` → **AND** (e.g. `--tag dsv4 --tag rocm`)
- No `--tag` → all `default=True` SOURCE patches (dsv4 + rocm + llama)

**CONFIG_BUILD / MODEL_BUILD tag filtering** uses subset (AND) semantics — a patch matches when `filter_tags ⊆ patch.tags`:

```python
apply_model_build(config=config, args=args, tags={"dsv4", "spec"})
# Runs MODEL_BUILD patches tagged with both dsv4 and spec
```

**Explicit `names=`** bypasses tag filtering and is used for opt-in MODEL_BUILD patches
(e.g. LoRA layernorm fix after adapter wrapping).

**`default_only=True`** (used by `install_all()`): skip patches with `default=False`. SOURCE CLI
`--tag lora` explicitly includes opt-in patches when that tag is requested.

### Naming and metadata conventions

- **Patch name:** `{workflow}_{concern}` — e.g. `dsv4_megatron_bootstrap`, `lora_layernorm_linear`.
- **Avoid ambiguous suffixes** — e.g. `_no_te` was renamed to `dsv4_megatron_bootstrap` because it
  describes Megatron core init shims for the DSV4 spec path, not “model without TE”.
- **`default=False`** for opt-in patches; document the CLI flag or env gate in `description`.
- **`enabled=lambda: ...`** for env-gated patches (both SOURCE and IMPORT).
- **`depends_on`** for SOURCE ordering only; cycles raise `ValueError`.
- **Backward-compatible aliases** — keep thin `_patch_*` wrappers in `megatron.py` /
  `megatron_model.py` when tests or external scripts still import them.

### Torch-less testing strategy

- SOURCE patches and registry logic run without PyTorch (`patch_megatron_source.py` stubs `atomrl.lumen`).
- `runtime/__init__.py` eagerly loads only `moe_fused_router` (no torch); `megatron_import.py`
  loads on `install_all()`.
- Tests use `PatchRegistry.clear()` + module `reload()` to re-register patches per case
  (`tests/patches/`).

### Adding a new patch (decision tree)

```
Need to change Megatron?
├─ Edit files under Megatron-LM checkout?
│  └─ SOURCE → lumen/patches/source/<workflow>.py
├─ Monkey-patch at Python import?
│  └─ IMPORT → lumen/patches/runtime/megatron_import.py (or new runtime module)
├─ New CLI flag?
│  └─ ARGS → lumen/patches/builders/<workflow>.py
├─ Mutate TransformerConfig?
│  └─ CONFIG_BUILD → builders
├─ Change during spec/model construction?
│  └─ MODEL_BUILD → builders
└─ Hook training loop / optimizer?
   └─ TRAINING → lumen/patches/training/
```

Then: register → `--list` or unit test → update this catalog.

### Patch inventory (current)

| Phase | Count | Module(s) |
|-------|------:|-----------|
| **SOURCE** | 21 | `source/dsv4.py` (12), `source/rocm.py` (2), `source/llama.py` (3), `source/llama_lora.py` (4) |
| **IMPORT** | 16 | `runtime/megatron_import.py` (15), `runtime/moe_fused_router.py` (1) |
| **ARGS** | 3 | `builders/megatron_args.py`, `builders/dsv4.py`, `builders/llama.py` |
| **CONFIG_BUILD** | 4 | `builders/dsv4.py`, `builders/llama.py` |
| **MODEL_BUILD** | 8 | `builders/dsv4.py`, `builders/megatron_model.py` |
| **TRAINING** | 4 | `training/megatron_hooks.py` |
| **Total** | **56** | |

IMPORT patches: **14 default-on**, **2 opt-in** (`split_along_dim`, `cross_entropy`).

---

## Patch phases

| Phase | When | Typical trigger |
|-------|------|-----------------|
| **SOURCE** | Megatron checkout on disk | `prepare_rocm_megatron.sh`, container bootstrap |
| **IMPORT** | Python import, before model classes load | `import atomrl.lumen.models.megatron` |
| **ARGS** | CLI argument registration | DSV4 pretrain entry scripts |
| **CONFIG_BUILD** | After `core_transformer_config_from_args` | `dsv4_gpt_builder`, `lumen_gpt_builder` |
| **MODEL_BUILD** | During spec / model construction | `get_dsv4_spec`, `lumen_gpt_builder`, `apply_lora` |
| **TRAINING** | Training loop hooks | `apply_training_patches(names={...})` in entry scripts |

Tag filtering uses **AND** semantics: `tags={"dsv4", "builder"}` matches patches whose tag
set is a **superset** of both tags.

---

## Discovery and apply

### List SOURCE patches (no PyTorch required)

```bash
# From Lumen repo root
PYTHONPATH=. python3 examples/dsv4/patch_megatron_source.py --list
PYTHONPATH=. python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
PYTHONPATH=. python3 examples/dsv4/patch_megatron_source.py --list --tag rocm
```

### Apply SOURCE patches

```bash
PYTHONPATH=. python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM

# Dry-run (no file changes)
PYTHONPATH=. python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag llama --dry-run
PYTHONPATH=. python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag llama,lora --dry-run
```

Inside a training container (prefer `patch_megatron_source.py` — no full Lumen import):

```bash
python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM
python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag llama,lora
```

Equivalent module entry (requires full Lumen env):

```bash
python3 -m atomrl.lumen.patches --list --tag dsv4
python3 -m atomrl.lumen.patches /path/to/Megatron-LM
```

Patches are **idempotent** — re-applying prints `skipped` for already-patched files.

### Runtime patches (automatic)

| Call site | Phases applied | Tags / names |
|-----------|----------------|--------------|
| `install_all()` / `import atomrl.lumen.models.megatron` | IMPORT (default set) | — |
| `_override_te_args_for_lumen` + `--lumen-cross-entropy` | IMPORT (opt-in) | `install_cross_entropy()` |
| `dsv4_gpt_builder()` | CONFIG_BUILD | `{dsv4, builder}` |
| `get_dsv4_spec()` | CONFIG_BUILD + MODEL_BUILD | `{dsv4, spec}` |
| `lumen_gpt_builder()` / `lumen_gpt_builder_with_spec()` | CONFIG_BUILD + MODEL_BUILD | `{lumen, builder}` + spec/model tags |
| `apply_lora()` | MODEL_BUILD | `names={"lora_layernorm_linear"}` |

---

## SOURCE patches (DSV4 Megatron on-disk)

Module: `lumen/patches/source/dsv4.py`
Applied by: `prepare_rocm_megatron.sh`, `setup_dsv4_container_env()`

| Name | Tags | Depends on | Env gate | Description |
|------|------|------------|----------|-------------|
| `dsv4_transformer_config` | dsv4, config, megatron | — | — | Add `dsv4` to `experimental_attention_variant`; add `dsv4_*` TransformerConfig fields |
| `moe_sqrtsoftplus` | dsv4, moe, megatron | `dsv4_transformer_config` | — | Add `sqrtsoftplus` MoE router score function |
| `dsv4_training_config` | dsv4, config, megatron | `dsv4_transformer_config` | — | Finetune flags: clamp shared expert, freeze gate / e-score bias |
| `moe_router_freeze` | dsv4, moe, megatron | `dsv4_training_config` | — | Honor `moe_router_freeze_gate` in TopKRouter |
| `dsv4_hash_routing` | dsv4, moe, megatron | `moe_sqrtsoftplus` | — | Hash routing via `tid2eid[input_ids]` for early MoE layers |
| `skip_none_router_expert_bias` | dsv4, moe, megatron | — | — | Skip expert-bias grad updates when `expert_bias is None` |
| `dist_ckpt_skip_dsv4_norms` | dsv4, checkpoint, megatron | — | `LUMEN_DSV4_SKIP_OPTIONAL_NORMS=1` (default) | Skip missing optional q/kv norm and router expert_bias ckpt keys |
| `shared_expert_clamp` | dsv4, moe, megatron | `dsv4_training_config` | — | Honor `activation_func_clamp_shared_expert` on shared experts |
| `dsv4_transformer_block` | dsv4, hc, megatron | `dsv4_transformer_config` | — | mHC expand/collapse hooks in TransformerBlock |
| `dsv4_transformer_layer` | dsv4, hc, megatron | `dsv4_transformer_config` | — | Per-layer HC params; mHC pre/post around attention and MLP |
| `dsv4_eav_specs` | dsv4, attention, megatron | `dsv4_transformer_config` | — | Add `dsv4` branch to experimental attention variant specs |
| `tp_layers_condition_init` | dsv4, tp, megatron, rocm | — | — | `condition_init_method` shim for Lumen parallel linears |

### SOURCE patches (ROCm platform)

Module: `lumen/patches/source/rocm.py`
Applied by: same entry points as DSV4 SOURCE patches (default set includes both)

| Name | Tags | Depends on | Env gate | Description |
|------|------|------------|----------|-------------|
| `disable_batch_p2p_comm` | rocm, pipeline, megatron | — | `MEGATRON_NO_BATCH_P2P_COMM=1` | Inject env check → `batch_p2p_comm=False` |
| `cpu_offload_torch_gpu_adam` | rocm, optimizer, megatron | — | — | Use CPUAdam for GPU hybrid-offload partitions (MI325 gfx950 TE Adam workaround) |

### SOURCE patches (LLaMA / GPT)

Module: `lumen/patches/source/llama.py`
Applied by: `run_pretrain_llama2_7b.sh`, `run_pretrain_llama31_8b.sh`, `run_pretrain.sh` (Megatron), Qwen3 pretrain (`--tag llama`)

| Name | Tags | Depends on | Env gate | Description |
|------|------|------------|----------|-------------|
| `llama_megatron_fused_rmsnorm` | llama, norm, megatron | — | — | Add `MegatronFusedRMSNorm` wrapper for apex FusedRMSNorm + sequence parallel |
| `llama_gpt_layer_specs_rmsnorm` | llama, norm, megatron | `llama_megatron_fused_rmsnorm` | — | Select MegatronFusedRMSNorm in `gpt_layer_specs` when normalization is RMSNorm |
| `llama_transformer_block_rmsnorm` | llama, norm, megatron | `llama_megatron_fused_rmsnorm` | — | Use MegatronFusedRMSNorm as `LayerNormImpl` in `transformer_block` |

```bash
# LLaMA pretrain bootstrap (inside container)
PYTHONPATH=/workspace/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag llama
```

Deprecated wrapper: `examples/llama2/scripts/patch_gpt_layer_specs.py` (forwards to registry).

### SOURCE patches (LLaMA LoRA finetune)

Module: `lumen/patches/source/llama_lora.py`
Applied by: `run_lora_finetune_llama2_70b.sh` (`--tag llama,lora`, opt-in / `default=False`)

| Name | Tags | Depends on | Env gate | Description |
|------|------|------------|----------|-------------|
| `lora_requires_grad` | lora, finetune, megatron | — | — | Force `hidden_states.requires_grad` before activation checkpointing |
| `lora_checkpoint_load` | lora, finetune, megatron | — | — | LoRA `base_layer` ckpt remap + `mmap=True` torch.load |
| `lora_adapter_scaling` | lora, finetune, megatron | — | — | LoRA `alpha/rank` scaling (NeMo / PEFT) |
| `lora_sft_loss_default` | lora, finetune, megatron | — | — | Default `--sft=True` for MLPerf val loss norm |

```bash
# MLPerf LoRA finetune bootstrap (RMSNorm + LoRA in one OR command)
PYTHONPATH=/workspace/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag llama,lora
```

### Config fields introduced (SOURCE)

| Field | Patch | Purpose |
|-------|-------|---------|
| `dsv4_mode`, `dsv4_hc_mult`, `dsv4_hc_sinkhorn_iters`, `dsv4_hc_eps` | `dsv4_transformer_config` | Hyper-Connection (mHC) |
| `dsv4_compress_ratios`, `dsv4_compress_rope_theta` | `dsv4_transformer_config` | Compressor layers |
| `dsv4_o_groups`, `dsv4_o_lora_rank`, `dsv4_n_hash_layers`, `dsv4_window_size` | `dsv4_transformer_config` | Attention / hash MoE |
| `activation_func_clamp_shared_expert` | `dsv4_training_config` | Shared expert clamp control |
| `freeze_e_score_correction_bias`, `moe_router_freeze_gate` | `dsv4_training_config` | Finetune freeze flags |
| `moe_router_score_function += 'sqrtsoftplus'` | `moe_sqrtsoftplus` | DeepSeek V4 default routing |

---

## IMPORT patches (Megatron runtime monkey-patches)

Module: `lumen/patches/runtime/megatron_import.py`
Re-export: `lumen/models/megatron_patches.py`
Applied by: `install_all()` on `import atomrl.lumen.models.megatron`

| Name | Tags | Default | Env gate | Description |
|------|------|---------|----------|-------------|
| `fused_layer_norm` | core, norm, megatron | yes | — | Replace `FusedLayerNorm` with Lumen RMSNorm / LayerNorm |
| `language_module_checkpoint_guard` | core, checkpoint, lora, megatron | yes | — | Guard `LanguageModule.sharded_state_dict` for LoRA `output_layer` |
| `mmap_checkpoint` | core, checkpoint, megatron | yes | — | Inject `mmap=True` into Megatron `torch.load` |
| `requires_grad_fix` | core, lora, recompute, megatron | yes | — | Fix `requires_grad` for LoRA + activation checkpointing |
| `moe_fused_router` | core, moe, megatron | yes | — | Lumen fused MoE router top-k and aux-loss ops in `moe_utils` |
| `swiglu_fp8` | core, fp8, fusion, megatron | yes | — | ROCm FP8 SwiGLU dtype (`e4m3fnuz`) + chunked backward |
| `fused_swiglu_triton` | fusion, fp8, megatron | yes | `LUMEN_FUSED_SWIGLU=1` | AITER Triton fused SwiGLU |
| `mlp_fp8_store` | fp8, mlp, megatron | yes | `LUMEN_MLP_FP8_STORE=1` | Store MLP SwiGLU intermediates in FP8 |
| `mlp_recompute` | recompute, mlp, megatron | yes | — | MLP-only recompute for non-recomputed layers (`LUMEN_MLP_RECOMPUTE=1` checked at forward time, not import time) |
| `fused_rope` | fusion, rope, megatron | yes | — | Apex fused RoPE in `rope_utils` |
| `eval_recompute` | recompute, eval, megatron | yes | `LUMEN_EVAL_RECOMPUTE=1` | Keep activation checkpointing during `model.eval()` |
| `post_eval_cache_clear` | eval, memory, megatron | yes | `LUMEN_POST_EVAL_CACHE_CLEAR=1` | `empty_cache` / gc after Megatron `evaluate()` |
| `fused_residual_norm` | fusion, norm, megatron | yes | `LUMEN_FUSED_RESIDUAL_NORM=1` | Deferred BDA + fused residual + RMSNorm |
| `optimizer_patches` | optimizer, megatron | yes | — | `DistributedOptimizer` batched `_foreach_copy_` |
| `split_along_dim` | te, attention, megatron | **no** | — | Lumen `SplitAlongDim` shim for TE attention (disabled in `install_all`) |
| `cross_entropy` | core, fusion, lumen, megatron | **no** | — | Lumen Triton parallel cross-entropy for `--lumen-cross-entropy` |

---

## TRAINING patches (Megatron setup / eval hooks)

Module: `lumen/patches/training/megatron_hooks.py`
Applied by: `apply_training_patches(names={...})` from entry scripts (not at import time)

Implementation helpers live in `lumen/models/fp8_param_storage.py` (meta materializer,
checkpoint quant, embedding/output dequant hooks).

| Name | Tags | Default | Depends on | Description |
|------|------|---------|------------|-------------|
| `fp8_param_gather_hook` | fp8, training, megatron | no | — | Optimizer post-step hook for FP8 param gather / weight quant once |
| `fp8_param_storage_hook` | fp8, training, megatron | no | `fp8_param_gather_hook` | Meta init + FP8 placeholders for frozen weights (`--fp8-param-storage`) |
| `hip_graphs_hook` | training, megatron | no | `fp8_param_gather_hook`, `fp8_param_storage_hook` | Lazy HIP graph capture on transformer layers (`--lumen-hip-graphs`) |
| `val_loss_early_stop_hook` | training, megatron, eval | no | — | Collective early stop when reduced validation loss hits `--val-loss-target` |

**Entry script usage**

```python
from atomrl.lumen.patches import apply_training_patches

# LLaMA 3.1 pretrain
apply_training_patches(names={"fp8_param_gather_hook"})

# LLaMA 2 LoRA finetune (all hooks; dependency order is automatic)
apply_training_patches(
    names={
        "fp8_param_gather_hook",
        "fp8_param_storage_hook",
        "hip_graphs_hook",
        "val_loss_early_stop_hook",
    }
)
```

`atomrl.lumen.models.megatron` re-exports the `install_*` helpers for backward compatibility.

---

## CONFIG_BUILD / ARGS / MODEL_BUILD patches

### Shared Megatron ARGS (`lumen/patches/builders/megatron_args.py`)

| Name | Phase | Tags | Trigger | Description |
|------|-------|------|---------|-------------|
| `common_megatron_args` | ARGS | megatron, lumen | `apply_args_patches(names={"common_megatron_args"})` | Shared CLI: backend, Lumen kernels, LoRA, linear FP8, checkpoint, experiment |

### LLaMA builder (`lumen/patches/builders/llama.py`)

| Name | Phase | Tags | Trigger | Description |
|------|-------|------|---------|-------------|
| `llama_pretrain_args` | ARGS | llama, pretrain, megatron | `add_pretrain_args` | mlperf Docker compatibility flags (`--size`, `--nodes`, …) |
| `lumen_gpt_config` | CONFIG_BUILD | lumen, builder | `lumen_gpt_builder` | `persist_layer_norm=False`, `bias_swiglu_fusion=False`; optional FP8 activation store |

### Generic GPT MODEL_BUILD (`lumen/patches/builders/megatron_model.py`)

| Name | Phase | Tags | Trigger | Description |
|------|-------|------|---------|-------------|
| `core_attention_spec` | MODEL_BUILD | megatron, lumen, builder, spec | `lumen_gpt_builder` (local spec) | Inject `LumenDotProductAttention` into layer spec |
| `norms_in_spec` | MODEL_BUILD | megatron, lumen, builder, spec | `lumen_gpt_builder` (local spec) | Inject Lumen norm factories into layer spec |
| `mla_attention_spec` | MODEL_BUILD | megatron, lumen, builder, spec, mla | `lumen_gpt_builder_with_spec` | Inject `LumenDotProductAttentionMLA` when `--multi-latent-attention` |
| `model_norms` | MODEL_BUILD | megatron, lumen, builder, model | `lumen_gpt_builder` | Replace built norms when `--lumen-rmsnorm` / `--lumen-norm` |
| `fused_swiglu_mlp` | MODEL_BUILD | megatron, lumen, builder, model, mlp | `lumen_gpt_builder_with_spec` | AITER fused SwiGLU when `--lumen-fused-mlp` |
| `lora_layernorm_linear` | MODEL_BUILD | lora, lumen, megatron, model | `apply_lora` | LoRA forward fix for `LumenLayerNormLinear` base layers |

### DSV4 builder (`lumen/patches/builders/dsv4.py`)

| Name | Phase | Tags | Trigger | Description |
|------|-------|------|---------|-------------|
| `dsv4_config_core` | CONFIG_BUILD | dsv4, builder | `dsv4_gpt_builder` | Set `dsv4_mode=True`, unpadded `vocab_size` for hash routing |
| `dsv4_config_pipeline` | CONFIG_BUILD | dsv4, builder | `dsv4_gpt_builder` (PP > 1) | `variable_seq_lengths=True`, `batch_p2p_comm=False`, mHC P2P shape exchange |
| `dsv4_spec_config` | CONFIG_BUILD | dsv4, spec | `get_dsv4_spec` | Set `dsv4_dsa_topk_backend` from CLI |
| `dsv4_pretrain_args` | ARGS | dsv4 | `add_dsv4_pretrain_args` | Register DSV4 CLI flags (YaRN, freeze gate, DSA backend, …) |
| `dsv4_megatron_bootstrap` | MODEL_BUILD | dsv4, spec | `get_dsv4_spec` | One-time Megatron core init shims (optimizer, MoE, JIT) |
| `dsv4_moe_mori` | MODEL_BUILD | dsv4, spec, moe | `get_dsv4_spec` | MoEMori token dispatcher when `LUMEN_DSV4_MOE_MORI=1` |

`get_dsv4_spec` delegates transformer block construction to
`build_dsv4_transformer_block_spec()` (temporary EAV module hooks for DSV4 attention).

**Entry script usage**

```python
from atomrl.lumen.patches.builders import apply_args_patches

# LLaMA 3.1 pretrain (after model-specific default overrides)
apply_args_patches(parser, names={"common_megatron_args", "llama_pretrain_args"})

# LLaMA 2 finetune
apply_args_patches(parser, names={"common_megatron_args"})

# DSV4 pretrain
apply_args_patches(parser, tags={"dsv4"})
```

`atomrl.lumen.models.megatron.add_common_megatron_args` remains a backward-compatible re-export.

---

## Per-patch details (selected)

### `dsv4_hash_routing` (SOURCE)

Threads `input_ids` from `GPTModel` through `TransformerBlock` → `TransformerLayer` →
`MoELayer` → `TopKRouter`. For layers `1 … dsv4_n_hash_layers`, expert selection uses
`tid2eid[input_ids]` while sqrt-softplus logits remain as combine weights.

**Why:** Converted checkpoints store per-token expert tables for early MoE layers.

### `dsv4_config_core` (CONFIG_BUILD)

Sets `config.vocab_size` to the **unpadded** vocab size from CLI. Hash-routed MoE
allocates `tid2eid` with shape `[vocab_size, topk]` matching the checkpoint, even though
GPT embeddings use `padded_vocab_size`.

### `mmap_checkpoint` (IMPORT)

Prevents every rank from loading a full checkpoint into CPU RAM during distributed
checkpoint restore. Critical for large models on ROCm.

### `fused_swiglu_triton` (IMPORT)

Replaces Megatron's decomposed SwiGLU (6–8 elementwise kernels) with AITER Triton
single-kernel forward/backward. Typical saving: ~400–600 ms/step at LLaMA-70B scale.

---

## HF checkpoint conversion (not in registry)

HF → Megatron config mapping for **mbridge** convert stays in
`examples/dsv4/tools/lumen_mbridge.py` via `@register_model("deepseek_v4")`. This path
is separate from the Megatron training patch registry.

---

## Adding a new patch

1. Choose the phase using the decision tree in **Refactoring framework** above.
2. Implement in the matching module under `lumen/patches/`.
3. Register:

```python
from atomrl.lumen.patches.registry import PatchPhase, register_patch

@register_patch(
    "my_patch",
    PatchPhase.IMPORT,
    description="What it does",
    tags=frozenset({"dsv4", "megatron"}),
    enabled=lambda: os.environ.get("MY_FLAG", "0") == "1",  # optional
    depends_on=("other_patch",),  # SOURCE only
    default=True,
)
def install_my_patch():
    ...
```

4. Wire a call site — IMPORT via `install_all()` or explicit installer; builders via
   `apply_*_build(tags=...)`; TRAINING via entry script.
5. Verify with `--list` (SOURCE) or a short training smoke test (runtime patches).
6. Update this catalog.

---

## Related files

| Path | Role |
|------|------|
| `lumen/patches/registry.py` | `@register_patch`, `apply_patches`, `list_patches` |
| `lumen/patches/source/dsv4.py` | DSV4 Megatron on-disk patches |
| `lumen/patches/source/rocm.py` | ROCm platform on-disk patches |
| `lumen/patches/source/llama.py` | LLaMA/GPT on-disk patches (fused RMSNorm) |
| `lumen/patches/source/llama_lora.py` | LLaMA LoRA finetune on-disk patches |
| `lumen/patches/builders/dsv4.py` | DSV4 CONFIG_BUILD / ARGS / MODEL_BUILD |
| `lumen/patches/runtime/megatron_import.py` | Megatron IMPORT monkey-patches |
| `lumen/patches/runtime/moe_fused_router.py` | MoE fused router IMPORT patch |
| `lumen/models/megatron_patches.py` | Backward-compatible re-export shim |
| `examples/dsv4/patch_megatron_source.py` | Torch-less host entry for SOURCE patches |
| `examples/dsv4/patch_rocm_megatron_dsv4.py` | Deprecated alias for `patch_megatron_source.py` |
| `examples/dsv4/PATCHES.md` | Quick reference (links here for full catalog) |
