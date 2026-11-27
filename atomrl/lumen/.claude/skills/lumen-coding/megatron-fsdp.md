# Megatron & FSDP Integration Guide

## Overview

Lumen integrates with two training frameworks:
- **Megatron-Core:** via `LumenSpecProvider` monkey-patch into TE spec path
- **FSDP/FSDP2:** via HF model + `FullyShardedDataParallel` wrapping

## Key Files

| File | Role |
|------|------|
| `lumen/models/spec_provider.py` | `LumenSpecProvider` -- returns Lumen module classes for Megatron |
| `lumen/models/megatron.py` | Norm patch, TE-arg overrides, `lumen_gpt_builder`, `apply_fp8_training`, CLI args |
| `lumen/models/fsdp.py` | `add_common_fsdp_args`, `patch_norms`, `apply_fp8_training`, `apply_fsdp2` |
| `lumen/models/utils.py` | `peek_backend()`, `safe_add_argument()` |
| `examples/llama31/pretrain_llama31.py` | Unified entry: dispatches `--backend megatron` vs `fsdp` |
| `examples/llama2/finetune_llama2.py` | Same pattern for LLaMA 2 SFT |

## LumenSpecProvider (Megatron Path)

`LumenSpecProvider` subclasses Megatron-Core's `BackendSpecProvider` and returns Lumen module classes:

| TE Spec | Lumen Replacement |
|---------|-------------------|
| `TEColumnParallelLinear` | `LumenColumnParallelLinear` |
| `TERowParallelLinear` | `LumenRowParallelLinear` |
| `TELayerNormColumnParallelLinear` | `LumenLayerNormLinear` |
| `TEGroupedLinear` | `LumenGroupedLinear` |
| `TEDotProductAttention` | `LumenDotProductAttention` |
| `TEFusedMLP` | `LumenFusedMLP` / `LumenGatedMLP` |
| Norms | `LumenRMSNorm` / `LumenLayerNorm` |

### How the Monkey-Patch Works

`lumen_gpt_builder` (in `megatron.py`):
1. Saves original `TESpecProvider` and `HAVE_TE`
2. Sets `megatron.core.models.gpt.gpt_layer_specs.TESpecProvider = LumenSpecProvider`
3. Sets `HAVE_TE = True`
4. Calls `get_gpt_layer_with_transformer_engine_spec(...)` -- Megatron builds the model using Lumen modules
5. Restores originals

This allows Lumen to plug into Megatron **without editing Megatron sources**.

### Without `--lumen-linear`

Falls back to local spec path:
- `_patch_core_attention()` swaps `LumenDotProductAttention` into the spec
- `_patch_norms_in_spec()` injects Lumen norm factories
- Uses `get_gpt_layer_local_spec` instead of TE spec

### Import-Time Patch

`_install_fused_layer_norm_patch()` runs at import time -- patches Megatron's `FusedLayerNorm` references.

## FP8 Configuration (Megatron)

`_override_te_args_for_lumen` maps Megatron args to Lumen:
- `--fp8-format e4m3|hybrid` -> `args.lumen_fp8_format`
- Sets `args.fp8 = None` (disables TE FP8)
- Maps `--lumen-fp8-attn dpa|mha`

`apply_fp8_training` (in `megatron.py`) builds `QuantConfig` from CLI args:

```python
config = QuantConfig(
    scaling_type=args.linear_fp8_scaling,
    fp8_format=args.lumen_fp8_format,
    history_len=args.linear_fp8_amax_history_len,
    amax_algo=args.linear_fp8_amax_algo,
    fp8_wgrad=args.linear_fp8_wgrad,
    first_last_layers_bf16=args.first_last_layers_bf16,
    block_size=args.linear_fp8_block_size,
    grad_quant_type=args.grad_quant_type,
)
quant.enable(model, config=config, dp_group=dp_group)
```

With `--lumen-linear` and `--linear-fp8`, also calls `enable_fp8_for_parallel_linear` and optionally attaches `Blockwise2DScaleManager` for FP8 MHA.

## FSDP1 (Current Example Trainers)

Example trainers (`lumen/models/llama31/fsdp/pretrain.py`) use:

```python
model = FullyShardedDataParallel(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy(LlamaDecoderLayer),
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, ...),
    device_id=rank,
)
```

**No standalone `apply_fsdp()`** -- wrapping is inline in `_wrap_fsdp` / `_build_and_wrap_model`.

FP8 for FSDP: `apply_fp8_training` in `fsdp.py` builds `QuantConfig` from `linear_fp8_*` args, calls `patch_norms` if `--lumen-norm`, then `quant.enable`.

## FSDP2

`apply_fsdp2()` in `lumen/models/fsdp.py`:

```python
def apply_fsdp2(model, args):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    # Shard child modules that expose .layers
    for child in model.children():
        if hasattr(child, 'layers'):
            for layer in child.layers:
                fully_shard(layer, ...)
    fully_shard(model, ...)
```

**Status:** Library helper + tests exist. **Not wired** into shipped LLaMA FSDP trainers. `--lumen-fsdp2` flag exists in `add_common_megatron_args` but is not referenced elsewhere.

## CLI Args

### Megatron (`add_common_megatron_args`)

| Group | Key Args |
|-------|----------|
| Backend | `--backend {megatron,fsdp}` |
| Lumen | `--lumen-linear`, `--lumen-norm`, `--lumen-cross-entropy`, `--lumen-attn-backend`, `--lumen-fp8-attn {dpa,mha}` |
| Linear FP8 | `--linear-fp8`, `--linear-fp8-scaling`, `--linear-fp8-block-size`, `--linear-fp8-amax-history-len`, `--linear-fp8-amax-algo`, `--linear-fp8-wgrad` |
| Quant | `--grad-quant-type`, `--first-last-layers-bf16` |
| MXFP8 | `--mxfp8-block-size-*` knobs |
| Training | `--lumen-warmup-steps`, `--lumen-early-stop-step` |

### FSDP (`add_common_fsdp_args`)

Shared template for `linear_fp8_*` naming + `--fsdp-version {1,2}`. Individual trainers duplicate a subset of flags.

## Example Entry Points

```bash
# Megatron pretrain
torchrun --nproc_per_node=8 examples/llama31/pretrain_llama31.py \
    --backend megatron --lumen-linear --linear-fp8 \
    --linear-fp8-scaling blockwise --lumen-fp8-attn mha

# FSDP finetune
torchrun --nproc_per_node=8 examples/llama2/finetune_llama2.py \
    --backend fsdp --fp8-training --fp8-scaling dynamic
```

`peek_backend()` reads `--backend` from `sys.argv` **before** imports, then branches to `_run_megatron()` or `_run_fsdp()`.

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `examples/llama31/run_pretrain.sh` | Launch Megatron pretrain with recommended args |
| `examples/llama2/run_finetune.sh` | Launch FSDP finetune |
| `examples/*/config_*.sh` | Model/hardware-specific config presets |
| `scripts/prepare_data_and_model.sh` | Download datasets and models |

## Adding a New Model Integration

1. Create `lumen/models/<model>/megatron/` and/or `lumen/models/<model>/fsdp/`
2. For Megatron: use `lumen_gpt_builder` or manual spec patching
3. For FSDP: wrap with `FullyShardedDataParallel` + call `apply_fp8_training`
4. Add CLI args via `add_common_megatron_args` / `add_common_fsdp_args`
5. Create example script under `examples/<model>/`
6. Test with `tests/models/test_<framework>.py`
