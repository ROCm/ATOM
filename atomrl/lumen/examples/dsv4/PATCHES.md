# Lumen DSV4 Patch Registry

> **Full catalog (recommended):** [Megatron Patch Registry](../../lumen-docs/docs/source/advance/patch_registry.md) in the Lumen docs.

Quick reference for DSV4 developers. Each patch is registered in `lumen/patches/` with a **phase** and optional **tags**.
Tag filtering on the SOURCE CLI:

- **Comma in one `--tag`** → OR (e.g. `--tag dsv4,rocm` = dsv4 patches **or** rocm patches)
- **Repeat `--tag`** → AND (e.g. `--tag dsv4 --tag rocm` = patches that have **both** tags)
- **`--tag-mode any|all`** → override the default mode above

## Phase overview

| Phase | When | DSV4 location | Typical trigger |
|-------|------|---------------|-----------------|
| **SOURCE** | Bootstrap | `lumen/patches/source/dsv4.py` | `prepare_rocm_megatron.sh`, container `setup_container_env.sh` |
| **IMPORT** | Python import | `lumen/models/megatron_patches.py` | `import atomrl.lumen.models.megatron` |
| **ARGS** | CLI setup | `lumen/patches/builders/dsv4.py` | `add_dsv4_pretrain_args` in pretrain entry |
| **CONFIG_BUILD** | After `core_transformer_config_from_args` | `lumen/patches/builders/dsv4.py` | `dsv4_gpt_builder`, `get_dsv4_spec` |
| **MODEL_BUILD** | Before module spec build | `lumen/patches/builders/dsv4.py` | `get_dsv4_spec` |

HF checkpoint → Megatron config for **mbridge convert** stays in
`examples/dsv4/tools/lumen_mbridge.py` (`@register_model("deepseek_v4")`), not in this
registry.

## List registered SOURCE patches (no PyTorch required)

On the host or in CI, from the Lumen repo root:

```bash
# All default SOURCE patches
PYTHONPATH="${ATOMRL_DIR:-/workspace/ATOM/atomrl}" python3 examples/dsv4/patch_megatron_source.py --list

# DSV4-tagged SOURCE patches only
PYTHONPATH="${ATOMRL_DIR:-/workspace/ATOM/atomrl}" python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4

# ROCm platform patches only (pipeline / optimizer workarounds)
PYTHONPATH="${ATOMRL_DIR:-/workspace/ATOM/atomrl}" python3 examples/dsv4/patch_megatron_source.py --list --tag rocm
```

In a training container (full Lumen env):

```bash
python3 examples/dsv4/patch_megatron_source.py --list
python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
python3 examples/dsv4/patch_megatron_source.py "${MEGATRON_PATH}" --tag dsv4,rocm
```

Example output:

```text
dsv4_transformer_config                  [enabled] default=True tags=config,dsv4,megatron
  Add dsv4 variant and dsv4_* fields to TransformerConfig
dsv4_hash_routing                        [enabled] default=True tags=dsv4,megatron,moe
  Hash routing via tid2eid[input_ids] for early MoE layers
...
```

## Apply SOURCE patches to Megatron checkout

**Host bootstrap** (clone + patch, used by `prepare_rocm_megatron.sh`):

```bash
PYTHONPATH="${ATOMRL_DIR:-/workspace/ATOM/atomrl}" python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM
```

**Container re-apply** (idempotent, used by `setup_dsv4_container_env`):

```bash
PYTHONPATH=/workspace/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py "${MEGATRON_PATH}"
```

Patches are idempotent: re-running prints `skipped` for already-applied changes.

## Runtime patches (IMPORT / CONFIG_BUILD / MODEL_BUILD)

These run automatically during training — no shell script needed:

```text
import atomrl.lumen.models.megatron          → IMPORT patches (megatron_patches.install_all)
dsv4_gpt_builder()                    → CONFIG_BUILD tags {dsv4, builder}
get_dsv4_spec()                       → CONFIG_BUILD {dsv4, spec} + MODEL_BUILD {dsv4, spec}
```

Key env gates (IMPORT phase):

| Env | Patch |
|-----|-------|
| `LUMEN_FUSED_SWIGLU=1` | AITER Triton SwiGLU |
| `LUMEN_MLP_FP8_STORE=1` | MLP FP8 activation store |
| `LUMEN_FUSED_RESIDUAL_NORM=1` | Fused residual + RMSNorm |
| `LUMEN_DSV4_MOE_MORI=1` | MoEMori token dispatcher (MODEL_BUILD) |

Key env gates (SOURCE phase):

| Env | Patch |
|-----|-------|
| `LUMEN_DSV4_SKIP_OPTIONAL_NORMS=1` (default) | Skip optional norm/router ckpt keys |
| `MEGATRON_NO_BATCH_P2P_COMM=1` | Force `batch_p2p_comm=False` in Megatron args |

## Adding a new patch

1. Pick the phase (SOURCE vs IMPORT vs CONFIG_BUILD).
2. Implement the function in the matching module under `lumen/patches/`.
3. Register with `@register_patch(name, PatchPhase.…, tags=frozenset({…}))`.
4. For SOURCE: run `--list` to verify; for IMPORT: append is automatic via `install_all()`.
5. Add a test under `tests/patches/` if the registration logic is non-trivial.

## Related files

```text
lumen/patches/registry.py              # @register_patch, apply_patches, list_patches
lumen/patches/source/dsv4.py           # DSV4 Megatron on-disk patches
lumen/patches/source/rocm.py           # ROCm platform on-disk patches
lumen/patches/source/llama.py          # LLaMA/GPT on-disk patches
lumen/patches/builders/dsv4.py         # CONFIG_BUILD / ARGS / MODEL_BUILD
lumen/patches/runtime/megatron_import.py  # IMPORT monkey-patches (14)
lumen/patches/runtime/moe_fused_router.py   # MoE fused router IMPORT
lumen/models/megatron_patches.py       # Backward-compatible re-export shim
examples/dsv4/patch_megatron_source.py   # Thin wrapper (torch-less host entry)
examples/dsv4/patch_rocm_megatron_dsv4.py   # Deprecated alias
examples/dsv4/prepare_rocm_megatron.sh      # Clone ROCm Megatron + apply SOURCE patches
```
