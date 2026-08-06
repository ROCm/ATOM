# Kimi-K3 Usage Guide

Kimi-K3 is a **KimiLinear hybrid-attention MoE** model (`KimiLinearForCausalLM`). Each decoder layer is either a **KDA linear-attention** layer or an **MLA full-attention** layer, on top of a large MXFP4 latent MoE. ATOM serves the **text-only** backbone.

Covers **MI355 (gfx950), `-tp 8`** and **MI450 (gfx1250), `-tp 4`**. Same checkpoint and model math; only the enablement differs.

| Variant | Quantization | Description |
|---------|-------------|-------------|
| **MXFP4** | MXFP4 (w4a4, e8m0 scales, group_size=32) | Routed MoE expert weights in microscale FP4. On gfx950 the SiTU experts run the FlyDSL **native SiTUv2** grouped-MoE path. Attention, shared experts, and dense MLP remain BF16. |

---

## gfx950 / MI355

### Launch server — MXFP4 on 8×MI355 GPUs (TP8)

```bash
#!/bin/bash

python -m atom.entrypoints.openai_server \
  --model Kimi-K3 \
  --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.93 \
  --block-size 128 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'
```

Kimi full-attention layers use true MLA with a compressed latent KV cache. Aiter MLA is selected by default; `ATOM_USE_TRITON_MLA=1` selects the Triton MLA implementation when that configuration has been validated.

Prefix caching remains disabled because the KDA recurrent state is maintained per request and cannot be reconstructed from the paged MLA cache alone. `-tp 8` is required for the model to fit. Use `gpu-memory-utilization 0.93` so the CUDA-graph pool fits alongside the KDA per-request state cache.

### Accuracy test

With that server running, execute the full 1319-question GSM8K 5-shot evaluation with base completions and seed 42:

```bash
lm_eval \
  --model local-completions \
  --model_args "model=Kimi-K3,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --seed 42
```

`model=` must match what the server reports at `/v1/models`, which is whatever was passed to `--model` -- a path if the checkpoint was launched by path.

Validated true-MLA result range on gfx950 TP8 across three clean-start runs:

```text
| Filter           | Minimum | Maximum |
|------------------|--------:|--------:|
| flexible-extract |  0.9538 |  0.9591 |
| strict-match     |  0.9538 |  0.9591 |
```

---

## gfx1250 / MI450

### Launch server — MXFP4 on 4×MI450 GPUs (TP4)

Measured in container image **`rocm/fw-bringup:gfx1250-atom-dev-20260729`**, whose
triton and torch builds are kept as-is:

| Component | Version | Notes |
|---|---|---|
| triton | `3.8.0+git5b5a3760` | the image's build at `/app/triton-mi450`; gfx1250 support depends on it |
| torch | `2.11.0+rocm7.15.0a20260712` | from the image |
| ROCm | `7.15.0` | from the image |
| flydsl | `0.2.4` | from the image |
| amd-aiter | `0.1.20.dev32+gb0b6945e7` plus ROCm/aiter#4482 | built in-image, see below |
| fla-core / flash-linear-attention | `0.5.1` | required by the KDA layers |

Install this tree's ATOM and aiter over whatever the image ships, plus the FLA
packages:

```bash
# aiter. AITER_USE_SYSTEM_TRITON=1 is not optional: aiter's setup.py defaults it
# to 0, at which point it uninstalls the image's triton and pulls its own pinned
# build, discarding the gfx1250-capable one. --no-deps keeps the rest of the
# image's stack (torch, flydsl) untouched for the same reason.
cd /path/to/aiter
AITER_USE_SYSTEM_TRITON=1 ENABLE_CK=0 PREBUILD_KERNELS=0 \
  pip install -e . --no-build-isolation --no-deps --break-system-packages

# ATOM. Unlike aiter this one needs build isolation: pyproject.toml uses the
# PEP 639 `license = "MIT"` string, which the image's setuptools 68 rejects
# ("`project.license` must be valid exactly by one definition"), so pip has to
# fetch a newer setuptools into the build env.
cd /path/to/ATOM
pip install -e . --no-deps --break-system-packages

# KDA kernels. atom/models/kimi_k3.py imports fla.ops.kda with no fallback, so
# the engine dies at the first KDA layer without these.
pip install --no-deps --break-system-packages \
  "fla-core==0.5.1" "flash-linear-attention==0.5.1"
```

`--break-system-packages` is what this image needs (PEP 668); drop it on images
that use a virtualenv.

FlyDSL resolves the device bitcode as `$ROCM_PATH/amdgcn/bitcode`. This image's
ROCm is a pip wheel that keeps it one level deeper, and the MLA prefill kernel
then fails to compile with `ROCm amdgcn bitcode path ... does not exist or is
not a directory`. Link it if it is missing:

```bash
[ -d "$ROCM_PATH/amdgcn/bitcode" ] || ln -s lib/llvm/amdgcn "$ROCM_PATH/amdgcn"
```

Verify before launching -- all three must point where you expect, and triton
must still be the image's build:

```bash
python -c "import aiter, atom, triton, fla; print(aiter.__file__, atom.__file__, triton.__version__, triton.__file__)"
```

```bash
#!/bin/bash
# ---- gfx1250 backends (no CK) ----
export ENABLE_CK=0                          # CK never registered gfx1250 -> flydsl/triton/hip
export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=1             # flydsl grouped MoE
export ATOM_MOE_GU_ITLV=1                   # GUGU weights -> the grouped MoE
export ATOM_USE_TRITON_MLA=1                # K3 MLA-latent Triton path

# ---- gfx1250 correctness ----
export ATOM_KDA_FORCE_RECURRENT=1           # chunk_kda NaNs here; run KDA prefill recurrently
export ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1   # avoid aiter's Gluon MXFP4 preshuffle GEMM

python -m atom.entrypoints.openai_server \
  --model Kimi-K3 \
  --kv_cache_dtype bf16 -tp 4 \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.93 \
  --no-enable_prefix_caching
```

`ATOM_MOE_GU_ITLV=1` is the one to not drop. It shuffles the MoE weights to
GUGU, which is what routes K3 to aiter's grouped MoE -- the only path with a
SiTUv2 (`hidden_act="situ"`) kernel on gfx1250. Without it the weights stay
GGUU, the dispatcher reads `GateMode.SEPARATED`, `AITER_USE_GROUPED_GEMM=1`
above becomes a no-op, and SiTUv2 lands on `flydsl_moe1_afp4_wfp4_bf16_*`, which
on current aiter fails to build (`LLVM ERROR: Do not know how to expand this
operator's operand`) and takes the server down during warmup.

`ATOM_K3_MOE_CHUNK` and `--level 0` are no longer needed. Both existed for a
grouped-MoE fault at K3's prefill sizes that was read as "only correct at small
M"; the actual cause was the expert count, not M. aiter's contiguous-M prefix
scan ran one thread per expert in a single 512-thread block and silently dropped
everything past expert 512, and K3 has 896. With that scan fixed the chunking is
unnecessary and level 3 (the default) captures and replays normally.

### Accuracy test

```bash
lm_eval \
  --model local-completions \
  --model_args "model=Kimi-K3,base_url=http://localhost:8000/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --seed 42
```

Run it with `--limit 100` first (expect `0.99`) before committing to the full set.

Full 1319 questions on the stack tabled above, with the launch command exactly as
printed. The same score was measured on the earlier configuration -- aiter
`56f56db7e` unmodified, with `ATOM_K3_MOE_CHUNK=128` and `--level 0` -- so
dropping those two costs nothing:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9591|±  |0.0055|
|     |       |strict-match    |     5|exact_match|↑  |0.9591|±  |0.0055|
```
