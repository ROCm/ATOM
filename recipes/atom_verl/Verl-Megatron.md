# Configure a veRL + Megatron + ATOM Training Environment on ROCm

veRL currently supports MTP training primarily through Megatron. However, Megatron is not yet supported in ROCm environments. This document describes a practical environment setup based on ROCm images and upstream repositories so that veRL + ATOM can be used to train MTP models on ROCm.

## Base Docker Image

Use the following image as the base environment:

```bash
rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev
```

## Start the Container

```bash
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=128G \
  -v "$HOME:$HOME" \
  -w "$PWD" \
  rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev \
  bash
```

## Installation Guide

### 1. TransformerEngine

Repository: [ROCm/TransformerEngine](https://github.com/ROCm/TransformerEngine) (`main`)

#### Option 1: Install from Wheel

This is faster, but you may need to resolve dependency conflicts manually.

```bash
pip install transformer_engine==2.1.0 \
  --index-url https://download.pytorch.org/whl/rocm7.0
```

`transformer_engine==2.1.0` has a version check that is incompatible with `flash-attn==2.8.3`. Update the upper bound in `attention.py` with:

```bash
sed -i 's/PkgVersion("2.8.0.post2")/PkgVersion("2.8.3")/' \
  /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention.py
```

#### Option 2: Install from Source

This is slower, but gives you tighter control over dependencies.

```bash
git clone --recursive https://github.com/ROCm/TransformerEngine.git
cd TransformerEngine
export NVTE_FRAMEWORK=pytorch,jax
export NVTE_ROCM_ARCH="gfx942;gfx950"
export NVTE_USE_ROCM=1
pip3 install --no-build-isolation .
```

Notes:

- `NVTE_FRAMEWORK` is optional.
- `NVTE_ROCM_ARCH` is optional. `gfx942` targets MI300/MI325, and `gfx950` targets MI350.
- `NVTE_USE_ROCM=1` is optional, but useful if you want to force a ROCm build.

### 2. Megatron-LM (including Megatron-Core)

Repository: [NVIDIA/Megatron-LM](https://github.com/nvidia/megatron-lm) (`main`)

```bash
git clone -b main https://github.com/nvidia/megatron-lm.git
cd megatron-lm
```

Choose one of the following installation methods:

```bash
# Editable install with dependencies
pip install -e .

# Editable install without pulling dependencies
pip install -e . --no-deps
```

### 3. MBridge (Megatron-Core <-> Hugging Face / RL)

Repository: [mbridge-project/mbridge](https://github.com/mbridge-project/mbridge) (`main`)

The PyPI release `0.15.1` has a compatibility issue related to `ModelType.encoder_and_decoder`. Install directly from GitHub and force reinstall:

```bash
pip install git+https://github.com/mbridge-project/mbridge.git --force-reinstall
```

### 4. veRL

Repository: [sijyang/verl](https://github.com/sijyang/verl) (`dev`)

The base image is mainly used to provide lower-level dependencies. You should still reinstall veRL from source inside the container.

```bash
git clone -b dev https://github.com/sijyang/verl.git
cd verl

# Remove any preinstalled veRL package first
pip uninstall -y verl veRL || true

# Reinstall from source
pip install -e .
```

### 5. AITER

Repository: [ROCm/aiter](https://github.com/ROCm/aiter) (`main`)

```bash
git clone --recursive -b main https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

If you forgot to add `--recursive` when cloning:

```bash
git submodule sync
git submodule update --init --recursive
```

### 6. ATOM

Repository: [ROCm/ATOM](https://github.com/ROCm/ATOM) (`main`)

```bash
git clone -b main https://github.com/ROCm/ATOM.git
cd ATOM
pip install -e .
```

## Appendix A: Repository and Branch Summary

| Package | Repository / Image | Branch |
| --- | --- | --- |
| Base image | `rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev` | N/A |
| TransformerEngine | [ROCm/TransformerEngine](https://github.com/ROCm/TransformerEngine) | `main` |
| Megatron-LM | [NVIDIA/Megatron-LM](https://github.com/nvidia/megatron-lm) | `main` |
| MBridge | [mbridge-project/mbridge](https://github.com/mbridge-project/mbridge) | `main` |
| veRL | [sijyang/verl](https://github.com/sijyang/verl) | `dev` |
| AITER | [ROCm/aiter](https://github.com/ROCm/aiter) | `main` |
| ATOM | [ROCm/ATOM](https://github.com/ROCm/ATOM) | `main` |
