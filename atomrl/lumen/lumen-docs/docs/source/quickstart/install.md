# Installation

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.4 |
| ROCm | >= 6.2 |
| Hardware | AMD Instinct MI300X (CDNA3) |

## User Install (recommended)

```bash
git clone https://github.com/ROCm/ATOM.git
cd ATOM
git checkout main
pip install torch triton amd-aiter
pip install --no-build-isolation --no-deps -e .
LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation -e ./atomrl
```

## Developer Install

```bash
git clone https://github.com/ROCm/ATOM.git
cd ATOM
git checkout main
pip install torch triton amd-aiter pytest pytest-xdist torchao
pip install --no-build-isolation --no-deps -e .
LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation -e ./atomrl
```

## Third-party Libraries

Lumen delegates kernel execution and communication to two AMD-native libraries:

| Library | PyPI Package | Purpose |
|---------|-------------|---------|
| [AITER](https://github.com/ROCm/aiter) | `amd-aiter` | GPU kernels: attention, GEMM, normalization, RoPE, MoE, fused MLP, cross-entropy, quantization (ASM / CK / Triton backends) |
| [MORI](https://github.com/ROCm/mori) | `mori` | RDMA + GPU communication: MORI-CCL (collective ops), MORI-EP (MoE expert dispatch) |

Case images install these dependencies explicitly before installing ATOM.

`LUMEN_BUILD_EXTENSIONS=1` is fail-fast: it requires ROCm Torch, `HIP_HOME`,
and `hipcc`. Omit it for CPU/general wheels; `atomrl.lumen` then uses its
Python/Triton runtime fallbacks.

## Verify Installation

```python
import atomrl.lumen
print(atomrl.lumen.__version__)

import atomrl.lumen.quantize as quant
from atomrl.lumen.quantize import QuantConfig, QuantFormat, ScalingType
print("Lumen ready.")
```
