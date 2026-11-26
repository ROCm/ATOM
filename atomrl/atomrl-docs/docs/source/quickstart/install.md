# Installation

This page covers hardware and software requirements, container-based setup, editable installs for development, third-party stack components, and a short verification snippet.

## Requirements

| Component | Version / hardware |
| --- | --- |
| Python | >= 3.10 |
| PyTorch | >= 2.4 (ROCm build recommended) |
| ROCm | >= 6.2 |
| GPU | AMD Instinct MI250 / MI300 / MI350 series |

```{note}
Match your PyTorch wheel to the ROCm version on the host or inside the container. Mismatched userspace stacks are the most common source of runtime failures on AMD clusters.
```

## Docker install (recommended)

The repository Docker build installs the root ATOM package and the separate
AtomRL distribution together, including Ray/OmegaConf and optional Lumen
extensions.

```bash
git clone https://github.com/ROCm/ATOM.git
cd ATOM
bash atomrl/docker/build.sh
```

Start the resulting image:

```bash
docker run -it --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=16G -v "$PWD":/workspace/ATOM \
  -w /workspace/ATOM/atomrl atomrl:latest
```

Inside the container, verify the installation:

```bash
python -c "import atom, atomrl, atomrl.lumen; print('ATOM/AtomRL/Lumen: OK')"
```

Do not add Ray or OmegaConf to the root ATOM install and do not install
`atomrl/lumen` as a standalone package.

The repository Dockerfiles build Lumen's optional HIP extensions when the
required ROCm toolchain is available; otherwise the Python/Triton fallbacks
remain available.

## Developer install

On a ROCm host with compatible PyTorch already available:

```bash
git clone https://github.com/ROCm/ATOM.git
cd ATOM
git checkout main

# Build the same supported development image from the ATOM monorepo root.
bash atomrl/docker/build.sh --dev
```

If you iterate on Ray workers or configs, keep the editable install so local changes are picked up without reinstalling.

## Third-party libraries

| Library | PyPI / import | Role in AtomRL |
| --- | --- | --- |
| [Lumen](https://github.com/ROCm/ATOM/tree/main/atomrl/lumen) | `atomrl.lumen` | Quantized training: hybrid FP8/MXFP8, AITER-backed kernels, MORI communication hooks |
| [ATOM](https://github.com/ROCm/ATOM) | `atom` | High-throughput inference: FP8, MXFP4, MoE expert parallel, speculative decoding |
| [AITER](https://github.com/ROCm/aiter) | `amd-aiter` | Low-level GPU kernels (attention, GEMM, MoE, norm) |
| [MORI](https://github.com/ROCm/mori) | `mori` | RDMA-aware GPU collectives and MoE expert dispatch |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | `mooncake` | Transfer engine: async RDMA/TCP hidden-state transport |
| [Ray](https://github.com/ray-project/ray) | `ray` | Controller/worker orchestration and resource placement |

## Verify installation

Run a lightweight import check and print key versions:

```bash
python - <<'PY'
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda/hip available:", torch.cuda.is_available())

import atomrl  # noqa: F401
from atomrl.core.config import AtomRLConfig

cfg = AtomRLConfig()
print("AtomRLConfig OK:", type(cfg).__name__)
PY
```

If `torch.cuda.is_available()` is false on a GPU node, reinstall a ROCm-enabled PyTorch build before debugging AtomRL itself.

Next: {doc}`/quickstart/quick_start` for a minimal GRPO launch and links into the deeper guides.
