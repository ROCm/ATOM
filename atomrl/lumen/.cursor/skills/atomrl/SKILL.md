---
name: atomrl
description: Use for AtomRL orchestration, ATOM/vLLM inference, the optional AtomRL VERL plugin, and Lumen quantized training on AMD GPUs.
---

# AtomRL and Lumen

- AtomRL orchestration lives under `ATOM/atomrl/atomrl/`.
- The optional AtomRL VERL plugin lives under `ATOM/atomrl/atomrl/plugin/verl/`.
- Lumen quantization, kernels, model patches, and distributed training live under `ATOM/atomrl/atomrl/lumen/`.
- Lumen's former framework-specific RL adapters were removed; do not recreate them.
- AtomRL supports ATOM and vLLM inference. SGLang is not an AtomRL backend.

Read `reference.md` and `getting-started.md` before changing ownership or support statements.
