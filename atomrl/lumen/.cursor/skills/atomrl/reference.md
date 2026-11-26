# AtomRL and Lumen Reference

## Project Ownership

- `ATOM/atomrl/atomrl/` owns AtomRL orchestration and post-training.
- `ATOM/atomrl/atomrl/lumen/` owns quantized model and kernel support.
- `ATOM/atomrl/atomrl/plugin/verl/` owns the optional AtomRL VERL plugin.

Lumen's former in-tree framework adapters were removed; use AtomRL's plugin integration for VERL workflows.

## Support Boundaries

- AtomRL native inference: ATOM and vLLM.
- AtomRL SGLang: unsupported and intentionally removed.
- Lumen training backends: Hugging Face FSDP/FSDP2 and Megatron.
- Lumen supplies quantization, operators, patches, models, and distributed training primitives, not an RL framework adapter.
