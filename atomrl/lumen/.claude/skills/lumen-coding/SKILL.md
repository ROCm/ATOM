---
name: lumen-coding
description: "Use when writing, reviewing, or designing any Lumen code. Covers architecture (Megatron -> modules -> ops -> kernels), AITER dispatch conventions, FP8 quantization, torch.compile compatibility, and code review checklists."
---

# Lumen Development Guide

## Mission

Lumen is a drop-in replacement for NVIDIA TransformerEngine (TE) within Megatron-Core, targeting AMD GPUs (gfx94x / gfx950). Every module, op, and kernel provides a feature-equivalent alternative to TE's interfaces.

## Architecture

```
Megatron-Core
  +- lumen/models/megatron.py   (LumenSpecProvider)
       +- lumen/modules/         (nn.Module wrappers)
            +- lumen/ops/        (stateless dispatch functions)
                 +- lumen/kernels/    (Lumen-owned Triton kernels)
                 +- external ROCm/aiter checkout (CK, ASM -- NOT Lumen code)
```

| Layer | Owns | Does NOT own |
|-------|------|-------------|
| `modules/` | State (params, buffers), Megatron API shape | Kernel calls |
| `ops/` | Backend dispatch, argument marshalling, fallback | Param management |
| `kernels/` | Lumen-specific Triton kernels | CK/ASM kernels (-> AITER) |

**New GPU kernels belong in AITER, not in Lumen.** Lumen only dispatches.

### AITER Public-API Boundary

AITER is part of the supported image contract. Lumen must call AITER's public API directly.

- Do not add `_probe_aiter_*`, `find_spec("aiter")`, guarded `try: import
  aiter`, symbol inspection, or private-attribute capability copies.
- Do not reimplement AITER kernels, tensor permutations, weight/scale shuffles,
  or communication setup in AtomRL/Lumen. Add or optimize that behavior in
  AITER, then consume its public function here.
- `try_backends()` chooses among supported public implementations. It is not a
  dependency or symbol availability probe.
- If a required public API is missing from the supported AITER checkout, fail
  clearly or update AITER first; do not add a compatibility shim in Lumen.

## Coding Conventions

### Constants -- derive from spec

```python
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
scale = amax / FP8_E4M3_MAX  # NOT: scale = 448.0
```

### No TE language in docstrings

```python
# BAD: """Drop-in replacement for TE's TEColumnParallelLinear."""
# GOOD: """Column-parallel linear using Lumen GEMM."""
```

### Fallback Logging

Every fallback path must log why and comment the intent:

```python
def my_op(x):
    try:
        return _asm_path(x)
    except RuntimeError as e:
        logger.warning("my_op: ASM failed (%s), falling back to Triton", e)
        return _triton_path(x)
```

`try_backends()` chains need per-lambda comments. Conditional fallbacks need comments explaining why.

### First-Principles Codegen

1. Clarify the problem in one sentence
2. Identify constraints (memory bandwidth, register pressure, API contract)
3. Derive solution from constraints, not analogy
4. Add abstraction only when it removes duplication across >=2 call sites

### Anti-Patterns

- Abstraction with one subclass -> use plain function
- Copying TE API blindly -> shape API by Lumen's own requirements

## Code Review Checklist

- [ ] Constants derived from hardware facts, not magic numbers
- [ ] No unnecessary abstraction (YAGNI)
- [ ] `try_backends()` ordered correctly (ASM -> CK -> Triton)
- [ ] AITER imports use the pinned public runtime API directly
- [ ] No AITER import/symbol/capability probes or private attributes
- [ ] No local copies of AITER kernels, shuffles, layouts, or communication code
- [ ] Fallback paths logged with reason
- [ ] FP8 scaling type handled for both forward and backward
- [ ] No TE "drop-in replacement" language in docstrings
- [ ] Tests use `compute_snr` / `check_close` against references
- [ ] Fallback paths covered by tests
- [ ] Performance-critical path free of Python-level overhead

### Handling Review Feedback

- **Verify before implementing** -- check against codebase reality
- **Push back with reasoning** if a suggestion breaks functionality or violates YAGNI
- **No performative agreement** -- state what changed or ask for clarification
- **One fix at a time**, test each before proceeding

## Feature Parity

Scorecard: ~61 features, ~47 supported (77%), 4 Lumen-only, ~2 missing, ~6 partial, ~2 deferred.

For complete feature matrix and TE mapping -> see [reference.md](reference.md).

---

## Subsection Guides

Detailed domain-specific guides are in separate files. Read when working in that area:

- **[AITER Dispatch](aiter-dispatch.md)** -- public API boundary, `try_backends()`, GEMM/attention backend dispatch tables, fallback chain design
- **[Quantize Manager](quantize-manager.md)** -- ScalingManager lifecycle, FP8 scaling types, blockwise2d dual behavior, amax history
- **[torch.compile Compatibility](torch-compile.md)** -- Compile Guard Dual-Mode design, 6 blockers, implementation phases, `LUMEN_BACKEND` env var
- **[Megatron & FSDP Integration](megatron-fsdp.md)** -- LumenSpecProvider, norm patching, FSDP1/FSDP2, CLI args, example scripts, QuantConfig wiring
