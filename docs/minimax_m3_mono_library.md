# MiniMax-M3 mono tensor library

`atom.models.minimax_m3.mono.library` runs sparse decoder layers without an
ATOM engine, model runner, plugin, global forward context, or distributed-state
singleton. The serving engine selects eligible batches and supplies its existing
TP group and native tensor storage.

```python
from atom.models.minimax_m3.mono.library import (
    AtomM3Mono, CacheSpec, LayerSpec, StepMetadata, TPContext,
)

mono = AtomM3Mono(layer_specs, cache_specs, tp_context)
mono.prepare_step(step_metadata)
for layer_id in sparse_layer_ids:
    hidden, residual = mono.forward_layer(layer_id, hidden, residual, positions)
# After destroying captured graphs, before destroying the TP group:
mono.close()
```

`LayerSpec` contains AITER-shuffled per-channel E4M3 QKV/O projections and FP32
scales, BF16 router weights, norm and rotary tensors, shuffled MXFP4 expert
tensors, and numerical constants. `validate_layer` checks this contract before
runtime allocation. ATOM borrows all weights without conversion or copies;
native execution and mono use the same storage. The library and
native ATOM runner share sparse allocation and execution through `SparseExecution`,
and use the common compilation, IPC, and per-step mailbox runtime.

`CacheSpec` retains native packed main cache storage, its K/V page16 views,
independent index storage, and positive FP32 scalar K/V scales. Main storage is
`[blocks, 2, 128, 128]`; each main block occupies sixteen physical page16 pages.
Index storage is `[index_blocks, 128, 128]`. No cache copy or dynamic scale sidecar
is allocated.

`StepMetadata` contains request-level tables and lengths, token-level slots,
the padded token count, and uniform query length. A main-table entry is
**physical main block ID × 2**, expanded internally into eight page16 IDs.
An index-table entry is a physical 128-token index block ID. Main and index slot
addresses are independent; `-1` denotes padding. Lengths include all query tokens.
ATOM expands request metadata into token rows, applies the causal length for each
query token, marks padding with `batch_ids=-1`, and resets/fences mailboxes once
at the start of every step.

## Supported contract

- gfx950 with 256 CUs; TP4 using the caller's existing CPU process group.
- Dedicated GPU capacity: the persistent kernel's forward-progress contract
  assumes all 256 CTAs can reside together. External GPU work can invalidate
  this assumption; concurrent workloads are not qualified by the library.
- `TPContext.device` must be the rank's current CUDA device. Every rank must
  construct, prepare the same step shape, execute the same layer order, and
  close together. Initialization failures are agreed before IPC or execution.
- Sparse indexed M3 layers: hidden size 6144, local Q/KV/index heads 16/1/1,
  Gemma norms, partial NeoX rotary dimension 64, top-16 blocks, init/local 0/1.
- BF16 router weights with FP32 logits, sigmoid, and correction bias. The library
  builds kernels with `router_logits_fp32=True`; native ATOM retains its existing
  BF16-logit rounding by default. Choosing BF16 router weights is lossy and is
  the caller's explicit model configuration, not an implicit library conversion.
- Packed scalar FP8 K/V and independent unit-scale E4M3 index caches, contexts
  up to 16384, at most four requests, token counts 1/4/8/16, query length 1 or 4.
- Eager execution and CUDA graph capture/replay. `torch.compile` is explicitly
  rejected because functionalization can replace storage referenced by device
  pointer descriptors. There are no host reads of metadata tensor contents on
  the step path.

Construction eagerly compiles all supported kernel widths and warms metadata
expansion. Before graph capture, call `prepare_step` eagerly with the actual
input views to warm any tensor-alignment specialization for those views.
Unsupported configuration or compilation failure raises; the library does not
silently fall back. Native ATOM's default cache mode remains available to its
runner and retains per-token scale behavior.

The vLLM adapter checks all sparse layers collectively, including the actual
AITER quantization backend and shuffled layout. Unsupported weights use native
execution without allocating this runtime. At each step, unsupported token/query
shapes or more than four padded request rows also use native execution. A larger
configured scheduler concurrency does not disable eligible smaller batches.

Returned activation buffers are reused after two layers. Clone auxiliary states
that must survive that reuse. Keep the runtime, weights, caches, scales, and all
captured tensor addresses stable; weight updates and cache replacement require
tearing down graphs and constructing a new runtime. Calls on one runtime must
be serialized on the same stream. Destroy every captured graph before calling
`close()`, which drains device work, synchronizes the ranks, and frees IPC once.

## Validation

The CPU suites `tests/test_mono_runtime.py` and
`tests/test_minimax_m3_mono_library.py` cover shared argument contracts,
collective refusals, import isolation, and unsupported compilation. The build-key
suite compiles both cache modes and checks that mode changes cannot reuse a
binary with different cache semantics.

`tests/minimax_m3_mono_replay.py` is an opt-in TP4 replay harness for captured
native layer tensors with online PTPC projections and a BF16/FP32-output router.
Each `rank-N` directory contains `weights.pt`, `info.json`
(the model's numerical configuration), and `case-*.pt` (native inputs, caches,
metadata, and expected outputs). Input captures must contain only live rows;
the harness injects padding itself. It reports precision differences from native
and asserts exact eager/graph equality, independent cache remapping, repeated
steps, and live-row invariance under poisoned padding. Run it on gfx950:

```bash
ATOM_DISABLE_VLLM_PLUGIN=1 VLLM_PLUGINS='' \
python -m torch.distributed.run --master-addr 127.0.0.1 --master-port 29631 \
    --nproc-per-node 4 tests/minimax_m3_mono_replay.py \
    --directory /path/to/native-layer --layer-id 3 --output /tmp/mono-replay
```

Repeat with `--k-scale-factor 2 --v-scale-factor 0.5` to exercise independent,
non-unit scalar scales; the harness rescales captured cache payloads accordingly.
Checkpoint captures are external fixtures and are not included in CPU CI.
Older BF16-projection captures must be regenerated; the harness does not convert
them into a different weight configuration.
