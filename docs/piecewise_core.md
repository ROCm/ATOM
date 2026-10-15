# AF piecewise cores

`@piecewise_core` places a core between compiled dense pieces. With
`piecewise=True`, its output uses a stable address for the downstream graph.
`capture=True` also captures/replays the core when eligible. FULL/eager calls
use `piecewise=False` and execute the original method directly.

## Builder-owned metadata

Live prefill uses the same ownership principle as V4's decode compressor:
the attention builder publishes fixed-address buffers before the layers run,
and graphs read those buffers directly. The model only decorates its core:

```python
@piecewise_core(capture_prefill=True)
def attention_forward(self, x: torch.Tensor, step: BatchStep):
    return self.attention(x, step)
```

The builder supplies a `PiecewiseState` once per forward or subset view:

```python
step.piecewise_state = PiecewiseState(
    key=(is_decode, query_width, kernel_variant),
    buffers=(positions, lengths, block_tables, kv_pool),
    caches={"indices": step.indices},
)

result = layer.attention_forward(
    hidden, step,
    piecewise=True, capture=True, forward_context=forward_context,
    capture_state=step.piecewise_state,
)
```

`key` must include all host decisions that affect the captured kernels.
`buffers` is an explicit tensor/scalar tree; addresses, shapes, strides,
dtypes, devices and scalar values are keyed automatically. Include persistent
resources such as the KV pool. The state retains these buffers for the graph's
lifetime. Other changing non-tensor arguments must be represented in the key.
Scalar core arguments are keyed automatically.

The core receives the live metadata unchanged. There is no schema, metadata
reconstruction, per-layer refresh, or model-specific graph runner. The builder
must publish contents before execution and preserve addresses for replay.
Replaced allocations select another graph. Tensor parameters are direct inputs
by default; `copy_per_step=("x",)` or per-call `zero_copy_inputs=()` allows
temporary producer tensors by copying them to runner-owned storage.

## Per-forward results

Use the common `ReplayCache` for Python dictionaries populated by a core:

```python
step.indices = ReplayCache()
step.selected = ReplayCache(stable=True)
```

The runner records cache reads and writes during warmup and CUDA recording.
Both executions see the same entry state, so lazy index-building kernels are
included in the graph. Only entries actually read before the core's own writes
are dependencies. Replay validates those dependencies and publishes only the
recorded writes into the current forward's caches. Upstream and unrelated
entries are preserved. Shared subset views should share the actual cache
objects; name each shared cache once in `PiecewiseState`.

Values are immutable tensor/scalar trees. Update the mapping rather than
mutating a nested container in place. Persistent mutable GPU state such as KV
belongs in the builder's buffers. Normal eager execution uses ordinary mapping
operations; no capture callbacks are needed. Clear caches at every forward,
including when reusing the same metadata object.

When downstream cores reuse a result, `stable=True` delivers tensor writes to
the state's optional `PiecewiseBuffers` store. Eager producers, first capture
and replay then publish the same address. Give independent or concurrently
live views distinct storage namespaces. Use `PiecewiseBuffers` only for
results/subsets without existing storage; do not copy the builder's existing
metadata into it again. For subset inputs, prepare the stable storage when
entering that view, before recording or replaying any of its layers.

The first live call warms once, records without replaying, and publishes the
warmup result. Later calls replay once and publish recorded results. GPU state
updates therefore occur exactly once per invocation. Startup capture, dummy
runs and FULL runtime graphs cannot start a nested live capture. Explicit
`capture_state=None` declines live capture. At the default limit of 32 graphs
per runner, new keys fall back eagerly before allocating graph inputs; cached
keys continue to replay.

The decorator lazily creates a `PiecewiseAFRunner` and output storage when
none are supplied. Explicit instances support fixed-capacity `StableOutputs`.
Graph-holder release drops recordings and retained resources; builder buffer
stores participate in the same lifecycle.

## Existing decode interface

V4 and other decode-only users retain `key=decode_bucket_key` and startup
capture. They need no `PiecewiseState` when their core only writes persistent
device state. The lower-level `capture_key`/`prepare` route remains available
for callers managing their own lifecycle, and cannot be combined with
`capture_state`.
