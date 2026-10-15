# AF piecewise cores

`@piecewise_core` places a core between compiled dense pieces. With
`piecewise=True`, its output uses a stable address for the downstream graph.
`capture=True` also captures/replays the core when eligible. FULL/eager calls
use `piecewise=False` and execute the original method directly.

## Live prefill metadata interface

Use `capture_prefill=True` and a metadata provider to opt into live prefill.
The provider receives the layer and bound arguments, including defaults, and
returns `PiecewiseMetadata` or `None` to fall back to eager execution. Put it
on the model's existing metadata or attention class; no graph runner subclass
or per-model capture-state class is needed.

```python
from dataclasses import replace
import torch
from atom.utils.attn_ffn_piecewise import piecewise_core
from atom.utils.piecewise_af import PiecewiseMetadata

# Method on the model's existing metadata class:
def piecewise_metadata(self, cache):
    def rebuild(values):
        return {"step": replace(self, **values, indices={})}

    def publish(arguments):
        self.indices.update(arguments["step"].indices)

    return PiecewiseMetadata(
        key=(self.is_decode, cache.pool.data_ptr()),
        inputs={"positions": self.positions, "lengths": self.lengths},
        rebuild=rebuild,
        publish=publish,
    )

# Method on the model's existing layer:
@piecewise_core(
    capture_prefill=True,
    metadata=lambda layer, args: args["step"].piecewise_metadata(args["cache"]),
)
def attention_forward(self, x: torch.Tensor, step: BatchStep, cache: Cache):
    return self.attention(x, step, cache)

# At the splitting boundary:
result = layer.attention_forward(
    hidden, step, cache,
    piecewise=True, capture=True, forward_context=forward_context,
)
```

The decorator lazily owns a runner/output store per layer method. Supply a
common `PiecewiseAFRunner` when sharing input storage across layers or using
fixed-capacity `StableOutputs`. Shared stores are for serial execution;
concurrent microbatches need separate stores. The existing graph-holder
lifecycle releases recordings and captured Python state.

The interface has four parts:

- `inputs`: explicit dict/list/tuple trees with tensors and Python scalar
  leaves (`None`, bool, int, float, str). The engine keys tree structure,
  scalar values, tensor shape/stride/dtype/device and direct input addresses.
  Do not include arbitrary request objects or workspaces.
- `key`: additional host decisions and persistent addresses outside that tree.
  Non-tensor core arguments remain baked into a recording; scalar arguments
  are keyed automatically, and the provider must key other changing config.
- `rebuild`: creates non-tensor keyword arguments from staged inputs, with
  fresh derived Python state separately for warmup and recording. It must
  preserve any metadata aliases the core relies on. Inputs are read-only;
  device writes belong in persistent state such as the KV pool.
- `publish`: updates the current live metadata from the executed state. Copy
  only this core's results, preserving upstream inputs. It is optional for
  cores without Python-visible results and is never cached across requests.

Tensor parameters of the core are inferred from annotations and directly read
by default. `copy_per_step=("x",)` stages selected parameters instead, or pass
`zero_copy_inputs=()` at the boundary to stage all of them for that call.
Metadata tensors are always staged. An optional `memo` skips repeated copies
across layers within one forward; its owner must clear it at every forward,
even when reusing metadata and source allocations. `copy_each_call` names
metadata leaf paths whose contents can change within a forward.

The first live call warms once and records without replaying, then publishes
the warmup result. Later calls refresh inputs, replay once and publish the
recorded results. This preserves exactly-once device state updates. Startup
capture, dummy runs and FULL runtime graphs cannot start a nested live capture.
The default limit is 32 graphs per runner; new keys beyond it run eagerly
without staging inputs, while existing keys continue to replay.

Decode-only users retain the existing `key=decode_bucket_key` startup-capture
API. The low-level `capture_key`/`prepare` route remains available, but cannot
be combined with a metadata provider.
