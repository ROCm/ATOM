# AF piecewise cores

`@piecewise_core` places a core between compiled dense pieces. With
`piecewise=True`, its output uses a stable address for the downstream graph.
`capture=True` also captures/replays the core when eligible. FULL/eager calls
use `piecewise=False` and execute the original method directly.

## Declarative live prefill metadata

Use `capture_prefill=True` and a `metadata` dictionary. The decorator compiles
it into the common `PiecewiseMetadataSchema`; models only declare their
fields and rules. No per-model staging, reconstruction, publication method
or graph runner subclass is needed.

```python
import torch
from atom.utils.attn_ffn_piecewise import piecewise_core

@piecewise_core(
    capture_prefill=True,
    metadata={
        "when": "step.enabled",
        "key": ("step.is_decode", lambda s: s["cache"].pool.data_ptr()),
        "objects": {
            "step": {
                "inputs": ("positions", "lengths"),
                "reset": {"indices": {}},
            },
        },
        "arguments": ("step",),
        "publish": {"step.indices": "update"},
    },
)
def attention_forward(self, x: torch.Tensor, step: BatchStep, cache: Cache):
    return self.attention(x, step, cache)

result = layer.attention_forward(
    hidden, step, cache,
    piecewise=True, capture=True, forward_context=forward_context,
)
```

Selectors are dotted paths rooted at bound core arguments (including their
defaults) or `layer`. Ordered `bindings` add reusable names. A callable taking
that scope may express a model's host predicate or derived scalar, such as a
KV pool address. Strings are field paths, never evaluated Python expressions.
A path through `None` yields `None`; missing fields raise an error.

| Declaration | Common behavior |
|---|---|
| `objects[name].source` | Live template path, defaulting to `name`; `None` creates no view. |
| `inputs` | Names of fields to stage. Dict/list/tuple trees, tensor and scalar leaves are supported. |
| `select` | Select dictionary fields with `{"keys": (selector, ...)}` or `{"suffix": selector}` and optional `when`. Missing requested keys fail; disabled selections are empty. |
| `reset` | Fresh scalar/container values for derived fields on each warmup and recording, for example `{"indices": {}, "window": None}`. |
| `bind` | Attach an earlier captured object by name, or use `{"object": name, "key": selector}` to attach a one-entry dictionary, empty if the object is `None`. |
| `template` | Clone an earlier captured object, with inputs from this object's `source`; unchanged fields keep their aliases. |
| `arguments` | Object names to pass as non-tensor keyword overrides to the core. |
| `publish` | Field paths mapped to `update`, `replace`, `if_not_none`, or `{"keys": (selector, ...)}` for scoped dictionary writes. |
| `when`, `key`, `memo`, `copy_each_call` | Eligibility, extra host decisions, optional per-forward copy memo and metadata paths refreshed on each call. |

Objects are declared in dependency order. Dataclasses (including frozen
ones), dictionaries and shallow-copyable attribute objects are supported.
Undeclared fields stay on the template; all mutable derived fields that the
core changes must be declared as `reset`, selected inputs or explicit aliases.
Tensor inputs are read-only. Persistent device state such as a KV pool remains
outside the staged input tree and its identity belongs in the key.

For a subset view sharing the main step's result dictionaries:

```python
"objects": {
    "step": {
        "inputs": ("positions",),
        "reset": {"selected": {}, "indices": {}},
    },
    "query_index": {
        "source": "queries.index",
        "template": "step",
        "inputs": ("batch_ids",),
    },
    "queries": {
        "inputs": ("rows",),
        "bind": {"index": "query_index"},
    },
},
"arguments": ("step", "queries"),
"publish": {
    "step.selected": {"keys": ("layer.layer_id",)},
    "step.indices": "update",
},
```

Warmup and recording get separate fresh dictionaries; each query index shares
its dictionaries with its own captured step. Keyed publication skips absent
outputs and preserves upstream entries in the current live request. Input
layouts/scalars, object presence/types, attachment keys and publication keys
are included in the graph key. Other host branches must be declared in `key`.

Reconstruction runs only on a cache miss, once for warmup and once for
recording. Replay refreshes tensors and publishes recorded results without
rebuilding metadata. Schema compilation happens at decoration, not per step.

## Runner and advanced callback interface

A callable metadata provider remains supported for unusual cases. It receives
`(layer, bound_arguments)` and returns `PiecewiseMetadata` or `None`. Prefer the
schema for standard field selection, cloning and updates. The low-level
callback contract below describes what the common schema implements.

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
