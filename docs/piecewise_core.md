# AF piecewise cores

`@piecewise_core` places a numerical region between compiled pieces. The common
`CudagraphCaptureRunner` owns recording, replay, graph lifetime and the output
address consumed by the next piece. An attention backend owns request metadata
and cross-layer result buffers, through its normal preparation lifecycle.

## Model interface

```python
@piecewise_core(capture_prefill=True, token_dim=1)
def attention_forward(self, hidden: torch.Tensor, cache: PagedAttentionCache,
                      step: BatchStep, rope: RotaryEmbedding):
    return self.attention(hidden, cache, step, rope)

# The current forward context supplies runtime mode and backend layout.
result = layer.attention_forward(hidden, cache, step, rope)
```

No graph runner or capture-state argument is required at the numerical call.
The decorator lazily owns one common runner per region on the layer. FULL/eager
runtime executes the original body. Plain PIECEWISE stabilizes the output while
running the body eagerly. A backend layout enables live attention capture.

Runtimes may configure boundaries once, using the shared interface:

```python
configure_piecewise_core(layer, outputs=shared_output_buffers,
                         copy_per_step=("hidden",))
```

This is optional. It preserves an existing fixed-capacity output workspace or
requests staging for a standalone eager producer. Declared tensor inputs are
direct by default; `copy_per_step` on the decorator is the ordinary static policy.
`ATOM_ATTN_FFN_ZC` remains the diagnostic override.

## Backend contract

An opting-in builder implements `AttentionMetadataBuilder.cudagraph_key(metadata)`
and publishes its result as `metadata.cudagraph_key` after normal preparation.
The default `None` runs attention eagerly. The key describes launch layout and
buffer allocation generation, including any host choice that changes kernel
dispatch or grid. Request values such as PAGE IDs or positions are tensor data.

The backend must:

- Update request inputs at stable device addresses before replay.
- Bind cross-layer outputs before execution, with explicit producer/consumer
  order. Producers write those tensors in place, including on eager fallback.
- Compute batch-only metadata once during normal preparation. GPU metadata that
  depends on an earlier layer's output executes after that producer.
- Give concurrent views separate storage and retain the normal stream fences.
- Change the key when a recorded layout or buffer generation changes, or return
  `None` for unsupported layouts.

Python wrappers may be recreated. Recorded GPU addresses and their lifetimes
must stay valid. The common layer does not scan metadata objects, maintain field
schemas, intercept dictionaries, or replay Python side effects. A backend may
use ordinary dictionaries to name its stable tensor slots.

## Boundary buffers and capture lifetime

`StableOutputs.empty(..., capacity=elements)` binds a fixed-capacity result
before its producer runs. It returns a contiguous active prefix, reshaped to
the requested dimensions. Different active widths reuse the same allocation;
grouped planes remain contiguous. Choose capacity from the execution view
budget, not the current prompt length. Separate namespaces isolate concurrent
views. Fixed metadata inputs need no additional AF copy, and tensors local
to a region need no output slot.

`deliver(..., capacity=elements)` copies an eager result into the same kind of
contiguous prefix. Derived views such as a bounded decoder tail use the backend's
token and request capacities. Packed metadata reserves each field at a fixed
offset; active lengths select views without moving the following fields. The
backend layout key includes those allocation bounds.

Only AF regions need prebound cross-layer results; ordinary eager, FULL and
plain PIECEWISE keep lazy attention results. Existing
`slot`/`deliver` methods preserve a split operation's output address across eager
and graphed execution. Prefer an operator's `out=` interface; use an explicit
copy when its implementation cannot write the destination directly.

V4.1 reuses the builder's request buffers and scorer workspace directly. Its
topology determines which selections, candidates and grouped indices have
consumers in other regions; only those results get persistent slots. A candidate
translation used by one REINDEX region and single-layer indices stay local.
The scorer's `out=(selected, candidates)` accepts either fixed output or `None`
for a local result, and its top-k kernels write the supplied tensors directly.
Publishing a selection only exposes these tensors; it performs no second copy.

V4.1's normal `begin_step` / `update_step` lifecycle builds indptrs, refreshes
batch-only tables and exposes shared scorer destinations. It also handles
microbatch and bounded-tail views; there is no separate AF preparation call.
Candidate translations and grouped indices acquire stable slots in their normal
accessors. An eager consumer can recover the same slot after a producer replayed
without running Python. Their GPU producers still run in layer order.

On first use, live capture executes warmup once, records without executing GPU
nodes, and returns the warmup result without an immediate replay. Later calls
replay once. The region must not contain host mutations needed by later calls.
Startup capture, dummy execution and FULL runtime cannot nest live capture.

The common runner keys tensor argument layouts and direct input addresses. It
retains the captured callable and its backend resources until graph release.
Each runner records at most `max_entries` live graphs (32 by default), including
calls with an explicit `capture_key`. New layouts then fall back eagerly, while
existing layouts continue replaying. The limit does not restrict the legacy
startup capture plan. Compiled pieces keep independent pools per token bucket.

## Existing decode interface

V4 keeps `key=decode_bucket_key` and its explicit startup-capture interface.
Existing `piecewise`, `capture`, `runner`, `outputs`, `forward_context` and the
lower-level `capture_key`/`prepare` arguments remain supported. A backend does
not need to adopt live-prefill metadata to keep its existing decode behavior.

## Token-only hybrid capture

`supports_piecewise_prefill` opts into both padded live prefill and
`build_for_piecewise_capture(num_tokens)`. Unsupported backends run eager
prefill in FULL_AND_PIECEWISE while keeping the same token-derived FULL/draft
decode ladders. Their ordinary PIECEWISE and AF_PIECEWISE startup paths remain
unchanged, and hybrid memory estimation reserves no unused prefill graph pool.
A mixed prefill/decode DP group uses the union BS ladder on every
rank; the per-Q ladder applies only when all ranks are decoding.
