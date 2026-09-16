# Engram — host-side n-gram embedding lookup

Engram (DeepSeek's https://github.com/deepseek-ai/Engram) augments a few decoder
layers with a lookup into very large n-gram embedding tables. The tables cannot
live in HBM (DeepSeek-V4.1-Flash carries two, ~101.5 GB each), so the lookup runs
on the **host** and its result is staged to the device.

This recipe documents how a **model** wires Engram in. The engram code is a
self-contained library; no in-tree model uses it today. The DeepSeek-V4.1 stub we
used to bring it up and verify correctness has been removed — it lives in git
history (`git show 00afbe32b`). If a model's config declares engram layers but the
model does not expose `build_engram_host`, the runner **fails closed** (it raises
rather than silently serving the base model and ignoring the engram weights).

## The library

- `atom/model_ops/engram.py` — the host path:
  - `EngramConfig.from_hf(text_config)` — parses the `engram_*` config block; `None`
    when the model has no engram.
  - `CompressedTokenizer` — token id → normalized compressed vocab (cached `.npz`).
  - `NgramHashMapping` — compressed ids → per-layer n-gram hashes → table row indices.
  - `HostEmbeddingTable` — a memory-mapped table; `gather(rows)` pulls rows on the
    host (fp8 with E8M0 block scale, decoded natively or from raw uint8 codes).
  - `EngramPrefetcher` / `EngramPrefetchCache` — async hash+gather on a worker
    thread, plus a per-request rolling n-gram window for decode.
  - `EngramHost` — the per-step driver the runner talks to: `prefetch_next`,
    `stage_embeddings` (decode), `stage_prefill` (prefill), `stage_dummy`
    (warmup/capture), `wait_for_embeddings`, `embeddings(layer_id)`,
    `seed_context`, `drop_requests`, `shutdown`.
- `atom/model_ops/engram_layer.py` — the device side:
  - `EngramOp(nn.Module)` — `forward(hidden_states, embeddings)`: `wkv` projection +
    context-aware gate, `[N, hc_mult, H] + [N, engram_width] -> [N, hc_mult, H]`.
  - `EngramModules(nn.Module)` — one op per engram layer + the host tables; the
    model-facing factory (`from_checkpoint`, `build_engram_host`).

## What the n-gram lookup needs, and where the tokens come from

Position `t`'s embedding is the lookup for the causal n-gram ending at `t` (tokens
`[t-max_ngram_size+1 .. t]`, left-padded at the prompt start). The lookup keys on
raw token ids, so those ids must be available regardless of how the request is
batched:

- **Decode** — one new token per step. `EngramHost` keeps a per-request rolling
  window (`seed_context` from the prompt tail, `prefetch_next` appends each sampled
  token). The next step's embedding is hashed + gathered on the worker during the
  current step (overlap), cached, then staged.
- **Prefill (incl. chunked)** — every prompt position needs its embedding, because
  its hidden state feeds the KV cache. `stage_prefill` gathers all positions of a
  chunk in one pass. A chunk that does not start at position 0 needs the
  `max_ngram_size-1` tokens *before* it to hash its leading positions; those live
  in `seq.token_ids`, not in this chunk's `scheduled_tokens`, so the **scheduler**
  carries them on `ScheduledBatch.prefill_context` (engram models only). Prefill
  runs synchronously — there is no prior step to overlap a prefetch with.

## Timeline (what the runner drives)

```
decode step N   postprocess:  sample -> prefetch_next(final rows) -> worker hashes
                              + gathers next step's embedding into the cache
        step N+1 prepare:     _stage_engram -> stage_embeddings: take from cache
                              (recompute a miss from the window), async H2D; wait
        step N+1 forward:     engram layers read embeddings(layer_id), wkv + gate

prefill step    prepare:      _stage_engram -> stage_prefill: gather every chunk
                              position (context from prefill_context), stage rows;
                              final chunk seeds the decode window
```

The decode gather is prefetched a step early on a worker thread and its small
result is H2D'd on a side stream, so it overlaps GPU work **when it finishes within
the step**; `stage_embeddings` waits for it otherwise. The host staging buffers are
double-buffered (a per-layer pinned ring) so the next step's fill never races the
previous step's in-flight H2D, while the device buffer stays fixed for CUDAGraph.

## Model-side contract (3 hooks)

A model opts in by doing three things; ModelRunner does the rest, gated on
`getattr(model, "build_engram_host", None)`.

**1. Build the modules in `__init__`, and hold a slot for the host.**

```python
self.engram = EngramModules.from_checkpoint(
    config.model, hf_config=hf.to_dict(), dtype=dtype
)
# Set by build_engram_host below; ModelRunner stages into it before each forward.
# Not a module attribute, so it stays out of the state dict.
self.engram_host = None
```

**2. Expose `build_engram_host` on the `*ForCausalLM`.** ModelRunner probes for
this name, builds the `EngramHost`, and holds it; keep a reference on the model too
so the layers can reach it (forward has a fixed `(input_ids, positions)` signature).

```python
def build_engram_host(self, device, max_num_tokens: int, max_num_seqs: int):
    if self.model.engram is None:
        return None
    host = self.model.engram.build_engram_host(device, max_num_tokens, max_num_seqs)
    self.model.engram_host = host
    return host
```

`max_num_tokens` sizes the per-step staging buffer; `max_num_seqs` sizes the
per-request prefetch cache and rolling-window store (different budgets).

**3. Gate the embedding into the residual, BEFORE the engram layer.** DeepSeek's
Engram sits *between* decoder layers: the module at `layer_id` reads the residual
coming out of layer `layer_id - 1` and feeds it into layer `layer_id`. So inject
**before** `layer(...)`, not after it.

```python
for layer_id, layer in enumerate(self.layers):
    if self.engram is not None and layer_id in self.engram:
        if self.engram_host is None:
            raise RuntimeError(
                f"layer {layer_id} carries engram but no host was supplied; "
                f"ModelRunner stages the embeddings before calling forward"
            )
        hidden = hidden + self.engram[layer_id](
            hidden, self.engram_host.embeddings(layer_id)
        )
    hidden = layer(hidden, positions)
```

The residual `hidden` is `[num_tokens, hc_mult, hidden]` — the multi-branch shape
`EngramOp` gates against.

## What ModelRunner does for you

All gated on the model exposing `build_engram_host`; inert otherwise.

- `_init_engram_host()` — builds the `EngramHost` (held as `self.engram`). Raises if
  the config declares engram but the model has no `build_engram_host`; rejects
  engram + speculative decoding and engram + prefill/decode disaggregation.
- `_stage_engram(batch)` — before each forward: drops finished/preempted requests
  (`batch.engram_dropped`), then `stage_prefill` for a prefill batch,
  `stage_embeddings` for decode, `stage_dummy` for a dummy/capture pass; then
  `wait_for_embeddings`.
- postprocess — `prefetch_next` on the just-sampled token, for **final prefill
  chunks and decode rows only** (a middle chunk's next input is the next prompt
  token, not the sampled value).
- `exit()` — `EngramHost.shutdown()` joins the worker before the tables are dropped.
- CUDAGraph capture — `stage_dummy` before the capture forwards.

The **scheduler** carries the engram-only fields the runner consumes:
`prefill_context` (the pre-chunk context tokens) and `engram_dropped`
(finished/preempted request ids).

## Config and checkpoint requirements

The `text_config` must carry the `engram_*` block `EngramConfig.from_hf` reads:
`engram_layer_ids`, `engram_num_embeddings`, `engram_max_ngram_size`,
`engram_vocab_size`, `engram_n_heads`, `engram_head_dim`, `engram_pad_token_id`,
`engram_compressed_vocab_size` (plus optional `engram_seed`, `engram_kernel_size`).

The checkpoint carries six tensors per engram layer (see `EngramOp`):
`embed.weight`/`embed.scale` (the ~98 GB fp8 table, stays on host),
`wkv.weight`/`wkv.scale`, `k_weight`, `q_weight`.

`EngramModules.from_checkpoint` memory-maps these itself, so the model must add
their prefixes to its `skip_weight_prefixes` — otherwise the normal weight loader
marks them wanted and (with `ATOM_DISABLE_MMAP=true`) deserializes the shard
holding the ~98 GB table wholesale, which OOMs.
