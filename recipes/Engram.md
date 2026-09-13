# Engram — host-side n-gram embedding lookup

Engram (DeepSeek's https://github.com/deepseek-ai/Engram) augments a few decoder
layers with a lookup into very large n-gram embedding tables. The tables cannot
live in HBM (DeepSeek-V4.1-Flash carries two, ~101.5 GB each), so the lookup runs
on the **host** and its result is staged to the device, overlapped with GPU work.

This recipe documents how a **model** wires Engram in. The engram code itself is a
self-contained, verified library; there is no live model using it in-tree today.
The DeepSeek-V4.1 stub that we used to bring it up and verify correctness has been
removed — it lives in git history (`git show 00afbe32b`), and its wiring is
reproduced below as the worked example.

## The library

- `atom/model_ops/engram.py` — the host path:
  - `EngramConfig.from_hf(text_config)` — parses the `engram_*` config block; `None`
    when the model has no engram.
  - `CompressedTokenizer` — token id → normalized compressed vocab (cached `.npz`).
  - `NgramHashMapping` — compressed ids → per-layer n-gram hashes → table row indices.
  - `HostEmbeddingTable` — a memory-mapped table; `gather(rows)` pulls rows on the host.
  - `EngramPrefetcher` / `EngramPrefetchCache` — async hash+gather on a worker thread.
  - `EngramHost` — the per-step driver the runner talks to: `prefetch_next`,
    `stage_embeddings`, `stage_dummy`, `wait_for_embeddings`, `embeddings(layer_id)`.
- `atom/model_ops/engram_layer.py` — the device side:
  - `EngramOp(nn.Module)` — `forward(hidden_states, embeddings)`: `wkv` projection +
    context-aware gate, `[N, hc_mult, H] + [N, engram_width] -> [N, hc_mult, H]`.
  - `EngramModules(nn.Module)` — one op per engram layer + the host tables; the
    model-facing factory.

## Timeline (what the runner drives)

```
step N   postprocess:  sample -> prefetch_next(sampled_tokens) -> worker hashes +
                        gathers next step's embedding into the cache (overlapped)
step N+1 prepare:      _stage_engram -> stage_embeddings: wait worker, take from
                        cache, async H2D to GPU; wait_for_embeddings
step N+1 forward:      engram layers read embeddings(layer_id), do wkv + gate
```

The gather (the slow 101 GB host read) is done a step early on a worker thread and
the small result is H2D'd on a side stream, so the decode step never stalls on it.

## Model-side contract (3 hooks)

A model opts in by doing three things. ModelRunner does the rest automatically,
gated on `getattr(model, "build_engram_host", None)` — a model without it pays
nothing.

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
so the layers can reach it (forward has a fixed `(input_ids, positions)` signature
with no room to pass it through).

```python
def build_engram_host(self, device: torch.device, max_num_tokens: int):
    if self.model.engram is None:
        return None
    host = self.model.engram.build_engram_host(device, max_num_tokens)
    self.model.engram_host = host
    return host
```

**3. Gate the embedding into the residual at each engram layer, in `forward`.**

```python
for layer_id, layer in enumerate(self.layers):
    hidden = layer(hidden, positions)
    if self.engram is not None and layer_id in self.engram:
        if self.engram_host is None:
            raise RuntimeError(
                f"layer {layer_id} carries engram but no host was supplied; "
                f"ModelRunner stages the embeddings before calling forward"
            )
        hidden = hidden + self.engram[layer_id](
            hidden, self.engram_host.embeddings(layer_id)
        )
```

The residual `hidden` is `[num_tokens, hc_mult, hidden]` — the multi-branch shape
`EngramOp` gates against.

## What ModelRunner does for you

All gated on the model exposing `build_engram_host`; inert otherwise.

- `_init_engram_host()` — calls the factory, holds the `EngramHost` as `self.engram`.
- `_stage_engram(batch, input_ids)` — before each forward: `stage_dummy` for
  dummy/prefill passes (stages zeros so the engram layers still run), otherwise
  `stage_embeddings` for real decode, then `wait_for_embeddings`.
- postprocess — `self.engram.prefetch_next(list(batch.req_ids), sampled_tokens)`
  on the just-sampled token.
- CUDAGraph capture — `stage_dummy` before the capture forwards.

## Config and checkpoint requirements

The `text_config` must carry the `engram_*` block `EngramConfig.from_hf` reads:
`engram_layer_ids`, `engram_num_embeddings`, `engram_max_ngram_size`,
`engram_vocab_size`, `engram_n_heads`, `engram_head_dim`, `engram_pad_token_id`,
`engram_compressed_vocab_size` (plus optional `engram_seed`, `engram_kernel_size`).

The checkpoint carries six tensors per engram layer (see `EngramOp`):
`embed.weight`/`embed.scale` (the ~98 GB fp8 table, stays on host),
`wkv.weight`/`wkv.scale`, `k_weight`, `q_weight`.
