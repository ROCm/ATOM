# Kimi-K3 Usage Guide (gfx950 / MI355)

Kimi-K3 is a **KimiLinear hybrid-attention MoE** model (`KimiK3ForConditionalGeneration`). Each decoder layer is either a **KDA linear-attention** layer or an **MLA full-attention** layer, on top of a large MXFP4 latent MoE. ATOM serves both the text backbone and the **MoonViT3d vision tower** — see [Multimodal](#multimodal) below.

This guide targets **AMD MI355 (gfx950) only**, `-tp 8`.

| Variant | Quantization | Description |
|---------|-------------|-------------|
| **MXFP4** | MXFP4 (w4a4, e8m0 scales, group_size=32) | Routed MoE expert weights in microscale FP4. On gfx950 the SiTU experts run the FlyDSL **native SiTUv2** grouped-MoE path. Attention, shared experts, and dense MLP remain BF16. |

**Validated (full 1319, GSM8K 5-shot, base completions, tp8, seed 42):**

- **flexible-extract 0.9538–0.9591 / strict-match 0.9538–0.9591** across three clean-start runs.

---

## Launching server

### MXFP4 on 8×MI355 GPUs (TP8)

```bash
#!/bin/bash

python -m atom.entrypoints.openai_server \
  --model Kimi-K3 \
  --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.93 \
  --block-size 128 \
  --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'
```

Kimi full-attention layers use true MLA with a compressed latent KV cache. Aiter MLA is selected by default; `ATOM_USE_TRITON_MLA=1` selects the Triton MLA implementation when that configuration has been validated.

Prefix caching remains disabled because the KDA recurrent state is maintained per request and cannot be reconstructed from the paged MLA cache alone. `-tp 8` is required for the model to fit. Use `gpu-memory-utilization 0.93` so the CUDA-graph pool fits alongside the KDA per-request state cache.

---

## CPU offload for KV and KDA state (opt-in)

The launch lines above pass `--no-enable_prefix_caching`, so they run none of the
offload path. To serve K3 with the `lmcache_offload` connector — dense MLA KV
chunks *plus* a CPU tier for the KDA recurrent state — use this shape instead:

```bash
export AITER_LOG_LEVEL=WARNING
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=96      # GiB, KV *and* state share this pool
export LMCACHE_CHUNK_SIZE=1024            # = block_size × decode_context_parallel_size

python -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 --trust-remote-code \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --kv_cache_dtype fp8 --block-size 128 \
  --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.86 \
  --enable_prefix_caching --state-checkpoint-interval-tokens -1 \
  --kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'
```

`--enable_prefix_caching` is **required**. The whole tier hangs off the
prefix-cache admission path: the joint boundary is chosen while the block manager
walks the HBM prefix cache, and with prefix caching off there is no walk, no
boundary and no load.

`--state-checkpoint-interval-tokens -1` keeps state checkpointing on but places
no fixed interval rungs — checkpoints are taken at the demand rung and the
prompt-end anchor only. `0` would disable state checkpointing outright and with
it the whole tier; a positive value is a rung every N tokens and is also valid,
at the cost of more checkpoints kept.

### `LMCACHE_CHUNK_SIZE` must be a multiple of the hash block size

The prefix-cache hash block size is `block_size × decode_context_parallel_size`
(`BlockManager.hash_block_size`) — 1024 in the configuration above. The KV leg of
a joint load moves whole LMCache chunks, so `BlockManager._joint_kv_boundary`
floors the claim to the chunk grid; if that floored claim is not also a multiple
of the hash block size the boundary is **refused** and counted as
`joint_skip_claim_off_hash_grid`. It refuses rather than re-flooring because the
tail between the two grids would land in a fresh, unfilled block that
`num_cached_tokens` then counts as computed — silent wrong output.

If your chunk size and hash block size do not divide, **raise** the chunk size to
a multiple of the hash block size. Do not lower it: a smaller chunk does not make
the grids divide, and every boundary is lost the same way.

### Sizing

`LMCACHE_MAX_LOCAL_CPU_SIZE` is the one size to tune. KV chunks and state objects
go into a single LMCache pool under a single LRU, deliberately: a state boundary
whose KV has been evicted is worthless, so the two should cool at the same rate.
There is no separate state-size knob.

### Pipeline parallelism is refused

The LMCache key carries no PP component, so two stages at the same TP rank would
overwrite each other's state images. `pipeline_parallel_size > 1` therefore
raises at startup with a message naming the value. Run with
`pipeline_parallel_size=1`, or drop `--kv-transfer-config`.

### Confirming it is actually running

At startup each rank logs

```text
kimi_k3 offload: state tier up, entry=... MiB rank=..., sharing the paged-KV CPU pool, layout=...
state offload: engine index attached (store=True load=True chunk=1024)
```

If the first line is missing, the worker built no tier — the same code path logs
a `kimi_k3 offload: ...` warning naming which probe failed (no attention backend,
no checkpoint layout id, no per-request state views, no `page_unit_views`, or an
image/slot byte mismatch).

Then read the funnel, which `BlockManager.checkpoint_funnel()` assembles and the
server exposes at `GET /debug/cache_stats`:

```bash
curl -s localhost:8000/debug/cache_stats | python3 -m json.tool
```

| Key | Reading |
|-----|---------|
| `joint_boundaries` | Joint KV+state boundaries committed. Zero means nothing was ever offered to the tier. |
| `state_hbm_boundaries` / `state_tier_boundaries` | Of those, how many were served from the HBM checkpoint pool versus the CPU tier. Only the second exercises this code. |
| `joint_skip_<reason>` | One bucket per refusal reason, e.g. `joint_skip_off`, `joint_skip_no_chunk_size`, `joint_skip_lmcache_within_hbm`, `joint_skip_hbm_off_chunk_grid`, `joint_skip_no_room_above_hbm`, `joint_skip_no_rung_above_hbm`, `joint_skip_covering_chunk_beyond_lookup`, `joint_skip_claim_off_hash_grid`. |
| `stores_attempted` / `stores_completed` / `stores_failed` / `stores_refused` | The store leg. `stores_completed` against `checkpoints_kept` is how much of what HBM keeps the CPU tier never received. |
| `loads_dispatched` / `loads_settled` / `loads_outstanding` | The load leg's lifecycle; `dispatched == settled + outstanding` always holds. |
| `loads_completed` / `loads_failed` | `loads_failed / loads_dispatched` is the index's false-positive rate — hashes it still advertised after LMCache's LRU dropped the bytes. |
| `indexed` / `hashes_evicted` | Hashes the engine believes are in LMCache, and how many the index itself dropped. |

The diagnostic reading is the pair: `joint_boundaries == 0` with one
`joint_skip_*` bucket carrying all the counts tells you exactly which gate
refused. `joint_skip_off` means no tier is attached at all;
`joint_skip_no_paged_checkpoints` means the sequence carries no per-request state
or the runtime published no paged checkpoint coordinator;
`joint_skip_claim_off_hash_grid` is the chunk size above; and
`joint_skip_lmcache_within_hbm` or `joint_skip_no_rung_above_hbm` is the benign
case where LMCache held nothing usable beyond what HBM already had.

---

## Accuracy test

Start the server as above, then run the full 1319-question GSM8K evaluation:


```bash
# download inferenceX gsm8k yaml from https://github.com/SemiAnalysisAI/InferenceX/blob/main/utils/evals/gsm8k.yaml

lm_eval --model local-chat-completions \
  --apply_chat_template \
  --tasks /path-to-gsm8k-yaml \
  --model_args "model=${MODEL},base_url=http://localhost:8000/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=10,timeout=1800,tokenized_requests=False,max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1
```

Validated true-MLA result range on gfx950 TP8:

```text
| Filter           | Minimum | Maximum |
|------------------|--------:|--------:|
| flexible-extract |  0.9659 |  0.9591 |
| strict-match     |  0.9666 |  0.9591 |
```

Run on an uncontended GPU set and verify the evaluation completes without server disconnects or worker failures.

---

## Multimodal

The checkpoint ships a **MoonViT3d** vision tower (27 layers, `vt_hidden_size=1024`, `qkv_hidden_size=1536`) plus a `patchmergerv2` projector into the 7168-wide text space. ATOM implements both natively in `atom/models/kimi_k3_vl.py`; they are built and loaded by the same server command as above — no extra flag.

The tower is replicated on every TP rank rather than sharded, costing ~0.9 GB bf16 per GPU (measured `peak_torch` 191.81 GB with it against 190.98 GB without) — a KV-cache budget the text-only GSM8K run above also pays.

### How images reach the model

Kimi-K3's processor differs from the Qwen convention in two ways that ATOM handles in `atom/model_engine/multimodal.py`:

- it takes `messages` plus a separate `medias` list (chat rendering is Python, not Jinja) and returns `grid_thws` rather than `image_grid_thw`;
- it emits **one** `<|media_pad|>` token per image, leaving the expansion to the model. ATOM expands it to `(h // 2) * (w // 2)` tokens up front so the scheduler, KV blocks and positions all see the real prompt length.

Multimodal prefills are never chunked — the vision embeddings cover the whole prompt — so `--max-num-batched-tokens` is a hard cap on an image prompt's length even with chunked prefill enabled. A 512x512 image is 256 tokens; the default 10240 leaves ample room. Lifting this is a known TODO (see the merge site in `ModelRunner.run_model` and the admission check in `Scheduler.schedule`); it needs the encoder output cached per request and sliced per chunk.

On a **cold AITER JIT cache**, the first image request builds the varlen attention kernel and pays ~35 s of TTFT; once that kernel is cached it drops to ~2 s for the first request of a fresh server and ~0.3 s steady-state. Send one throwaway image before timing anything or starting an eval.

### Image request

```bash
IMAGE_BASE64=$(base64 -w 0 /app/image.png)

curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kimi-K3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url",
           "image_url": {"url": "data:image/png;base64,'"$IMAGE_BASE64"'"}},
          {"type": "text", "text": "Describe this image in detail."}
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0,
    "stream": false
  }' | python3 -m json.tool
```

Content parts are rendered in the order the client sent them, so text before/after an image lands where you put it.

### Offline inference

Export the same environment as the server launch script above, then:

```bash
python -m atom.examples.multimodal_inference \
  --model /data/Kimi-K3 \
  --image /app/image.png \
  --prompt "Describe this image in detail." \
  --trust-remote-code \
  --kv_cache_dtype fp8 -tp 8 \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 10240 \
  --gpu-memory-utilization 0.93 \
  --block-size 128 \
  --no-enable_prefix_caching \
  --temperature 0 --max-tokens 512
```

Repeat `--image` for a multi-image prompt. Keep `--block-size 128` and `--gpu-memory-utilization 0.93`: the same MLA-layout and per-request-cache constraints as the server apply.

### Multimodal accuracy test

Driving the server over HTTP needs far less than the full `lmms-eval` dependency set in [Qwen3.5_multimodel.md](Qwen3.5_multimodel.md) (that list also covers video tasks, local model backends and caption/math metrics). Install with `--no-deps` so the ROCm torch stack is untouched:

```bash
python3 -m pip install --no-deps --force-reinstall \
  "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git"
python3 -m pip install --no-deps \
  accelerate datasets loguru pytablewriter openai sniffio distro jiter
```

Then:

```bash
OPENAI_API_KEY=EMPTY \
PYTHONPATH="${LMMS_EVAL_PATH:-/app/lmms-eval}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m lmms_eval \
  --model openai \
  --model_args "model=Kimi-K3,base_url=http://127.0.0.1:8000/v1,api_key=EMPTY,timeout=900,max_retries=3,num_concurrent=16,max_size_in_mb=50" \
  --tasks mmstar \
  --batch_size 1 \
  --process_with_media \
  --gen_kwargs "temperature=0,max_new_tokens=8192" \
  --log_samples \
  --output_path /tmp/atom_k3_mmstar
```
