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

## Accuracy test

Start the server as above, then run the full 1319-question GSM8K evaluation:

```bash
lm_eval \
  --model local-completions \
  --model_args "model=Kimi-K3,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --seed 42
```

Validated true-MLA result range on gfx950 TP8:

```text
| Filter           | Minimum | Maximum |
|------------------|--------:|--------:|
| flexible-extract |  0.9538 |  0.9591 |
| strict-match     |  0.9538 |  0.9591 |
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
