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
  --cudagraph-mode FULL \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'
```

Kimi full-attention layers use true MLA with a compressed latent KV cache. Aiter MLA is selected by default; `ATOM_USE_TRITON_MLA=1` selects the Triton MLA implementation when that configuration has been validated.

Prefix caching remains disabled because the KDA recurrent state is maintained per request and cannot be reconstructed from the paged MLA cache alone. `-tp 8` is required for the model to fit. Use `gpu-memory-utilization 0.93` so the CUDA-graph pool fits alongside the KDA per-request state cache.

### Mono decode

Two fused kernels replace the KDA layers of a speculative verify step: K1, the
attention seam through the KDA recurrence, and K2, `o_proj` plus the latent
MoE. The MLA layers keep the production paged path.

`ATOM_MONO_ENABLE` defaults to `1`, so a qualifying deployment gets mono
without being asked. It needs TP8, DSpark, FP8 KV, DCP1, a FULL cudagraph and
`ATOM_ENABLE_REPLAYSSM=0`. The launch above supplies TP8 and FP8 KV and takes
DCP1 and the FULL cudagraph by default; **DSpark and `ATOM_ENABLE_REPLAYSSM`
it does not**, and without both mono stays off:

- FULL cudagraph **is** the CLI default (`cudagraph_mode: str = "FULL"`,
  `atom/model_engine/arg_utils.py:145`, passed through at :768), so a launch
  from the CLI arms mono without passing `--cudagraph-mode` at all. The
  `None -> PIECEWISE` default at `atom/config.py:2393` applies only to a
  config that never went through `arg_utils` -- i.e. plugin mode -- and K3
  mono refuses PIECEWISE, logging `Kimi-K3 mono decode off: piecewise
  cudagraph`. The examples below still pass the flag explicitly, because it
  costs nothing and makes the requirement visible in the command.
- **DSpark is not in the launch above.** Mono replaces the KDA layers of a
  speculative *verify* step, so a server with no drafter has no such step to
  replace and says `Kimi-K3 mono decode off: no speculative decoding`. The
  draft model is a separate checkpoint and a local directory, not a repo id.
- `ATOM_ENABLE_REPLAYSSM` defaults ON whenever spec decode is used, and K1
  resumes a request from a per-token snapshot slot that ReplaySSM does not
  keep live.
- **Five projections must stay out of the online quant.** The kernels read
  them as plain BF16 (`kimi_k3/mono/weights.py`, five `_plain_bf16` call
  sites): `self_attn.in_proj`, `self_attn.f_b_proj`, `self_attn.o_proj`, and
  the shared experts' `gate_up_proj` and `down_proj`. The baseline
  `exclude_layer` list above covers none of them, and
  `*block_sparse_moe.experts*` does **not** match `shared_experts`. Miss any
  one and mono refuses the whole model -- the server answers correctly on the
  production path and says so once, e.g. `mono decode off: layer 1 shared
  gate_up_proj is torch.float8_e4m3fn`.

```bash
export ATOM_ENABLE_REPLAYSSM=0
export DRAFT_MODEL_PATH=/path/to/Kimi-K3-DSpark   # a directory, not a repo id
```

Add these to the launch command:

```
  --method dspark \
  --draft-model "${DRAFT_MODEL_PATH}" \
  --num-speculative-tokens 7 \
```

Seven is what the row cap wants: a step is served only at 8 rows or fewer, and
bs=1 with seven speculative tokens is exactly 8 (see below).

Add these to the launch's `exclude_layer` list:

```
"*self_attn.in_proj*", "*self_attn.f_b_proj*", "*self_attn.o_proj*", "*shared_experts*"
```

The server log says which way it went -- `Kimi-K3 mono decode armed (TP8)`, or
`Kimi-K3 mono decode off: <why>` naming the first condition that failed. Read
it before concluding anything about a run.

**A step is served only if it is at most 8 rows** (batch x query length).
bs=1 with seven speculative tokens is 8 rows and is served; bs=2 at the same
width is 16, which every step declines back to the production path -- the
server still answers, just without mono.

The cap is a constant (`atom/models/kimi_k3/mono/config.py:MAX_ROWS`), not a
switch. 16 rows is correct -- the kernels stride over their columns rather
than assuming one pass covers them -- and measurably faster (-13.4 % target
forward against stock). It stays at 8 because no task-level accuracy run
exists at any width and K2's latent MoE carries a real, unsettled precision
cost, so raising it is a decision to make on accuracy evidence rather than on
speed.

For a performance A/B, keep the launch command and the workload identical and
change only `ATOM_MONO_ENABLE` between `0` and `1`. Pin
`--spec-decode-acceptance-length` on both arms, or the two runs do different
amounts of work and the comparison means nothing.

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

- Kimi-K3's processor takes `messages` plus a separate `medias` list (chat rendering is Python, not Jinja) and returns `grid_thws` rather than `image_grid_thw`;
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
