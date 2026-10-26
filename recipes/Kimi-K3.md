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

- FULL cudagraph **is** the CLI default (`EngineArgs.cudagraph_mode: str =
  "FULL"` in `atom/model_engine/arg_utils.py`), so a launch from the CLI arms
  mono without passing `--cudagraph-mode` at all. The `None -> TR_HYBRID`
  default in `Config.__post_init__` (`atom/config.py`) applies only to a
  config that never went through `arg_utils` -- i.e. plugin mode -- and K3
  mono refuses TR_HYBRID, logging `Kimi-K3 mono decode off: TR_HYBRID
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

Seven draft tokens make one request exactly 8 rows. The buffers hold 16 rows
(`atom/models/kimi_k3/mono/config.py:MAX_ROWS`), so up to 15 draft tokens are
accepted for a single request; every number below was measured at 7.

Add these to the launch's `exclude_layer` list:

```
"*self_attn.in_proj*", "*self_attn.f_b_proj*", "*self_attn.o_proj*", "*shared_experts*"
```

The server log says which way it went -- `Kimi-K3 mono decode armed (TP8)`, or
`Kimi-K3 mono decode off: <why>` naming the first condition that failed. Read
it before concluding anything about a run.

**A step is served only if it is at most 16 rows** (batch x query length). bs=1
with seven speculative tokens is 8 rows and bs=2 at the same width is 16; both
are served, including the bs=2 CUDA graph. A step past 16 rows declines back to
the production path -- the server still answers, just without mono. The log
names a declined row count once per distinct count (`Kimi-K3 mono decode
declined a step: 24 rows > the 16-row buffers`), so the number of such lines is
the number of distinct widths turned away, not the number of steps.

The 16-row cap and K2 on the 24 full-attention (MLA) layers are fixed; there is
no switch for either. Only `ATOM_MONO_ENABLE=0` turns them off, and it turns
off mono on the KDA layers with them. What this means for a deployment that
used to run the 8-row, KDA-only mono:

- bs=2 steps (16 rows) and every CUDA-graph capture size up to 16 rows now run
  on the kernels instead of the production path. The log line above no longer
  appears for them.
- The peer buffer doubles, 4.76 MB to 9.52 MB a rank, and is zeroed at every
  step.
- K2 on the MLA layers dequantizes those layers' latent-MoE weights: about
  1.4 GB a rank (24 layers x about 58 MB, computed from the per-layer size in
  `mono/weights.py`; not measured). The runner is built on the first eligible
  decode step, which in the code is after the KV-cache block count is fixed:
  `warmup_model` runs a prefill-only dummy batch (mono declines prefill),
  `get_num_blocks` runs no forward, and the first decode forward that is not a
  graph capture is the per-size warm-up forward in `capture_cudagraph` (the
  first real decode step under `--enforce-eager`). So these weights
  are not part of the KV-cache budget and come out of the memory left over
  (`1 - gpu_memory_utilization` and the 2 % safety margin). That is read from
  the code, not observed; whether the margin holds on a tight deployment is not
  tested.
- K2's latent MoE has a documented MXFP4 precision cost (rel_L2 vs stock median
  0.24) and now covers 24 more layers. The task-level checks are paired GSM8K
  runs (3-shot, 1319 questions, two concurrent requests, same configuration, no
  synthetic acceptance length): 16 rows scored 0.9591 against 0.9439 at the
  8-row cap (paired z = +2.4); MLA K2 against the 8-row control was +0.0038
  (z = +0.6). Neither run showed a regression; the second is not distinguishable
  from no change.
- An MLA layer whose K2 cannot bind (for example a quantized `o_proj`; the
  `exclude_layer` list above keeps `o_proj` bf16) stays on the production K2
  alone and logs `Kimi-K3 mono: layer N keeps its production K2`; the other
  layers keep mono. A KDA layer that cannot bind still turns the whole runner
  off, as before.
- At bs=1 K2 on the MLA layers cut the step ITL from 16.5 to 14.8 ms. The
  benchmark below has no C1 number for the old default (8 rows, no K2).

`ATOM_PREFILL_DECODE_INTERVAL=1000000` delays admitting a new prefill while any
request is decoding, using the existing `PrefillDelayer`
(`atom/model_engine/prefill_delayer.py`) in its local mode (DP=1, PP=1). After a
prefill runs, the interval holds further prefill until the running-decode count
reaches zero, so for the agentic trace requests mostly decode one at a time. It
is not a hard bs=1 guarantee: while another request is in chunked prefill the
running-decode count is zero, so a second request can be admitted then and two
can decode together afterward. Launch with
`--max-num-seqs 2 --cudagraph-capture-sizes "[1,2]"`; `--max-num-seqs 1` fails
at startup on a DSpark deployment (the block drafter's warmup needs two
sequences). Queued requests wait, so TTFT rises (mean 0.9 s to about 1.8 s on
the agentic trace).

Agentic trace, TP8, DSpark k=7, 16-row cap and MLA K2 on, official client 3600 s
each, p90 interactivity (tok/s/user). This is a synthetic benchmark: the
acceptance length is forced to 3.84, so throughput is meaningful but the
generated text is not, and it says nothing about accuracy. Native vLLM (one
request at a time) 177.4; ATOM without the delayer or the scheduler cap 177.1;
with a scheduler admission cap (since removed) 216.6 and 217.5 on two machines;
with `ATOM_PREFILL_DECODE_INTERVAL=1000000` instead 215.6 (0 errors; delayer held
about 8 % of scheduler ticks). The 215 to 217 results need the delayer or the
cap; without one ATOM is level with vLLM. The delayer run is 0.5 to 0.9 % below
the cap runs; this is one run per arm, not a statistical equivalence test, and
these runs were measured on the code before the cap was removed.
At two-request concurrency (C2) ATOM gives 137.4 against vLLM's 130.1; that run
also predates the removal of the switches, and this recipe does not record
whether MLA K2 was on in it.

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
