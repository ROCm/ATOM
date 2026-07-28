# Kimi-K3 Usage Guide (gfx950 / MI355)

Kimi-K3 is a **KimiLinear hybrid-attention MoE** model (`KimiLinearForCausalLM`). Each decoder layer is either a **KDA linear-attention** layer or an **MLA full-attention** layer, on top of a large MXFP4 latent MoE. ATOM serves the **text-only** backbone.

This guide targets **AMD MI355 (gfx950) only**, `-tp 8`.

| Variant | Quantization | Description |
|---------|-------------|-------------|
| **MXFP4** | MXFP4 (w4a4, e8m0 scales, group_size=32) | Routed MoE expert weights in microscale FP4. On gfx950 the SiTU experts run the FlyDSL **native SiTUv2** grouped-MoE path. Attention / shared experts / dense MLP stay BF16. |

**Validated (full 1319, GSM8K 5-shot, base completions, tp8):**

- **flexible-extract 0.9621 / strict-match 0.9621** — above Kimi-K2-Thinking (0.9363).
- Decode ~20 tok/s.

---

## Launching server

### MXFP4 on 8×MI355 GPUs (TP8)

```bash
#!/bin/bash
# ---- load (load-only) ----
export ATOM_LOADER_USE_THREADPOOL=1
export ATOM_LOADER_THREADPOOL_WORKERS=16
export ATOM_SYNC_AFTER_LOAD=1               # one-off TP barrier after load
export ATOM_DIST_TIMEOUT_SECONDS=3600

# ---- gfx950 MoE: FlyDSL native SiTUv2 ----
export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=0
export ATOM_USE_TRITON_MOE=0                 # use FlyDSL native SiTUv2 (in-kernel), not the torch-fp32 triton path
export AITER_FLYDSL_FORCE=1
export AITER_FORCE_GFX1250=0

# ---- attention: aiter Triton unified_attention (head_dim=192) ----
export ATOM_USE_UNIFIED_ATTN=1
export ATOM_FORCE_ATTN_TRITON=1

python -m atom.entrypoints.openai_server \
  --model Kimi-K3 \
  --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 10240 \
  --gpu-memory-utilization 0.93 \
  --block-size 128 \
  --no-enable_prefix_caching
```

MLA layers use the flash KV-cache layout to support `head_dim = 192` (dodges the aiter SHUFFLE-read bug at non-power-of-two head dim), and MLA prefill runs torch SDPA, so prefix caching and chunked prefill stay off — a cached or chunked prefix would be missed by the in-batch SDPA prefill. `-tp 8` is required (tp4 OOMs: MoE weights ~175 GB/GPU). `gpu-memory-utilization 0.93` (not 0.90) so the CUDA-graph pool fits alongside the KDA per-request state cache; at 0.90 startup fails with "Per-request cache tensor exceeds available KV budget".

---

## Accuracy test

Start the server as above, then run the full 1319-question GSM8K eval:

```bash
lm_eval \
  --model local-completions \
  --model_args "model=Kimi-K3,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot 5
```

Validated GSM8K result (gfx950 tp8):

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9621|±  |0.0067|
|     |       |strict-match    |     5|exact_match|↑  |0.9621|±  |0.0067|
```

Run on a **clean GPU set**. A competing job stealing VRAM mid-run corrupts the score (a one-off 0.53 was traced to exactly that). Verify no other container holds VRAM and the server stays up (grep the eval output for `ServerDisconnected`).
