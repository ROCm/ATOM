# Kimi-K3 Usage Guide (gfx950 / MI355)

Kimi-K3 is a **KimiLinear hybrid-attention MoE** model (`KimiLinearForCausalLM`). Each decoder layer is either a **KDA linear-attention** layer or an **MLA full-attention** layer, on top of a large MXFP4 latent MoE. ATOM serves the **text-only** backbone.

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
# ---- load ----
export ATOM_LOADER_USE_THREADPOOL=1
export ATOM_LOADER_THREADPOOL_WORKERS=16
export ATOM_DIST_TIMEOUT_SECONDS=3600

# ---- gfx950 MoE: FlyDSL native SiTUv2 ----
export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=0
export ATOM_USE_TRITON_MOE=0
export AITER_FLYDSL_FORCE=1
export AITER_FORCE_GFX1250=0

python -m atom.entrypoints.openai_server \
  --model Kimi-K3 \
  --kv_cache_dtype fp8 -tp 8 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.93 \
  --block-size 128 \
  --no-enable_prefix_caching
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
