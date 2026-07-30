# Kimi-K3 Usage Guide

Kimi-K3 is a **KimiLinear hybrid-attention MoE** model (`KimiLinearForCausalLM`). Each decoder layer is either a **KDA linear-attention** layer or an **MLA full-attention** layer, on top of a large MXFP4 latent MoE. ATOM serves the **text-only** backbone.

Covers **MI355 (gfx950), `-tp 8`** and **MI450 (gfx1250), `-tp 4`**. Same checkpoint and model math; only the enablement differs.

| Variant | Quantization | Description |
|---------|-------------|-------------|
| **MXFP4** | MXFP4 (w4a4, e8m0 scales, group_size=32) | Routed MoE expert weights in microscale FP4. On gfx950 the SiTU experts run the FlyDSL **native SiTUv2** grouped-MoE path. Attention, shared experts, and dense MLP remain BF16. |

**Validated on gfx950 (full 1319, GSM8K 5-shot, base completions, tp8, seed 42):**

- **flexible-extract 0.9538–0.9591 / strict-match 0.9538–0.9591** across three clean-start runs.

---

## Launching server — gfx950 / MI355

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
  --no-enable_prefix_caching
```

Kimi full-attention layers use true MLA with a compressed latent KV cache. Aiter MLA is selected by default; `ATOM_USE_TRITON_MLA=1` selects the Triton MLA implementation when that configuration has been validated.

Prefix caching remains disabled because the KDA recurrent state is maintained per request and cannot be reconstructed from the paged MLA cache alone. `-tp 8` is required for the model to fit. Use `gpu-memory-utilization 0.93` so the CUDA-graph pool fits alongside the KDA per-request state cache.

---

## Launching server — gfx1250 / MI450

### MXFP4 on 4×MI450 GPUs (TP4)

```bash
#!/bin/bash
# ---- gfx1250 backends (no CK) ----
export ENABLE_CK=0                          # CK never registered gfx1250 -> flydsl/triton/hip
export ATOM_USE_TRITON_GEMM=1
export AITER_USE_GROUPED_GEMM=1             # flydsl grouped MoE
export ATOM_USE_TRITON_MLA=1                # K3 MLA-latent Triton path
export ATOM_USE_TRITON_MLA_SHUFFLE_KV=1     # shuffled KV gather
export ATOM_USE_CUSTOM_ALL_GATHER=0

# ---- gfx1250 correctness ----
export ATOM_K3_MOE_CHUNK=128                # grouped MoE/MXFP4 only correct at small M; sub-batch to <=128

python -m atom.entrypoints.openai_server \
  --model Kimi-K3 \
  --kv_cache_dtype bf16 -tp 4 \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.93 \
  --level 0 \
  --no-enable_prefix_caching
```

Chunked prefill is ON (bounds the MLA prefill activation to the chunk); prefix caching stays off for the KDA hybrid. `-tp 4` because that box has 4 GPUs, not a model requirement.

`ATOM_KDA_FORCE_RECURRENT` and `ATOM_WARMUP_MAX_TOKENS` from the original bring-up scripts no longer exist and must not be set. `ATOM_K3_ATTN_RES_NS` already defaults to `1`; `2` restores the pipelined attn-residual H loop, which faults on gfx1250.

Validated GSM8K result (gfx1250 tp4, `num_concurrent=8`), measured on the pre-upstream bring-up branch and not re-run on this tree:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9613|±  |0.0053|
|     |       |strict-match    |     5|exact_match|↑  |0.9598|±  |0.0054|
```

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

Keep `num_concurrent` at or below the server's `--max-num-seqs`: `64` for the gfx950 command above, `8` for the gfx1250 one.

Validated true-MLA result range on gfx950 TP8 (for the gfx1250 reference numbers see that section above):

```text
| Filter           | Minimum | Maximum |
|------------------|--------:|--------:|
| flexible-extract |  0.9538 |  0.9591 |
| strict-match     |  0.9538 |  0.9591 |
```

Run on an uncontended GPU set and verify the evaluation completes without server disconnects or worker failures.
