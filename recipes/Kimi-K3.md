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
export ATOM_USE_CUSTOM_ALL_GATHER=0

# ---- gfx1250 correctness ----
export ATOM_K3_MOE_CHUNK=128                # grouped MoE/MXFP4 only correct at small M; sub-batch to <=128
export ATOM_KDA_FORCE_RECURRENT=1           # chunk_kda NaNs here; run KDA prefill recurrently
export ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1   # avoid aiter's Gluon MXFP4 preshuffle GEMM

python -m atom.entrypoints.openai_server \
  --model Kimi-K3 \
  --kv_cache_dtype bf16 -tp 4 \
  --block-size 64 \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.93 \
  --level 0 \
  --no-enable_prefix_caching
```

Chunked prefill is ON (bounds the MLA prefill activation to the chunk); prefix caching stays off for the KDA hybrid. `-tp 4` because that box has 4 GPUs, not a model requirement. `--block-size 64` is not a hard requirement with shuffled KV off (see below), but it is the value the gsm8k number below was measured with.

`ATOM_WARMUP_MAX_TOKENS` from the original bring-up scripts no longer exists and is not needed: the uncapped 2048-token warmup prefill passes on this stack. `ATOM_K3_ATTN_RES_NS` already defaults to `1`; `2` restores the pipelined attn-residual H loop, which faults on gfx1250.

Three of the settings above are gfx1250 workarounds for kernels reached through aiter, and each one is load-bearing for either startup or accuracy:

- **`ATOM_KDA_FORCE_RECURRENT=1`** — `chunk_kda` NaNs on gfx1250 for prompts shorter than its chunk size, and its `transpose_state_layout` output can mismatch what the decode-time `fused_recurrent_kda` reader expects, so the first decode step goes NaN too. NaN logits make argmax pick token id 0, and the server emits `"!"` forever at 0.0 gsm8k. Running KDA prefill on `fused_recurrent_kda` keeps the state layout consistent across prefill and decode.
- **`ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1`** — on gfx1250 `gemm_afp4wfp4_preshuffle` dispatches to a Gluon kernel with no usable M: it asserts `M >= 32`, and its only tuned config for `M >= 32` is `BLOCK_SIZE_M=256`, which memory-faults. K3 reaches it because the checkpoint's quant config marks the routed latent projections `per_1x32`/`fp4x2`, and `ATOM_K3_MOE_CHUNK=128` puts them at `M=128`. This flag keeps the same MXFP4 weights and scales but takes the plain `gemm_afp4wfp4` kernel on an unshuffled layout. aiter disabled the Gluon dispatch deliberately in `0730b33fc` ("disable 1250 gluon path", TODO: revert after upstream triton is fixed); `9312ef7c0` re-enabled it.
- **`ATOM_USE_TRITON_MLA_SHUFFLE_KV` left at its `0` default** — do not set it to `1` on gfx1250. The shuffled-KV path calls aiter's `fused_qk_rope_cat_and_cache_mla`, which prepends an extra positional argument for the Gluon kernel. That kernel's signature has no such slot, so every later positional shifts by one and the last one lands on the `k_scale_ptr` keyword: `TypeError: dynamic_func() got multiple values for argument 'k_scale_ptr'`. Setting it also makes `aiter_mla.py` require `--block-size 64`.

Validated GSM8K result (gfx1250 tp4, `num_concurrent=8`), measured on this tree with the command above against aiter `main` (`56f56db7e`, unmodified) and triton `3.8.0`:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9575|±  |0.0056|
|     |       |strict-match    |     5|exact_match|↑  |0.9575|±  |0.0056|
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
