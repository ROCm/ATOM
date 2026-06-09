# GPT-OSS gfx1250 PA ASM Shuffle KV Repro

This branch pins the ATOM side needed to run GPT-OSS 120B on gfx1250 with:

- Triton MHA backend selected by `ATOM_USE_UNIFIED_ATTN=1`.
- FP8 KV cache with block size 256.
- Shuffle KV cache layout for Triton unified attention.
- `aiter.pa_decode_bf16_asm` for eligible non-SWA decode layers.
- `ATOM_FORCE_ATTN_TRITON=1` to force all attention back to Triton.
- Triton unified attention for prefill and SWA layers.

Use this branch together with the matching AITER branch:

```bash
cd /app/ATOM
git checkout yhl/gptoss-pa-asm-shuf-repro-20260611

cd /app/aiter
git checkout yhl/gptoss-pa-asm-shuf-repro-20260611
```

## Launch

Use a healthy gfx1250 device. Device 1 was used for the smoke tests below.

```bash
cd /app/ATOM

HIP_VISIBLE_DEVICES=1 \
ATOM_USE_UNIFIED_ATTN=1 \
HSA_ENABLE_SDMA=1 \
HSA_USE_SVM=1 \
HSA_XNACK=1 \
ATOM_LOADER_USE_THREADPOOL=0 \
ATOM_USE_TRITON_MLA=1 \
AITER_ROPE_TRITON_BACKEND=1 \
ATOM_ENABLE_DS_QKNORM_FUSION=0 \
ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0 \
ATOM_USE_TRITON_GEMM=1 \
ENABLE_DS_QKNORM_FUSION=0 \
ENABLE_CK=0 \
AITER_USE_OPUS_MOE_SORTING=1 \
PYTHONPATH=/app/aiter:/app/aiter/aiter/jit/utils:/app/ATOM:/app/triton/python \
python -m atom.entrypoints.openai_server \
  --model /data/models/gpt-oss-120b \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.75 \
  --block-size 256 \
  --server-port 8013
```

## Expected Logs

During cudagraph capture, the full-attention decode path should load the
gfx1250 PA ASM kernel:

```text
ATOM_USE_UNIFIED_ATTN on gfx1250: routing matching decode layers through aiter.pa_decode_bf16_asm
LoadKernel: _ZN5aiter31pa_decode_bf16_d64_page256_gqa8E hsaco: /app/aiter/hsa//gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8.co
```

SWA and prefill are expected not to use PA ASM:

```text
PA decode BF16 ASM fallback: sliding-window layer
PA decode BF16 ASM fallback: prefill
```

The server should then report:

```text
Engine Core: EngineCore fully initialized and ready
Server started successfully and ready to accept requests
```

## Smoke Test

```bash
curl -sS http://127.0.0.1:8013/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Only output the final answer: 12*13=?"}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

In the validated container, 10 short factual prompts returned HTTP 200 and the
final answers after the model's `assistantfinal` marker were correct. The raw
response may still include the model template text (`analysis...assistantfinal`);
that is a serving/template issue, not evidence of PA ASM numerical corruption.

## Notes

- `/data/models/openai/gpt-oss-120b` was not present in the validation
  container; `/data/models/gpt-oss-120b` was used.
- Device 2 previously hit a GPU hang during model loading in this container.
  Device 1 was used for the successful validation run.
- The model loader may warn that `model.embedding.weight` was not loaded. That
  warning was present in both baseline and ASM validation runs.
