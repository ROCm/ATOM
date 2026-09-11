# DeepSeek-V4.1-Flash (MI355X)

Text-only support. The vision tower, DSpark drafting and prefix-cache reuse of
the decoder half are not implemented.

## Model

552B backbone parameters plus 196B Engram parameters, 40 layers, hidden 5120,
64 query heads over a single 512-wide KV latent, 384 routed experts plus one
shared, top-6. Dense weights are fp8 with [32,32] ue8m0 block scales; the
routed experts are MXFP4. The checkpoint is 476 GB.

Three structures differ from DeepSeek-V4:

- **Causal encoder-decoder.** Layers 0-1 are sliding-window only, layers 2-19
  compress KV 2:1, layers 20-39 compress 1:1. Layer 20 builds the whole
  decoder-half KV from the encoder output.
- **Cross-layer reuse.** Only layers 2, 8, 14 and 20 own a compressed KV cache
  and an indexer key cache. Only 2, 8, 14, 20, 24, 28, 32 and 36 compute a
  top-k. Every other layer reads its owner's. Every layer keeps its own
  128-entry sliding window.
- **Engram.** Layers 1 and 14 look up 24 n-gram hashes per token in a 384M-row
  fp8 table and gate the result into the residual stream. The tables are
  sharded by row across ranks.

## Run

    HIP_VISIBLE_DEVICES=0,1,2,3 AITER_LOG_LEVEL=WARNING \
    python -m atom.entrypoints.openai_server \
      --model /data/DeepSeek-V4.1-Flash \
      --server-port 8000 -tp 4 --enforce-eager \
      --max-model-len 4096 --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.90

`--enforce-eager` is required: the model refuses to build without it, because
the indexer reads per-sequence spans back off the device and a captured graph
would freeze one step's shapes.

Memory per rank: tp8 ~60 GB, tp4 ~119 GB, tp2 ~238 GB of weights. tp2 fits in
288 GB but leaves little for the KV cache. tp1 does not fit.

## Accuracy

    lm_eval --model local-completions \
      --model_args model=/data/DeepSeek-V4.1-Flash,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
      --tasks gsm8k --num_fewshot 3

## Validation

`tools/dsv41/` holds the parity harness. `torch_kernel.py` is a pure-torch
stand-in for the six kernels of the official implementation, so that
implementation runs as an oracle without tilelang. `dump_reference.py` writes a
fixture from a 6-layer model that has the same structure as the released one,
`make_mini_ckpt.py` turns it into an HF-layout checkpoint, and `run_parity.py`
and `run_paged_parity.py` compare ATOM against it. Every attention output
agrees to one bf16 ulp.

## Known limits

- Performance is bring-up grade. RoPE, the compressor pooling and the indexer
  scoring all run in torch rather than through their fused kernels, and there
  is one device-to-host sync per forward. The paged MQA-logits kernels need a
  64-row KV block, which neither compression class gives.
- The owner of a compressed cache writes its latent into every reader's
  envelope rows, which raises KV write traffic by about 70%. Permuting the
  class layer order so an owner sits above its readers would remove the copies
  and let `UnifiedPoolGeometry.compress_bias` carry the offset instead.
- fp8 KV cache, CUDA graph capture, MTP and prefix caching are untested.
