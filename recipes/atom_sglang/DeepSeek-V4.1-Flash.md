# DeepSeek-V4.1-Flash with ATOM SGLang Plugin

This recipe serves DeepSeek-V4.1-Flash (FP4 PTPC checkpoint) through the ATOM SGLang plugin, without speculative
decoding and with SGLang's radix cache off, with decode CUDA graphs and the overlap scheduler. The configuration below
is TP4 on MI308X with SGLang 0.5.22.

SGLang keeps the server, scheduler, page allocation and sampling. ATOM owns the model, its weights, the CSA2 KV pool
(which replaces SGLang's DeepSeek-V4 pool) and the attention kernels; the plugin translates each SGLang batch into
ATOM's step metadata.

The FP4PTPC checkpoint runs only with its routed experts converted to per-channel FP8 at load, as in native ATOM: the
launch passes ATOM's online quantization through `--model-loader-extra-config` (below). Without it the plugin refuses
the checkpoint before loading weights, as native ATOM refuses it without `--online_quant_config`. Use aiter a53213dc4
or later, the commit native ATOM's DeepSeek-V4.1-Flash recipe pins.

## Launching Server

```bash
MODEL_PATH=${MODEL_PATH:-/data/models/DeepSeek-V4.1-Flash-FP4PTPC}
PORT=${PORT:-30000}

export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export AITER_LOG_LEVEL=WARNING
rm -rf /root/.cache/atom/*

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --trust-remote-code \
    --tp-size 4 --host 0.0.0.0 --port "${PORT}" \
    --language-model-only --kv-cache-dtype bfloat16 --page-size 256 \
    --disable-radix-cache \
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
    --cuda-graph-max-bs-decode 64 --cuda-graph-bs-decode 1 2 4 8 16 32 48 64 \
    --context-length 65536 \
    --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --model-loader-extra-config '{"online_quant_config": {"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}}'
```

The plugin refuses, before loading weights:

- the FP4PTPC checkpoint without the `--model-loader-extra-config` above;
- `--chunked-prefill-size -1` (unchunked prefill): SGLang would admit a first prompt whole, past the step size the
  plugin's metadata buffers hold;
- prefill CUDA graphs (launch with `--cuda-graph-backend-prefill disabled`);
- `--enable-decoder-swa-bounded-replay` with `ATOM_SGLANG_V41_DECODER_REPLAY` off (below): decoder SWA bounded replay
  is on by default, so the flag restates the default and contradicts the variable;
- `--enable-encoder-swa-bounded-replay`;
- `--disable-hybrid-swa-memory`: the pool is sized by SGLang's hybrid-SWA full token count;
- `--enable-memory-saver`: the ATOM pool is allocated outside SGLang's memory-saver region;
- radix cache (launch with `--disable-radix-cache`), hierarchical cache, HiSparse, speculative decoding, PD
  disaggregation, DP attention, pipeline parallelism, decode context parallelism (`--dcp-size` above 1), and
  multimodal requests (launch with `--language-model-only`).

Two-batch overlap (`--enable-two-batch-overlap`) is refused by SGLang itself, in its DeepSeek-V4.1 feature check
(`validate_deepseek_v41_features`).

`--kv-cache-dtype` takes `auto` / `bfloat16` (BF16 pool) or `fp8_e4m3` (FP8 pool).

## Environment variables

Read by the plugin (`atom/plugin/sglang/deepseek_v41_envs.py`):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_SGLANG_V41_DECODER_REPLAY** | bool | 1 (true) | Decoder SWA bounded replay: a prefill runs V4.1's decoder layers on each request's last 128 rows only, as native ATOM does. `0` / `false` / `no` / `off` turn it off (every prefill row runs every layer), as native's `--no-decoder-swa-bounded-replay` does. Unlike native, which backs a short resumed chunk off to a full window, a chunk SGLang resumes with fewer rows than the window computes every row, and so does a step in which SGLang reads more than each request's last row (input logprobs, a full hidden-state capture, a prefill-only batch); the plugin's prefill output therefore depends on where SGLang cuts a prompt. |
