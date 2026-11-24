# DeepSeek-V4.1-Flash with ATOM SGLang Plugin

This recipe serves DeepSeek-V4.1-Flash (FP4 PTPC checkpoint) through the ATOM SGLang plugin, without speculative
decoding and with SGLang's radix cache off, eagerly (no CUDA graphs yet). The configuration below is TP4 on MI308X
with SGLang 0.5.22.

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
    --disable-decode-cuda-graph --disable-prefill-cuda-graph \
    --context-length 65536 \
    --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --model-loader-extra-config '{"online_quant_config": {"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}}'
```

The plugin refuses, before loading weights:

- the FP4PTPC checkpoint without the `--model-loader-extra-config` above;
- `--chunked-prefill-size -1` (unchunked prefill): SGLang would admit a first prompt whole, past the step size the
  plugin's metadata buffers hold;
- CUDA graphs (launch with `--disable-decode-cuda-graph --disable-prefill-cuda-graph`, as above, or
  `--disable-cuda-graph`);
- `--kv-cache-dtype` other than `bfloat16`: the pool is BF16. That includes the default `auto`, which SGLang resolves
  to `fp8_e4m3` for DeepSeek-V4.1, so the launch passes `--kv-cache-dtype bfloat16` (above);
- `--enable-decoder-swa-bounded-replay`: every prefill row runs every layer (no decoder SWA bounded replay yet);
- `--enable-encoder-swa-bounded-replay` (SGLang itself refuses it on hardware other than CUDA and gfx950, MI308X
  included);
- `--disable-hybrid-swa-memory`: the pool is sized by SGLang's hybrid-SWA full token count;
- `--enable-memory-saver`: the ATOM pool is allocated outside SGLang's memory-saver region;
- `--enable-unified-memory`: not validated with the ATOM pool;
- radix cache (launch with `--disable-radix-cache`), hierarchical cache, speculative decoding (SGLang admits DSpark
  only for DeepSeek-V4.1), PD disaggregation, DP attention, decode context parallelism (`--dcp-size` above 1), and
  multimodal requests (launch with `--language-model-only`).

SGLang itself refuses, in its DeepSeek-V4.1 feature check (`validate_deepseek_v41_features`), before the plugin is
asked: two-batch overlap (`--enable-two-batch-overlap`), HiSparse, pipeline parallelism, and speculative decoding other
than DSpark. The plugin adds no environment knobs of its own.
