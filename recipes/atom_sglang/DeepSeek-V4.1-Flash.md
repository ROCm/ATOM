# DeepSeek-V4.1-Flash with ATOM SGLang Plugin

This recipe serves DeepSeek-V4.1-Flash (FP4 PTPC checkpoint) through the ATOM SGLang plugin, without speculative
decoding or with DSpark (below), with SGLang's radix cache and decode CUDA graphs (SGLang's overlap scheduler stays
on, its default). The configuration below is TP4 on MI308X with SGLang 0.5.22.

SGLang keeps the server, scheduler, page allocation and sampling. ATOM owns the model, its weights, the CSA2 KV pool
(which replaces SGLang's DeepSeek-V4 pool) and the attention kernels; the plugin translates each SGLang batch into
ATOM's step metadata.

The FP4PTPC checkpoint runs only with its routed experts converted to per-channel FP8 at load, as in native ATOM: the
launch passes ATOM's online quantization through `--model-loader-extra-config` (below). Without it the plugin refuses
the checkpoint before loading weights, as native ATOM refuses it without `--online_quant_config`. Use aiter a53213dc4
or later (the commit native ATOM's DeepSeek-V4.1-Flash recipe pins).

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
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
    --cuda-graph-max-bs-decode 64 --cuda-graph-bs-decode 1 2 4 8 16 32 48 64 \
    --context-length 65536 \
    --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --model-loader-extra-config '{"online_quant_config": {"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}}'
```

The plugin refuses, before loading weights:

- an SGLang other than 0.5.22 (an exact match on `sglang.__version__`, so `0.5.22.post1` is refused too),
  before the plugin's DeepSeek-V4.1 rewiring of SGLang: that rewiring is written against 0.5.22 and has no fallback;
- the FP4PTPC checkpoint without the `--model-loader-extra-config` above;
- `--chunked-prefill-size -1` (unchunked prefill): SGLang would admit a first prompt whole, past the step size the
  plugin's metadata buffers hold;
- prefill CUDA graphs (launch with `--cuda-graph-backend-prefill disabled`);
- with decode CUDA graphs on: `SGLANG_ENABLE_METADATA_GLUE_GRAPH=1` (the plugin stages every replay on the host,
  outside the captured region), and `ATOM_ENGRAM_UVA=0` or `ATOM_ENGRAM_OVERLAP=0` (a decode graph records Engram's
  device staging; either variable moves it to the host). If Engram's tables cannot be page-locked at load, Engram falls
  back to the host too, and the decode-graph setup refuses that after the weights load (launch with
  `--cuda-graph-backend-decode disabled`);
- `--kv-cache-dtype` other than `bfloat16` or `fp8_e4m3`. That includes the default `auto`, which SGLang resolves to
  `fp8_e4m3` for DeepSeek-V4.1, so the launch names the format (`--kv-cache-dtype bfloat16` above);
- `--enable-decoder-swa-bounded-replay` with `ATOM_SGLANG_V41_DECODER_REPLAY` off (below): decoder SWA bounded replay
  is on by default, so the flag restates the default and contradicts the variable; the same flag with DSpark (below),
  which runs without decoder SWA bounded replay;
- `--enable-encoder-swa-bounded-replay` (SGLang itself refuses it on hardware other than CUDA and gfx950, MI308X
  included);
- `--disable-hybrid-swa-memory`: the pool is sized by SGLang's hybrid-SWA full token count;
- `--enable-memory-saver`: the ATOM pool is allocated outside SGLang's memory-saver region;
- a MoE all-to-all backend (`--moe-a2a-backend` other than `none`, which `--moe-runner-backend megamoe` and
  `--enable-waterfill` also set), and `--moe-dense-tp-size`: either makes SGLang gather attention inputs
  across attention-TP ranks, which the ATOM model does not read, and round its decode-graph batch sizes up past
  `--max-running-requests`;
- hierarchical cache, PD disaggregation, DP attention, decode context parallelism (`--dcp-size` above 1), and
  multimodal requests (launch with `--language-model-only`).

SGLang itself refuses, before the plugin is asked: in its DeepSeek-V4.1 feature check
(`validate_deepseek_v41_features`), two-batch overlap (`--enable-two-batch-overlap`), HiSparse, pipeline parallelism,
and speculative algorithms other than DSPARK (below); and `--enable-unified-memory`, whose attention-backend allow-list
does not hold this model's (`dsv4`).

`--kv-cache-dtype` takes `bfloat16` (BF16 pool) or `fp8_e4m3` (FP8 pool). `fp8_e4m3` is refused before the weights
load when aiter's `get_mla_sparse_prefill_v41_backend` reports no FP8 backend for this model's attention on the device;
with aiter a53213dc4 that covers gfx950 / gfx1250 only, so on MI308X (gfx942) launch with `bfloat16`. The indexer's FP8
index plane is separate and is not affected.

## Radix cache

Radix is on by default for V4.1 on the plugin: without `--disable-radix-cache` the plugin serves SGLang's radix
(prefix) cache. When a request hits the cache at `P` tokens, the 128 rows below `P` are recomputed into its fresh STATE
slot (window ring, compressor rings, Engram history) inside the step's own forward, prepended to the request's rows, as
native ATOM's encoder SWA bounded replay does. The plugin refuses radix cache beside mixed chunks, session radix cache,
a custom radix cache backend, and a KV storage backend (LMCache, FlexKV, the external cache linker). Unchunked prefill
and unified memory stay refused for every launch (above). With radix cache on, a request that carries positional
embedding overrides gets a 400: SGLang never matches or caches such a request's prefix, but a prefill step holding it
hands the model input embeddings for the whole batch, which a prefix hit's replay rows in the same step are not part
of. SGLang's tokenizer already refuses `input_embeds` with radix cache, and `--language-model-only` refuses multimodal
inputs. Launch with `--disable-radix-cache` to serve such requests. `ATOM_SGLANG_V41_RADIX=0` turns SGLang's radix
cache off for the launch, with one warning, as `--disable-radix-cache` does.

## DSpark speculative decoding

DSpark (block 5, the draft bundled in the checkpoint) adds to the launch above:

```bash
export SGLANG_RAGGED_VERIFY_MODE=static
# appended to the sglang.launch_server arguments:
--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
```

DSpark runs eagerly here: launch with `--cuda-graph-backend-decode disabled` instead of `full`; the plugin refuses
decode CUDA graphs with DSpark. DSpark also runs without decoder SWA bounded replay: every prefill row runs every
layer. The plugin turns SGLang's radix cache off for DSpark launches, with one warning. The plugin refuses a draft
token count other than 6 (only a `--speculative-dspark-block-size` other than 5 reaches it: with block 5, SGLang itself
refuses any other `--speculative-num-draft-tokens`), a draft checkpoint other than the target's, non-static verify,
block verification, `--enable-decoder-swa-bounded-replay`, `SGLANG_DSPARK_EMBED_IN_GRAPH=0` (the ATOM draft embeds its
own block), and `SGLANG_SIMULATE_ACC_LEN` above 1 (unless `ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC=1`, below).

## Environment variables

Read by the plugin (`atom/plugin/sglang/deepseek_v41_envs.py`):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_SGLANG_V41_DECODER_REPLAY** | bool | 1 (true) | Decoder SWA bounded replay: a prefill runs V4.1's decoder layers on each request's last 128 rows only, as native ATOM does. `0` / `false` / `no` / `off`, or an empty value, turn it off (every prefill row runs every layer), as native's `--no-decoder-swa-bounded-replay` does. Unlike native, which backs a short resumed chunk off to a full window, a chunk SGLang resumes with fewer rows than the window computes every row, and so does a step in which SGLang reads more than each request's last row (input logprobs, a full hidden-state capture, a prefill-only batch); the plugin's prefill output therefore depends on where SGLang cuts a prompt. DSpark launches run without it whatever the variable says. |
| **ATOM_SGLANG_V41_RADIX** | bool | 1 (true) | Serve SGLang's radix cache with the replay above whenever it is on. `0` / `false` / `no` / `off`, or an empty value, turn SGLang's radix cache off for the launch (declared through SGLang's model-override registry when the server arguments are resolved, so the tokenizer manager and the schedulers agree; one warning). |
| **ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC** | bool | 0 (false) | Performance runs only: allow `SGLANG_SIMULATE_ACC_LEN` > 1 (a fixed mean accept length; the output text is then not the model's). |
