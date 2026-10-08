# DeepSeek-V4.1-Flash with ATOM vLLM Plugin Backend

This recipe shows how to run `deepseek-ai/DeepSeek-V4.1-Flash` with the ATOM
vLLM plugin backend. For background on the plugin backend, see
[ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md); for the
model itself and its native ATOM path, see the
[DeepSeek-V4.1-Flash recipe](../DeepSeek-V4.1-Flash.md).

The plugin path serves the **text backbone only**. Vision, DSpark speculative
decoding and prefix caching are refused at startup rather than silently
degraded — run the native ATOM engine for those. The reasons are in
[What the plugin path does not do](#what-the-plugin-path-does-not-do).

## Step 1: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and
general usage flow compatible with upstream vLLM. For general server options
and API usage, refer to the
[official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
TP=4

export AITER_LOG_LEVEL=WARNING

vllm serve "${MODEL}" \
    --host localhost \
    --port 8001 \
    --dtype auto \
    --kv-cache-dtype auto \
    --tensor-parallel-size "${TP}" \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 64 \
    --no-enable-prefix-caching
```

Notes:

- `--tokenizer-mode deepseek_v4` is **required**. The V4.1 checkpoint ships no
  HF `tokenizer.json`; it carries the same DeepSeek-V4 encoding under
  `encoding/`, and vLLM only auto-selects that mode for the V4 architecture
  string.
- `--no-enable-prefix-caching` is **required**; see below.
- `--block-size` is negotiated to `256` whatever you pass. V4.1's PAGE is 256
  tokens, and the bridge relies on a vLLM block id being the same id as a V4.1
  page id.
- CUDA graphs are forced off (`cudagraph_mode=NONE`), so `--enforce-eager` and
  any `--compilation-config` cudagraph setting are redundant. The server logs
  the override at startup.
- `--max-num-seqs` is what the per-request state region is sized for, and that
  region comes out of the same pool as cached history — see
  [How the KV pool is sized](#how-the-kv-pool-is-sized). Raise it only
  together with `--gpu-memory-utilization`.

Feature flags:

- `--kv-cache-dtype auto` (equivalently `bfloat16`) runs the unpacked BF16
  pool. `--kv-cache-dtype nvfp4` selects ATOM's packed FP4 pool, which roughly
  halves the bytes a cached token costs. The fp8 spellings are rejected: V4.1
  has no fp8 main pool, and silently downgrading would leave vLLM's allocation
  and the kernels' geometry describing different bytes.
- Expert parallelism (`--enable-expert-parallel`) and the usual parallel
  options behave as they do for any other ATOM plugin model.

## Step 2: Performance Benchmark

Users can use the default vLLM bench command for performance benchmarking.

```bash
vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8001 \
    --endpoint /v1/completions \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 100 \
    --max-concurrency 4 \
    --num-prompts 40 \
    --trust_remote_code \
    --num-warmups 8 \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el
```

## Step 3: Accuracy Validation

The accuracy can be verified on the GSM8K dataset with `lm_eval`, the same way
the native path is validated:

```bash
lm_eval \
  --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4.1-Flash,base_url=http://localhost:8001/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
```

The native TP4 reference is GSM8K `0.9204` over 1,319 questions
(see the [model recipe](../DeepSeek-V4.1-Flash.md)). The plugin path runs the
same backbone against the same weights, so it is judged against that number.
Measured on the plugin path at TP4, BF16 pool, `--max-model-len 8192`:
`0.9242 +/- 0.0073` over the same 1,319 questions (strict-match and
flexible-extract agree).

Pass `tokenizer_backend=none` as well if you point `lm_eval` at a local
checkpoint directory: with no HF `tokenizer.json` beside the weights, the
harness otherwise tries to resolve the served model name on the Hub.

## How the KV pool is sized

V4.1 buys its pool in two currencies out of one contiguous allocation:

- **PAGE** bytes scale with cached history — one 256-token page of main latent
  per owner plus that page's index rows.
- **STATE** bytes scale with *in-flight requests* — each one owns a fixed-size
  entry holding its sliding-window ring, the compressor's rings and its Engram
  cursor.

vLLM can only size the first: it buys `num_blocks` uniform blocks of
`page_size_bytes` and hands every one of them to its `BlockPool`. ATOM declares
the proxy layer's `page_size_bytes` as exactly one V4.1 PAGE and then withholds
the STATE tail's worth of block *ids* from the scheduler, while leaving the
allocation's byte size alone. The profiled memory budget is therefore
unchanged; what shrinks is how many of those blocks may hold history.

The tail costs `--max-num-seqs` STATE entries. If it does not leave room for a
single `--max-model-len` request, the server fails at startup with a message
naming both numbers — raise `--gpu-memory-utilization`, or lower
`--max-num-seqs` or `--max-model-len`.

## What the plugin path does not do

| Capability | On the plugin path |
|---|---|
| Text generation, chat, tools, reasoning effort | Supported |
| BF16 and packed FP4 main pool | Supported (`--kv-cache-dtype auto` / `nvfp4`) |
| Image requests | Refused — text only |
| DSpark speculative decoding | Refused — needs ATOM's tentative staging, which the proxy bridge does not drive |
| Prefix caching | Refused — a block-table hit restores the compressed pages but not the per-request window ring, compressor rings or Engram cursor that CSA2 attention reads alongside them |
| KV transfer / offload connectors | Refused — V4.1's cached prefix is PAGE bytes *and* a per-request STATE image, and no connector on this path carries the second; a PAGE prefix restored without its STATE is a dead engine, not a degraded answer |
| CUDA graphs | Forced off — a V4.1 step does host-side Engram staging, state reset and cursor advance every forward that no captured graph replays. See [CUDA graphs: what blocks them](#cuda-graphs-what-blocks-them) |

Each of these is refused by `enforce_deepseek_v41_constraints` -- which the
worker applies to the config it is about to run, ahead of cudagraph capture and
the first forward -- or by
`atom.models.deepseek_v41.config.validate_runtime_config`, with a message
naming the flag to drop. Use the
[native ATOM engine](../DeepSeek-V4.1-Flash.md) for any of them.

## CUDA graphs: what blocks them

Graphs were attempted, measured and stood down. The record, so the next
attempt does not start from the beginning:

**What was removed.** The stated reason V4.1 ran eager was that ATOM's plugin
models are not fx-split, so vLLM's PIECEWISE mode had nothing to split on and
would swallow the backbone whole. `VLLM_USE_BREAKABLE_CUDAGRAPH=1` ends the
stream capture at runtime instead of splitting an fx graph, so that reason
does not survive it. With it:

- `v41_stage_step` (`deepseek_v41_bridge.py`) carries the break around the
  step's host work -- `_prepare` *and* `prepare_model_inputs` in one break,
  because `_prepare` stages the step's index/indptr/slot tensors with kernels
  of its own.
- `cudagraph_mode` must be **PIECEWISE exactly**, not merely non-NONE:
  `eager_break_during_capture` *skips the break* when the forward context
  reports a FULL runtime mode. Under FULL the step work is recorded into the
  graph and never runs again -- a frozen cursor, every replayed step
  re-answering the first, with nothing raised.
- The proxy builder's `AttentionCGSupport.NEVER` needs no change: vLLM's three
  NEVER gates in `resolve_cudagraph_mode_and_sizes` all test FULL
  (`mixed_mode`, `decode_mode`, `has_full_cudagraphs`), so PIECEWISE passes
  them untouched and the honest declaration stays.
- The dummy-batch cache must be reused, not rebuilt per call. `_prepare`'s
  dummy branch allocated a scratch `PagedAttentionCache` each time; captured
  kernels hold its addresses, it is freed when the capture returns, and the
  first replay reads freed memory. Symptom: `illegal memory access`, reported
  asynchronously inside an unrelated `copy_to_gpu`. Fixed by `_dummy_cache`.

**What still blocks it.** With all of the above, V4.1 captures and serves
without raising, and the answers degenerate into noise after the first few
tokens (measured: 8/8 prompts, greedy, against an eager arm that answers all 8
correctly). The attention is what stays behind: its kernels are launched with
per-step host values -- the batch's longest KV extent among them -- which a
capture freezes at whatever length it recorded while every decode step grows
past it.

The ordinary remedy, an eager break on the attention op the way vLLM does for
`unified_attention_with_output`, **does not apply as a decoration**: the
decorator requires an in-place output buffer ("a fresh tensor returned by `fn`
would change address each replay"), and V4.1's `Attention.forward` returns a
fresh tensor. Applied anyway, it faults *inside capture*, not at replay.

So capturing V4.1 is attention-level work -- a persistent per-layer output
buffer plus a device-side length bound -- not configuration. Two routes, both
real:

1. V4's road: hoist every per-step host value into fixed-address buffers and
   declare `UNIFORM_BATCH` (see `deepseek_v4_bridge.py`). Large, and V4.1's
   step work is the thing its own builder docstring says must run inside the
   forward.
2. Give attention a persistent output buffer so it can legally carry the
   break, and move the KV length bound to device memory. Costs one buffer per
   layer at the captured width.

**The gate is closed by default and deliberately.** `_breakable_cudagraph_available`
returns False unless `ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1` is set, *in addition*
to vLLM's own flag. vLLM auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH` for some
architectures; without the second condition, a vLLM upgrade that adds V4.1 to
that list would turn correct answers into noise with nothing in the log.
`tests/plugin/test_vllm_deepseek_v41_cudagraph_mode.py` holds that shut. The
numbers in this recipe are therefore all eager, and must not be extrapolated
to a graph deployment.
