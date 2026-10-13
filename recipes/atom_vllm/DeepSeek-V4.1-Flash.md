# DeepSeek-V4.1-Flash with ATOM vLLM Plugin Backend

This recipe shows how to run `deepseek-ai/DeepSeek-V4.1-Flash` with the ATOM
vLLM plugin backend. For background on the plugin backend, see
[ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md); for the
model itself and its native ATOM path, see the
[DeepSeek-V4.1-Flash recipe](../DeepSeek-V4.1-Flash.md).

The plugin path serves the **text backbone only**. Vision, prefix caching and
KV-transfer connectors are refused at startup rather than silently degraded —
run the native ATOM engine for those. DSpark speculation and CUDA graphs are
supported, both behind an explicit flag. The reasons, and what each refusal
costs, are in
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
- CUDA graphs are **off unless you ask for them**, with
  `ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1` *and* `VLLM_USE_BREAKABLE_CUDAGRAPH=1`.
  Both are required: vLLM auto-enables its own flag for some architectures, and
  the second condition keeps a vLLM upgrade that adds V4.1 to that list from
  changing this path's behaviour with nothing in the log. See
  [CUDA graphs](#cuda-graphs).
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
flexible-extract agree) -- **on the tree this work was first written against,
not on this one.** That run predates a rebase onto a newer `main` that moved
three things on the numerical path: the aiter the kernels come from, the
`workspace` argument `PagedAttentionCache` now takes (the bridge hands it the
builder's own scorer scratch, as the native path does), and the `index_fp4`
field the pool geometry gained. Treat the number as the expectation to confirm,
not as this tree's result, until it is re-measured here.

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
| DSpark speculative decoding | Supported — 5 draft tokens, fixed by the model's own config gate; the draft is the target checkpoint's `mtp.*` stages, so there is no separate draft model to pass |
| Prefix caching | Refused — a block-table hit restores the compressed pages but not the per-request window ring, compressor rings or Engram cursor that CSA2 attention reads alongside them |
| KV transfer / offload connectors | Refused — V4.1's cached prefix is PAGE bytes *and* a per-request STATE image, and no connector on this path carries the second; a PAGE prefix restored without its STATE is a dead engine, not a degraded answer |
| CUDA graphs | Supported behind `ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1` — the host-side step work and attention each run in an eager break, so the rest of the forward is captured. See [CUDA graphs](#cuda-graphs) |

Each refusal above is made by `enforce_deepseek_v41_constraints` -- which the
worker applies to the config it is about to run, ahead of cudagraph capture and
the first forward -- or by
`atom.models.deepseek_v41.config.validate_runtime_config`, with a message
naming the flag to drop. Use the
[native ATOM engine](../DeepSeek-V4.1-Flash.md) for any of them.

## CUDA graphs

Graphs are off by default and enabled by two flags together:

```bash
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1
```

Both are required. vLLM auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH` for some
architectures; without ATOM's own second condition, a vLLM upgrade that adds
V4.1 to that list would change this path's behaviour with nothing in the log.
`tests/plugin/test_vllm_deepseek_v41_cudagraph_mode.py` holds that shut.

Measured on TP4, GSM8K 20-shot, the full 1319 questions:

| Arm | Accuracy |
|---|---|
| Graphs off, no speculation | 0.9280 |
| Graphs on (FULL + PIECEWISE), no speculation | 0.9280 FULL / 0.9325 PIECEWISE |
| Graphs on, DSpark 5 tokens | 0.9295 ± 0.0071, mean acceptance 3.55–3.58 |

A V4.1 serving run captures a mix: 47 FULL graphs and 110 PIECEWISE in the
DSpark configuration above, with no capture failures.

### How a V4.1 forward gets captured at all

A V4.1 step does host-side work every forward that no captured graph can
replay -- Engram staging, state reset, cursor advance -- and its attention
kernels are launched with per-step host values, including the batch's longest
KV extent, which a capture would freeze at whatever length it recorded. Three
things make it work, and each is load-bearing:

- **The step work moved out of the forward.** The metadata builder stages the
  step before the model is called (`_v41_stage_outside_forward` in
  `deepseek_v41_bridge.py`), so nothing that must run per step is inside a
  region a graph records.
- **Attention runs in an eager break** (`v41_core_overrides.py`).
  `eager_break_during_capture` ends the capture segment at the decorated call,
  runs it eagerly and starts a fresh one, which is what keeps the per-step
  launch bounds live. The decorator requires the broken-out region to write
  through a caller-owned buffer -- a fresh tensor would change address on each
  replay while the captured segments after it still point at the first -- so
  `_attn_out_buffer` allocates that buffer in what stays inside the graph.
- **Engram's prefetch keeps its side stream, with the orderings in breaks.**
  A fork held open across a segment boundary is refused
  (`hipErrorStreamCaptureUnjoined`), and a captured stream waiting on an event
  recorded off-capture is too (`hipErrorStreamCaptureIsolation`). Decorating
  `_prefetch`, `_await_layer` and `_await_fork` issues each where no segment
  owns it, which is what upstream's own V4.1 does for `_start_prefetch` and
  `_finish_prefetch`. Standing the fork down instead also works and costs the
  overlap between a largely-UVA gather and the layers consuming it.

Under FULL the picture differs in one way worth knowing: `eager_break_during_capture`
returns the function unwrapped, so there is no break. The Engram fork is
therefore joined immediately on that path rather than left open across layers
(`_v41_join_covers_the_layers`), which keeps FULL correct at the cost of that
overlap.

### Dead ends, so they are not re-walked

Each of these was measured, and none of them is the explanation for a capture
that will not finish: the compressor/indexer side streams
(`ATOM_DSV41_SIDE_STREAMS=0`); `in_hipgraph` opening ATOM's fork gate under a
foreign capture; the MoE's dual-stream overlap
(`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0`, verified to take effect); an
exception inside an eager-break body; unsynchronised staging left on another
stream; the cudagraph mode and the path that sets it; and 0.31's wrapper
nesting sharing one `_capturing` flag.

Two notes on reading a failure here. `capture_end()` reporting
`hipErrorStreamCaptureWrongThread` names a thread that is not the problem: the
first `_end_segment` raises inside `capture_end()` before it reaches
`self._capturing = False`, so the context manager's `__exit__` ends an
already-dead graph and reports *that*, hiding the first failure. And a capture
that completes is not a capture that is correct -- an early arm captured and
served without raising while the answers degenerated into noise after the
first few tokens (8/8 prompts, greedy, against an eager arm answering all 8).
Check accuracy, not the absence of an exception.
