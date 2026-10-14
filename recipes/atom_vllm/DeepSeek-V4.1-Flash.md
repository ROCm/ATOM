# DeepSeek-V4.1-Flash with ATOM vLLM Plugin Backend

This recipe shows how to run `deepseek-ai/DeepSeek-V4.1-Flash` with the ATOM
vLLM plugin backend. For background on the plugin backend, see
[ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md); for the
model itself and its native ATOM path, see the
[DeepSeek-V4.1-Flash recipe](../DeepSeek-V4.1-Flash.md).

The plugin path serves the **text backbone only**, without prefix caching or
KV-transfer connectors. Do not configure a connector: startup rejection does
not cover every connector name. DSpark speculation and CUDA graphs are
supported, both behind an explicit flag. The scope and restrictions are in
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
- This integration targets the **V2 model runner**. The validated deployment
  is TP4 with pipeline, decode-context and prefill-context parallel sizes all
  equal to 1; other parallel layouts are not covered by this recipe.

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

Historical reference: ATOM `2a89e29135`, based on main `661ef7661`, with
vLLM `db9527a46`, V2 TP4, BF16 pool, DSpark5 and FULL_AND_PIECEWISE graphs
scored `0.9219105383` on all 1,319 GSM8K questions (5-shot, concurrency 32,
`--max-model-len 8192`, `max_gen_toks=256`; strict and flexible extraction
agree). The sampling standard error was `0.00739065`, not a measured
run-to-run variation. This result predates the review fixes to STATE ownership,
FULL plan capacities and H2D publication; it does not validate those changes.
Re-run the same configuration when evaluating a newer tree.

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

The tail reserves **twice `--max-num-seqs`** STATE entries: one half for real
requests and one for synthetic warmup/capture steps. Real requests use V2's
persistent `idx_mapping_np` indices, not an LRU over the scheduled subset;
a live but temporarily unscheduled request keeps its state. Position-zero
recomputation uses the cache's existing reset path.

The target has one proxy KV group; DSpark also introduces draft KV storage.
Available history capacity must leave room for all those groups, the STATE
tail, and the shared null block. Do not estimate concurrent-request capacity
from target proxy pages alone.

## What the plugin path does not do

| Capability | On the plugin path |
|---|---|
| Text generation, chat, tools, reasoning effort | Supported |
| BF16 and packed FP4 main pool | Supported (`--kv-cache-dtype auto` / `nvfp4`) |
| Image requests | Refused — text only |
| DSpark speculative decoding | Supported — 5 draft tokens, fixed by the model's own config gate; the draft is the target checkpoint's `mtp.*` stages, so there is no separate draft model to pass |
| Prefix caching | Refused — a block-table hit restores the compressed pages but not the per-request window ring, compressor rings or Engram cursor that CSA2 attention reads alongside them |
| KV transfer / offload connectors | Unsupported in this integration — restoring PAGE bytes without the matching per-request STATE is unsafe. Do not rely on every connector alias being rejected at startup |
| CUDA graphs | Opt-in with both graph flags; FULL and PIECEWISE use different staging boundaries. See [CUDA graphs](#cuda-graphs) |

Existing refusals are made by `enforce_deepseek_v41_constraints` or
`atom.models.deepseek_v41.config.validate_runtime_config`. These checks are not
a complete connector admission gate. For capabilities outside this recipe,
consult the [native ATOM recipe](../DeepSeek-V4.1-Flash.md) for its supported
configuration rather than bypassing a plugin refusal.

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

See [Accuracy validation](#step-3-accuracy-validation) for the revision-pinned
historical result. Graph counts depend on vLLM's descriptors and configuration;
record them for each run rather than treating a previous count as a contract.
Successful capture or aggregate accuracy alone does not prove that an inactive
compression branch was recorded or that padded replay rows were cleared.

### How a V4.1 forward gets captured at all

A V4.1 step does host-side work every forward that no captured graph can
replay -- Engram staging, state reset, cursor advance -- and its attention
kernels are launched with per-step host values, including the batch's longest
KV extent, which a capture would freeze at whatever length it recorded. The
PIECEWISE path handles these boundaries in three ways:

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

Under FULL there is no eager break. The metadata builder joins Engram before
capture/replay and the consumer skips a duplicate external-event wait. V2
passes runtime mode `NONE` while preparing a FULL capture, so the explicit
capture flag must take precedence over the mode. This path gives up Engram's
layer overlap; PIECEWISE retains the per-layer eager waits.

FULL compression plans use both the padded request capacity and fixed query
width. A ratio-2 step with no active compression rows still needs a nonzero
capture capacity; replay initializes inactive rows with `position=-1`. This
is separate from whether the graph can be captured without an exception.

### Host buffers and scratch lifetime

The snapshot keeps exact device-to-host reads for sequence lengths and PAGE
tables. V2's `seq_lens_cpu_upper_bound` is not an exact replacement during
asynchronous speculation. This path is not synchronization-free.

Metadata and cache-owned cursor/reset/tentative H2D sources have publication
owners: before reusing pinned host storage they wait for its previous transfer;
partial preparation failures poison the owner rather than allow unsafe reuse.
Tentative host prefixes remain in the same epoch through sampler commit.
The separate D2H state probe still defaults to blocking; H2D ownership does not
prove that deferred probe mode is safe.

Capture scratch is retained only when the effective model configuration enables
graphs. `--enforce-eager` and mode `NONE` use shape-sized temporary scratch,
even if ROCm sets `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. Storage already referenced
by a captured graph is not released on a later temporary-cache request.

For a capture failure, retain the first exception as well as any teardown
exception: a second `capture_end()` failure can obscure the original one.
Check targeted replay writes and end-to-end accuracy, not just startup.
