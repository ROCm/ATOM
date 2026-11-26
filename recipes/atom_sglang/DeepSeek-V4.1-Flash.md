# DeepSeek-V4.1-Flash with ATOM SGLang Plugin

This recipe serves DeepSeek-V4.1-Flash (FP4 PTPC checkpoint) through the ATOM SGLang plugin, without speculative
decoding or with DSpark (below), with SGLang's radix cache and decode CUDA graphs (SGLang's overlap scheduler stays
on, its default). The configuration below is TP4 on MI308X with SGLang 0.5.23.

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

- an SGLang other than 0.5.23 (an exact match on `sglang.__version__`, so `0.5.23.post1` is refused too),
  before the plugin's DeepSeek-V4.1 rewiring of SGLang: that rewiring is written against 0.5.23 and has no fallback;
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
  is on by default, so the flag restates the default and contradicts the variable;
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
export SGLANG_DSPARK_USE_GREEDY_DRAFT=1  # native ATOM's draft and verify rule, at any temperature (below)
# appended to the sglang.launch_server arguments:
--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6
```

Decode CUDA graphs cover the target's verify and the draft's block (`--cuda-graph-backend-decode full`, as above);
with DSpark the plugin also refuses `--disable-cuda-graph-padding`, and warns when the decode buckets stop below
`--max-running-requests` (larger rounds run eagerly). Decoder SWA bounded replay stays on, as natively. The radix cache
stays on with DSpark: a hit's replay rows rebuild the draft's context windows too. The plugin refuses a draft token
count other than 6 (only a `--speculative-dspark-block-size` other than 5 reaches it: with block 5, SGLang itself
refuses any other `--speculative-num-draft-tokens`), a draft checkpoint other than the target's, non-static verify,
block verification, `SGLANG_DSPARK_EMBED_IN_GRAPH=0` (the ATOM draft embeds its own block),
`SGLANG_SIMULATE_ACC_LEN` above 1 (unless `ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC=1`, below), and full hidden-state
returns (`--enable-return-hidden-states` or `--return-hidden-states-mode full`) under decoder SWA bounded replay: SGLang
returns every prompt row of the target's DSpark capture, which the replay leaves zero outside each request's last 128
rows (`--return-hidden-states-mode last` is served; `ATOM_SGLANG_V41_DECODER_REPLAY=0` serves full returns).

### Greedy draft (`SGLANG_DSPARK_USE_GREEDY_DRAFT=1`)

The launch sets `SGLANG_DSPARK_USE_GREEDY_DRAFT=1` for every temperature: the draft always takes the argmax, as native
ATOM's draft does. The graph-folded draft block then runs ATOM's sharded fused Markov argmax wherever native ATOM's
DSpark would (a TP-sharded head; native's `ATOM_DSPARK_FUSED_MARKOV_SAMPLE=0` turns it off): each rank's LM-head
shard, the ranks exchanging only each row's `(max, id)`, instead of SGLang's Markov chain over all-gathered
full-vocabulary logits. Greedy requests draft the same tokens up to numerics.

Every verify rule involved is lossless, so a sampled request's output follows the target distribution (temperature,
top-p) with or without the flag; only acceptance differs. A round whose batch holds a sampled request accepts eagerly,
outside the verify CUDA graph. For a top-p batch with no greedy, top-k or min-p request, the plugin then verifies as
native ATOM does: each verify row draws a token from the target's top-p distribution (`softmax(logits / T)` in fp32 and
aiter's `top_p_sampling_from_probs`, native's kernels), and the draft is accepted while the drawn token equals it. For
an argmax draft this is the same output distribution as SGLang's rejection sampler given the draft as one-hot (a draft
token is accepted with its target probability, a rejected row emits a draw from the target without it), without
SGLang's one-hot `[bs, 5, V]` draft distribution, per-row top-p renormalisation and rejection kernel. Batches with a
greedy, top-k or min-p request and temperature-only sampling keep SGLang's accept.

Unset the flag only to take SGLang's sampled draft: the draft samples its block from its own distribution and
SGLang's rejection sampler accepts by the probability ratio, which accepts more as the distributions flatten (high
temperature) but costs more a round (the draft samples over the all-gathered full-vocabulary logits, and the verify
computes the draft's full-vocabulary probabilities). Native ATOM's lossless sampled draft
(`ATOM_SPEC_LOSSLESS_SAMPLING=1`) has no counterpart flag in the plugin.

### Sampled verify with the sampled draft: exact top-p without the vocabulary sort (temporary)

With the sampled draft (`SGLANG_DSPARK_USE_GREEDY_DRAFT` unset), a top-p batch with no top-k request goes through
SGLang's rejection sampler, which renormalises the target distribution of every verify row by sorting its whole
vocabulary (`top_p_renorm_probs_triton`). For DeepSeek-V4.1 the plugin replaces that one branch: it finds the pivot
SGLang's sort / cumsum / searchsorted would find, bitwise, from the top 1280 values of each row (PyTorch's fp32 cumsum
replayed addition for addition, the rest of the row entering as its exact sum within a rigorous rounding bracket),
and renormalises with SGLang's own kernel, so the accept sees SGLang's target distribution bit for bit. Rows the top
1280 cannot settle are sorted on the device; nothing is read back on the host. A cost guard compares, from device
counts copied without waiting, what those rows cost with what the exact calls saved, and takes SGLang's sort for the
next 1024 calls when that is cheaper (same probabilities either way; its cost table is MI308X's).

The DeepSeek-V4.1 DSpark draft wrapper installs it only with the sampled draft (with
`SGLANG_DSPARK_USE_GREEDY_DRAFT=1` a top-p verify takes the draw rule above, and one INFO line says the patch is not
installed), and only when the source of the five SGLang functions it replaces or relies on matches pinned sha256
fingerprints, PyTorch is the pinned build whose HIP cumsum it replays, and a GPU self-check at 6, 192 and 258 verify
rows is bitwise equal to SGLang's; otherwise one WARNING line names the reason and SGLang's path stays. It covers at
most 85 requests (510 verify rows) and static verify. Temporary: remove `atom/plugin/sglang/patches/dspark_topp_verify_probs_patch.py`,
`dspark_topp_pivot.py` and their test once SGLang's top-p renorm no longer sorts the full vocabulary.

## Prefill

After SGLang's own graph capture, ATOM records prefill token graphs for the target's layers 0-19 at native ATOM's
buckets (`ATOM_TOKEN_GRAPH_SIZES` up to the chunk size; 256 to 3072 tokens by default). A prefill step pads to the
smallest bucket that holds it and replays the recording; a larger step runs eagerly. A radix hit step pads on the rows
ATOM runs (its new rows plus 128 replay rows per hit). Without decode CUDA graphs no prefill token graphs are recorded
(one log line says so). With DSpark they are recorded only under decoder SWA bounded replay (the default). SGLang
0.5.22 ranks its deprecated `--disable-cuda-graph` below an explicit `--cuda-graph-backend-decode`: to run without
decode graphs, replace `--cuda-graph-backend-decode full` with `disabled`.

With or without graphs, ATOM then runs one eager prefill at the chunk budget (`--chunked-prefill-size`, at most the
builder's budget; split over two sequences under DSpark), as native ATOM's start-up warm-up does, on the graph-capture
STATE slots and SGLang's padding page, so no request state is touched. Without it the first chunk-sized prefill pays
the eager path's first-use kernel loads.

An arriving prefill takes its STATE slot and page rows from a copy queued where SGLang's `alloc_for_extend` wrote
them, instead of a read-back behind the step in flight.

A long prompt is prefilled in chunks of `--chunked-prefill-size`. A chunk that leaves at least 128 rows of its prompt
for later chunks runs layers 0-19 and layer 20's compression only; the decoder layers (20-39) and the DSpark draft
context write run on the chunk that ends the prompt, as in native ATOM. If the step that continues such a prompt
computes every row instead (another request in it asks input logprobs, or SGLang cuts the next chunk below 128 rows),
that prompt's decoder window starts at the chunk boundary instead of reading rows the skipped chunk never wrote.

## Decode step tail

- An all-greedy batch (every request at temperature 0, no logprobs, no sampling mask) gets its tokens from aiter's
  `topk_select(k=1, tie="low")`, native ATOM's greedy sampler, instead of SGLang's `torch.argmax` over the full fp32
  vocabulary: same lowest-id tie rule, ids as int64; SGLang's NaN check, logit preprocessing and TP token sync run
  around it, and every other sampler call is SGLang's own. It is installed only where `Sampler.forward` has the
  expected signature (otherwise one log line and SGLang's path). With DSpark it only picks each request's first token
  after prefill.
- Step metadata goes to the GPU on native ATOM's packed transport: one pinned arena, one copy and one scatter per phase
  group, instead of one copy per member.

## Prefill admission

`--max-running-requests <concurrency> --min-free-slots-delay N` holds new admissions until `N` running slots have
freed, which coalesces arrivals into fewer prefill steps: shorter wall time for a higher TTFT. It only engages when
the running batch is full, so set `--max-running-requests` to the serving concurrency. SGLang then captures decode
graphs only up to that size (it drops larger buckets and adds one at `--max-running-requests`).

## Environment variables

Read by the plugin (`atom/plugin/sglang/deepseek_v41_envs.py`). Every one is a bool with one parser: `0`, `false`,
`no`, `off` or an empty value turn it off, anything else on.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_SGLANG_V41_DECODER_REPLAY** | bool | 1 (true) | Decoder SWA bounded replay: a prefill runs V4.1's decoder layers on each request's last 128 rows only, as native ATOM does. `0` / `false` / `no` / `off`, or an empty value, turn it off (every prefill row runs every layer), as native's `--no-decoder-swa-bounded-replay` does. Unlike native, which backs a short resumed chunk off to a full window, a chunk SGLang resumes with fewer rows than the window computes every row, and so does a step in which SGLang reads more than each request's last row (input logprobs, a full hidden-state capture, a prefill-only batch); the plugin's prefill output therefore depends on where SGLang cuts a prompt. |
| **ATOM_SGLANG_V41_RADIX** | bool | 1 (true) | Serve SGLang's radix cache with the replay above whenever it is on. `0` / `false` / `no` / `off`, or an empty value, turn SGLang's radix cache off for the launch (declared through SGLang's model-override registry when the server arguments are resolved, so the tokenizer manager and the schedulers agree; one warning). |
| **ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS** | bool | 1 (true) | Prefill token graphs (see Prefill). `0`: every prefill runs eagerly. |
| **ATOM_SGLANG_V41_PREFILL_WARMUP** | bool | 1 (true) | The start-up eager prefill at the chunk budget (see Prefill), with or without graphs. `0` skips it. |
| **ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC** | bool | 1 (true) | An arriving prefill takes its STATE slot and page rows from a copy made when SGLang allocated them. `0`: read back at staging, behind the step in flight. |
| **ATOM_SGLANG_V41_FAST_GREEDY** | bool | 1 (true) | All-greedy batches pick tokens with aiter's top-1 (see Decode step tail). `0`: SGLang's sampler. |
| **ATOM_SGLANG_V41_DSPARK_DEFERRED_SEQ_LENS** | bool | 1 (true) | A DSpark decode round launches its draft block before the host waits for the previous round's published lengths, and the verify's staging takes them. `0`: SGLang's wait at the top of the round. |
| **ATOM_SGLANG_V41_DSPARK_PREFETCH_ROWS** | bool | 1 (true) | A DSpark round's page rows that the verify would read back are copied before the draft launch. `0`: read back behind the draft block. |
| **ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY** | bool | 1 (true) | With the sampled draft (`SGLANG_DSPARK_USE_GREEDY_DRAFT` unset), a top-p verify (no top-k) takes PyTorch's exact top-p pivot without the vocabulary sort, bitwise SGLang's probabilities (see Sampled verify; temporary patch). `0`: SGLang's sort. |
| **ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY_GUARD** | bool | 1 (true) | The cost guard of the above: SGLang's sort for the next 1024 calls when the device fallback cost more than the exact calls saved (MI308X cost table; same probabilities either way). `0`: always the exact path. |
| **ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC** | bool | 0 (false) | Performance runs only: allow `SGLANG_SIMULATE_ACC_LEN` > 1 (a fixed mean accept length; the output text is then not the model's). |
