# DSpark Speculative Decoding with the ATOM vLLM Plugin Backend

DSpark is a semi-autoregressive *block* drafter. Instead of running `k`
sequential MTP passes, it drafts a whole block of `k` tokens in one parallel
backbone pass (parallel backbone + Markov sequential head), and the target then
verifies the block in a single `1 + k` token forward.

This recipe covers DSpark under the **ATOM vLLM out-of-tree plugin**. For DSpark
on ATOM's native engine (`atom.entrypoints.openai_server`) — including
DeepSeek-V4-Pro and the confidence-scheduled ragged verify — see
[`recipes/DSpark.md`](../DSpark.md).

In plugin mode the validated target is **Kimi-K3** with the standalone
[Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) draft
checkpoint, on eight MI355 (gfx950) GPUs at TP8.

## Prerequisites

Identical to the [Kimi-K3 recipe](./Kimi-K3.md) — the ATOM vLLM OOT image plus
the KDA dependency, with the target ATOM checkout installed over it:

```bash
docker pull rocm/atom-dev:vllm-latest
pip install "fla-core==0.5.1" "flash-linear-attention==0.5.1"
pip install -e /path/to/ATOM --no-deps
```

The draft checkpoint is downloaded on startup, so no separate fetch is needed.

## Launch

```bash
MODEL=/path/to/Kimi-K3

vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --language-model-only \
    --kv-cache-dtype fp8 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"dspark","model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":7}'
```

`model` inside `--speculative-config` takes a local path or an HF repo id.
`num_speculative_tokens` is the block width; the draft checkpoint does not pin
it, and 7 is the validated setting.

## How the plugin differs from upstream vLLM DSpark

### Causal block drafting

vLLM's DSpark speculator drafts **non-causally**: every position in the block
attends to the whole block. ATOM's `mla_decode_fwd` masks tail-aligned-causally
whatever it is asked for, so producing that upper triangle means merging a
second partial attention in by log-sum-exp.

The plugin instead flips the draft to **causal** for the `K3DSparkModel`
architecture only, matching what ATOM's native DSpark has always done: position
`i` sees `context + block[:i+1]`. One drafting semantics across both engines,
and no merge to build. That costs the block's tail — the last three of the seven
positions contribute almost no accepted tokens — so acceptance lands near 3.8
rather than the non-causal ceiling. Nothing else in the DSpark path changes, and
no other model's drafting is touched.

Non-causal remains open as a follow-up. The persistent decode kernel does return
a usable LSE on gfx950 — DCP merges cross-rank on it — so the merge is
buildable; it is the split-KV kernel that is not usable here, its stage2 reduce
leaving `final_lse` untouched for any request whose valid split count collapses,
which is exactly what a large batch produces.

### Separate bf16 draft KV cache

The draft is a standalone MLA model, so it gets its own KV cache group, pinned
to **bf16** regardless of the target's `--kv-cache-dtype fp8`. Its pages are
therefore twice the target's per token. Budget for that before raising
`--max-model-len` or `--max-num-seqs`.

### Model Runner V2 is selected automatically

DSpark is implemented only by vLLM's V2 GPU model runner. vLLM turns V2 on for
`"method": "dspark"` by itself, so `VLLM_USE_V2_MODEL_RUNNER` does not need to
be set.

### Asynchronous scheduling stays enabled

vLLM disables async scheduling for most speculative methods but exempts DSpark,
and every run reported below kept it on. This is the one place the DSpark launch
departs from the target-only Kimi-K3 recipe, which turns it off.

### Full CUDA graphs cover both models

Under `FULL_AND_PIECEWISE`, the target's 8-token verify and the draft's 7-token
block both replay from full graphs. ATOM's MLA metadata builder declares
`UNIFORM_BATCH` graph support for exactly this reason: vLLM takes the minimum
across every attention backend, so anything lower here would downgrade the whole
model — the target's verify included — to piecewise.

### Prefix caching must stay off

Inherited from Kimi-K3 rather than from DSpark: KDA recurrent state cannot be
reconstructed from the paged MLA cache alone.

## Accuracy validation

Same protocol as the [Kimi-K3 recipe](./Kimi-K3.md) — over the served endpoint,
against a freshly started server launched exactly as above:

```bash
lm_eval \
    --model local-completions \
    --model_args "model=${MODEL},base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --output_path /app/logs/kimi_k3_dspark_gsm8k
```

Full 1319-example GSM8K test set, TP8, fp8 KV, `FULL_AND_PIECEWISE` CUDA Graph,
async scheduling on:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9545|±  |0.0057|
|     |       |strict-match    |     5|exact_match|↑  |0.9538|±  |0.0058|
```

Verification always emits the target-greedy token, so drafting is accuracy
neutral. The target alone scores `0.9553 ± 0.0057` under the same harness and
concurrency (see [Kimi-K3](./Kimi-K3.md)); the two are within each other's error
bars.

Check the run for `TimeoutError` before reporting a number. A run that hits the
`local-completions` client timeout retries the request, and the retry both
doubles the server's work and re-generates the answer under a different batch
composition — enough numeric difference to flip a chain of thought that was
already marginal. One such run here read `0.9424` while its acceptance figures
stayed normal, which is the tell: DSpark was fine and the harness was not.

### Acceptance

Aggregated over the same run — 36,119 verify forwards, 252,833 drafted tokens:

- **Mean accepted length: 3.777 tokens/forward** — 1 verified token plus 2.78
  accepted draft tokens.
- **Acceptance rate: 39.7%** — accepted draft tokens / total drafted tokens.

Per-position acceptance rate across the 7-token block:

```text
position   |    1  |    2  |    3  |    4  |    5  |    6  |    7
rate       | 0.923 | 0.774 | 0.602 | 0.442 | 0.034 | 0.002 | 0.000
```

The collapse after position 4 is the cost of causal drafting: the block's tail
positions see the least context and contribute roughly 1% of accepted tokens
between them. With positions 6 and 7 already at zero, widening the block past 7
has no headroom left to buy — it would only widen the verify forward.

The server logs these as it runs (`Mean acceptance length`, `Draft acceptance
rate` per position), and Prometheus exposes them as
`vllm:spec_decode_num_{accepted,drafted}_tokens_total` and
`vllm:spec_decode_num_accepted_tokens_per_pos_total`.

## Profiling

Torch profiling is a server-side flag in vLLM 0.25.x, not an environment
variable. Add to the launch command:

```bash
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/app/prof_dspark","torch_profiler_with_stack":true}'
```

Then drive it from the benchmark client, which starts and stops the profiler
around the measured window:

```bash
vllm bench serve \
    --backend vllm --base-url http://localhost:8000 \
    --model "${MODEL}" --trust-remote-code \
    --dataset-name random \
    --random-input-len 8192 --random-output-len 100 --random-range-ratio 0 \
    --max-concurrency 16 --num-prompts 16 \
    --ignore-eos --profile
```

`--random-range-ratio 0` pins every prompt to exactly the requested length; in
this version the ratio is the *width of the variation*, so `1.0` would let
sampled lengths reach zero and the run aborts. Run the same command once without
`--profile` first — the first pass pays autotuning and graph-replay warmup that
otherwise lands in the trace.

This writes one gzipped trace per TP rank plus one for the API server front end,
and a `profiler_out_<rank>.txt` CUDA-time summary. Expect them to be large:
`torch_profiler_with_stack` records every Python frame, which for 8k/100/16 at
TP8 is about 3.6 GB gzipped and ~30M Python-frame events per rank. Drop
`torch_profiler_with_stack` if you only need kernel timings.

Both models show up in the trace under their own MLA kernels —
`aiter::mla_a8w8_*` for the fp8 target and `aiter::mla_a16w16_*` for the bf16
draft — which is the quickest way to confirm the draft is running on its own
cache.

## Current scope

- Kimi-K3 is the only DSpark target validated in plugin mode. DeepSeek-V4-Pro
  DSpark runs on ATOM's native engine only.
- The verify length is batch-uniform `1 + num_speculative_tokens`. The draft
  checkpoint's confidence head is not used here — confidence-scheduled ragged
  verify (`--dspark-config`) exists only on the native engine.
- Drafting is causal, so acceptance is below the non-causal upper bound.
- TP8 on MI355/gfx950 is the validated deployment. DP attention and decode
  context parallel are not exercised with DSpark.
- Serving throughput has not been swept in plugin mode. The 8k/100/16 point
  above exists to produce a trace, not as a reported number; only accuracy and
  acceptance are tracked here.
