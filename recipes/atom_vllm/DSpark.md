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
second partial attention in by log-sum-exp — and no aiter decode kernel hands
back an LSE that holds up at serving batch sizes (the persistent kernel leaves
`final_lse` at `FLT_MAX` for part of the batch once sequence lengths vary).

The plugin therefore flips the draft to **causal** for the `K3DSparkModel`
architecture only, matching what ATOM's native DSpark has always done: position
`i` sees `context + block[:i+1]`. That costs the block's tail — the last three
of the seven positions contribute almost no accepted tokens — so acceptance
lands near 3.8 rather than the non-causal ceiling, and it needs no LSE at all.
Nothing else in the DSpark path changes, and no other model's drafting is
touched.

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

## Validation

Accuracy and acceptance were measured with the engine driven directly by
`lm_eval`'s vLLM class rather than over HTTP: TP8, fp8 KV, block size 128, 64
concurrent sequences, `FULL_AND_PIECEWISE` CUDA Graph, async scheduling on. The
context length is pinned short here only because 5-shot GSM8K prompts are short;
it is an evaluation setting, not a serving constraint.

```python
from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM

model = VLLM(
    pretrained="/path/to/Kimi-K3",
    tensor_parallel_size=8,
    trust_remote_code=True,
    max_model_len=4096,
    batch_size=64,
    max_num_seqs=64,
    gpu_memory_utilization=0.90,
    block_size=128,
    kv_cache_dtype="fp8",
    enable_prefix_caching=False,
    speculative_config={
        "method": "dspark",
        "model": "Inferact/Kimi-K3-DSpark",
        "num_speculative_tokens": 7,
    },
)
results = simple_evaluate(model=model, tasks=["gsm8k"], num_fewshot=5,
                          bootstrap_iters=0)
print(results["results"]["gsm8k"])
```

Full 1319-example GSM8K test set:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |
|-----|------:|----------------|-----:|-----------|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9530|
|     |       |strict-match    |     5|exact_match|↑  |0.9530|
```

- **Mean accepted length: 3.776 tokens/forward** — 1 verified token plus ~2.78
  accepted draft tokens.
- **Acceptance rate: 39.7%** — accepted draft tokens / total drafted tokens.

Verification always emits the target-greedy token, so drafting is accuracy
neutral; the target alone scores `0.9553` on the same benchmark, though that
number was measured over the served endpoint at a longer context (see
[Kimi-K3](./Kimi-K3.md)) and is a reference point rather than a paired control.

Use a freshly started engine for each reported accuracy run, matching the
Kimi-K3 validation protocol.

## Current scope

- Kimi-K3 is the only DSpark target validated in plugin mode. DeepSeek-V4-Pro
  DSpark runs on ATOM's native engine only.
- The verify length is batch-uniform `1 + num_speculative_tokens`. The draft
  checkpoint's confidence head is not used here — confidence-scheduled ragged
  verify (`--dspark-config`) exists only on the native engine.
- Drafting is causal, so acceptance is below the non-causal upper bound.
- TP8 on MI355/gfx950 is the validated deployment. DP attention and decode
  context parallel are not exercised with DSpark.
- Serving throughput has not been benchmarked in plugin mode; only accuracy and
  acceptance are tracked here.
