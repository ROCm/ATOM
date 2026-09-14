# Qwen3.8-Flash-Next

The checkpoint declares `qwen4_exp` / `Qwen4ExpForConditionalGeneration`.
ATOM supports its text and image path: 36 GDN layers, 12 QSA layers, four
hyper-connection streams, PLE n-gram memory and a Qwen3-compatible vision tower.

## Environment and support

Use `transformers==5.16.1`, as pinned in `pyproject.toml`. Earlier versions do
not include this model's config and processor. Use an AITER build containing
the shared MLA split-K reducer.

| Capability | Status |
|---|---|
| BF16 weights, BF16 activations/KV, TP2 + EP | Covered by full-model smoke and profiler runs |
| Pure TP2, EP disabled | Fails startup warmup on the tested gfx950/AITER build: default MoE dispatch does not support the 320-wide expert shard; not full-model validated |
| Text and image questions | Full-model validation |
| Chunked text prefill, padded decode graphs, prefix reuse | Implemented for text requests |
| PLE cold state, forks, relocation, EOS/chunk history | Implemented using per-request state slots |
| FP8 weights, BF16 activations/KV, TP1 | Full GSM8K and text/image smoke validation |
| Quantized GDN input projections (source or online) | Rejected: packed B/A rows do not align with independent block scales; keep GDN excluded as in the published FP8 configuration |
| MTP/speculation, PP, DP attention, PCP/DCP, TBO | Not implemented; checked by the model/backend, with no Qwen-specific blacklist in shared Config |
| Ordinary DP, RapidServe, plugin mode | Not end-to-end validated; no model-specific blanket rejection or support claim |
| Piecewise compilation/graphs (including FULL_AND_PIECEWISE and AF_PIECEWISE) | Not implemented; rejected when active. Eager ignores the inactive graph setting; full decode graphs remain available |
| Image prefix caching | Known engine issue; fix belongs to a separate PR, not this model adaptation |
| Preemption/recompute | Known image-payload and deferred-output issues; fixes belong to separate PRs |
| External KV transfer / CPU offload | Rejected; connectors do not own the full hybrid state |
| Video and long-context quality beyond the tested context | Not validated |

Only one PLE layer is supported. MTP weights are explicitly skipped. The GDN
recurrent state follows `mamba_ssm_dtype` (FP32 in this checkpoint); the
convolution history remains BF16. PLE's previous `ngram_size - 1` token IDs
live in an int64 GPU window on that same per-request state slot. The model
advances it from the actual forward input; EngineCore, Scheduler and startup
messages do not carry a token-history interface.

## Launch: FP8 on one gfx950 GPU with 288 GiB

The local validation checkpoint is `/shareddata/Qwen3.8-Flash-Next-FP8`.
PLE keeps its embedding table in FP8 with the checkpoint's global scale;
only the selected rows are dequantized to BF16. Select an available GPU;
`HIP_VISIBLE_DEVICES` uses HIP
ordinals, which need not match `rocm-smi` display order.

```bash
HIP_VISIBLE_DEVICES=0 AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /shareddata/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1 --kv-cache-dtype bf16 --block-size 64 \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,48,64]' --server-port 18389
```

The block size must divide evenly by `indexer_compress_ratio` (4). BF16 is the
only supported KV format. Use the same checkpoint path as the `model` field
in API requests; this recipe does not set a served-model alias.

Do not use an unexplained 0.98 memory-utilization setting to compensate for
profiling. Startup now executes PLE and QSA workspaces before sizing the pool.
Check the logged memory budget for the actual hardware and workload. Image
prefills are not chunked by the current multimodal scheduler, so image prompts
must fit `max-num-batched-tokens`.
This PR does not change image-prefix caching or the scheduler's preemption
policy. Their known correctness issues are handled by separate PRs.
Until those fixes are present, basic image
validation should use the existing `--no-enable_prefix_caching` option and
enough free capacity to avoid preemption. This is a validation condition, not
a new model-specific startup requirement or a claim that those bugs are fixed.

## GSM8K evaluation

Full GSM8K, all 1,319 test questions, 5-shot chat:

```bash
lm_eval --model local-chat-completions \
  --model_args 'model=/shareddata/Qwen3.8-Flash-Next-FP8,base_url=http://localhost:18389/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=900,tokenized_requests=False,trust_remote_code=True' \
  --tasks gsm8k --num_fewshot 5 \
  --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' \
  --log_samples --output_path /tmp/qwen38-gsm8k
```

Do not set `--limit` for the full evaluation. The model emits reasoning before
the final answer; 256 generated tokens can truncate it. Override GSM8K's
default `Question:` stop string, which can stop a reasoning model while it is
restating the problem. The harness evaluates final `content`, not
`reasoning_content`, and `--log_samples` saves per-question outputs. Set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` only when the dataset is already cached.
`timeout=900` allows long reasoning
requests to finish under load without hitting the harness's default five-minute
timeout; it does not change generation length or scoring. These parameters are
kept in this model's recipe; the shared GSM8K script is unchanged.

## Implementation boundaries

- Network composition and checkpoint mappings:
  [qwen4_exp.py](../atom/models/qwen4_exp.py).
- Cache geometry, shared-arena allocation and per-forward metadata:
  [qwen4_exp_attn.py](../atom/model_ops/attentions/qwen4_exp_attn.py).
- Common ATOM operations: GDN, FusedMoE, parallel embeddings/linears,
  Qwen3Next shared MLP and the Gemma RMSNorm kernel. A small Qwen-local wrapper
  supplies grouped affine weights; the public GemmaRMSNorm class is unchanged.
- Qwen-local PLE: `qwen4_exp/ple.py` contains the n-gram embedding/history,
  fused signed-sqrt gate and flat-varlen dilated convolution with persistent
  state. It does not change or extend the shared `mamba_ops/causal_conv1d.py`
  interface.
- AITER reuse: top-k, main K/V cache writes, RoPE frequency tables and split-K softmax reduction.
- Model-specific math retained: HC mixing/injection, n-gram hashing,
  mean-pool/normalize/rotate index keys, BF16 paged index scoring,
  group-to-token expansion and sparse separate-K/V GQA.

QSA uses two modules in the flat `qwen4_exp/` package:

| Module | Responsibility |
|---|---|
| `qsa_attention.py` | Attention and replicated indexer projections, RoPE and cache wiring |
| `qsa_ops.py` | Cache writes and pooling, partial mRoPE, slot mapping, index scoring/selection and sparse paged GQA; kernels and launchers together |

`qsa_ops.py` contains seven model-specific GPU kernels and reuses the shared
top-k and split-K reduction. HC and PLE remain separate modules; PLE keeps its
embedding, state updates, gate and convolution together. Qwen cache allocation
and metadata stay in its attention backend.

The local mRoPE launch replaces the reference's short indexing/multiply/add/cat
chain and supports an indexer Q-only call without dummy K. AITER's fused
norm/mRoPE/cache candidate was not adopted: its FP32 products omit HF's BF16
intermediate rounding. Norm and main KV writes still use ATOM/AITER.
`gated_ops.py` groups four local HC/gate/norm kernels. The PLE-specific
signed-sqrt gate is in `ple.py` and preserves the same intermediate casts.
GDN QKV/Z/B/A projections share one existing merged linear operator with
per-projection TP checkpoint loading.

Quantization policy lookup accepts both checkpoint and ATOM module names,
including prefix/glob/regex exclusions, without rewriting the original rules.
GDN's four input policies are checked independently for source and online
quantization before packing; an exclusion of A alone cannot hide quantized
QKV/Z/B. Its packed-name mapping is installed on the model instance for weight
loading, not on the class-level mapping used for early quant-policy remapping.
The sigmoid RMSNorm wrapper is Qwen-local and calls the existing fused kernel;
the shared `RMSNormGated` SiLU interface is unchanged.

QSA index expansion stays in `qwen4_exp/qsa_ops.py`. It is derived from the
same algorithm as GLM's pooled-index expansion, but this adaptation does not
move GLM code into a common module or change GLM's existing call path. Prefer
existing shared interfaces; cross-model extraction is deferred.

QSA's raw/compressed index caches and optional mRoPE position cache use the same
block IDs as K/V, declared as fields of one arena. PLE shares GDN's request-slot
lifetime. Selection scratch is forward-local instead of permanently allocated
once per layer. PLE hashing/convolution use flat tokens, so a ragged batch does
not allocate a `[requests, longest_request, channels]` tensor.
Hash IDs are produced by one kernel in `ple.py`, preserving signed int64
wraparound, nonnegative remainder, EOS resets and graph padding. Compressed
QSA slots are derived directly from the scheduler's physical slots in one
launch; the page-size divisibility and `block_ratio=1` contracts make another
block-table lookup redundant. These optimizations add no implementation files
and do not change the shared GDN or GLM implementations.
