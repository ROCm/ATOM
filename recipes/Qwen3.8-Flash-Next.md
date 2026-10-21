# Qwen3.8-Flash-Next Usage Guide

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a
multimodal MoE model combining Gated DeltaNet (GDN), sparse attention (QSA),
and n-gram memory. ATOM supports text and image inference with BF16 or FP8
weights and BF16 activations/KV cache.

## Preparing the environment

Follow the [ATOM installation instructions](../README.md). Use the
dependencies pinned by ATOM, including `transformers==5.16.1`, which provides
the model configuration and multimodal processor.

## Serving FP8 on one GPU

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1
```

The API listens on port 8000 by default. Use
`Qwen/Qwen3.8-Flash-Next-FP8` as the `model` in API requests.

BF16 is the only supported KV cache format. The default block size already
satisfies `indexer_compress_ratio` (4); no block-size override is needed.

Choose a GPU with enough memory for the weights, recurrent state, and KV
cache. Adjust `--max-num-seqs`, `--max-model-len`, and
`--max-num-batched-tokens` for the available memory and workload.
In particular, `--max-num-seqs` defaults to 512; lowering it reduces the
preallocated recurrent-state memory. These are resource controls, not
model-specific requirements.

## Native MTP

Use the draft weights included in the same checkpoint:

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1 \
  --method mtp --num-speculative-tokens 2 \
  --no-enable_prefix_caching
```

Draft depths 1, 2, and 3 are supported. The checkpoint contains one draft
layer, reused at each step; no separate draft model is needed. The target
and draft share the embedding and output head.

## Fused decode (mono) on MXFP4

On MI355X (gfx950), a decode step of up to 4 requests runs one FlyDSL kernel
per decoder layer (`atom/models/qwen4_exp/mono/`): the layer's QSA or
GatedDeltaNet attention, the PLE layer's embedding mix, the MoE with its
shared expert, and the hyper-connection streams. With MTP it also serves the
verify of up to 4 requests (k + 1 tokens each) and their draft steps. The
embedding is one launch and `lm_head` runs from MXFP4 weights. Prefill, the
final hyper-connection mix and sampling stay on the original path, as does
every step it does not serve (more than 4 requests). It is on by default;
`ATOM_MONO_ENABLE=0` turns it off.

It needs an MXFP4 W4A8 checkpoint: MXFP4 weights with MXFP8 activations.
`tools/qwen38_flash_next/convert_mxfp4.py` writes it from the BF16
checkpoint, without a GPU. The conversion is deterministic, and
`Qwen3.8-Flash-Next-MXFP4.sha256` lists the 63 files the results below ran
on:

```bash
hf download Qwen/Qwen3.8-Flash-Next --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --local-dir Qwen3.8-Flash-Next
python tools/qwen38_flash_next/convert_mxfp4.py --hf Qwen3.8-Flash-Next --out Qwen3.8-Flash-Next-MXFP4
cd Qwen3.8-Flash-Next-MXFP4 && sha256sum -c /path/to/ATOM/tools/qwen38_flash_next/Qwen3.8-Flash-Next-MXFP4.sha256
```

The download is 335 GiB and the result 163 GiB. `amd/Qwen3.8-Flash-Next-Quark-MXFP4`
is a different checkpoint (MXFP4 experts only); the mono decode does not
serve it.

```bash
ATOM_ENABLE_REPLAYSSM=0 python -m atom.entrypoints.openai_server \
  --model Qwen3.8-Flash-Next-MXFP4 \
  --trust-remote-code \
  -tp 1 \
  --max-num-seqs 4 --max-model-len 262141 \
  --hf-overrides '{"_multimodal_config": null}' \
  --method mtp --num-speculative-tokens 3
```

- `ATOM_ENABLE_REPLAYSSM=0`: with MTP, ReplaySSM is on by default, and the
  mono decode does not serve it.
- `--max-model-len`: at most 262144, less the draft count with MTP.
- `--hf-overrides '{"_multimodal_config": null}'`: text only.
- `--max-num-seqs 4` keeps every decode step on the fused kernels; a step of
  more requests runs the original path.
- No MTP: drop `--method` and `--num-speculative-tokens`, and
  `ATOM_ENABLE_REPLAYSSM`.

The server logs `Qwen3.8 mono decode on` once it has bound the weights. A
deployment it does not serve logs `Qwen3.8 mono decode off: <reason>` once
and runs the original path; the conditions are listed under
`ATOM_MONO_ENABLE` in the
[environment variable guide](../docs/environment_variables.md).

Measured on one MI355X (2026-10-20; ROCm 10.0, AITER c53e9b3f, FlyDSL
0.3.4.1) with `atom.benchmarks.benchmark_serving`: random prompts of 1,024
tokens and 1,024 output tokens (`--random-range-ratio 1.0 --ignore-eos`), the
serving line above with `--max-model-len 10240`, the mean of two runs of
max(10, 4 × requests) requests. Original is the same server with
`ATOM_MONO_ENABLE=0`. TPOT and the step (the median inter-token latency) are
in ms; tokens a step is the mean accepted length of a verify:

| Requests | MTP k = 3: TPOT, mono / original | Step, mono / original | Tokens a step, mono / original | No MTP: TPOT, mono / original |
|---|---|---|---|---|
| 1 | 1.67 / 4.73 | 4.46 / 14.18 | 2.94 / 3.05 | 2.75 / 9.44 |
| 2 | 2.23 / 5.58 | 6.25 / 15.17 | 2.95 / 2.80 | 3.14 / 10.31 |
| 3 | 2.86 / 5.74 | 7.63 / 16.14 | 2.90 / 2.93 | 3.54 / 12.11 |
| 4 | 3.53 / 6.68 | 8.98 / 16.80 | 2.87 / 2.71 | 3.90 / 12.43 |

GSM8K on all 1,319 questions ([the command below](#gsm8k-evaluation) with
`model=Qwen3.8-Flash-Next-MXFP4` and `num_concurrent=4`), flexible-extract:
0.9583 mono / 0.9621 original with MTP k = 3 (3.25 / 3.26 tokens a verify),
0.9575 mono without MTP; the standard error is 0.0055.

## Usage notes and limitations

- For image requests, use `--no-enable_prefix_caching` and sufficient cache
  capacity to avoid preemption until the image caching/recompute limitations
  are resolved. Image prompts must fit within `--max-num-batched-tokens`.
- Preserve the checkpoint's quantization exclusions and `ple_embedding_dtype`
  when present. GDN inputs support unquantized weights or per-channel FP8 QKV/Z
  with unquantized B/A. GDN input online quantization is not supported.
- PTPC (`compressed-tensors`) checkpoint loading is supported, but end-to-end
  execution on gfx950 is blocked by the default small-batch MoE dispatch:
  its second-stage kernel requires a 256-aligned intermediate size, whereas
  this model uses 640. The FP8 example above uses block-wise quantization.
- Chunked text prefill and full decode CUDA graphs are supported. Piecewise
  compilation/CUDA graphs are not supported.
- Only native MTP speculation is supported. Pipeline parallelism, context
  parallelism, DP attention, TBO, and external KV transfer/offload are not
  supported.
- This example covers FP8 with TP1. Pure TP2 is not currently validated;
  it can fail MoE warmup on gfx950 due to an MoE dispatch limitation.
- Video inference and long-context accuracy have not been validated.

## GSM8K evaluation

Install the evaluation client:

```bash
python -m pip install 'lm-eval[api]'
```

With the server running, evaluate all 1,319 test questions using 5-shot chat:

```bash
lm_eval --model local-chat-completions \
  --model_args 'model=Qwen/Qwen3.8-Flash-Next-FP8,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=900,tokenized_requests=False,trust_remote_code=True' \
  --tasks gsm8k --num_fewshot 5 \
  --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' \
  --log_samples --output_path ./results/gsm8k
```

The model emits reasoning before its final answer, so allow enough output
tokens to avoid truncation. The explicit stop string replaces GSM8K's default
`Question:` stop, which can prematurely end a reasoning response.
The harness scores the final `content`, not `reasoning_content`.
The timeout accommodates long requests under load; it does not change the
generation limit or scoring. Set evaluation concurrency to suit the server's
capacity.
