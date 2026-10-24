# Muse Spark Open Usage Guide

[Muse Spark Open](https://huggingface.co/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4) is an 816B-total / 42B-active Mixture-of-Experts model (256 routed experts, top-8, 62 layers). ATOM serves the text stack of the multimodal checkpoint via `MuseSparkForConditionalGeneration`; the vision and audio towers are skipped at load time.

Architecture points that matter for serving:
- 46 of 62 layers use a 2048-token sliding window; the other 16 are full-attention NoPE layers.
- Gated attention (an extra query-sized output gate) with weightless per-head q/k/o RMSNorm.
- Each expert output is RMS-normalized before the routing-weighted sum, so the experts run as a top-1 FusedMoE over tokens repeated top-k times.
- fp32 residual stream with norm-preserving gated residuals; fp32 logits with `output_multiplier` and a tanh softcap of 20.
- The AMD Quark checkpoint quantizes only the routed experts to MXFP4 (layers 0 and 61 keep BF16 experts); attention, router and the expert projections stay BF16 through the checkpoint's exclude list.

## Preparing environment

Muse Spark support is in ATOM main from #2641 on, and not in a release image yet. Use a nightly dev image built after that change:
```bash
docker pull rocm/atom-dev:latest
```
All the operations below will be executed inside the container.

## Launching server

### TP4

The MXFP4 checkpoint is ~509 GB and fits in four MI355X GPUs (~136 GB of weights per GPU):

```bash
python -m atom.entrypoints.openai_server \
  --model amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
  --kv_cache_dtype bf16 -tp 4 \
  --max-model-len 32768
```

### TP8, optionally with expert parallel

We recommend TP8 with expert parallel. Under plain TP each layer all-reduces the per-slot expert outputs (`[tokens * top_k, 4096]`, about 4x the bytes of a usual MoE reduce); with EP the combine is local and only `[tokens, 4096]` is reduced (about 1/8 of that).

```bash
python -m atom.entrypoints.openai_server \
  --model amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
  --kv_cache_dtype bf16 -tp 8 \
  --max-model-len 32768 [--enable-expert-parallel]
```

Tips on server configuration:
- Pass `--max-model-len`: the config advertises a 1M-token context.
- Only the bf16 KV cache is validated so far.
- There is no speculative decoding yet: the checkpoint has no MTP layers, and the DFlash2 draft (`RadixArk/Muse-Spark-Open-DFlash2`) is not supported in ATOM.
- There is no Muse reasoning parser, so chat completions return the reasoning text in `content`.
- The tokenizer does not add `<|begin_of_text|>` to raw completion prompts (the same as other serving stacks with this tokenizer).

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=0.8 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

The current implementation is accuracy-first (eager torch arithmetic around the aiter attention and MoE kernels).

### Accuracy test

We verified the lm_eval accuracy on the gsm8k dataset with this command:
```bash
lm_eval --model local-completions \
        --model_args model=amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 5
```

Measured on MI355X (1319 questions, flexible-extract, bf16 KV). Compiled is the default level 3 + CUDA graphs; eager is `--enforce-eager --level 0`:

| Config | Compiled 3-shot | Compiled 5-shot | Eager 5-shot |
|---|---|---|---|
| TP4 | 0.9409 | 0.9439 | 0.9439 |
| TP8 | | 0.9431 | 0.9484 |
| TP8 + expert parallel | | 0.9431 | 0.9424 |

CI accuracy threshold: `flexible-extract ≥ 0.92` (3-shot, TP4).
