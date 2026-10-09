# Qwen3.8-2.4T with ATOM SGLang Plugin

[Qwen3.8-2.4T-A95B-Quark-MXFP4](https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4)
is a text-only MoE model using the `Qwen3_5MoeForCausalLM` architecture. It
combines Gated DeltaNet (GDN) linear-attention layers with periodic full MHA
layers.

The validated configuration requires eight MI355 (gfx950) GPUs with TP8. The
Quark MXFP4 checkpoint uses approximately 180 GB of model memory per GPU,
leaving enough memory for an FP8 KV cache. The BF16 checkpoint does not fit on
one 8-GPU MI355 node.

## Preparing Environment

Pull the latest SGLang-ATOM image:

```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container with all eight GPUs visible and run the remaining commands
inside it.

## Launching Server

```bash
MODEL_PATH=${MODEL_PATH:-amd/Qwen3.8-2.4T-A95B-Quark-MXFP4}
PORT=${PORT:-8000}
TP=${TP:-8}

export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export SGLANG_USE_AITER=1
export AITER_LOG_LEVEL=WARNING

# Use the Native ATOM FP8 pure-prefill MHA path when its dispatch constraints
# are satisfied.
export ATOM_AITER_FP8_PREFILL_ATTN=1

# Keep tc_piecewise FX/Inductor compilation while executing prefill without
# per-bucket CUDA Graph capture, token padding, or static input staging.
export ATOM_SGLANG_PREFILL_COMPILE_ONLY=1

export TORCHINDUCTOR_COMPILE_THREADS=128
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.9 \
    --max-running-requests 256 \
    --context-length 32768 \
    --disable-radix-cache \
    --page-size 16 \
    --reasoning-parser qwen3 \
    --enable-torch-compile \
    --torch-compile-max-bs 64 \
    --cuda-graph-backend-prefill tc_piecewise \
    --cuda-graph-max-bs-prefill 16384 \
    --cuda-graph-bs-prefill 8192 16384 \
    --cuda-graph-tc-compiler inductor \
    2>&1 | tee qwen38-2.4t-tp8-sglang-server.log
```

SGLang continues to own request scheduling and KV/Mamba pool allocation. The
ATOM plugin binds those pools as zero-copy Native ATOM cache views and executes
full MHA and GDN with ATOM attention implementations.

`ATOM_SGLANG_PREFILL_COMPILE_ONLY=1` is optional. Set it to `0` to use the real
prefill tc_piecewise CUDA Graph path; that path uses graph-stable mutable
attention outputs between captured segments.

Prefix caching and speculative decoding are not validated for this
configuration and should remain disabled.

## Smoke Test

```bash
curl "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL_PATH}\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"What is 17 + 25?\"}
      ],
      \"max_tokens\": 128,
      \"temperature\": 0,
      \"min_tokens\": 1,
      \"reasoning_effort\": \"low\"
    }"
```

The response should conclude that the answer is `42`.

## Accuracy Validation

Run the same 5-shot GSM8K chat-completions protocol used by the SGLang nightly
CI:

```bash
OUTPUT_PATH=${OUTPUT_PATH:-./qwen38-sglang-gsm8k}

lm_eval \
    --model local-chat-completions \
    --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=1200,max_gen_toks=8192,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --output_path "${OUTPUT_PATH}" \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_tokens=1,reasoning_effort=low" \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --trust_remote_code
```

The initial SGLang-ATOM CI threshold is `0.97`. Recalibrate the threshold only
after recording a complete SGLang-specific baseline with the same server and
evaluation settings.

## Serving Benchmark

The nightly SGLang benchmark CI covers the standard `1024x1024` and
`8192x1024` workloads at concurrency 4, 8, 16, 32, and 64. The catalog also
supports the `8192x64` / concurrency-64 workload used during Native ATOM
performance alignment; select it manually with:

```text
param_lists=8192,64,64,1.0
```

The equivalent local command is:

```bash
ISL=${ISL:-8192}
OSL=${OSL:-64}
CONC=${CONC:-64}
NUM_PROMPTS=${NUM_PROMPTS:-64}

python -m atom.benchmarks.benchmark_serving \
    --model="${MODEL_PATH}" \
    --backend=sglang \
    --base-url="http://127.0.0.1:${PORT}" \
    --dataset-name=random \
    --random-input-len="${ISL}" \
    --random-output-len="${OSL}" \
    --random-range-ratio=1.0 \
    --num-prompts="${NUM_PROMPTS}" \
    --max-concurrency="${CONC}" \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics="ttft,tpot,itl,e2el"
```

## Current Scope

- Text generation with the Quark MXFP4 checkpoint.
- TP8 on one 8x MI355/gfx950 node.
- FP8 KV cache with 16-token pages.
- Native ATOM MHA and GDN under the SGLang scheduler.
- Full decode CUDA Graph with Torch Compile coverage through batch size 64.
- tc_piecewise Inductor prefill, optionally executed in compile-only mode.
- Radix/prefix cache and speculative decoding disabled.
