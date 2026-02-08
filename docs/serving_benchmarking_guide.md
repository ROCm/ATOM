# ATOM Serving & Benchmarking Guide

ATOM (AiTer Optimized Model) is AMD's lightweight LLM inference engine built on
[AITER](https://github.com/ROCm/aiter) kernels for ROCm/HIP GPUs.  This guide
covers the OpenAI-compatible serving API, programmatic engine usage, benchmarking
tools, profiling, and speculative decoding.

---

## Quick Reference

```bash
# Start the OpenAI-compatible server
python -m atom.entrypoints.openai_server --model <model_name_or_path> --kv_cache_dtype fp8

# Run the online serving benchmark
python -m atom.benchmarks.benchmark_serving \
    --backend vllm --model <model_name_or_path> \
    --base-url http://localhost:8000 \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --num-prompts 1000 --request-rate inf --ignore-eos

# Simple inference example
python -m atom.examples.simple_inference --model <model_name_or_path> --kv_cache_dtype fp8

# Offline profiling
python -m atom.examples.profile_offline --model <model_name_or_path> --kv_cache_dtype fp8

# Accuracy validation with lm-eval
lm_eval --model local-completions \
    --model_args model=<model>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
    --tasks gsm8k --num_fewshot 5
```

---

## 1. OpenAI-Compatible Server

The server is implemented in `atom/entrypoints/openai_server.py` using FastAPI
and Uvicorn.  It exposes OpenAI-compatible HTTP endpoints so that existing
clients (curl, OpenAI SDK, lm-eval) work without modification.

### 1.1 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (ChatCompletionRequest -> ChatCompletionResponse) |
| `POST` | `/v1/completions` | Text completion (CompletionRequest -> CompletionResponse) |
| `GET`  | `/v1/models` | List available models |
| `GET`  | `/health` | Health check (returns `{"status": "ok"}`) |
| `POST` | `/start_profile` | Start torch profiler on the engine |
| `POST` | `/stop_profile` | Stop torch profiler and flush traces |

### 1.2 Request Models

**ChatCompletionRequest** fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `Optional[str]` | `None` | Model name (validated against the loaded model) |
| `messages` | `Optional[List[ChatMessage]]` | `None` | List of chat messages (`role`, `content`) |
| `prompt` | `Optional[List[ChatMessage]]` | `None` | Alias for `messages` |
| `temperature` | `Optional[float]` | `1.0` | Sampling temperature |
| `top_p` | `Optional[float]` | `1.0` | Nucleus sampling threshold |
| `max_tokens` | `Optional[int]` | `256` | Maximum tokens to generate |
| `stop` | `Optional[List[str]]` | `None` | Stop strings |
| `ignore_eos` | `Optional[bool]` | `False` | Ignore end-of-sequence token |
| `stream` | `Optional[bool]` | `False` | Enable server-sent events streaming |
| `seed` | `Optional[int]` | `None` | Random seed |

**CompletionRequest** fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `Optional[str]` | `None` | Model name |
| `prompt` | `str` | (required) | Text prompt |
| `temperature` | `Optional[float]` | `1.0` | Sampling temperature |
| `top_p` | `Optional[float]` | `1.0` | Nucleus sampling threshold |
| `max_tokens` | `Optional[int]` | `256` | Maximum tokens to generate |
| `stop` | `Optional[List[str]]` | `None` | Stop strings |
| `ignore_eos` | `Optional[bool]` | `False` | Ignore end-of-sequence token |
| `stream` | `Optional[bool]` | `False` | Enable SSE streaming |

### 1.3 Response Models

Both `ChatCompletionResponse` and `CompletionResponse` include:

- `id` -- unique request identifier (e.g. `chatcmpl-<uuid>` or `cmpl-<uuid>`)
- `object` -- `"chat.completion"` or `"text_completion"`
- `created` -- Unix timestamp
- `model` -- model name
- `choices` -- list of generated completions
- `usage` -- token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`)
  plus `ttft_s`, `tpot_s`, and `latency_s` timing fields

Streaming responses use the SSE (Server-Sent Events) protocol with
`data: [DONE]\n\n` as the termination signal.

### 1.4 Server Startup

```bash
python -m atom.entrypoints.openai_server \
    --model <model_name_or_path> \
    --kv_cache_dtype fp8 \
    --host 0.0.0.0 \
    --server-port 8000
```

Server-specific CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--server-port` | `8000` | HTTP port (note: `--port` is for internal engine communication) |

All `EngineArgs` arguments are also accepted (see Section 7 for the full list).

### 1.5 Example: curl

```bash
# Non-streaming chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'

# Streaming text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 64,
    "stream": true
  }'
```

---

## 2. Programmatic API (LLMEngine)

The `LLMEngine` class in `atom/model_engine/llm_engine.py` provides a
Python-native interface for inference without running an HTTP server.

### 2.1 Initialization

```python
from atom import LLMEngine, SamplingParams

engine = LLMEngine(model="deepseek-ai/DeepSeek-R1", kv_cache_dtype="fp8",
                   tensor_parallel_size=8)
```

`LLMEngine.__init__(model, **kwargs)` accepts all `Config` field names as
keyword arguments (e.g. `tensor_parallel_size`, `kv_cache_dtype`,
`max_model_len`, `data_parallel_size`, `gpu_memory_utilization`).

### 2.2 SamplingParams

Defined in `atom/sampling_params.py`:

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_strings: Optional[list[str]] = None
```

### 2.3 Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `(prompts: list[str], sampling_params) -> list[dict]` | Synchronous batch generation; blocks until all prompts complete |
| `add_request` | `(prompt_or_tokens_list, sampling_params_list, stream_callback=None)` | Submit requests for asynchronous processing |
| `step` | `() -> list[Sequence]` | Retrieve completed sequences |
| `is_finished` | `() -> bool` | Check whether all pending requests have completed |
| `start_profile` | `()` | Start torch profiler on all workers |
| `stop_profile` | `()` | Stop torch profiler and write traces |
| `print_mtp_statistics` | `()` | Print speculative decoding acceptance statistics |

### 2.4 Synchronous Generation Example

```python
from atom import LLMEngine, SamplingParams

engine = LLMEngine(model="meta-llama/Meta-Llama-3-8B", kv_cache_dtype="fp8")
params = SamplingParams(temperature=0.6, max_tokens=256)

outputs = engine.generate(["Explain quantum computing in simple terms."], params)
for out in outputs:
    print(out["text"])
```

Each output dictionary contains: `text`, `token_ids`, `latency`,
`finish_reason`, `num_tokens_input`, `num_tokens_output`, `ttft`, and `tpot`.

### 2.5 Asynchronous / Streaming Usage

```python
engine.add_request(
    prompt_or_tokens_list=["Hello world", "How are you?"],
    sampling_params_list=SamplingParams(temperature=0.8, max_tokens=128),
    stream_callback=my_callback,  # called per-token with RequestOutput
)

while not engine.is_finished():
    completed = engine.step()
    # process completed sequences
```

---

## 3. Simple Inference

The `atom/examples/simple_inference.py` script provides a quick way to validate
model loading and generation.

### 3.1 Usage

```bash
python -m atom.examples.simple_inference \
    --model meta-llama/Meta-Llama-3-8B \
    --kv_cache_dtype fp8 \
    --temperature 0.6
```

### 3.2 What It Does

1. Parses all `EngineArgs` plus `--temperature` (default `0.6`).
2. Creates an `LLMEngine` via `EngineArgs.from_cli_args(args).create_engine()`.
3. Applies the model's chat template to four built-in prompts (English and
   Chinese) with `enable_thinking=True`.
4. Runs a warmup generation, then generates completions for the batch.
5. Calls `llm.print_mtp_statistics()` to report speculative decoding stats
   (if MTP is enabled).

---

## 4. Benchmarking

ATOM ships a comprehensive online serving benchmark in
`atom/benchmarks/benchmark_serving.py` (adapted from vLLM's benchmarking
tooling).

### 4.1 Metrics

The `BenchmarkMetrics` dataclass tracks:

| Metric | Abbreviation | Description |
|--------|--------------|-------------|
| Time to First Token | **TTFT** | Latency from request submission to the first generated token |
| Time per Output Token | **TPOT** | Average latency per output token (excluding the first) |
| Inter-Token Latency | **ITL** | Latency between successive output tokens |
| End-to-End Latency | **E2EL** | Total latency from request send to full response receipt |
| Request Throughput | -- | Completed requests per second |
| Output Token Throughput | -- | Generated tokens per second |
| Total Token Throughput | -- | (input + output) tokens per second |
| Request Goodput | -- | Requests per second meeting SLO targets |

For each latency metric, mean, median, standard deviation, and configurable
percentiles (default: P99) are reported.

### 4.2 Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backend` | `vllm` | Backend type. Choices: `tgi`, `vllm`, `lmdeploy`, `deepspeed-mii`, `openai`, `openai-chat`, `tensorrt-llm`, `scalellm`, `sglang` |
| `--model` | (required) | Model name or path |
| `--base-url` | `None` | Server base URL (e.g. `http://localhost:8000`) |
| `--host` | `127.0.0.1` | Server host (used when `--base-url` is not set) |
| `--port` | `8000` | Server port (used when `--base-url` is not set) |
| `--endpoint` | `/v1/completions` | API endpoint path |
| `--dataset-name` | `sharegpt` | Dataset type: `sharegpt`, `burstgpt`, `sonnet`, `random`, `hf` |
| `--dataset-path` | `None` | Path to dataset file or HuggingFace dataset ID |
| `--num-prompts` | `1000` | Number of prompts to benchmark |
| `--request-rate` | `inf` | Requests per second (`inf` = send all at once) |
| `--burstiness` | `1.0` | Burstiness factor (1.0 = Poisson process) |
| `--max-concurrency` | `None` | Maximum concurrent requests |
| `--ignore-eos` | `False` | Ignore EOS token in generation |
| `--save-result` | `False` | Save results to JSON |
| `--result-dir` | `None` | Directory for result JSON files |
| `--result-filename` | `None` | Custom filename for results |
| `--percentile-metrics` | `ttft,tpot,itl` | Comma-separated metrics to report percentiles for |
| `--metric-percentiles` | `99` | Comma-separated percentile values (e.g. `25,50,75,99`) |
| `--goodput` | `None` | SLO targets as `KEY:VALUE` pairs (e.g. `ttft:100 tpot:50`) |
| `--profile` | `False` | Enable torch profiler during the benchmark run |
| `--tokenizer` | `None` | Custom tokenizer name or path |
| `--seed` | `0` | Random seed |

**Random dataset options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--random-input-len` | `1024` | Input token length |
| `--random-output-len` | `128` | Output token length |
| `--random-range-ratio` | `1.0` | Length variation ratio |
| `--random-prefix-len` | `0` | Fixed prefix token length |
| `--use-chat-template` | `False` | Apply chat template to random prompts |

### 4.3 Backend Request Functions

Defined in `atom/benchmarks/backend_request_func.py`:

| Backend Key | Function | Protocol |
|-------------|----------|----------|
| `vllm` | `async_request_openai_completions` | OpenAI Completions API (streaming) |
| `openai` | `async_request_openai_completions` | OpenAI Completions API (streaming) |
| `openai-chat` | `async_request_openai_chat_completions` | OpenAI Chat Completions API (streaming) |
| `tgi` | `async_request_tgi` | TGI `generate_stream` |
| `tensorrt-llm` | `async_request_trt_llm` | TRT-LLM `generate_stream` |
| `deepspeed-mii` | `async_request_deepspeed_mii` | DeepSpeed-MII |
| `lmdeploy` | `async_request_openai_completions` | OpenAI Completions API |
| `scalellm` | `async_request_openai_completions` | OpenAI Completions API |
| `sglang` | `async_request_openai_completions` | OpenAI Completions API |

Each function uses `RequestFuncInput` and returns a `RequestFuncOutput` with
timing data (`ttft`, `itl`, `latency`, `tpot`).

### 4.4 Full Benchmark Example

```bash
# 1. Start the server
python -m atom.entrypoints.openai_server \
    --kv_cache_dtype fp8 -tp 8 --model deepseek-ai/DeepSeek-R1

# 2. Run benchmark
MODEL=deepseek-ai/DeepSeek-R1
ISL=1024
OSL=1024
CONC=128
PORT=8000
RESULT_FILENAME=Deepseek-R1-result

python -m atom.benchmarks.benchmark_serving \
    --model=$MODEL --backend=vllm --base-url=http://localhost:$PORT \
    --dataset-name=random \
    --random-input-len=$ISL --random-output-len=$OSL \
    --random-range-ratio 0.8 \
    --num-prompts=$(( $CONC * 10 )) \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=./ --result-filename=$RESULT_FILENAME.json
```

---

## 5. Profiling

ATOM supports PyTorch profiling via environment variables, HTTP endpoints, and
the programmatic API.

### 5.1 Configuration

| Mechanism | Description |
|-----------|-------------|
| `--torch-profiler-dir <dir>` | CLI arg to set the trace output directory |
| `ATOM_TORCH_PROFILER_DIR` env var | Sets the default `torch_profiler_dir` in `Config` |
| `ATOM_PROFILER_MORE=1` env var | Enables detailed profiling: `record_shapes`, `with_stack`, `profile_memory` |

When a profiler directory is configured, each worker saves traces to a
rank-specific subdirectory:

- Multi-GPU with DP: `{profiler_dir}/dp{dp_rank}_tp{rank}/`
- Single-GPU / TP-only: `{profiler_dir}/rank_{rank}/`

Traces are saved in gzip-compressed TensorBoard format and can be viewed with
`tensorboard --logdir <profiler_dir>` or Chrome's `chrome://tracing`.

### 5.2 Online Profiling (HTTP)

While the server is running, start and stop profiling with HTTP requests:

```bash
# Start profiling
curl -s -S -X POST http://127.0.0.1:8000/start_profile

# ... run your workload ...

# Stop profiling and flush traces
curl -s -S -X POST http://127.0.0.1:8000/stop_profile
```

The server must be started with `--torch-profiler-dir` or with
`ATOM_TORCH_PROFILER_DIR` set for these endpoints to produce traces.

### 5.3 Programmatic Profiling

```python
engine = LLMEngine(model="Qwen/Qwen3-0.6B", torch_profiler_dir="./traces")

engine.start_profile()
outputs = engine.generate(prompts, sampling_params)
engine.stop_profile()
# Traces written to ./traces/rank_0/
```

### 5.4 Offline Profiling Script

`atom/examples/profile_offline.py` provides a self-contained offline profiling
workflow:

```bash
python -m atom.examples.profile_offline \
    --model Qwen/Qwen3-0.6B \
    --kv_cache_dtype fp8 \
    --torch-profiler-dir ./profiler_traces \
    --input-length 128 \
    --output-length 32 \
    --bs 4
```

Script-specific arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-length` | `128` | Approximate input prompt length in tokens |
| `--output-length` | `32` | Output generation length in tokens |
| `--bs` | `1` | Batch size (number of parallel requests) |
| `--random-input` | `False` | Use random token input instead of predefined text |

If `--torch-profiler-dir` is not specified, the script defaults to
`./profiler_traces`.

### 5.5 Profiling During Benchmarks

The benchmark tool can trigger profiling automatically via `--profile`:

```bash
python -m atom.benchmarks.benchmark_serving \
    --model <model> --backend vllm \
    --base-url http://localhost:8000 \
    --dataset-name random --num-prompts 100 \
    --profile
```

This sends `POST /start_profile` before the benchmark and
`POST /stop_profile` after completion.

---

## 6. Speculative Decoding (MTP)

ATOM supports Multi-Token Prediction (MTP) for DeepSeek models using the
Eagle-style speculative decoding framework.

### 6.1 Architecture

- **EagleProposer** (`atom/spec_decode/eagle.py`): Loads and runs the draft
  (MTP) model to propose speculative tokens.  Supports the `DeepSeekMTPModel`
  architecture via `DeepSeekMTP`.
- **RejectionSampler** (`atom/model_ops/rejection_sampler.py`): Implements
  greedy rejection sampling with a Triton kernel.  Compares draft token IDs
  against target model argmax and accepts matching prefixes; appends a bonus
  token if all drafts are accepted.

### 6.2 Configuration

Enable MTP via CLI arguments:

```bash
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    --kv_cache_dtype fp8 -tp 8 \
    --method mtp \
    --num-speculative-tokens 1
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `None` | Speculative method; currently only `mtp` is supported |
| `--num-speculative-tokens` | `1` | Number of draft tokens per iteration (draft model runs this many autoregressive steps) |

### 6.3 MTP Statistics

ATOM tracks acceptance statistics at runtime:

- **total_draft_tokens**: Total number of draft tokens proposed
- **total_accepted_tokens**: Number of draft tokens accepted by rejection sampling
- **acceptance_rate**: Ratio of accepted to draft tokens

Statistics are logged every 1000 draft tokens and can be printed on demand:

```python
engine.print_mtp_statistics()
```

Example output:
```
MTP Statistics:
  Total draft tokens: 5000
  Accepted tokens:    4250
  Acceptance rate:    85.00%
```

### 6.4 How Rejection Sampling Works

1. The draft model generates `num_speculative_tokens` token predictions
   autoregressively using argmax.
2. The target model verifies all draft tokens in a single forward pass.
3. The `rejection_greedy_sample_kernel` (Triton) compares each draft token
   against the target model's argmax:
   - If they match, the token is accepted.
   - On the first mismatch, the target model's token replaces it and all
     subsequent draft tokens are discarded.
   - If all draft tokens match, a bonus token from the target model is
     appended.

---

## 7. Deployment Examples

### 7.1 Single-GPU

```bash
python -m atom.entrypoints.openai_server \
    --model Qwen/Qwen3-0.6B \
    --kv_cache_dtype fp8
```

### 7.2 Multi-GPU with Tensor Parallelism

```bash
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    --kv_cache_dtype fp8 \
    -tp 8
```

### 7.3 Docker Deployment

```bash
# Pull the ROCm PyTorch image
docker pull rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Launch container
docker run -it --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME:/home/$USER \
    -v /mnt:/mnt \
    -v /data:/data \
    --shm-size=16G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Inside the container
pip install amd-aiter
git clone https://github.com/ROCm/ATOM.git && cd ATOM && pip install .

# Start serving
python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-R1 \
    --kv_cache_dtype fp8 -tp 8
```

### 7.4 Engine CLI Arguments (EngineArgs)

These arguments are available for all entrypoints (server, examples, and any
script using `EngineArgs.add_cli_args`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3-0.6B` | Model name or path |
| `--trust-remote-code` | `False` | Trust remote code from HuggingFace |
| `--tensor-parallel-size`, `-tp` | `1` | Tensor parallel size |
| `--data-parallel-size`, `-dp` | `1` | Data parallel size |
| `--enforce-eager` | `False` | Disable CUDA graph capture; use eager execution |
| `--enable_prefix_caching` | `False` | Enable prefix caching |
| `--port` | `8006` | Internal engine communication port |
| `--kv_cache_dtype` | `bf16` | KV cache dtype: `bf16` or `fp8` |
| `--block-size` | `16` | KV cache block size |
| `--max-model-len` | `None` | Maximum context length (defaults to HF config) |
| `--max-num-batched-tokens` | `16384` | Maximum tokens per batch |
| `--max-num-seqs` | `512` | Maximum sequences per batch |
| `--gpu-memory-utilization` | `0.9` | GPU memory utilization (0.0 to 1.0) |
| `--scheduler-delay-factor` | `0.0` | Delay factor before scheduling next prompt |
| `--cudagraph-capture-sizes` | `[1,2,4,...,256]` | Batch sizes for CUDA graph capture |
| `--level` | `3` | Compilation level (0-3); 3 = torch.compile |
| `--load_dummy` | `False` | Skip loading model weights (for testing) |
| `--enable-expert-parallel` | `False` | Enable expert parallelism for MoE |
| `--enable-dp-attention` | `False` | Enable data-parallel attention |
| `--torch-profiler-dir` | `None` | Directory for torch profiler traces |
| `--method` | `None` | Speculative decoding method (`mtp`) |
| `--num-speculative-tokens` | `1` | Number of speculative tokens per step |

---

## 8. Accuracy Validation

ATOM supports accuracy validation through the
[lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) framework via
the OpenAI-compatible API.

### 8.1 Setup

```bash
pip install lm-eval[api]
```

### 8.2 Run Evaluation

Start an ATOM server, then run lm-eval against it:

```bash
# Start server
python -m atom.entrypoints.openai_server \
    --model meta-llama/Meta-Llama-3-8B \
    --kv_cache_dtype fp8

# Run evaluation
lm_eval --model local-completions \
    --model_args model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
    --tasks gsm8k \
    --num_fewshot 5
```

Any lm-eval task can be used.  The `local-completions` model type sends
requests to the `/v1/completions` endpoint, making it compatible with the ATOM
server without modification.

---

## Source Files

| File | Description |
|------|-------------|
| `atom/entrypoints/openai_server.py` | OpenAI-compatible API server (FastAPI + Uvicorn) |
| `atom/model_engine/llm_engine.py` | `LLMEngine` programmatic API |
| `atom/sampling_params.py` | `SamplingParams` dataclass |
| `atom/model_engine/arg_utils.py` | `EngineArgs` CLI argument definitions and engine factory |
| `atom/examples/simple_inference.py` | Simple batch inference example |
| `atom/examples/profile_offline.py` | Offline profiling tool |
| `atom/benchmarks/benchmark_serving.py` | Online serving benchmark (`BenchmarkMetrics`, dataset sampling, result reporting) |
| `atom/benchmarks/backend_request_func.py` | Async HTTP request functions for each backend (`RequestFuncInput`, `RequestFuncOutput`, `ASYNC_REQUEST_FUNCS`) |
| `atom/benchmarks/benchmark_utils.py` | `convert_to_pytorch_benchmark_format` utility |
| `atom/spec_decode/eagle.py` | `EagleProposer` -- MTP draft model for DeepSeek speculative decoding |
| `atom/model_ops/rejection_sampler.py` | `RejectionSampler` with Triton greedy rejection kernel |
| `atom/config.py` | `Config`, `CompilationConfig`, `SpeculativeConfig` dataclasses |
| `atom/model_engine/model_runner.py` | `ModelRunner` with `start_profiler`/`stop_profiler` and MTP statistics |
