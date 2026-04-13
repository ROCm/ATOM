<div align="center" id="logo">
<img src="docs/assets/atom_logo.png" alt="logo" width="400" margin="10px"></img>

[![CI](https://github.com/ROCm/ATOM/actions/workflows/atom-test.yaml/badge.svg)](https://github.com/ROCm/ATOM/actions/workflows/atom-test.yaml)
[![Benchmark](https://github.com/ROCm/ATOM/actions/workflows/atom-benchmark.yaml/badge.svg)](https://github.com/ROCm/ATOM/actions/workflows/atom-benchmark.yaml)
[![Dashboard](https://img.shields.io/badge/Performance-Dashboard-blue)](https://rocm.github.io/ATOM/benchmark-dashboard/)

</div>

--------------------------------------------------------------------------------

**ATOM** (AiTer Optimized Model) is a lightweight vLLM-like implementation, focusing on integration and optimization based on [AITER](https://github.com/ROCm/aiter).

## 🚀 Features

- **ROCm Optimized**: Built on AMD's ROCm platform with [AITER](https://github.com/ROCm/aiter) kernels (ASM, CK, Triton)
- **OpenAI-Compatible API**: Drop-in server with `/v1/chat/completions` and `/v1/completions` endpoints
- **Piecewise torch.compile**: 4 compilation levels with CUDA graph capture for low-latency decode
- **Multi-GPU Parallelism**: Tensor parallelism (TP), data parallelism (DP), and expert parallelism (EP) with MORI all-to-all
- **Quantization**: FP8, MXFP4, INT8, INT4 with auto-detection from HuggingFace configs
- **Speculative Decoding**: Multi-Token Prediction (MTP) with EAGLE proposer
- **Prefix Caching**: xxhash64-based KV cache block sharing across sequences

### Supported Models

| Model Family | HF Architecture | Dense/MoE | Notes |
|---|---|---|---|
| [Llama](https://huggingface.co/meta-llama) | `LlamaForCausalLM` | Dense | Llama 2, Llama 3, Llama 3.1 |
| [Qwen3](https://huggingface.co/Qwen) | `Qwen3ForCausalLM` | Dense | |
| [Qwen3-MoE](https://huggingface.co/Qwen) | `Qwen3MoeForCausalLM` | MoE | 128 experts, top-8 routing |
| [Qwen3-Next](https://huggingface.co/Qwen) | `Qwen3NextForCausalLM` | MoE | Hybrid full attention + Gated DeltaNet |
| [DeepSeek V2/V3](https://huggingface.co/deepseek-ai) | `DeepseekV3ForCausalLM` | MoE | MLA attention, MTP speculative decoding |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | `MixtralForCausalLM` | MoE | 8 experts, top-2 routing |
| [GLM-4-MoE](https://huggingface.co/THUDM) | `Glm4MoeForCausalLM` | MoE | |
| [GLM-5](https://huggingface.co/zai-org/GLM-5-FP8) | `GlmMoeDsaForCausalLM` | MoE | MLA attention, similar to DeepSeek V3.2. See [recipe](recipes/GLM-5.md) |
| [GPT-OSS](https://huggingface.co/openai) | `GptOssForCausalLM` | MoE | Sliding window + attention sinks |
| [Kimi-K2](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | via `--trust-remote-code` | MoE | See [recipe](recipes/Kimi-K2-Thinking.md) |

## 📋 Requirements

- AMD GPU with ROCm support
- Docker

## 🛠️ Installation

### Option A: Nightly Image (Recommended)

Pre-built image with AITER + ATOM ready to use:

```bash
docker pull rocm/atom-dev:latest

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
  rocm/atom-dev:latest
```

### Option B: Build from Base ROCm Image

#### 1. Pull and run the base image

```bash
docker pull rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

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
```

#### 2. Install AITER and ATOM inside the container

```bash
pip install amd-aiter
git clone https://github.com/ROCm/ATOM.git && pip install ./ATOM
```

## 💡 Usage

### Basic Example

Before running the example, please install ninja and the Hugging Face CLI, and log in to your account.
```bash
pip install ninja
pip install -U "huggingface_hub"
hf auth login
```

The default optimization level is 3 (piecewise torch.compile with CUDA graphs).

```bash
python -m atom.examples.simple_inference --model meta-llama/Meta-Llama-3-8B --kv_cache_dtype fp8
```

> **Note:** First-time execution may take approximately 10 minutes for model compilation.

### Serving

Start an OpenAI-compatible server:

```bash
# Single GPU
python -m atom.entrypoints.openai_server --model Qwen/Qwen3-0.6B --kv_cache_dtype fp8

# Multi-GPU with tensor parallelism
python -m atom.entrypoints.openai_server --model deepseek-ai/DeepSeek-R1 --kv_cache_dtype fp8 -tp 8

# With MTP speculative decoding
python -m atom.entrypoints.openai_server --model deepseek-ai/DeepSeek-R1 --kv_cache_dtype fp8 -tp 8 \
  --method mtp --num-speculative-tokens 3
```

## 📊 Performance

### Live Benchmark Dashboard

**[rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/)**

The dashboard tracks nightly performance across models and configurations:

- **Interactive vs Throughput** — tok/s/user vs tok/s/gpu tradeoff across concurrency levels
- **Throughput & Latency trends** — Output throughput, TTFT, TPOT over time, grouped by model
- **Regression detection** — Automatic alerts when throughput drops >5% or latency increases >10%
- **Profiler trace collection** — On regression, automatically re-runs with PyTorch profiler and uploads traces

Models tracked: DeepSeek-R1-0528 (FP8 & MTP3), GLM-5-FP8, gpt-oss-120b

### Online Serving Throughput

![DS R1 Performance](./docs/assets/ds_r1_performance.png)

For more information, visit [InferenceX](https://inferencex.semianalysis.com/).

### Benchmarking

Run an online throughput benchmark against a running server:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-R1 --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=1024 --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=1280 --max-concurrency=128 \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

### Profiling & Trace Analysis

#### Collect a Trace

Launch the server with `--torch-profiler-dir` and `--mark-trace`:

```bash
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-R1 --kv_cache_dtype fp8 -tp 8 \
  --torch-profiler-dir ./trace --mark-trace
```

Collect traces via benchmark `--profile` flag (auto start/stop):

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-R1 --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random --random-input-len=1024 --random-output-len=1024 \
  --num-prompts=128 --max-concurrency=128 \
  --request-rate=inf --ignore-eos --profile
```

Or control profiling manually on a running server:

```bash
curl -X POST http://127.0.0.1:8000/start_profile
# ... run your workload ...
curl -X POST http://127.0.0.1:8000/stop_profile
```

#### Analyze the Trace

```bash
# Kernel breakdown per layer → Excel
python tools/parse_trace.py ./trace/rank_0/DeepSeek-R1_ts_*.json.gz --layer 3

# Performance summary → Markdown report
python tools/analyze_trace_summary.py ./trace/rank_0/DeepSeek-R1_ts_*.json.gz
```

| Output | Description |
|---|---|
| `prefill_breakdown.xlsx` | Per-kernel duration, call count, pct%, module grouping, cross-layer averages |
| `decode_breakdown.xlsx` | Same for decode phase, with CUDAGraph kernel mapping |
| `performance_summary.md` | Prefill/decode/draft step timing, iteration breakdown |

### Accuracy Validation

```bash
pip install lm-eval[api]

# Start server, then run evaluation
lm_eval --model local-completions \
  --model_args model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5
```

## 📚 Documentation

**Full documentation: [rocm.github.io/ATOM/docs](https://rocm.github.io/ATOM/docs)**

| Topic | Description | Guide |
|---|---|---|
| Architecture | System overview, request lifecycle, component design | [Architecture Guide](docs/architecture_guide.md) |
| Configuration | Config classes, CLI arguments, environment variables | [Configuration Guide](docs/configuration_guide.md) |
| Model Support | Supported models, weight loading, adding new architectures | [Model Support Guide](docs/model_support_guide.md) |
| Model Operations | AITER kernel integration, linear/attention/MoE/norm wrappers | [Model Ops Guide](docs/model_ops_guide.md) |
| Scheduling & KV Cache | Batch scheduling, block allocation, prefix caching | [Scheduling Guide](docs/scheduling_kv_cache_guide.md) |
| Compilation | torch.compile levels, CUDA graphs, piecewise compilation | [Compilation Guide](docs/compilation_cudagraph_guide.md) |
| Distributed | Tensor/data/expert parallelism, multi-GPU deployment | [Distributed Guide](docs/distributed_guide.md) |
| Serving & Benchmarks | OpenAI API server, benchmarking, profiling, speculative decoding | [Serving Guide](docs/serving_benchmarking_guide.md) |
| Environment Variables | All `ATOM_*` variable definitions | [Env Vars](docs/environment_variables.md) |

**Deployment Recipes:**

- [DeepSeek-R1](recipes/DeepSeek-R1.md) — FP8/MXFP4 with MTP speculative decoding on 8 GPUs
- [Qwen3-235B-A22B](recipes/Qwen3-235b.md) — TP8 + EP with FP8 KV cache
- [Qwen3-Next](recipes/Qwen3-Next.md) — Hybrid GDN + MoE architecture
- [Kimi-K2-Thinking](recipes/Kimi-K2-Thinking.md) — MXFP4 MoE on 4 GPUs
- [GLM-5](recipes/GLM-5.md) — FP8 MoE with MLA on 8 GPUs
- [GPT-OSS-120B](recipes/GPT-OSS.md) — Single GPU or DP+EP on 2 GPUs

**Framework Integration:**

- [vLLM Plugin Backend](docs/vllm_plugin_backend_guide.md) — ATOM as the out-of-tree plugin backend for vLLM

## Acknowledgements

This project was adapted from [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## Support & Reporting Issues

We welcome issues and contributions! Please use the GitHub Issues page to report bugs or request features: https://github.com/ROCm/ATOM/issues
