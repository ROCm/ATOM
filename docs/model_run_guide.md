# Model Run Guide

Ready-to-use commands for serving models on ATOM with AMD Instinct MI355X / MI300X GPUs. Each model recipe below is validated in nightly CI.

## Quick Start

```bash
# Pull the latest ATOM container
docker pull rocm/atom:latest

# Start the container
docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16G \
  --privileged --cap-add=SYS_PTRACE \
  -e HF_TOKEN=$HF_TOKEN \
  -p 8000:8000 \
  rocm/atom:latest
```

## Supported Models

| Model | Type | Precision | TP | Recipe |
|-------|------|-----------|-----|--------|
| DeepSeek-R1-0528 | MoE + MLA | FP8 / MXFP4 | 8 | [recipes/DeepSeek-R1.md](../recipes/DeepSeek-R1.md) |
| GLM-5 | MoE + MLA | FP8 | 8 | [recipes/GLM-5.md](../recipes/GLM-5.md) |
| GPT-OSS-120B | MoE | FP8 | 1 | [recipes/GPT-OSS.md](../recipes/GPT-OSS.md) |
| Kimi-K2.5 | MoE | MXFP4 | 4 | [recipes/Kimi-K2.5.md](../recipes/Kimi-K2.5.md) |
| Kimi-K2-Thinking | MoE | FP8 | 8 | [recipes/Kimi-K2-Thinking.md](../recipes/Kimi-K2-Thinking.md) |
| Qwen3-235B | MoE | FP8 | 8 | [recipes/Qwen3-235b.md](../recipes/Qwen3-235b.md) |
| Qwen3-Next | MoE | FP8 | 8 | [recipes/Qwen3-Next.md](../recipes/Qwen3-Next.md) |

### vLLM Plugin Backend

ATOM also runs as a vLLM plugin backend. See recipes under [recipes/atom_vllm/](../recipes/atom_vllm/) for vLLM-integrated serving.

## Nightly CI Benchmark Configurations

The nightly CI sweeps these configurations for every model:

| ISL | OSL | Concurrency Levels |
|-----|-----|--------------------|
| 1024 | 1024 | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| 8192 | 1024 | 1, 2, 4, 8, 16, 32, 64, 128, 256 |

Run a benchmark against a running ATOM server:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model <model_name_or_path> \
  --backend vllm --base-url http://localhost:8000 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --max-concurrency 128 --num-prompts 1280 \
  --random-range-ratio 0.8 \
  --request-rate inf --ignore-eos
```

Key parameters:
- `--random-range-ratio 0.8` — adds ±20% jitter to sequence lengths
- `--num-prompts` — typically `concurrency × 10`
- `--request-rate inf` — closed-loop benchmarking (no inter-request delay)
- `--ignore-eos` — forces full output length generation

## Live Dashboard

Nightly benchmark results are published to the [ATOM Benchmark Dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/).

Competitive comparison (MI355X vs B200/B300) is available on the [AI Frameworks Dashboard](https://rocm.github.io/AI-Frameworks-Dashboard/atom-benchmark/).
