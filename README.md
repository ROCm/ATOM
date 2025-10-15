# ATOM

**ATOM** (AI Tensor Offline Model) is a lightweight vLLM-like implementation, focusing on integration and optimization based on [aiter](https://github.com/ROCm/aiter).

## 🚀 Features

- **ROCm Optimized**: Built on AMD's ROCm platform with torch compile support
- **Model Support**: Compatible with Qwen, Llama, and Mixtral models
- **Easy Integration**: Simple API for quick deployment

## 📋 Requirements

- AMD GPU with ROCm support
- Docker

## 🛠️ Installation

### 1. Pull Docker Image

```bash
docker pull rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

### 2. Run Docker Container

```bash
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

### 3. Clone and Setup

```bash
git clone https://github.com/valarLip/atom.git
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
cd ../atom
pip install .
```

## 💡 Usage

### Basic Example

The default optimization level is 3 (running with torch compile). Supported models include **Qwen**, **Llama**, and **Mixtral**.

```bash
python -m atom.bench.example.py --model meta-llama/Meta-Llama-3-8B
```

> **Note:** First-time execution may take approximately 10 minutes for model compilation.

### Performance Benchmarking

Run performance tests to compare ATOM against vLLM:

```bash
python -m atom.bench.one_batch.py --model Qwen/Qwen3-0.6B
```

## 📊 Performance Comparison

ATOM demonstrates significant performance improvements over vLLM:

| Model | Framework | Tokens | Time | Throughput |
|-------|-----------|--------|------|------------|
| **Qwen3-0.6B** | ATOM | 4096 | 0.25s | **16,643.74 tok/s** |
| Qwen3-0.6B | vLLM | 4096 | 0.63s | 6,543.06 tok/s |
| **Llama-3.1-8B-Instruct-FP8-KV** | ATOM | 4096 | 0.68s | **5,983.37 tok/s** |
| Llama-3.1-8B-Instruct-FP8-KV | vLLM | 4096 | 1.68s | 2,432.62 tok/s |