<div align="center" id="logo">
<img src="docs/atom_logo.png" alt="logo" width="400" margin="10px"></img>
</div>

--------------------------------------------------------------------------------

**ATOM** (AiTer Optimized Model) is a lightweight vLLM-like implementation, focusing on integration and optimization based on [aiter](https://github.com/ROCm/aiter).

## 🚀 Features

- **ROCm Optimized**: Built on AMD's ROCm platform with torch compile support
- **Model Support**: Compatible with **[Deepseek](https://huggingface.co/deepseek-ai)**, **[Qwen](https://huggingface.co/Qwen)**, **[Llama](https://huggingface.co/meta-llama)**, **[Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)**, **[GPTOSS](https://huggingface.co/openai)**.
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
pip install amd-aiter
git clone https://github.com/ROCm/ATOM.git; cd ./ATOM; pip install .
```

## 💡 Usage

### Basic Example

The default optimization level is 3 (running with torch compile). Supported models include **[Deepseek](https://huggingface.co/deepseek-ai)**, **[Qwen](https://huggingface.co/Qwen)**, **[Llama](https://huggingface.co/meta-llama)**, **[Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)**, **[GPTOSS](https://huggingface.co/openai)**.

```bash
python -m atom.examples.simple_inference --model meta-llama/Meta-Llama-3-8B --kv_cache_dtype fp8 
```

> **Note:** First-time execution may take approximately 10 minutes for model compilation.

### Performance profiling

Profile offline inference
```bash
python -m atom.examples.profile_offline --model Qwen/Qwen3-0.6B --kv_cache_dtype fp8 
```
Or profile offline with custom input length
```bash
python -m atom.examples.profile_offline --model Qwen/Qwen3-0.6B --kv_cache_dtype fp8 --random-input --input-length 1024 --output-length 32
```

Profile online inference, after starting the server
```bash
python -m atom.examples.profile_online 
```
Or profile online with custom input length
```bash
python -m atom.examples.profile_online --model Qwen/Qwen3-0.6B --random-input --input-length 1024 --output-length 32
```

Or directly send start profile and stop profile reuqest
```bash
curl -s -S -X POST http://127.0.0.1:8000/start_profile
```
```bash
curl -s -S -X POST http://127.0.0.1:8000/stop_profile
```

### Performance Benchmarking

Run online throughput benchmark:

start the server
```bash
python -m atom.entrypoints.openai_server --kv_cache_dtype fp8  --model Qwen/Qwen3-0.6B
python -m atom.entrypoints.openai_server --kv_cache_dtype fp8 -tp 8  --model deepseek-ai/DeepSeek-R1
```
run benchmark
```bash
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


## 📊 Performance Comparison

ATOM demonstrates significant performance improvements over vLLM:

| Model | Framework | Tokens | Time | Throughput |
|-------|-----------|--------|------|------------|
| **Qwen3-0.6B** | ATOM | 4096 | 0.25s | **16,643.74 tok/s** |
| Qwen3-0.6B | vLLM | 4096 | 0.63s | 6,543.06 tok/s |
| **Llama-3.1-8B-Instruct-FP8-KV** | ATOM | 4096 | 0.68s | **5,983.37 tok/s** |
| Llama-3.1-8B-Instruct-FP8-KV | vLLM | 4096 | 1.68s | 2,432.62 tok/s |


### Online serving throughput:

Deepseek-V3
| concurrency | IPS/QPS | prompts num | vLLM Throughput | ATOM Throughput |
|-------------|---------|-------------|-----------------|-----------------|
| 16 | 1024/1024 | 128 | 423.68 tok/s | **922.03 tok/s** |
| 32 | 1024/1024 | 128 | 629.06 tok/s | **1488.52 tok/s** |
| 64 | 1024/1024 | 128 | 760.22 tok/s | **2221.25 tok/s** |
| 128 | 1024/1024 | 128 | 1107.93 tok/s | **2254.88 tok/s** |

### Accuracy Benchmarking

First, install `lm-eval` to test model accuracy:

```bash
pip install lm-eval[api]
```

Next, start an OpenAI-compatible server using `openai_server.py`:

```bash
python -m atom.entrypoints.openai_server --model meta-llama/Meta-Llama-3-8B --kv_cache_dtype fp8 
```

Finally, run the evaluation by choosing your datasets:

```bash
lm_eval --model local-completions \
        --model_args model=meta-llama/Meta-Llama-3-8B,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```
## Acknowledgements
This project was adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)

## Support & Reporting Issues 

We welcome issues and contributions!  Please use the GitHub Issues page to report bugs or request features: https://github.com/ROCm/ATOM/issues
