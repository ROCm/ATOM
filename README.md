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

Profile online inference
```bash
curl -s -S -X POST http://127.0.0.1:8000/start_profile
```
Run your task
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


## 📊 Performance

### Online serving throughput

![DS R1 Performance](./docs/ds_r1_performance.png)

For more information, visit [InferenceMAX](https://inferencemax.semianalysis.com/).

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
        --num_fewshot 5
```
## Acknowledgements
This project was adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)

## Support & Reporting Issues 

We welcome issues and contributions!  Please use the GitHub Issues page to report bugs or request features: https://github.com/ROCm/ATOM/issues
