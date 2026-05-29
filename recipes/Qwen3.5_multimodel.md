# Qwen3.5 Multimodal Usage Guide

[Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) is a multimodal Mixture-of-Experts (MoE) model from the Qwen3.5 family. ATOM supports the native Qwen3.5 multimodal path, including Hugging Face processor/chat-template preprocessing, the vision encoder, and Qwen3.5 MRoPE positions for language-model attention.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### Serving on 1xMI355X GPU

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --server-port 8000 \
  --no-enable_prefix_caching
```

For single-GPU testing on a multi-GPU node, pin the server to one GPU:

```bash
HIP_VISIBLE_DEVICES=0 python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --server-port 8000 \
  --no-enable_prefix_caching
```

Tips on server configuration:
- Use `--no-enable_prefix_caching` for multimodal requests.
- ATOM has built-in support for `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`.
- Qwen3.5 multimodal models use MRoPE positions on the language-model side. Text-only and non-MRoPE models still use one-dimensional positions.
- Set `AITER_LOG_LEVEL=WARNING` before starting to suppress aiter kernel log noise.
- Use `HIP_VISIBLE_DEVICES` to run independent tests on different GPUs.

## Image request

Send an OpenAI-compatible image request to the server:

```bash
IMAGE_BASE64=$(base64 -w 0 /app/image.png)

curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B-FP8",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image in detail."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,'"$IMAGE_BASE64"'"
            }
          }
        ]
      }
    ],
    "max_tokens": 1024,
    "temperature": 0,
    "top_p": 1,
    "stream": false
  }' | python3 -m json.tool
```

For offline image-to-text inference, use the multimodal example:

```bash
python /app/ATOM/atom/examples/multimodal_inference.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --image /app/image.png \
  --prompt "Describe this image in detail." \
  --temperature 0 \
  --max-tokens 1024 \
  -tp 1
```

## Multimodal accuracy test

Install `lmms-eval` first. The commands below install `lmms-eval` with `--no-deps`, then install only the runtime packages used by the multimodal evaluators to avoid upgrading the ROCm PyTorch stack:

```bash
python3 -m pip install --no-deps --force-reinstall \
  "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git"

python3 -m pip install --no-deps \
  "accelerate>=0.29.1" "av<16.0.0" "datasets>=2.19.0" "evaluate>=0.4.0" \
  "jsonlines" "aiohttp" "numexpr" "peft>=0.2.0" "pybind11>=2.6.2" \
  "pytablewriter" "sacrebleu>=1.5.0" "scikit-learn>=0.24.1" "timm" \
  "einops" "ftfy" "openai" "opencv-python-headless" "hf-transfer" "nltk" \
  "sentencepiece" "yt-dlp" "pycocoevalcap" "tqdm-multiprocess" \
  "transformers-stream-generator" "zstandard" "pillow" "pyyaml" "sympy" \
  "latex2sympy2" "mpmath" "Jinja2" "openpyxl" "loguru" "math-verify" \
  "wandb==0.25.0" "tiktoken" "pydantic" "packaging" "pre-commit" "zss" \
  "protobuf" "python-dotenv" "qwen-vl-utils>=0.0.14" "decord"
```

### MMStar

```bash
OPENAI_API_KEY=EMPTY \
PYTHONPATH="${LMMS_EVAL_PATH:-/app/lmms-eval}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m lmms_eval \
  --model openai \
  --model_args "model=Qwen/Qwen3.5-35B-A3B-FP8,base_url=http://127.0.0.1:8000/v1,api_key=EMPTY,timeout=900,max_retries=3,num_concurrent=64,max_size_in_mb=50" \
  --tasks mmstar \
  --batch_size 1 \
  --process_with_media \
  --gen_kwargs "temperature=0,max_new_tokens=8192" \
  --log_samples \
  --output_path /tmp/atom_qwen35_mmstar
```

### MMMU validation

```bash
OPENAI_API_KEY=EMPTY \
PYTHONPATH="${LMMS_EVAL_PATH:-/app/lmms-eval}${PYTHONPATH:+:${PYTHONPATH}}" \
python -m lmms_eval \
  --model openai \
  --model_args "model=Qwen/Qwen3.5-35B-A3B-FP8,base_url=http://127.0.0.1:8000/v1,api_key=EMPTY,timeout=900,max_retries=3,num_concurrent=16,max_size_in_mb=50" \
  --tasks mmmu_val \
  --batch_size 1 \
  --process_with_media \
  --gen_kwargs "temperature=0,max_new_tokens=8192" \
  --log_samples \
  --output_path /tmp/atom_qwen35_mmmu_val
```
