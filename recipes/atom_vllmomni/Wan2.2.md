# Wan2.2 with ATOM vLLM-Omni Plugin Backend Usage Guide

[Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) is an advanced large-scale video generation foundation model developed by Alibaba's Wan-AI team. It represents a major upgrade with innovations including a Mixture-of-Experts (MoE) architecture for efficient high-quality video generation, cinematic-level aesthetics with detailed controllable styles, and enhanced complex motion generation capabilities. The model supports both text-to-video and image-to-video generation at 720P resolution with 24fps, achieving top performance among open-source and closed-source models while being efficient enough to run on consumer-grade GPUs.


| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| T2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)    | Text-to-Video MoE model, supports 480P & 720P |
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
| TI2V-5B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)     🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)     | High-compression VAE, T2V+I2V, supports 720P |


## Launching server

### BF16 on 1xMI300X/MI355X GPUs

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
    --host localhost \
    --port 8091 \
    --tensor-parallel-size 1
```

### Interact with the model

1. Text to video. The command is extracted from https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_video

```bash
#!/bin/bash
# Wan2.2 text-to-video curl example using the async video job API.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-wan22_output.mp4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" \
    -H "Accept: application/json" \
    -F "prompt=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    -F "seconds=2" \
    -F "size=832x480" \
    -F "negative_prompt=色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    -F "fps=16" \
    -F "num_inference_steps=40" \
    -F "guidance_scale=4.0" \
    -F "guidance_scale_2=4.0" \
    -F "boundary_ratio=0.875" \
    -F "flow_shift=5.0" \
    -F "seed=42"
)

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
echo "${create_response}" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq .
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"
echo "Saved video to ${OUTPUT_PATH}"

```

2. Image to video. You can find the command from https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/image_to_video .

### Benchmark tools

vLLM-Omni offers performance benchmarking scripts through https://github.com/vllm-project/vllm-omni/tree/main/benchmarks/diffusion .