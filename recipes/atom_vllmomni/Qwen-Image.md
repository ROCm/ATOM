# Qwen-Image with ATOM vLLM-Omni Plugin Backend Usage Guide

[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) is an image generation foundation model in the Qwen series developed by Alibaba. It achieves significant advances in complex text rendering and precise image editing. The model demonstrates strong general capabilities in both image generation and editing, with exceptional performance in text rendering.

## Launching server

### BF16 on 1xMI300X/MI355X GPUs

```bash
vllm serve Qwen/Qwen-Image --omni \
    --host localhost \
    --port 8091 \
    --tensor-parallel-size 1
```

### Interact with the model

The command is extracted from https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_image

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.chat.completions.create(
    model="Qwen/Qwen-Image",
    messages=[{"role": "user", "content": "A beautiful landscape painting"}],
    extra_body={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "seed": 42,
    },
)

img_url = response.choices[0].message.content[0]["image_url"]["url"]
_, b64_data = img_url.split(",", 1)
with open("output.png", "wb") as f:
    f.write(base64.b64decode(b64_data))
```