# FLUX.2-dev with ATOM vLLM-Omni Plugin Backend Usage Guide

[FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) is a text-to-image diffusion foundation model from Black Forest Labs. ATOM integrates with vLLM-Omni through the plugin backend and replaces the transformer's tensor-parallel linear layers with ATOM's AITER-backed implementations.

## Before launching

`black-forest-labs/FLUX.2-dev` is a gated Hugging Face repository. Before serving the model:

1. Make sure your Hugging Face account has accepted the model license.
2. Export a valid `HF_TOKEN`, or serve from a fully downloaded local snapshot path.

## Launching server

### BF16 on 1xMI300X/MI355X GPUs

```bash
export HF_TOKEN=YOUR_HF_TOKEN

vllm serve black-forest-labs/FLUX.2-dev --omni \
    --host localhost \
    --port 8091 \
    --tensor-parallel-size 1
```


## Interact with the model

The request pattern is adapted from the vLLM-Omni online-serving diffusion examples and FLUX.2-dev E2E coverage.

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.chat.completions.create(
    model="black-forest-labs/FLUX.2-dev",
    messages=[
        {
            "role": "user",
            "content": (
                "A cinematic mountain landscape at sunrise, dramatic clouds, "
                "ultra-detailed, realistic photography."
            ),
        }
    ],
    extra_body={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "negative_prompt": "low quality, blurry, distorted, deformed, watermark",
        "true_cfg_scale": 4.0,
        "seed": 42,
    },
)

img_url = response.choices[0].message.content[0]["image_url"]["url"]
_, b64_data = img_url.split(",", 1)
with open("flux2_output.png", "wb") as f:
    f.write(base64.b64decode(b64_data))
```
