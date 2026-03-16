# SPDX-License-Identifier: Apache-2.0
"""OpenAI Images API compatible server for Flux."""

import argparse
import base64
import io
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.diffusion_runner import DiffusionModelRunner

app = FastAPI()
runner = None


class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: int = 50
    guidance_scale: float = 3.5


def tensor_to_b64(tensor: torch.Tensor) -> str:
    img = Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/v1/images/generations")
async def create_image(req: ImageRequest):
    w, h = map(int, req.size.split("x"))
    images = runner.generate(
        [req.prompt] * req.n, h, w, req.num_inference_steps, req.guidance_scale
    )
    return JSONResponse(
        {
            "created": int(time.time()),
            "data": [{"b64_json": tensor_to_b64(img)} for img in images],
        }
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    global runner
    config = EngineArgs.from_cli_args(args).create_atom_config()
    runner = DiffusionModelRunner(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
