# SPDX-License-Identifier: MIT
# Image grid and patch math adapted from the released DeepSeek reference.
"""Image preprocessing.

An image becomes a `n_vit_h x n_vit_w` patch grid for the ViT and a `n_llm_h x n_llm_w` token grid
after the 3x3 aligner downsample, which the LLM sees as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every one of those positions carries `image_token_id` in `input_ids`; only the token type tells them
apart. The IMAGE slots are filled with aligner rows in reading order.
"""

import math

import numpy as np
import torch
from PIL import Image, ImageOps

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, args):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = args.patch_size
    if args.max_wh_ratio is not None and width > height * args.max_wh_ratio:
        width = height * args.max_wh_ratio
    if 0 < width * height < args.min_pixels:
        ratio = (args.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        args.downsample_ratio,
        args.max_image_tokens,
    )


def preprocess_image(image: Image.Image, args):
    """Load and transform one image record into ViT patches."""
    p = args.patch_size
    image = image.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
        image.width, image.height, args
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if (
        args.max_wh_ratio is not None
        and image.width >= args.max_wh_ratio * image.height
    ):
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


class DeepseekV41ImageProcessor:
    """Official chat encoding followed by deterministic image span expansion."""

    def __init__(self, atom_config, tokenizer, encoder):
        if encoder is None or encoder.name != "encoding_dsv41":
            raise ValueError(
                "DeepSeek-V4.1 images require the official message encoder"
            )
        self.tokenizer, self.encoder = tokenizer, encoder
        self.config = atom_config.hf_config
        root = getattr(atom_config, "multimodal_config", None)
        if root is None:
            root = self.config._multimodal_config
        self.vision_config = root.vision_config

    def prepare(self, messages, images, template_kwargs, tools=None):
        from atom.entrypoints.openai.chat_encoders import apply_chat_template
        from atom.model_engine.multimodal_runtime import multimodal_cache_seed

        # The encoder renders placeholders without opening media. The serving
        # layer has already loaded these PIL images in exactly this order.
        encoded_messages = []
        image_count = 0
        for message in messages:
            message = dict(message)
            if isinstance(message.get("content"), list):
                parts = []
                for part in message["content"]:
                    if part.get("type") == "image":
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"atom-image:{image_count}"},
                            }
                        )
                        image_count += 1
                    else:
                        parts.append(part)
                message["content"] = parts
            encoded_messages.append(message)
        if image_count != len(images):
            raise ValueError("Image records and decoded images disagree")
        prompt = apply_chat_template(
            self.tokenizer,
            self.encoder,
            encoded_messages,
            tools=tools,
            **template_kwargs,
        )
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if prompt_tokens.count(self.config.image_token_id) != len(images):
            raise ValueError("Image placeholder count does not match supplied images")
        tokens, types, patches, grids, spans = [], [], [], [], []
        image_iter = iter(images)
        for token in prompt_tokens:
            if token != self.config.image_token_id:
                tokens.append(token)
                types.append(TEXT)
                continue
            x, h, w, lh, lw = preprocess_image(next(image_iter), self.vision_config)
            image_types = image_token_types(lh, lw).tolist()
            spans.append((len(tokens), len(image_types)))
            tokens.extend([token] * len(image_types))
            types.extend(image_types)
            patches.append(x)
            grids.append((1, h, w))
        data = {
            "pixel_values": torch.cat(patches),
            "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
            "token_types": np.asarray(types, dtype=np.int8),
            "embedding_spans": tuple(spans),
        }
        data["cache_seed"] = multimodal_cache_seed(data)
        return tokens, data


def build_inputs(
    atom_config, processor, messages, images, chat_template_kwargs, tools=None
):
    return processor.prepare(messages, images, chat_template_kwargs, tools)
