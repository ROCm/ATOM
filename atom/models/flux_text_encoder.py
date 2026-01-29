# SPDX-License-Identifier: Apache-2.0
"""Text encoder wrapper for Flux."""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class FluxTextEncoder(nn.Module):
    def __init__(self, model_name: str = "google/t5-v1_1-xxl", device: str = "cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list, max_length: int = 512) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return self.model(**tokens).last_hidden_state
