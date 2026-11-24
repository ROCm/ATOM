# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""HuggingFace-backed inference engine for rollout generation + log-prob computation.

Used when ATOM is not available or when features ATOM doesn't expose
(log-probs, weight-sync) are needed. Will be replaced by ATOM as its API matures.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HFEngine:
    """Generate sequences and compute log-probs using HuggingFace transformers.

    This engine loads a model in eval mode for rollout. For colocated
    actor+rollout setups, the same model instance can be shared.
    """

    def __init__(
        self,
        model_name: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.3,
    ) -> None:
        self._model_name = model_name
        self._torch_dtype = torch_dtype
        self._device_str = device
        self._gpu_mem_util = gpu_memory_utilization
        self._model: torch.nn.Module | None = None
        self._tokenizer: Any = None

    @property
    def model(self) -> torch.nn.Module | None:
        return self._model

    def init(self) -> None:
        """Load the model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "HFEngine: loading model %s (dtype=%s)", self._model_name, self._torch_dtype
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=self._torch_dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self._model.to(self._device_str)
        self._model.eval()
        logger.info(
            "HFEngine: model loaded (%d params)",
            sum(p.numel() for p in self._model.parameters()),
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate completions for a batch of prompts.

        Returns:
            sequences: [B, prompt_len + gen_len] full sequences
            attention_mask: [B, prompt_len + gen_len] corresponding mask
        """
        if self._model is None:
            raise RuntimeError("Call init() before generate().")

        input_ids = input_ids.to(self._device_str)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device_str)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature
            if top_p < 1.0:
                gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        outputs.shape[1]
        new_mask = torch.ones(outputs.shape, dtype=torch.long, device=outputs.device)
        if attention_mask is not None:
            prompt_len = attention_mask.shape[1]
            new_mask[:, :prompt_len] = attention_mask
            pad_id = self._tokenizer.pad_token_id
            if pad_id is not None:
                new_mask[:, prompt_len:] = (outputs[:, prompt_len:] != pad_id).long()

        return outputs.cpu(), new_mask.cpu()

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for the given sequences.

        Returns: [B, T-1] log-prob of each token given its prefix.
        """
        if self._model is None:
            raise RuntimeError("Call init() before compute_log_probs().")

        input_ids = input_ids.to(self._device_str)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device_str)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
            targets = input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.cpu()

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """Decode token IDs back to strings."""
        if self._tokenizer is None:
            raise RuntimeError("Call init() before decode().")
        return self._tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load new weights (for actor → rollout sync)."""
        if self._model is None:
            raise RuntimeError("Call init() before update_weights().")
        self._model.load_state_dict(state_dict, strict=False)
        logger.info("HFEngine: loaded %d weight tensors.", len(state_dict))

    def sleep(self) -> None:
        """Free GPU memory by moving model to CPU."""
        if self._model is not None:
            self._model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("HFEngine: model offloaded to CPU (sleep).")

    def wake(self) -> None:
        """Restore model to GPU."""
        if self._model is not None:
            self._model.to(self._device_str)
            logger.info("HFEngine: model restored to GPU (wake).")

    def shutdown(self) -> None:
        """Release all resources."""
        if self._model is not None:
            del self._model
            self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("HFEngine: shutdown complete.")
