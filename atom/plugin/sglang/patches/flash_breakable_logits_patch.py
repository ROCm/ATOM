# SPDX-License-Identifier: MIT
"""Teach SGLang breakable CUDA graphs to buffer LogitsProcessorOutput.

Flash EntryClass.forward returns logits, not a hidden tensor. BCG's
``_alloc_full_buffer`` only knows Tensor / PPProxyTensors / list / tuple, so
``--cuda-graph-backend-decode breakable`` dies at first capture. PLE first-knife
needs breakable so PLE can stay an eager island.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _is_logits_output(obj: Any) -> bool:
    return type(obj).__name__ == "LogitsProcessorOutput"


def _clone_logits_output(output: Any, *, size: int | None, num_tokens: int | None):
    cls = type(output)
    kwargs = {}
    for field in dataclasses.fields(cls):
        val = getattr(output, field.name)
        if torch.is_tensor(val) and val.dim() >= 1:
            if size is not None:
                kwargs[field.name] = val.new_empty((size, *val.shape[1:]))
            elif num_tokens is not None:
                kwargs[field.name] = val[:num_tokens]
            else:
                kwargs[field.name] = val
        else:
            kwargs[field.name] = val
    return cls(**kwargs)


def apply_flash_breakable_logits_patch() -> None:
    try:
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend,
        )
    except Exception:  # noqa: BLE001
        logger.debug("BreakableCudaGraphBackend unavailable; skip logits patch")
        return
    if getattr(BreakableCudaGraphBackend, "_atom_flash_logits_patch", False):
        return

    orig_rows = BreakableCudaGraphBackend._output_rows
    orig_alloc = BreakableCudaGraphBackend._alloc_full_buffer
    orig_slice = BreakableCudaGraphBackend._slice_output
    orig_copy = BreakableCudaGraphBackend._copy_output_to_buffer

    def _output_rows(self, output: Any, cap: int) -> int:
        if _is_logits_output(output) and torch.is_tensor(output.next_token_logits):
            return min(cap, int(output.next_token_logits.shape[0]))
        return orig_rows(self, output, cap)

    def _alloc_full_buffer(self, output: Any, size: int) -> Any:
        if _is_logits_output(output):
            return _clone_logits_output(output, size=size, num_tokens=None)
        return orig_alloc(self, output, size)

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if _is_logits_output(output):
            return _clone_logits_output(output, size=None, num_tokens=num_tokens)
        return orig_slice(self, output, num_tokens)

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if _is_logits_output(output) and _is_logits_output(output_buffer):
            for field in dataclasses.fields(type(output)):
                src = getattr(output, field.name)
                dst = getattr(output_buffer, field.name)
                if torch.is_tensor(src) and torch.is_tensor(dst):
                    dst[:num_tokens].copy_(src[:num_tokens])
            return
        return orig_copy(self, output, output_buffer, num_tokens)

    BreakableCudaGraphBackend._output_rows = _output_rows
    BreakableCudaGraphBackend._alloc_full_buffer = _alloc_full_buffer
    BreakableCudaGraphBackend._slice_output = _slice_output
    BreakableCudaGraphBackend._copy_output_to_buffer = _copy_output_to_buffer
    BreakableCudaGraphBackend._atom_flash_logits_patch = True
    logger.info("Patched BreakableCudaGraphBackend for LogitsProcessorOutput")
