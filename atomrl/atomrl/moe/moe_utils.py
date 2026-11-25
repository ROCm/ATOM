# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MoE diagnostics: load balancing, entropy, and lightweight router utilities."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


def _extract_router_logits(output: Any) -> Tensor | None:
    """Best-effort extraction of router logits from a MoE layer output."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple):
        for part in reversed(output):
            if isinstance(part, Tensor) and part.ndim >= 2:
                return part
    return None


def iter_moe_modules(model: nn.Module) -> Iterator[tuple[int, str, nn.Module]]:
    """Yield ``(layer_index, qualified_name, module)`` for likely MoE blocks.

    Handles HF-style blocks (``experts`` + ``gate``/``w_gate``) and Megatron-Core
    ``MoELayer`` (``experts`` + ``router``). The layer index is parsed from the
    module path (``decoder.layers.{i}.mlp``) when available, else a running count.
    """
    running = 0
    for name, module in model.named_modules():
        cls = type(module).__name__
        is_moe = (
            "MoE" in cls
            or "MoeBlock" in cls
            or "SparseMoe" in cls
            or (
                hasattr(module, "experts") and hasattr(module, "router")
            )  # Megatron MoELayer
            or (
                hasattr(module, "experts")
                and (hasattr(module, "gate") or hasattr(module, "w_gate"))
            )
        )
        if not is_moe:
            continue
        m = _LAYER_IDX_RE.search(name)
        layer_idx = int(m.group(1)) if m else running
        yield layer_idx, name, module
        running += 1


def iter_megatron_routers(model: nn.Module) -> Iterator[tuple[int, nn.Module]]:
    """Yield ``(layer_index, router_module)`` for Megatron-Core MoE routers.

    A router is detected by the presence of ``routing`` + ``gating`` callables
    (``megatron.core.transformer.moe.router.TopKRouter``). The layer index is
    parsed from the module path (local per-PP-stage numbering)."""
    for name, module in model.named_modules():
        if callable(getattr(module, "routing", None)) and callable(
            getattr(module, "gating", None)
        ):
            m = _LAYER_IDX_RE.search(name)
            if m:
                yield int(m.group(1)), module


@contextmanager
def megatron_record_router_logits(model: nn.Module, store: dict[int, list[Tensor]]):
    """Record each Megatron MoE layer's router logits into ``store`` (by layer idx).

    Wraps every router's ``routing(logits, ...)`` so the pre-routing gating logits
    are captured (detached) as they flow through the forward. ``store[layer]`` is a
    LIST that grows by one entry per routing call, i.e. one per microbatch in call
    order -- this keeps multi-microbatch (dynamic-batch / pipeline) forwards aligned
    for replay. Works under EP/CP/TP (each rank records its LOCAL token logits).
    Restores the methods on exit.
    """
    saved: list[tuple[nn.Module, Any]] = []

    def make_wrap(layer_idx: int, orig):
        def wrapped(logits, *args, **kwargs):
            # store on CPU: over many microbatches x layers the fp32 router logits
            # (~[tokens, num_experts]) can add up to GBs; keep them off the GPU.
            store.setdefault(layer_idx, []).append(logits.detach().to("cpu"))
            return orig(logits, *args, **kwargs)

        return wrapped

    try:
        for layer_idx, router in iter_megatron_routers(model):
            orig = router.routing
            saved.append((router, orig))
            router.routing = make_wrap(layer_idx, orig)
        if not saved:
            logger.warning("megatron_record_router_logits: no Megatron routers found.")
        yield store
    finally:
        for router, orig in saved:
            router.routing = orig


@contextmanager
def megatron_replay_router_logits(model: nn.Module, recorded: dict[int, list[Tensor]]):
    """Replay recorded router logits into each Megatron MoE layer's ``routing``.

    Substitutes the passed ``logits`` with the next ``recorded[layer_idx]`` entry
    (consumed in call order, matching the recording forward's microbatch order) so
    training reuses the recorded routing decisions. Falls back to the natural
    logits (with a one-time warning) when the recording is missing/exhausted or the
    shape does not match. Restores the methods on exit.
    """
    saved: list[tuple[nn.Module, Any]] = []
    call_idx: dict[int, int] = {}
    warned: dict[int, bool] = {}

    def make_wrap(layer_idx: int, orig):
        def wrapped(logits, *args, **kwargs):
            seq = recorded.get(layer_idx)
            i = call_idx.get(layer_idx, 0)
            if seq is not None and i < len(seq) and seq[i].numel() == logits.numel():
                logits = (
                    seq[i]
                    .to(device=logits.device, dtype=logits.dtype)
                    .reshape(logits.shape)
                )
            elif not warned.get(layer_idx):
                logger.warning(
                    "megatron_replay_router_logits: layer %d no aligned recording "
                    "(have=%s call=%d); using natural routing.",
                    layer_idx,
                    (len(seq) if seq is not None else None),
                    i,
                )
                warned[layer_idx] = True
            call_idx[layer_idx] = i + 1
            return orig(logits, *args, **kwargs)

        return wrapped

    try:
        for layer_idx, router in iter_megatron_routers(model):
            orig = router.routing
            saved.append((router, orig))
            router.routing = make_wrap(layer_idx, orig)
        yield
    finally:
        for router, orig in saved:
            router.routing = orig


def compute_load_balance_loss(
    router_logits: Tensor, num_experts: int, top_k: int
) -> Tensor:
    """Switch-style load balancing loss on router assignments.

    Args:
        router_logits: Tensor shaped ``[..., num_experts]``.
        num_experts: Number of routed experts.
        top_k: Top-k routing width (1 reduces to hard argmax counts).
    """
    logits = router_logits.reshape(-1, num_experts)
    probs = F.softmax(logits, dim=-1)
    k = max(1, min(int(top_k), num_experts))
    top = torch.topk(probs, k, dim=-1)
    counts = torch.zeros_like(probs)
    counts.scatter_(1, top.indices, top.values)
    f = counts.sum(dim=0) / float(counts.shape[0])
    loss = float(num_experts) * (f * f).sum()
    return loss


def compute_router_entropy(router_logits: Tensor) -> Tensor:
    """Mean Shannon entropy of the router softmax over tokens."""
    logits = router_logits.reshape(-1, router_logits.shape[-1])
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)
    return ent.mean()


def check_expert_utilization(router_logits: Tensor, num_experts: int) -> dict[str, Any]:
    """Summarize how uniformly experts are used under a softmax router."""
    logits = router_logits.reshape(-1, num_experts)
    p = F.softmax(logits, dim=-1)
    mean_p = p.mean(dim=0)
    std_p = p.std(dim=0)
    max_expert = int(mean_p.argmax().item())
    min_expert = int(mean_p.argmin().item())
    util = {
        "num_tokens": int(logits.shape[0]),
        "num_experts": int(num_experts),
        "mean_softmax_mass_per_expert": mean_p.detach().cpu().tolist(),
        "std_softmax_mass_per_expert": std_p.detach().cpu().tolist(),
        "argmax_expert_mass_mean": float(mean_p[max_expert].item()),
        "argmin_expert_mass_mean": float(mean_p[min_expert].item()),
    }
    logger.debug(
        "Expert utilization summary: max_mean=%.4f min_mean=%.4f",
        util["argmax_expert_mass_mean"],
        util["argmin_expert_mass_mean"],
    )
    return util
