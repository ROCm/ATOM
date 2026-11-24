# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/trainer/ppo/metric_utils.py

# Derived from verl (verl-project/verl):
#   verl/trainer/ppo/metric_utils.py L89-310 (compute_data_metrics, compute_timing_metrics,
#                                              compute_throughput_metrics)
#
# Licensed under the Apache License, Version 2.0.
"""Comprehensive metrics for RL training steps."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from atomrl.core.protocol import DataProto

logger = logging.getLogger(__name__)


def _response_info(batch: DataProto) -> dict[str, Tensor]:
    """Compute prompt / response lengths from batch tensors."""
    t = batch.tensors
    if "prompt_length" in t and "response_length" in t:
        return {
            "prompt_length": t["prompt_length"],
            "response_length": t["response_length"],
        }

    response_mask = t.get("response_mask", t.get("attention_mask"))
    if response_mask is None:
        raise KeyError("Batch must contain 'response_mask' or 'attention_mask'.")
    response_length = response_mask.sum(-1).float()

    prompt_mask = t.get("prompt_mask")
    if prompt_mask is not None:
        prompt_length = prompt_mask.sum(-1).float()
    else:
        # prompt_length = (all real tokens) - (response tokens). Robust to left/
        # right padding and to response_mask being the shifted [B, S-1] mask
        # (prompt zeroed); avoids the buggy ``attention_mask[:, :-S+1]`` slice
        # that collapsed prompt_length to ~1.
        attention_mask = t.get("attention_mask")
        if attention_mask is not None and attention_mask is not response_mask:
            total_length = attention_mask.sum(-1).float()
            prompt_length = (total_length - response_length).clamp(min=0.0)
        else:
            prompt_length = torch.zeros_like(response_length)
    return {"prompt_length": prompt_length, "response_length": response_length}


def _safe_stats(values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": float("nan"), "max": float("nan"), "min": float("nan")}
    return {
        "mean": values.mean().detach().item(),
        "max": values.max().detach().item(),
        "min": values.min().detach().item(),
    }


# ---------------------------------------------------------------------------
# compute_data_metrics  (verl/trainer/ppo/metric_utils.py L89-268)
# ---------------------------------------------------------------------------


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """Compute score/reward/advantage/value/length metrics from a training batch."""
    t = batch.tensors
    response_mask = t.get("response_mask", t.get("attention_mask"))
    if response_mask is None:
        return {}
    response_mask_bool = response_mask.bool()

    info = _response_info(batch)
    prompt_length = info["prompt_length"]
    response_length = info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted = ~aborted_mask

    metrics: dict[str, Any] = {}

    # --- Scores ---
    token_level_scores = t.get("token_level_scores", t.get("rewards"))
    if token_level_scores is not None:
        if token_level_scores.dim() > 1:
            seq_score = token_level_scores.sum(-1)
        else:
            seq_score = token_level_scores
        na_score = seq_score[non_aborted] if non_aborted.any() else seq_score
        for k, v in _safe_stats(na_score).items():
            metrics[f"critic/score/{k}"] = v

    # --- Rewards ---
    rewards = t.get("rewards")
    if rewards is not None:
        if rewards.dim() > 1:
            seq_reward = rewards.sum(-1)
        else:
            seq_reward = rewards
        na_reward = seq_reward[non_aborted] if non_aborted.any() else seq_reward
        for k, v in _safe_stats(na_reward).items():
            metrics[f"critic/rewards/{k}"] = v

    # --- Advantages ---
    # verl reports critic/advantages as a token-level masked mean (advantages are
    # broadcast over response tokens, so longer responses carry more weight). Lumen
    # stores sample-level [B] advantages, so broadcast to [B, T] over the response
    # mask before aggregating to match verl's statistic.
    advantages = t.get("advantages")
    if advantages is not None:
        if advantages.dim() == 1 and advantages.shape[0] == response_mask_bool.shape[0]:
            adv_tok = advantages.unsqueeze(-1).expand_as(response_mask_bool)
            valid_adv = torch.masked_select(adv_tok, response_mask_bool)
        elif advantages.dim() > 1:
            valid_adv = torch.masked_select(advantages, response_mask_bool)
        else:
            valid_adv = advantages
        for k, v in _safe_stats(valid_adv).items():
            metrics[f"critic/advantages/{k}"] = v

    # --- Returns ---
    # In verl's GRPO path token-level returns are the normalized advantages
    # (there is no learned value baseline), so its W&B returns and advantages
    # panels are identical. Mirror that fallback when no explicit GAE returns
    # tensor exists.
    returns = t.get("returns", advantages)
    if returns is not None:
        if returns.dim() == 1 and returns.shape[0] == response_mask_bool.shape[0]:
            returns_tok = returns.unsqueeze(-1).expand_as(response_mask_bool)
            valid_ret = torch.masked_select(returns_tok, response_mask_bool)
        else:
            valid_ret = (
                torch.masked_select(returns, response_mask_bool)
                if returns.dim() > 1
                else returns
            )
        for k, v in _safe_stats(valid_ret).items():
            metrics[f"critic/returns/{k}"] = v

    # --- Values (critic only) ---
    if use_critic and "values" in t:
        values = t["values"]
        valid_val = torch.masked_select(values, response_mask_bool)
        for k, v in _safe_stats(valid_val).items():
            metrics[f"critic/values/{k}"] = v
        if returns is not None:
            valid_ret_for_var = torch.masked_select(returns, response_mask_bool)
            if valid_ret_for_var.numel() > 1 and valid_val.numel() > 1:
                ret_var = torch.var(valid_ret_for_var)
                diff_var = torch.var(valid_ret_for_var - valid_val)
                metrics["critic/vf_explained_var"] = (
                    (1.0 - diff_var / (ret_var + 1e-5)).detach().item()
                )

    # --- Response length ---
    metrics["response_length/mean"] = response_length.mean().detach().item()
    metrics["response_length/max"] = response_length.max().detach().item()
    metrics["response_length/min"] = response_length.min().detach().item()

    # verl's response tensors are max_response_length wide; AtomRL's
    # response_mask spans the whole left-padded sequence, so the trainer passes
    # the budget in meta.
    max_resp_len = float(batch.meta.get("max_response_length") or 0)
    if max_resp_len <= 0:
        max_resp_len = response_mask.shape[-1] if response_mask.dim() > 1 else 0
    if max_resp_len > 0:
        metrics["response_length/clip_ratio"] = (
            (response_length == max_resp_len).float().mean().detach().item()
        )

    na_resp_len = response_length[non_aborted] if non_aborted.any() else response_length
    if na_resp_len.numel() > 0:
        metrics["response_length_non_aborted/mean"] = na_resp_len.mean().detach().item()
        metrics["response_length_non_aborted/max"] = na_resp_len.max().detach().item()
        metrics["response_length_non_aborted/min"] = na_resp_len.min().detach().item()
        if max_resp_len > 0:
            metrics["response_length_non_aborted/clip_ratio"] = (
                (na_resp_len == max_resp_len).float().mean().detach().item()
            )

    metrics["response/aborted_ratio"] = aborted_mask.float().mean().detach().item()

    # --- Prompt length ---
    metrics["prompt_length/mean"] = prompt_length.mean().detach().item()
    metrics["prompt_length/max"] = prompt_length.max().detach().item()
    metrics["prompt_length/min"] = prompt_length.min().detach().item()
    max_prompt_len = batch.meta.get("max_prompt_length")
    if max_prompt_len is not None and float(max_prompt_len) > 0:
        metrics["prompt_length/clip_ratio"] = (
            (prompt_length == float(max_prompt_len)).float().mean().detach().item()
        )

    return metrics


# ---------------------------------------------------------------------------
# compute_timing_metrics  (verl/trainer/ppo/metric_utils.py L271-310)
# ---------------------------------------------------------------------------


def compute_timing_metrics(
    batch: DataProto, timing_raw: dict[str, float]
) -> dict[str, Any]:
    """Compute per-stage raw and per-token timing metrics."""
    info = _response_info(batch)
    num_prompt_tokens = info["prompt_length"].sum().item()
    num_response_tokens = info["response_length"].sum().item()
    num_all_tokens = num_prompt_tokens + num_response_tokens

    tok_map = {"gen": max(num_response_tokens, 1)}
    for name in ("ref", "values", "adv", "update_critic", "update_actor"):
        tok_map[name] = max(num_all_tokens, 1)

    metrics: dict[str, Any] = {}
    for name, sec in timing_raw.items():
        metrics[f"timing_s/{name}"] = sec
        if name in tok_map:
            metrics[f"timing_per_token_ms/{name}"] = sec * 1000 / tok_map[name]
    return metrics


# ---------------------------------------------------------------------------
# compute_throughput_metrics  (verl/trainer/ppo/metric_utils.py L313-346)
# ---------------------------------------------------------------------------


def compute_throughput_metrics(
    batch: DataProto,
    timing_raw: dict[str, float],
    n_gpus: int,
) -> dict[str, Any]:
    """Compute throughput metrics: tokens/s/GPU."""
    info = _response_info(batch)
    total_tokens = (info["prompt_length"].sum() + info["response_length"].sum()).item()

    step_time = timing_raw.get("step", timing_raw.get("total", 0.0))
    if step_time <= 0:
        return {"perf/total_num_tokens": total_tokens}

    return {
        "perf/total_num_tokens": total_tokens,
        "perf/time_per_step": step_time,
        "perf/throughput": total_tokens / (step_time * max(n_gpus, 1)),
    }
