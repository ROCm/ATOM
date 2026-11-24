# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/trainer/ppo/core_algos.py

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pluggable advantage estimator registry.

Each estimator has signature ``(batch: DataProto, config: AtomRLConfig) -> DataProto``
and populates ``batch.tensors["advantages"]`` (and optionally ``batch.tensors["returns"]``).

New estimators derived from verl/trainer/ppo/core_algos.py L334-865.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from atomrl.algorithms.dapo_sampling import overlong_buffer_penalty
from atomrl.core.config import AtomRLConfig
from atomrl.core.protocol import DataProto

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADV_ESTIMATOR_REGISTRY: dict[str, Callable[[DataProto, AtomRLConfig], DataProto]] = {}


def register_adv_est(name: str) -> Callable:
    """Decorator that registers an advantage estimator function by *name*."""

    def wrapper(fn: Callable[[DataProto, AtomRLConfig], DataProto]) -> Callable:
        if name in ADV_ESTIMATOR_REGISTRY:
            logger.warning("Overwriting advantage estimator '%s'.", name)
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _masked_whiten(values: Tensor, mask: Tensor, shift_mean: bool = True) -> Tensor:
    """Whiten *values* using only the entries where *mask* is nonzero."""
    valid = values[mask.bool()]
    if valid.numel() < 2:
        return values
    mean = valid.mean() if shift_mean else 0.0
    std = valid.std().clamp(min=1e-8)
    return (values - mean) / std * mask.float()


def _response_mask_from_batch(batch: DataProto) -> Tensor | None:
    """Return ``response_mask`` if available, else fall back to ``attention_mask``."""
    if "response_mask" in batch.tensors:
        return batch.tensors["response_mask"].to(dtype=torch.bool)
    if "attention_mask" in batch.tensors:
        return batch.tensors["attention_mask"].to(dtype=torch.bool)
    return None


def _last_token_indices(attention_mask: Tensor) -> Tensor:
    """Return index of the last valid token per row (0-based)."""
    lengths = attention_mask.long().sum(dim=-1)
    return torch.clamp(lengths - 1, min=0)


def _scalar_rewards_to_token_rewards(rewards: Tensor, attention_mask: Tensor) -> Tensor:
    """Scatter final scalar rewards onto the last valid timestep."""
    b, t = attention_mask.shape
    device = attention_mask.device
    dtype = rewards.dtype
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    out = torch.zeros(b, t, device=device, dtype=dtype)
    rows = torch.arange(b, device=device)
    last = _last_token_indices(attention_mask)
    out[rows, last] = rewards.to(dtype=dtype)
    return out


def _gae_returns(
    token_rewards: Tensor,
    values: Tensor,
    attention_mask: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Vectorized GAE-Lambda advantages and TD(lambda) returns."""
    m = attention_mask.to(dtype=values.dtype)
    next_values = torch.roll(values, shifts=-1, dims=-1)
    lengths = attention_mask.long().sum(dim=-1)
    batch_idx = torch.arange(values.shape[0], device=values.device)
    next_values[batch_idx, lengths - 1] = 0.0

    delta = token_rewards + gamma * next_values * m - values
    b, seq = delta.shape
    adv = torch.zeros_like(delta)
    last_gae = torch.zeros(b, device=values.device, dtype=values.dtype)
    for t in range(seq - 1, -1, -1):
        last_gae = delta[:, t] + gamma * lam * m[:, t] * last_gae
        adv[:, t] = last_gae
    returns = adv + values
    return adv * m, returns * m


def _apply_overlong_shaping(
    rewards: Tensor,
    batch: DataProto,
    max_len: int,
    penalty: float,
) -> Tensor:
    """Subtract a linear penalty when sequence length exceeds *max_len*."""
    if penalty <= 0.0:
        return rewards
    lengths = batch.meta.get("response_lengths")
    if lengths is None:
        return rewards
    lens_t = torch.as_tensor(lengths, device=rewards.device, dtype=torch.float32)
    if lens_t.shape[0] != rewards.shape[0]:
        return rewards
    over = torch.clamp(lens_t - float(max_len), min=0.0)
    shaped = rewards.to(dtype=torch.float32) - penalty * over
    return shaped.to(dtype=rewards.dtype)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


@register_adv_est("grpo")
def compute_grpo_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """Group-based mean/std normalization (GRPO)."""
    if "rewards" not in batch.tensors:
        raise KeyError("GRPO requires tensor key 'rewards' on the batch.")
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    cfg = config.algorithm.grpo
    g = cfg.num_generations
    if rewards.shape[0] % g != 0:
        raise ValueError(
            f"Batch size {rewards.shape[0]} not divisible by num_generations={g}."
        )
    grouped = rewards.view(-1, g)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
    adv = (grouped - mean) / std
    adv_flat = adv.reshape(-1)
    batch.tensors["advantages"] = adv_flat
    logger.debug(
        "GRPO advantages: mean=%.6f std=%.6f",
        adv_flat.mean().item(),
        adv_flat.std().item(),
    )
    return batch


@register_adv_est("dapo")
def compute_dapo_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """DAPO: group-relative advantages with dynamic sampling and overlong shaping."""
    if "rewards" not in batch.tensors:
        raise KeyError("DAPO requires tensor key 'rewards' on the batch.")
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)

    cfg = config.algorithm.dapo
    g = cfg.num_generations
    if rewards.shape[0] % g != 0:
        raise ValueError(
            f"Batch size {rewards.shape[0]} not divisible by num_generations={g}."
        )

    ob = getattr(cfg, "overlong_buffer", None)
    if ob is not None and getattr(ob, "enable", False):
        lengths = batch.meta.get("response_lengths")
        if lengths is not None:
            max_resp = int(getattr(cfg, "max_resp_len", 0)) or int(
                config.policy.max_response_length
            )
            penalty = overlong_buffer_penalty(
                lengths,
                max_resp_len=max_resp,
                buffer_len=int(ob.len),
                penalty_factor=float(ob.penalty_factor),
            ).to(device=rewards.device)
            if penalty.shape[0] == rewards.shape[0]:
                rewards = rewards.to(dtype=torch.float32) + penalty
    elif cfg.overlong_reward_shaping:
        rewards = _apply_overlong_shaping(
            rewards,
            batch,
            max_len=int(config.policy.max_total_sequence_length),
            penalty=float(batch.meta.get("overlong_penalty", 1.0)),
        )

    grouped = rewards.view(-1, g)
    std = grouped.std(dim=1, unbiased=False)

    if cfg.dynamic_sampling:
        keep = std > 1e-6
        if not torch.any(keep):
            logger.warning("DAPO dynamic sampling removed all groups; keeping all.")
            keep = torch.ones_like(keep, dtype=torch.bool)
        row_mask = keep.unsqueeze(-1).expand(-1, g).reshape(-1)
    else:
        row_mask = torch.ones(rewards.shape[0], dtype=torch.bool, device=rewards.device)

    mean = grouped.mean(dim=1, keepdim=True)
    std_safe = grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
    adv = (grouped - mean) / std_safe
    adv_flat = adv.reshape(-1)

    batch.tensors["advantages"] = adv_flat
    batch.tensors["dapo_sample_mask"] = row_mask.to(dtype=torch.float32)
    if adv_flat.numel() > 0:
        logger.info(
            "NaN-DEBUG DAPO advantages: active_frac=%.4f, adv nan=%d inf=%d "
            "min=%.4f max=%.4f mean=%.4f, rewards min=%.4f max=%.4f mean=%.4f, "
            "std min=%.6f max=%.6f",
            float(row_mask.float().mean().cpu()),
            adv_flat.isnan().sum().item(),
            adv_flat.isinf().sum().item(),
            adv_flat.min().item(),
            adv_flat.max().item(),
            adv_flat.mean().item(),
            rewards.min().item(),
            rewards.max().item(),
            rewards.mean().item(),
            std.min().item(),
            std.max().item(),
        )
    else:
        logger.warning(
            "DAPO advantages: adv_flat is empty (batch_size=%d, num_gen=%d)",
            rewards.shape[0],
            g,
        )
    return batch


@register_adv_est("gae")
def compute_gae_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """Generalized Advantage Estimation (GAE-Lambda) for PPO."""
    if "values" not in batch.tensors:
        raise KeyError("GAE requires critic 'values' tensor [B, T].")
    if "rewards" not in batch.tensors:
        raise KeyError("GAE requires 'rewards' (scalar or token-level).")

    values = batch.tensors["values"]
    # Prefer response_mask, fall back to attention_mask
    mask = _response_mask_from_batch(batch)
    if mask is None:
        raise KeyError("GAE requires 'response_mask' or 'attention_mask' for masking.")
    mask_f = mask.to(dtype=torch.float32)
    rewards = batch.tensors["rewards"]

    if rewards.dim() == 1 or (rewards.dim() == 2 and rewards.shape[-1] == 1):
        token_rewards = _scalar_rewards_to_token_rewards(
            rewards.squeeze(-1) if rewards.dim() == 2 else rewards, mask
        )
    else:
        token_rewards = rewards.to(dtype=values.dtype)
        if token_rewards.shape != values.shape:
            raise ValueError(
                f"Token rewards shape {tuple(token_rewards.shape)} != "
                f"values {tuple(values.shape)}."
            )

    cfg = config.algorithm.ppo
    adv, ret = _gae_returns(
        token_rewards,
        values,
        mask_f,
        gamma=float(cfg.discount),
        lam=float(cfg.gae_lambda),
    )
    batch.tensors["advantages"] = adv
    batch.tensors["returns"] = ret
    mb = mask.bool()
    adv_mean = float((adv * mask_f).sum() / mask_f.sum().clamp(min=1.0))
    adv_std = float(torch.std(adv[mb])) if int(mask_f.sum()) > 0 else 0.0
    logger.debug("GAE: adv_mean=%.6f adv_std=%.6f", adv_mean, adv_std)
    return batch


@register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_advantage(
    batch: DataProto, config: AtomRLConfig
) -> DataProto:
    """REINFORCE++ with discounted cumulative returns and masked whitening."""
    if "rewards" not in batch.tensors:
        raise KeyError("REINFORCE++ requires tensor key 'rewards' on the batch.")

    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)

    response_mask = _response_mask_from_batch(batch)
    if response_mask is None:
        raise KeyError("REINFORCE++ requires 'response_mask' or 'attention_mask'.")
    response_mask_f = response_mask.to(dtype=torch.float32)

    gamma = float(config.algorithm.ppo.discount)

    # Scatter scalar rewards to last response token
    token_rewards = _scalar_rewards_to_token_rewards(rewards, response_mask)

    # Compute discounted returns by reverse iteration over response tokens
    b, t = token_rewards.shape
    returns = torch.zeros_like(token_rewards)
    running = torch.zeros(b, device=token_rewards.device, dtype=token_rewards.dtype)
    for step in range(t - 1, -1, -1):
        running = token_rewards[:, step] + gamma * running * response_mask_f[:, step]
        returns[:, step] = running

    # Apply masked whitening (zero-mean, unit-var within response_mask)
    advantages = _masked_whiten(returns, response_mask_f, shift_mean=True)

    batch.tensors["advantages"] = advantages
    logger.debug(
        "REINFORCE++ advantages: mean=%.6f std=%.6f",
        advantages[response_mask].mean().item() if response_mask.any() else 0.0,
        advantages[response_mask].std().item() if response_mask.any() else 0.0,
    )
    return batch


@register_adv_est("rloo")
def compute_rloo_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """Leave-one-out baseline (RLOO) advantage estimation."""
    if "rewards" not in batch.tensors:
        raise KeyError("RLOO requires tensor key 'rewards' on the batch.")

    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)

    # Determine num_generations from meta or config
    g = batch.meta.get("num_generations", config.algorithm.grpo.num_generations)
    if rewards.shape[0] % g != 0:
        raise ValueError(
            f"Batch size {rewards.shape[0]} not divisible by num_generations={g}."
        )

    response_mask = _response_mask_from_batch(batch)
    if response_mask is None:
        raise KeyError("RLOO requires 'response_mask' or 'attention_mask'.")
    response_mask_f = response_mask.to(dtype=torch.float32)

    # Group by prompt: [num_prompts, g]
    grouped = rewards.view(-1, g)
    # Leave-one-out: for each sample, baseline = (group_sum - self) / (N - 1)
    # advantage = reward - baseline = reward * N/(N-1) - group_mean * N/(N-1)
    #           = N/(N-1) * (reward - group_mean)
    #
    # Equivalently: score - (sum - score) / (N - 1)
    #             = score - sum/(N-1) + score/(N-1)
    #             = score * N/(N-1) - sum/(N-1)
    group_sum = grouped.sum(dim=1, keepdim=True)
    n = float(g)
    # baseline_i = (group_sum - score_i) / (N - 1)
    # advantage_i = score_i - baseline_i
    advantages_grouped = rewards.view(-1, g) - (group_sum - grouped) / (n - 1.0)
    adv_flat = advantages_grouped.reshape(-1)

    # Scatter to token level and apply masked whitening
    # Expand [B] -> [B, T] using response_mask
    adv_tokens = adv_flat.unsqueeze(-1) * response_mask_f
    advantages = _masked_whiten(adv_tokens, response_mask_f, shift_mean=True)

    batch.tensors["advantages"] = advantages
    logger.debug(
        "RLOO advantages: mean=%.6f std=%.6f",
        adv_flat.mean().item(),
        adv_flat.std().item(),
    )
    return batch


# ---------------------------------------------------------------------------
# Helpers for group-based estimators
# ---------------------------------------------------------------------------


def _get_group_index(batch: DataProto, config: AtomRLConfig) -> np.ndarray:
    """Build per-sample group index from num_generations."""
    g = batch.meta.get("num_generations", config.algorithm.grpo.num_generations)
    bs = next(iter(batch.tensors.values())).shape[0]
    return np.arange(bs) // g


def _get_token_level_rewards(batch: DataProto) -> Tensor:
    """Get token-level rewards from batch, handling scalar vs token-level."""
    rewards = batch.tensors["rewards"]
    mask = _response_mask_from_batch(batch)
    if rewards.dim() == 1 or (rewards.dim() == 2 and rewards.shape[-1] == 1):
        return _scalar_rewards_to_token_rewards(
            rewards.squeeze(-1) if rewards.dim() == 2 else rewards, mask
        )
    return rewards


# ---------------------------------------------------------------------------
# New estimators  (verl/trainer/ppo/core_algos.py L334-865)
# ---------------------------------------------------------------------------


@register_adv_est("grpo_vectorized")
def compute_grpo_vectorized_advantage(
    batch: DataProto, config: AtomRLConfig
) -> DataProto:
    """Vectorized GRPO with optional Dr.GRPO mode.
    (verl/trainer/ppo/core_algos.py L334-358)
    """
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    cfg = config.algorithm.grpo
    g = cfg.num_generations
    grouped = rewards.view(-1, g)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)

    norm_by_std = batch.meta.get("norm_adv_by_std_in_grpo", True)
    if norm_by_std:
        adv = (grouped - mean) / std
    else:
        adv = grouped - mean

    mask = _response_mask_from_batch(batch)
    adv_flat = adv.reshape(-1)
    if mask is not None and mask.dim() > 1:
        batch.tensors["advantages"] = adv_flat.unsqueeze(-1) * mask.float()
    else:
        batch.tensors["advantages"] = adv_flat
    return batch


@register_adv_est("gdpo")
def compute_gdpo_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """GDPO: Group reward-Decoupled normalization (arxiv 2601.05242).
    Normalizes each reward dimension independently before aggregation.
    (verl/trainer/ppo/core_algos.py L361-468)
    """
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    cfg = config.algorithm.grpo
    g = cfg.num_generations

    mask = _response_mask_from_batch(batch)
    mask_f = (
        mask.float()
        if mask is not None
        else (
            torch.ones_like(rewards).unsqueeze(-1)
            if rewards.dim() == 1
            else torch.ones_like(rewards)
        )
    )

    token_rewards = _get_token_level_rewards(batch)
    score_list = [token_rewards]

    adv_sum = None
    for scores in score_list:
        s = scores.sum(dim=-1) if scores.dim() > 1 else scores
        grouped = s.view(-1, g)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
        normed = ((grouped - mean) / std).reshape(-1)
        if mask is not None and mask.dim() > 1:
            normed_tok = normed.unsqueeze(-1) * mask_f
        else:
            normed_tok = normed
        adv_sum = normed_tok if adv_sum is None else adv_sum + normed_tok

    if mask is not None and mask.dim() > 1:
        advantages = _masked_whiten(adv_sum, mask_f, shift_mean=True) * mask_f
    else:
        advantages = adv_sum
    batch.tensors["advantages"] = advantages
    return batch


@register_adv_est("grpo_passk")
def compute_grpo_passk_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """Pass@K variant: only best response per group gets advantage.
    (verl/trainer/ppo/core_algos.py L471-530, arxiv 2503.19595)
    """
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    index = _get_group_index(batch, config)

    scores = rewards.clone()
    advantages = torch.zeros_like(scores)
    id2scores: dict[int, list] = defaultdict(list)
    id2indices: dict[int, list] = defaultdict(list)

    with torch.no_grad():
        for i in range(scores.shape[0]):
            id2scores[index[i]].append(scores[i])
            id2indices[index[i]].append(i)

        for idx, group_scores in id2scores.items():
            r = torch.stack(group_scores)
            if r.numel() < 2:
                continue
            topk, topk_idx = torch.topk(r, 2)
            i_max = id2indices[idx][topk_idx[0].item()]
            adv = topk[0] - topk[1]
            std = torch.std(r).clamp_min(1e-8)
            advantages[i_max] = adv / std

    mask = _response_mask_from_batch(batch)
    if mask is not None and mask.dim() > 1:
        batch.tensors["advantages"] = advantages.unsqueeze(-1) * mask.float()
    else:
        batch.tensors["advantages"] = advantages
    return batch


@register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_pp_baseline_advantage(
    batch: DataProto, config: AtomRLConfig
) -> DataProto:
    """RF++ with group baseline subtraction and whitening.
    (verl/trainer/ppo/core_algos.py L533-584, arxiv 2501.03262)
    """
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    index = _get_group_index(batch, config)
    mask = _response_mask_from_batch(batch)
    mask_f = mask.float() if mask is not None else None

    scores = rewards.clone()
    id2score: dict[int, list] = defaultdict(list)
    id2mean: dict[int, Tensor] = {}

    with torch.no_grad():
        for i in range(scores.shape[0]):
            id2score[index[i]].append(scores[i])
        for idx, group_scores in id2score.items():
            if len(group_scores) <= 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            else:
                id2mean[idx] = torch.mean(torch.stack(group_scores))
        for i in range(scores.shape[0]):
            scores[i] = scores[i] - id2mean[index[i]]

        if mask_f is not None and mask_f.dim() > 1:
            adv = scores.unsqueeze(-1).expand_as(mask_f) * mask_f
            adv = _masked_whiten(adv, mask_f, shift_mean=True) * mask_f
        else:
            adv = scores

    batch.tensors["advantages"] = adv
    return batch


@register_adv_est("remax")
def compute_remax_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """ReMax with reward baseline subtraction.
    (verl/trainer/ppo/core_algos.py L732-765, arxiv 2310.10505)
    """
    rewards = _get_token_level_rewards(batch)
    mask = _response_mask_from_batch(batch)
    mask_f = mask.float() if mask is not None else torch.ones_like(rewards)

    baselines = batch.tensors.get("reward_baselines")
    if baselines is None:
        baselines = torch.zeros(rewards.shape[0], device=rewards.device)
    if baselines.dim() > 1:
        baselines = baselines.squeeze(-1)

    with torch.no_grad():
        returns = (rewards * mask_f).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - baselines.unsqueeze(-1) * mask_f

    batch.tensors["advantages"] = advantages
    batch.tensors["returns"] = returns
    return batch


@register_adv_est("opo")
def compute_opo_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """OPO length-weighted baseline (arxiv 2505.23585).
    (verl/trainer/ppo/core_algos.py L639-690)
    """
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    index = _get_group_index(batch, config)
    mask = _response_mask_from_batch(batch)
    resp_len = (
        mask.float().sum(dim=-1)
        if mask is not None and mask.dim() > 1
        else torch.ones_like(rewards)
    )

    scores = rewards.clone()
    id2score: dict[int, list] = defaultdict(list)
    id2len: dict[int, list] = defaultdict(list)
    id2bsl: dict[int, Tensor] = {}

    with torch.no_grad():
        for i in range(scores.shape[0]):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(resp_len[i])
        for idx, group_scores in id2score.items():
            if len(group_scores) <= 1:
                id2bsl[idx] = torch.tensor(0.0, device=scores.device)
            else:
                st = torch.stack(group_scores)
                lt = torch.stack(id2len[idx])
                id2bsl[idx] = (lt * st).sum() / lt.sum().clamp(min=1)
        for i in range(scores.shape[0]):
            scores[i] = scores[i] - id2bsl[index[i]]

    if mask is not None and mask.dim() > 1:
        batch.tensors["advantages"] = scores.unsqueeze(-1) * mask.float()
    else:
        batch.tensors["advantages"] = scores
    return batch


@register_adv_est("gpg")
def compute_gpg_advantage(batch: DataProto, config: AtomRLConfig) -> DataProto:
    """GPG advantage with alpha scaling.
    (verl/trainer/ppo/core_algos.py L768-828)
    """
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    index = _get_group_index(batch, config)
    mask = _response_mask_from_batch(batch)

    scores = rewards.clone()
    id2score: dict[int, list] = defaultdict(list)
    id2mean: dict[int, Tensor] = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = float(bsz) / max(int(m.item()), 1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx, group_scores in id2score.items():
            if len(group_scores) <= 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            else:
                id2mean[idx] = torch.mean(torch.stack(group_scores))
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]])

    if mask is not None and mask.dim() > 1:
        batch.tensors["advantages"] = scores.unsqueeze(-1) * mask.float()
    else:
        batch.tensors["advantages"] = scores
    return batch


@register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_advantage(
    batch: DataProto, config: AtomRLConfig
) -> DataProto:
    """Vectorized RLOO advantage estimation.
    (verl/trainer/ppo/core_algos.py L831-865)
    """
    rewards = batch.tensors["rewards"]
    if rewards.dim() > 1:
        rewards = rewards.squeeze(-1)
    index = _get_group_index(batch, config)
    mask = _response_mask_from_batch(batch)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(
            rewards.device
        )
        c = torch.bincount(inv)[inv].float()
        group_sum = torch.zeros(
            inv.max() + 1, device=rewards.device, dtype=rewards.dtype
        )
        group_sum.scatter_add_(0, inv, rewards)
        adv = ((c * rewards - group_sum[inv]) / (c - 1).clamp_min(1)) * (c > 1)

    if mask is not None and mask.dim() > 1:
        batch.tensors["advantages"] = adv.unsqueeze(-1) * mask.float()
    else:
        batch.tensors["advantages"] = adv
    return batch
