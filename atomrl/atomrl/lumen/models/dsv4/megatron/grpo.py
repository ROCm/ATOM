# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Native GRPO rollout loading + policy loss for DSV4 Megatron finetune (no Ray)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core import mpu


@dataclass
class RolloutSample:
    group_index: int
    tokens: list[int]
    response_length: int
    reward: float
    rollout_log_probs: list[float]


def load_fake_rollout(path: str) -> list[RolloutSample]:
    payload = torch.load(path, weights_only=False)
    samples: list[RolloutSample] = []
    for raw in payload["samples"]:
        reward = raw.get("reward", 0.0)
        if isinstance(reward, dict):
            reward = float(next(iter(reward.values()), 0.0))
        samples.append(
            RolloutSample(
                group_index=int(raw.get("group_index", 0)),
                tokens=list(raw["tokens"]),
                response_length=int(raw["response_length"]),
                reward=float(reward),
                rollout_log_probs=list(raw.get("rollout_log_probs") or []),
            )
        )
    return samples


def compute_grpo_advantages(samples: Sequence[RolloutSample]) -> list[torch.Tensor]:
    """Group-wise reward normalization (GRPO), returns per-sample 1D advantage tensors."""
    groups: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(sample.reward)

    group_stats: dict[int, tuple[float, float]] = {}
    for group_id, rewards in groups.items():
        tensor = torch.tensor(rewards, dtype=torch.float32)
        std = tensor.std(unbiased=False)
        group_stats[group_id] = (tensor.mean().item(), std.item() if std > 0 else 1.0)

    advantages: list[torch.Tensor] = []
    for sample in samples:
        mean, std = group_stats[sample.group_index]
        adv = (sample.reward - mean) / std
        advantages.append(
            torch.full((sample.response_length,), adv, dtype=torch.float32)
        )
    return advantages


def _pad_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def build_bshd_rollout_batch(
    samples: Sequence[RolloutSample],
    advantages: Sequence[torch.Tensor],
    *,
    pad_multiplier: int,
    device: torch.device,
    min_seq_len: int = 0,
) -> dict[str, Any]:
    """Build a bshd rollout dict for CP=1 (Megatron GRPO actor layout)."""
    total_lengths = [len(sample.tokens) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    max_seq_len = _pad_to_multiple(max(total_lengths), pad_multiplier)
    if min_seq_len > max_seq_len:
        max_seq_len = _pad_to_multiple(min_seq_len, pad_multiplier)

    token_rows: list[torch.Tensor] = []
    loss_mask_rows: list[torch.Tensor] = []
    rollout_logprob_rows: list[torch.Tensor] = []
    full_loss_masks: list[torch.Tensor] = []

    for sample, adv, total_len, resp_len in zip(
        samples, advantages, total_lengths, response_lengths, strict=True
    ):
        prompt_len = total_len - resp_len
        tokens = torch.tensor(sample.tokens, dtype=torch.long, device=device)
        tokens = F.pad(tokens, (0, max_seq_len - total_len), value=0)

        response_mask = torch.zeros(resp_len, dtype=torch.float32, device=device)
        if resp_len > 0:
            response_mask[:] = 1.0

        old_log_probs = sample.rollout_log_probs
        if len(old_log_probs) != resp_len:
            old_log_probs = ([-0.5] * resp_len) if resp_len else []
        rollout_lp = torch.tensor(old_log_probs, dtype=torch.float32, device=device)

        padded_loss_mask = F.pad(
            response_mask, (prompt_len - 1, max_seq_len - total_len + 1), value=0.0
        )
        token_rows.append(tokens)
        loss_mask_rows.append(response_mask)
        rollout_logprob_rows.append(rollout_lp)
        full_loss_masks.append(padded_loss_mask)

    return {
        "tokens": token_rows,
        "unconcat_tokens": [
            row[:total_len]
            for row, total_len in zip(token_rows, total_lengths, strict=True)
        ],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "loss_masks": loss_mask_rows,
        "full_loss_masks": torch.stack(full_loss_masks),
        "advantages": [adv.to(device) for adv in advantages],
        "rollout_log_probs": rollout_logprob_rows,
        "max_seq_lens": [max_seq_len] * len(samples),
    }


class RolloutDataIterator:
    """Micro-batch iterator over a rollout dict (lists per sample)."""

    def __init__(self, rollout_data: dict[str, Any], micro_batch_size: int) -> None:
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.offset = 0
        self.num_samples = len(rollout_data["tokens"])

    def reset(self) -> None:
        self.offset = 0

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.micro_batch_size)

    def get_next(self, keys: Sequence[str]) -> dict[str, Any]:
        start = self.offset
        end = min(start + self.micro_batch_size, self.num_samples)
        self.offset = end
        batch: dict[str, Any] = {}
        for key in keys:
            value = self.rollout_data.get(key)
            if value is None:
                batch[key] = None
            elif (
                key == "full_loss_masks"
                or key == "max_seq_lens"
                or isinstance(value, list)
            ):
                batch[key] = value[start:end]
            else:
                batch[key] = value
        if "tokens" in keys:
            batch["tokens"] = torch.stack(batch["tokens"])
            flm = batch.get("full_loss_masks")
            if flm is not None and not isinstance(flm, torch.Tensor):
                batch["full_loss_masks"] = torch.stack(flm)
        return batch


def _extract_response_log_probs_bshd(
    logits: torch.Tensor,
    batch: dict[str, Any],
) -> list[torch.Tensor]:
    """Slice policy logits to response segments (bshd, CP=1)."""
    logits = logits.view(-1, logits.size(-1)).float()
    log_probs: list[torch.Tensor] = []
    max_seq_lens = batch["max_seq_lens"]
    for row, total_len, resp_len, max_seq_len in zip(
        range(len(batch["unconcat_tokens"])),
        batch["total_lengths"],
        batch["response_lengths"],
        max_seq_lens,
        strict=True,
    ):
        end = max_seq_len * row + total_len
        start = end - resp_len
        logits_chunk = logits[start - 1 : end - 1]
        tokens_chunk = batch["unconcat_tokens"][row][-resp_len:]
        if resp_len == 0:
            log_probs.append(logits_chunk.new_zeros((0,)))
            continue
        log_probs.append(_vocab_parallel_log_probs(logits_chunk, tokens_chunk))
    return log_probs


def _vocab_parallel_log_probs(
    logits: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    from megatron.core.fusions.fused_cross_entropy import (
        fused_vocab_parallel_cross_entropy,
    )

    if logits.numel() == 0:
        return logits.new_zeros((0,))
    logits_ = logits.unsqueeze(1)
    tokens_ = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(
        logits_, tokens_, mpu.get_tensor_model_parallel_group()
    ).squeeze(1)


def grpo_policy_loss(
    logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO/GRPO clipped policy loss on response tokens."""
    log_probs = _extract_response_log_probs_bshd(logits, batch)
    old_log_probs = batch["rollout_log_probs"]
    advantages = batch["advantages"]

    pg_chunks: list[torch.Tensor] = []
    clip_fracs: list[torch.Tensor] = []
    kl_chunks: list[torch.Tensor] = []

    for new_lp, old_lp, adv, mask in zip(
        log_probs, old_log_probs, advantages, batch["loss_masks"], strict=True
    ):
        if new_lp.numel() == 0:
            continue
        ppo_kl = old_lp - new_lp
        ratio = torch.exp(-ppo_kl)
        pg1 = -ratio * adv
        pg2 = -ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_high) * adv
        clipped = torch.maximum(pg1, pg2)
        pg_chunks.append((clipped * mask).sum() / torch.clamp_min(mask.sum(), 1.0))
        clip_fracs.append((pg2 > pg1).float() * mask)
        kl_chunks.append((ppo_kl * mask).sum() / torch.clamp_min(mask.sum(), 1.0))

    if not pg_chunks:
        zero = logits.new_zeros(())
        return zero, {"loss": 0.0, "pg_loss": 0.0, "pg_clipfrac": 0.0, "ppo_kl": 0.0}

    loss = torch.stack(pg_chunks).mean()
    clipfrac = torch.stack(
        [
            frac.sum() / torch.clamp_min(mask.sum(), 1.0)
            for frac, mask in zip(clip_fracs, batch["loss_masks"], strict=True)
        ]
    ).mean()
    ppo_kl = torch.stack(kl_chunks).mean()
    metrics = {
        "loss": float(loss.detach()),
        "pg_loss": float(loss.detach()),
        "pg_clipfrac": float(clipfrac.detach()),
        "ppo_kl": float(ppo_kl.detach()),
    }
    return loss, metrics
