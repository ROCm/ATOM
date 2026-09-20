# SPDX-License-Identifier: MIT
"""Side-effect-free native PAGE reuse semantics, shared with router fixtures."""

from dataclasses import dataclass

PLANNER_VERSION = "page-reuse-v1"


@dataclass(frozen=True)
class ReusePlan:
    kind: str
    precompute_start: int
    precompute_end: int
    load_start: int
    transfer_end: int
    reuse_end: int
    eligible: bool
    reject_reason: str = ""
    planner_version: str = PLANNER_VERSION


def loadable_prefix(cpu_prefix: int, prompt_tokens: int, chunk: int) -> int:
    """Native retrieve requires complete chunks and recomputes the final token."""
    if chunk <= 0 or prompt_tokens < 0 or cpu_prefix < 0:
        raise ValueError("invalid PAGE prefix geometry")
    return max(0, min(cpu_prefix, prompt_tokens - 1)) // chunk * chunk


def plan_reuse(
    prompt_tokens: int,
    hbm_prefix: int,
    cpu_prefix: int,
    chunk: int,
    min_load: int,
    *,
    allow_cpu: bool = True,
) -> ReusePlan:
    """Plan a continuous CPU prefix; suffix-only residency is never an input hit."""
    if min_load < 0 or chunk <= 0 or min(prompt_tokens, hbm_prefix, cpu_prefix) < 0:
        raise ValueError("invalid PAGE reuse inputs")
    hbm = min(hbm_prefix, max(0, prompt_tokens - 1))
    transfer_end = loadable_prefix(cpu_prefix, prompt_tokens, chunk)
    boundary = (hbm + chunk - 1) // chunk * chunk
    reason = ""
    if not allow_cpu:
        reason = "cpu_disabled"
    elif transfer_end <= boundary:
        reason = "no_loadable_suffix"
    elif transfer_end - boundary < min_load:
        reason = "too_small"
    if reason:
        return ReusePlan("hbm", hbm, hbm, hbm, hbm, hbm, False, reason)
    return ReusePlan(
        "cpu" if hbm == boundary else "precompute_cpu",
        hbm,
        boundary,
        boundary,
        transfer_end,
        transfer_end,
        True,
    )


def load_decision(hbm: int, cpu: int, chunk: int, min_load: int) -> tuple[bool, str]:
    """Allocator-time decision; the existing handoff path handles misalignment."""
    if cpu <= hbm:
        return False, "hbm_satisfies_after_alloc"
    if hbm % chunk:
        return False, "unaligned_hbm_prefill"
    if cpu - hbm < min_load:
        return False, "too_small"
    return True, "aligned_large_hit"


def cache_load_policy(hints) -> str:
    """Unknown planner versions retain the engine's default policy."""
    if not hints or hints.get("planner_version") != PLANNER_VERSION:
        return "auto"
    policy = hints.get("cache_load_policy", "auto")
    if policy not in ("auto", "skip"):
        raise ValueError("cache_load_policy must be auto or skip")
    return policy
