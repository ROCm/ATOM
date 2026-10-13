# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which DeepSeek-V4 forwards take the mono path.

``install_mono_decode`` wraps the loaded model, the way V4.1's does, and every
other attribute passes through to it. Unlike V4.1's, the runner is created at
install, not on the first eligible step: it packs ~0.1 GB of each layer's
weights from the checkpoint, which must be resident before the KV cache is
sized. Each launch width's kernels and buffers are still built on its first step
(``LazyRunner`` -> ``runner.prepare``).

The configuration is checked once (``config_refusal``). A step is then routed
only when it is a decode of up to ``MAX_TOKENS`` rows (a graph's pad rows
included), however they split into requests: a token a request, or an MTP /
DSpark verify (rectangular, q-bucketed or ragged), the MTP / DSpark draft
staying on the original path.
Every per-step input read is TP-uniform: the kernel waits on every peer rank,
so ranks that disagreed would wait on each other for ever.
"""

import functools
import logging
from dataclasses import dataclass

import torch
from torch import nn

from atom.models.deepseek_v4.mono.config import SUPPORTED_HEADS
from atom.models.deepseek_v4.mono.runner import MAX_TOKENS, Dsv4MonoDecodeRunner
from atom.mono.plan.shard import Shard, ShardError
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.mono.runtime.route import LazyRunner
from atom.utils import envs
from atom.utils.forward_context import AttnState, get_forward_context

logger = logging.getLogger("atom")

# the TP sizes served (the kernel's per-rank widths are checked by ``Dims``)
SUPPORTED_TP = (8,)


@dataclass(frozen=True)
class Dims:
    """A TP rank's share of a V4 checkpoint's widths: the kernel holds 8 or 16
    attention heads a rank and whole o_proj groups."""

    hf_config: object
    tp: int

    def __post_init__(self):
        shard = Shard(self.tp)
        heads = shard.split(self.hf_config.num_attention_heads, "attention heads")
        if heads not in SUPPORTED_HEADS:
            raise ShardError(
                f"{heads} attention heads a rank at TP {self.tp}: the kernel holds"
                f" {SUPPORTED_HEADS}"
            )
        shard.split(self.hf_config.o_groups, "o_proj groups")
        shard.split(self.hf_config.moe_intermediate_size, "expert intermediate")


# the speculative methods served: MTP, and DSpark, whose taps of the target's layers
# the runner fills (``runner.AuxTaps``); another drafter's hooks on the target layers
# would never fire under mono
SPEC_METHODS = ("mtp", "dspark")


def spec_steps(atom_config) -> int:
    """Drafted tokens a request verifies a step at most (0 without a drafter)."""
    spec = atom_config.speculative_config
    return (spec.num_speculative_tokens or 0) if spec is not None else 0


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use mono, or None."""
    spec = atom_config.speculative_config
    k = spec_steps(atom_config)
    dims = functools.partial(Dims, atom_config.hf_config)
    checks = common_refusals(atom_config, dims, SUPPORTED_TP) + [
        (atom_config.kv_cache_dtype == "fp8", f"kv cache {atom_config.kv_cache_dtype}"),
        (
            atom_config.index_cache_dtype == "fp4",
            f"index cache {atom_config.index_cache_dtype}",
        ),
        (not atom_config.load_dummy, "dummy weights (packed from the checkpoint)"),
        (
            spec is None or spec.method in SPEC_METHODS,
            f"speculative method {spec.method if spec else None}",
        ),
        # a request's verify (its anchor and drafted tokens) fits one launch
        (k + 1 <= MAX_TOKENS, f"a verify of {k + 1} rows > {MAX_TOKENS}"),
    ]
    return first_refusal(checks)


def step_supported(input_ids: torch.Tensor) -> bool:
    """Is this forward a decode step of up to ``MAX_TOKENS`` rows, however they
    split into requests, with its caches bound?"""
    rows = input_ids.numel()
    if not 0 < rows <= MAX_TOKENS:
        return False
    fwd = get_forward_context()
    c = fwd.context
    # is_dummy_run is TP-uniform here: no DP (common_refusals)
    if c is None or c.is_prefill or c.is_dummy_run or c.is_draft:
        return False
    if fwd.ubatch_slices is not None:
        return False
    md = fwd.attn_metadata
    return (
        md is not None
        and getattr(md, "state", None) is AttnState.DECODE
        and md.swa_dest_rows is not None
    )


class MonoDecodeModel(nn.Module):
    """The loaded V4 model, with its supported decode steps routed to mono."""

    def __init__(self, model: nn.Module, runner: Dsv4MonoDecodeRunner) -> None:
        super().__init__()
        self.model = model
        self._runner = runner
        # the runner exists already; each width is prepared on its first step
        self._mono = LazyRunner(lambda: runner, "DeepSeek-V4 mono decode")

    def forward(self, input_ids, positions, inputs_embeds=None):
        if (
            inputs_embeds is None
            # the engine's warmup runs before the KV cache is bound
            and getattr(self.model.model.layers[0].attn, "unified_kv", None) is not None
            and step_supported(input_ids)
            # a step mono does not serve (more requests than its scope) takes the
            # original path: LazyRunner turns mono off for good on a refused width
            and self._runner.launch_width(input_ids.numel()) is not None
            and self._mono.ready(input_ids.numel())
        ):
            return self._mono.runner.forward(input_ids, positions)
        if inputs_embeds is None:
            return self.model(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(
    model: nn.Module, atom_config, drafter, metadata_builder
) -> nn.Module:
    """``model`` wrapped for mono decode, or ``model`` itself when this
    deployment is one mono does not serve. Collective: every TP rank installs."""
    del metadata_builder
    if not envs.ATOM_MONO_ENABLE:
        return model
    why = config_refusal(atom_config)
    if why is not None:
        logger.info("DeepSeek-V4 mono decode off: %s", why)
        return model
    try:
        # the rings ATOM sized: C_ROWS + the drafter's mtp_k (max_spec_steps)
        steps = int(drafter.mtp_k) if drafter is not None else 0
        runner = Dsv4MonoDecodeRunner(model, atom_config, steps, drafter)
    except MonoUnsupported as why:
        logger.warning("DeepSeek-V4 mono decode off: %s", why)
        return model
    return MonoDecodeModel(model, runner)
