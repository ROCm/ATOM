# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Qwen3.5-MoE forwards take the mono path.

``install_mono_decode`` wraps the loaded model (and the MTP drafter's); runners view ATOM's weights
in place and build each launch width on its first step. The configuration is checked once
(``config_refusal``); a step is routed only when it is a pure decode of up to ``MAX_TOKENS`` rows.
Every per-step input read must be TP-uniform: the kernel waits on every peer rank, so a per-rank flag
such as ``is_dummy_run`` must never gate routing (ranks would deadlock).
"""

import functools
import logging
from dataclasses import dataclass, fields

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.models.qwen3_5.mono.check import forward_checked
from atom.models.qwen3_5.mono.config import (
    GLUON_VERIFY_TOKENS,
    MAX_TOKENS,
    MAX_VERIFY_TOKENS,
    Q35Config,
    StepInputError,
)
from atom.models.qwen3_5.mono.layout import BS
from atom.mono.plan.shard import Shard, ShardError
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.mono.runtime.route import LazyRunner
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# the TP sizes served
SUPPORTED_TP = (4,)

# the checkpoint widths the kernel is built for: hf_config field -> Q35Config field
_WIDTHS = {
    "hidden_size": "hidden",
    "linear_num_key_heads": "linear_k_heads",
    "linear_num_value_heads": "linear_v_heads",
    "linear_key_head_dim": "linear_k_dim",
    "linear_value_head_dim": "linear_v_dim",
    "linear_conv_kernel_dim": "conv_kernel",
    "num_attention_heads": "heads",
    "num_key_value_heads": "kv_heads",
    "head_dim": "head_dim",
    "num_experts": "n_experts",
    "num_experts_per_tok": "top_k",
    "moe_intermediate_size": "inter",
    "shared_expert_intermediate_size": "shared_inter",
}

# settings the kernel bakes in: hf_config field -> (kernel's value, ATOM's value when absent; None:
# must be present)
_FIXED = {
    "rms_norm_eps": (Q35Config().eps, None),
    "hidden_act": ("silu", None),
    "attn_output_gate": (True, None),
    "attention_bias": (False, False),
    # Qwen3NextAttention's q / k / v projection bias
    "qkv_bias": (False, False),
    # the router renormalizes its top-k (qwen3_next's default)
    "norm_topk_prob": (True, True),
    "mlp_only_layers": ([], []),
    "decoder_sparse_step": (1, 1),
    # Qwen3-Next's per-layer output scales: the kernel has none
    "layer_scale": (False, False),
}


def rotary_factor(hf):
    """The share of a head RoPE rotates, read as ``qwen3_next`` reads it (1.0 when absent)."""
    rope = getattr(hf, "rope_parameters", None) or {}
    return rope.get("partial_rotary_factor", 1.0)


def baseline_switches() -> list[tuple[str, bool]]:
    """ATOM switches that change what the original decode computes, as (name and setting, whether set
    so); the kernel reproduces the defaults only."""
    return [
        (
            "ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0",
            not envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE,
        ),
        (
            "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0",
            not envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION,
        ),
        ("ATOM_USE_UNIFIED_ATTN=1", envs.ATOM_USE_UNIFIED_ATTN),
        ("ATOM_FORCE_ATTN_TRITON=1", envs.ATOM_FORCE_ATTN_TRITON),
        ("ATOM_ENABLE_GDN_DECODE_LOSSY_FAST=1", envs.ATOM_ENABLE_GDN_DECODE_LOSSY_FAST),
        ("ATOM_USE_TRITON_GEMM=1", envs.ATOM_USE_TRITON_GEMM),
        ("ATOM_USE_TRITON_MOE=1", envs.ATOM_USE_TRITON_MOE),
        ("ATOM_USE_TRITON_MOE_DECODE=1", envs.ATOM_USE_TRITON_MOE_DECODE),
        ("ATOM_MOE_GU_ITLV=1", envs.ATOM_MOE_GU_ITLV),
        ("ATOM_USE_MODEL_SENSITIVE_RMSNORM=1", envs.ATOM_USE_MODEL_SENSITIVE_RMSNORM),
        # the UT-transform replay (``auto`` takes it only past 11 verify tokens, beyond MTP-5)
        ("ATOM_REPLAYSSM_ROUTE=ut", envs.ATOM_REPLAYSSM_ROUTE == "ut"),
        # past this many rows the original path stops using the fused q / k norm + RoPE + cache kernel
        (
            f"ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS < {MAX_TOKENS} with the QK fusion",
            envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION
            and envs.ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS < MAX_TOKENS,
        ),
    ]


@dataclass(frozen=True)
class Dims:
    """Checks the checkpoint has the 397B-A17B shapes the kernel is built for and that they split
    evenly over ``tp``."""

    hf_config: object
    tp: int

    def __post_init__(self):
        hf, ref = self.hf_config, Q35Config()
        names = {f.name for f in fields(Q35Config)}
        for theirs, ours in _WIDTHS.items():
            assert ours in names, ours
            got = getattr(hf, theirs, None)
            if got != getattr(ref, ours):
                raise ShardError(
                    f"{theirs} {got}: the kernel is built for {getattr(ref, ours)}"
                )
        for theirs, (want, absent) in _FIXED.items():
            got = getattr(hf, theirs, absent)
            if got != want:
                raise ShardError(f"{theirs} {got!r}: the kernel is built for {want!r}")
        got = rotary_factor(hf)
        if got is None or got * ref.head_dim != ref.rope_dim:
            raise ShardError(
                f"partial_rotary_factor {got!r}: the kernel rotates {ref.rope_dim} of {ref.head_dim} dims"
            )
        shard = Shard(self.tp)
        shard.split(hf.linear_num_key_heads, "GDN key heads")
        shard.split(hf.num_attention_heads, "attention heads")
        shard.split(hf.moe_intermediate_size, "expert intermediate")


def final_norm(norm, x):
    """Gemma RMSNorm of the f32 residual stream ``x`` in f32, rounded once to bf16 (as the original
    fused all-reduce norm)."""
    return norm.forward_static(norm.weight.data, norm.variance_epsilon, x, None).to(
        torch.bfloat16
    )


def spec_steps(atom_config) -> int:
    """Drafted tokens a request verifies a step at most (0 without a drafter)."""
    spec = atom_config.speculative_config
    return (spec.num_speculative_tokens or 0) if spec is not None else 0


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use mono, or None."""
    spec = atom_config.speculative_config
    k = spec_steps(atom_config)
    verify_max = MAX_VERIFY_TOKENS if envs.ATOM_PA_FLYDSL else GLUON_VERIFY_TOKENS
    hf = getattr(atom_config.hf_config, "text_config", atom_config.hf_config)
    dims = functools.partial(Dims, hf)
    checks = common_refusals(atom_config, dims, SUPPORTED_TP) + [
        (atom_config.kv_cache_dtype == "fp8", f"kv cache {atom_config.kv_cache_dtype}"),
        (
            atom_config.kv_cache_block_size == BS,
            f"KV block size {atom_config.kv_cache_block_size}",
        ),
        (not atom_config.load_dummy, "dummy weights"),
        (
            getattr(atom_config, "torch_dtype", torch.bfloat16) == torch.bfloat16,
            f"model dtype {getattr(atom_config, 'torch_dtype', None)}",
        ),
        (
            spec is None or spec.method == "mtp",
            f"speculative method {spec.method if spec else None}",
        ),
        (
            k + 1 <= verify_max,
            f"a verify of {k + 1} tokens a request > {verify_max}"
            + (
                ""
                if envs.ATOM_PA_FLYDSL
                else " (pa_decode_gluon's; past it ATOM_PA_FLYDSL's)"
            ),
        ),
        (
            k == 0 or getattr(hf, "mtp_num_hidden_layers", 1) == 1,
            f"mtp_num_hidden_layers {getattr(hf, 'mtp_num_hidden_layers', None)}",
        ),
        # plain decode on ReplaySSM keeps records the kernel's plain path does not
        (
            k > 0 or not envs.ATOM_ENABLE_REPLAYSSM,
            "ReplaySSM without speculative decoding",
        ),
    ]
    checks += [(not on, name) for name, on in baseline_switches()]
    # ATOM_MONO_CHECK reruns the original layers from the host: eager plain decode only
    checks += [
        (
            not envs.ATOM_MONO_CHECK or (atom_config.enforce_eager and k == 0),
            "ATOM_MONO_CHECK without --enforce-eager or with a drafter",
        )
    ]
    return first_refusal(checks)


def replay_cap(atom_config) -> int:
    """ReplaySSM's record buffer length (gdn_attn's rule), or 0 for snapshot GDN state."""
    k = spec_steps(atom_config)
    on = k > 0 if envs.ATOM_ENABLE_REPLAYSSM is None else envs.ATOM_ENABLE_REPLAYSSM
    return max(envs.ATOM_REPLAYSSM_CACHE_LEN, 2 * (k + 1)) if on and k > 0 else 0


def step_supported(input_ids, T: int, replay: bool) -> bool:
    """Is this forward a pure decode of whole requests, up to ``MAX_TOKENS`` rows?"""
    rows = input_ids.numel()
    if not 0 < rows <= MAX_TOKENS or rows % T:
        return False
    fwd = get_forward_context()
    c = fwd.context
    # no per-rank flag such as is_dummy_run may gate this (see the module docstring)
    if c is None or c.is_prefill or getattr(c, "is_draft", False):
        return False
    if getattr(fwd, "ubatch_slices", None) is not None:
        return False
    g = getattr(fwd.attn_metadata, "gdn_metadata", None)
    if g is None or g.num_prefills != 0:
        return False
    if T == 1:
        return g.spec_sequence_masks is None and not getattr(g, "replayssm", False)
    # an MTP verify on the GDN state kind the runner was built for
    if (
        g.spec_sequence_masks is None
        or g.num_decodes != 0
        or g.spec_state_indices_tensor is None
        or bool(getattr(g, "replayssm", False)) != replay
    ):
        return False
    # ReplaySSM: the kernel's flush rule matches the commit's only while both count the same window
    return (
        g.slot_idx is not None and getattr(g, "replayssm_max_query_len", None) == T
        if replay
        else g.spec_state_indices_tensor.shape[1] == T
    )


@functools.cache
def _step_refused(path: str, why: str) -> None:
    """Log once that a step was refused before launch (same on every rank)."""
    logger.warning("Qwen3.5 mono %s: a step keeps the original path: %s", path, why)


class MonoDecodeModel(nn.Module):
    """The loaded Qwen3.5-MoE model, with its supported decode steps routed to mono."""

    def __init__(self, model: nn.Module, runner) -> None:
        super().__init__()
        self.model = model
        self._runner = runner
        self._mono = LazyRunner(lambda: runner, "Qwen3.5 mono decode")

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw
    ):
        if (
            inputs_embeds is None
            and intermediate_tensors is None
            and not kw
            # the engine's warmup runs before the KV cache is bound
            and get_forward_context().kv_cache_data
            and step_supported(input_ids, self._runner.T, self._runner.replay_cap > 0)
            and self._mono.ready(input_ids.numel())
        ):
            try:
                if envs.ATOM_MONO_CHECK:
                    return forward_checked(self._mono.runner, input_ids, positions)
                return self._mono.runner.forward(input_ids, positions)
            except StepInputError as e:
                _step_refused("decode", str(e))
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kw
        )

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class MonoDraftModel(nn.Module):
    """The loaded MTP draft (``Qwen3_5MTP``) with its decoder layer on mono; ATOM keeps the rest."""

    def __init__(self, draft: nn.Module, runner) -> None:
        super().__init__()
        self.model = draft
        self._runner = runner
        self._mono = LazyRunner(lambda: runner, "Qwen3.5 mono draft")

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kw,
    ):
        fwd = get_forward_context()
        c = fwd.context
        rows = input_ids.numel()
        if (
            inputs_embeds is None
            and intermediate_tensors is None
            and not kw
            and c is not None
            and c.is_draft
            and not c.is_prefill
            and fwd.kv_cache_data
            and self._runner.step_t(rows) is not None
            and self._mono.ready(rows)
        ):
            # Qwen3_5MultiTokenPredictor.forward, its decoder layer on mono
            m = self.model.model
            e = m.pre_fc_norm_embedding(m.embed_input_ids(input_ids))
            h = m.pre_fc_norm_hidden(hidden_states)
            h = get_tp_group().all_gather(m.fc(torch.cat([e, h], dim=-1)))
            try:
                return final_norm(m.norm, self._runner.layer(h, positions))
            except StepInputError as e:
                _step_refused("draft", str(e))
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            **kw,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(
    model: nn.Module, atom_config, drafter, metadata_builder
) -> nn.Module:
    """``model`` wrapped for mono decode, or ``model`` itself when mono does not serve this
    deployment. Collective."""
    del metadata_builder
    if not envs.ATOM_MONO_ENABLE:
        return model
    why = config_refusal(atom_config)
    if why is not None:
        logger.info("Qwen3.5 mono decode off: %s", why)
        return model
    # import FlyDSL only for a deployment mono serves
    try:
        from atom.models.qwen3_5.mono.draft_runner import Qwen35MonoDraftRunner
        from atom.models.qwen3_5.mono.runner import Qwen35MonoDecodeRunner
    except ImportError as e:
        logger.warning("Qwen3.5 mono decode off: %s", e)
        return model
    try:
        steps = int(drafter.mtp_k) if drafter is not None else 0
        runner = Qwen35MonoDecodeRunner(
            model, atom_config, steps, replay_cap(atom_config)
        )
    except MonoUnsupported as why:
        logger.warning("Qwen3.5 mono decode off: %s", why)
        return model
    draft = getattr(drafter, "model", None) if drafter is not None else None
    if draft is not None and hasattr(
        getattr(draft, "model", None), "pre_fc_norm_hidden"
    ):
        try:
            draft_runner = Qwen35MonoDraftRunner(draft, atom_config, steps)
            draft_runner.prebuild()
            drafter.model = MonoDraftModel(draft, draft_runner)
        except MonoUnsupported as why:
            logger.warning("Qwen3.5 mono draft off: %s", why)
    if atom_config.enforce_eager:
        runner.prebuild()
    return MonoDecodeModel(model, runner)
