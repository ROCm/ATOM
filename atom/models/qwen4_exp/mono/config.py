# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The deployments Qwen3.8-Flash-Next mono serves, and why any other is refused.

These are the checks a configuration settles on its own; what only the
allocated caches show (their dtypes, shapes, the recurrent state's layout) is
checked when the runner binds them (``state.bind_state``)."""

from atom.mono.plan.shard import Shard
from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.utils import envs

from .kernels.config import from_dict
from .kernels.stages.qsa import PAGE
from .kernels.stages.qsa_fast import MAX_CONTEXT

# the kernels are TP1 / EP1 builds
SUPPORTED_TP = (1,)
# the rows (requests, one token each) a step serves at most: the batched
# kernel builds' (``kernels.layout.MAX_ROWS``)
MAX_ROWS = 4
# MTP draft tokens a verify step (one request's k + 1 tokens, a seq build)
# serves; the seq builds take up to ``MAX_ROWS`` rows, so k <= 3
SPEC_TOKENS = (1, 2, 3)


def longest_context(atom_config) -> int:
    """The longest context a step decodes: a verify step's k draft tokens sit
    past the request's last token."""
    return atom_config.max_model_len + getattr(
        atom_config.speculative_config, "num_speculative_tokens", 0
    )


def spec_refusal(spec) -> str | None:
    """Why the speculative decoding ``spec`` is not the mono's, or None."""
    if spec is None:
        return None
    if spec.method != "mtp":
        return f"speculative method {spec.method}"
    if spec.num_speculative_tokens not in SPEC_TOKENS:
        return f"{spec.num_speculative_tokens} speculative tokens"
    return None


def replayssm_on(atom_config) -> bool:
    """Is ReplaySSM on? Its default follows speculative decoding and its
    switch overrides it either way (``GDNAttentionMetadataBuilder``)."""
    switch = envs.ATOM_ENABLE_REPLAYSSM
    if switch is None:
        return atom_config.speculative_config is not None
    return switch


def geometry_refusal(hf_config) -> str | None:
    """Why the model's sizes are not the ones the kernels are built for."""
    try:
        g = from_dict(hf_config.to_dict())
    except (KeyError, ValueError) as e:
        return f"geometry: {e}"
    if len(g.ple_layers) != 1:
        return f"PLE layers {g.ple_layers}"
    try:
        [g.kind(i) for i in range(g.layers)]
    except AssertionError as e:
        return f"layer kinds: {e}"
    return None


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use Qwen3.8-Flash-Next mono, or None."""
    hf = atom_config.hf_config
    spec = spec_refusal(atom_config.speculative_config)
    longest = longest_context(atom_config)
    checks = [(envs.ATOM_MONO_ENABLE, "switched off")]
    # TP 1 only (``SUPPORTED_TP``): no width is split, so ``Shard`` stands in
    # for a ``Dims``
    checks += common_refusals(atom_config, Shard, SUPPORTED_TP)
    geometry = geometry_refusal(hf)
    checks += [
        (geometry is None, geometry),
        (
            atom_config.kv_cache_dtype in ("auto", "bf16", "bfloat16"),
            f"kv cache {atom_config.kv_cache_dtype}",
        ),
        # Q2-Q7 read ATOM's pages directly: 16 tokens, 4 index blocks a page
        (
            atom_config.kv_cache_block_size == PAGE,
            f"kv cache block size {atom_config.kv_cache_block_size} != {PAGE}",
        ),
        (
            getattr(hf, "mamba_ssm_dtype", "float32") == "float32",
            f"mamba_ssm_dtype {getattr(hf, 'mamba_ssm_dtype', None)}",
        ),
        # the QSA selection ranks contexts up to ``MAX_CONTEXT`` tokens
        (
            longest <= MAX_CONTEXT,
            f"max_model_len: contexts up to {longest} > {MAX_CONTEXT}",
        ),
        # MTP verify steps run the seq builds on ATOM's spec state layout
        (spec is None, spec),
        # an image's mRoPE positions are not its tokens' cache rows, and the
        # QSA caches then keep each token's positions, which mono does not
        # write; a text-only deployment (``--hf-overrides
        # '{"_multimodal_config": null}'``) has neither
        (getattr(atom_config, "multimodal_config", None) is None, "multimodal"),
        # the lossy decode keeps its state otherwise than B3 does
        (
            not envs.ATOM_ENABLE_GDN_DECODE_LOSSY_FAST,
            "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST",
        ),
        # ReplaySSM addresses the recurrent state as a contiguous KV pool,
        # which the VK storage B3 needs would break; with MTP it is on unless
        # switched off (``ATOM_ENABLE_REPLAYSSM=0``)
        (not replayssm_on(atom_config), "ReplaySSM"),
    ]
    return first_refusal(checks)
