# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 Mono installation and deployment admission.

The executor stays in :mod:`runner` and its serving lifecycle in :mod:`session`;
this module owns the policy boundary
that selects ordinary or MTP target decode and keeps unsupported deployments on
the native path.
"""

from __future__ import annotations

import logging

from atom.models.deepseek_v4.mono.profile import (
    MTP3_TARGET_PROFILE,
    ORDINARY_PROFILE,
)
from atom.models.deepseek_v4.mono.session import V4MonoSession
from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")


def install_mono_decode(model, atom_config, drafter, metadata_builder):
    """Install the pure-TP8 executor under the configured routing policy."""
    policy = str(getattr(atom_config, "mono_decode", "auto")).strip().lower()
    if policy not in ("auto", "off", "force"):
        raise ValueError(
            "mono_decode must be one of {'auto', 'off', 'force'}, " f"got {policy!r}"
        )
    if policy == "off":
        logger.info("V4 Mono off: mono_decode policy")
        return model
    why = None
    spec = atom_config.speculative_config
    if spec is None and drafter is None:
        profile = ORDINARY_PROFILE
    elif (
        spec is not None
        and spec.method == "mtp"
        and int(spec.num_speculative_tokens or 0) == 3
        and drafter is not None
        and not getattr(drafter, "is_block_drafter", False)
        and int(getattr(drafter, "mtp_k", 3)) == 3
    ):
        profile = MTP3_TARGET_PROFILE
    else:
        profile = None
        why = "only ordinary decode or the exact MTP3 target profile is supported"
    if why is None:
        if (
            atom_config.tensor_parallel_size != 8
            or atom_config.enable_expert_parallel
            or atom_config.enable_dp_attention
            or atom_config.enable_tbo
            or atom_config.pipeline_parallel_size != 1
            or atom_config.prefill_context_parallel_size != 1
            or atom_config.decode_context_parallel_size != 1
            or atom_config.parallel_config.data_parallel_size != 1
        ):
            why = "only pure TP8 without TBO/context parallel is supported"
        elif (
            atom_config.kv_cache_dtype != "fp8"
            or atom_config.index_cache_dtype != "fp4"
        ):
            why = "FP8 KV / FP4 index is required"
        elif getattr(atom_config, "enable_prefix_caching", False):
            why = "prefix caching is outside the qualified V4 Mono profile"
        elif profile is ORDINARY_PROFILE and (
            not getattr(atom_config.hf_config, "use_index_cache", False)
            or int(getattr(atom_config.hf_config, "index_topk_freq", 1)) != 4
        ):
            why = (
                "ordinary profile requires use_index_cache=true and "
                "index_topk_freq=4"
            )
        elif profile is MTP3_TARGET_PROFILE and (
            getattr(atom_config.hf_config, "use_index_cache", True)
            or int(getattr(atom_config.hf_config, "index_topk_freq", 1)) != 1
        ):
            why = (
                "MTP3 target profile requires use_index_cache=false and "
                "index_topk_freq=1"
            )
        elif getattr(atom_config, "enable_rapidserve", False) or getattr(
            atom_config, "disagg_is_decode", False
        ):
            why = (
                "IPC/disaggregated cache binding is outside the first exact-row "
                "profile"
            )
        elif atom_config.compilation_config.cudagraph_mode.name not in (
            "FULL",
            "NONE",
        ):
            why = "only FULL graph or eager is supported"
        elif atom_config.compilation_config.level == 1:
            why = (
                "outer fullgraph torch.compile would include the host dispatch; "
                "use level 0/3"
            )
    if why is not None:
        if policy == "force":
            raise MonoUnsupported(f"V4 Mono force policy: {why}")
        logger.info("V4 Mono auto kept native path: %s", why)
        return model
    model._v4_mono = V4MonoSession(model, atom_config, metadata_builder, profile)
    return model
