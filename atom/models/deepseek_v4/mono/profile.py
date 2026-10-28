# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Serving profiles admitted by the DeepSeek-V4 Mono backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V4MonoProfile:
    """One serving contract; request width and token rows stay distinct."""

    name: str
    tokens_per_seq: int
    auto_request_widths: tuple[int, ...]
    max_requests: int
    continuation: bool
    # Explicit candidate selection is separate from deployment qualification.
    launch_rows: tuple[int, ...] = tuple(range(1, 9))
    ragged: bool = False
    variable_q: bool = False
    deployment_qualified: bool = True

    def capture_request_widths(self, declared) -> tuple[int, ...]:
        """Prepare C4's exact three-request tail without changing its ladder.

        The extra graph uses one exact S3 stack instead of serial S2+S1.
        It does not admit a standalone C3 deployment; other profiles keep
        their ladder. The cache adapter prepares the same execution widths.
        """
        widths = set(declared)
        if self.name == "ordinary" and 4 in widths:
            widths.add(3)
        return tuple(sorted(widths, reverse=True))

    def accepts_graph(self, running_bs: int, max_seqlen_q: int) -> bool:
        if self.variable_q:
            valid_q = 1 <= max_seqlen_q <= self.tokens_per_seq
        else:
            valid_q = max_seqlen_q == self.tokens_per_seq
        if not valid_q:
            return False
        if not 1 <= running_bs <= self.max_requests:
            return False
        return running_bs in self.auto_request_widths


ORDINARY_PROFILE = V4MonoProfile(
    "ordinary",
    tokens_per_seq=1,
    auto_request_widths=(1, 2, 4),
    max_requests=16,
    continuation=True,
)

MTP3_TARGET_PROFILE = V4MonoProfile(
    "mtp3_target",
    tokens_per_seq=4,
    auto_request_widths=(1, 2),
    # Two q4 requests fit the existing S8 target launch; wider multi-lane
    # serving regresses against native and is not automatically selected.
    max_requests=2,
    continuation=True,
)

# Native-resolved index schedules qualified for each serving profile. The
# dashboard default refreshes every CSA layer; ordinary decode also retains
# its qualified four-CSA reuse schedule. The runner binds the actual native
# layer decisions, so this table never rewrites checkpoint/model arguments.
_INDEX_POLICIES = {
    ORDINARY_PROFILE.name: ((False, 1), (True, 4)),
    MTP3_TARGET_PROFILE.name: ((False, 1),),
}


def dspark_target_profile(
    draft_tokens: int, *, confidence_schedule: bool, ragged: bool
) -> V4MonoProfile:
    """Describe target geometry without granting deployment qualification.

    K is the draft width; target verify includes its anchor, hence q <= K+1.
    Dynamic q is needed by confidence scheduling even without ragged packing.
    Draft execution and prefix transactions require independent qualification.
    """
    # The original PD recipe uses DSpark K3, independently of MTP3's
    # autoregressive drafter. Geometry selection does not qualify PD transfer.
    if type(draft_tokens) is not int or draft_tokens not in (3, 6, 7):
        raise ValueError("V4 DSpark target profiles require draft K3, K6 or K7")
    if ragged and not confidence_schedule:
        raise ValueError("ragged DSpark requires confidence scheduling")
    return V4MonoProfile(
        name=f"dspark_k{draft_tokens}_target",
        tokens_per_seq=draft_tokens + 1,
        auto_request_widths=tuple(range(1, 17)),
        max_requests=16,
        continuation=False,
        ragged=ragged,
        variable_q=confidence_schedule,
        deployment_qualified=False,
    )


def select_profile(atom_config, drafter) -> V4MonoProfile | None:
    """Select the decode contract independently of deployment admission."""
    spec = atom_config.speculative_config
    if spec is None and drafter is None:
        return ORDINARY_PROFILE
    if (
        spec is not None
        and spec.method == "mtp"
        and int(spec.num_speculative_tokens or 0) == 3
        and drafter is not None
        and not getattr(drafter, "is_block_drafter", False)
        and int(getattr(drafter, "mtp_k", 3)) == 3
    ):
        return MTP3_TARGET_PROFILE
    if (
        spec is not None
        and spec.method == "dspark"
        and drafter is not None
        and getattr(drafter, "is_block_drafter", False)
    ):
        k = getattr(drafter, "mtp_k", None)
        declared = spec.num_speculative_tokens
        dspark = atom_config.dspark
        if (
            type(k) is not int
            or k not in (3, 6, 7)
            or (declared is not None and declared != k)
            or bool(getattr(drafter, "uses_confidence_schedule", False))
            != bool(dspark.confidence_schedule)
            or (dspark.ragged and not dspark.confidence_schedule)
        ):
            return None
        return dspark_target_profile(
            k,
            confidence_schedule=bool(dspark.confidence_schedule),
            ragged=bool(dspark.ragged),
        )
    return None


def deployment_refusal(
    atom_config, profile: V4MonoProfile | None, *, model_args
) -> str | None:
    """First deployment refusal, in the serving policy's diagnostic order.

    These host-only checks stay outside the device configuration and source
    digest: reorganizing admission must not invalidate compiled kernels.
    Model policy comes from the native model's resolved arguments, so missing
    checkpoint fields cannot acquire different defaults at this boundary.
    """
    if profile is None:
        return "only ordinary decode or the exact MTP3 target profile is supported"
    if not profile.deployment_qualified:
        return (
            f"{profile.name} target contract is not deployment-qualified; "
            "DSpark state/draft and full-network accuracy gates remain open"
        )
    if (
        atom_config.tensor_parallel_size != 8
        or atom_config.enable_expert_parallel
        or atom_config.enable_dp_attention
        or atom_config.enable_tbo
        or getattr(atom_config, "enable_tbo_decode", False)
        or atom_config.pipeline_parallel_size != 1
        or atom_config.prefill_context_parallel_size != 1
        or atom_config.decode_context_parallel_size != 1
        or atom_config.parallel_config.data_parallel_size != 1
    ):
        return "only pure TP8 without TBO/context parallel is supported"
    if atom_config.kv_cache_dtype != "fp8" or atom_config.index_cache_dtype != "fp4":
        return "FP8 KV / FP4 index is required"
    if getattr(atom_config, "enable_prefix_caching", False):
        return "prefix caching is outside the qualified V4 Mono profile"
    index_policy = (model_args.use_index_cache, model_args.index_topk_freq)
    if index_policy not in _INDEX_POLICIES[profile.name] or (
        model_args.use_index_cache and model_args.index_topk_pattern is not None
    ):
        # A pattern overrides the native frequency. It must not acquire
        # qualification merely because its unused frequency happens to match.
        return (
            f"{profile.name} index schedule is outside the qualified native policies "
            f"{_INDEX_POLICIES[profile.name]}; custom index reuse patterns are unsupported"
        )
    if getattr(atom_config, "enable_rapidserve", False) or getattr(
        atom_config, "disagg_is_decode", False
    ):
        return "IPC/disaggregated cache binding is outside the first exact-row profile"
    graph_mode = atom_config.compilation_config.cudagraph_mode
    if graph_mode.uses_tr_hybrid() or graph_mode.decode_mode().name not in (
        "FULL",
        "NONE",
    ):
        return "only native FULL graph or eager is supported"
    if atom_config.compilation_config.level == 1:
        return (
            "outer fullgraph torch.compile would include the host dispatch; "
            "use level 0/3"
        )
    return None
