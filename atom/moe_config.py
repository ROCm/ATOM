# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE configuration shared by native and plugin execution.

Config snapshots process settings and resolves the logical mapping and transport
before workers start. Workers project the same mapping onto their launch rank
and validate their process groups. The resolvers never create a process group
or examine tensors.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from atom.config import Config


@dataclass(frozen=True)
class MoEBackendConfig:
    backend: Literal["standard", "mega"] = "standard"
    all2all_backend: Literal["auto", "mori", "rccl", "none"] = "auto"
    low_latency: bool = False
    mori_available: bool = False
    disable_mori: bool = False
    mori_v2: bool = False
    fp4_dispatch: bool = False
    combine_quant: str = "none"
    pcp_merge: bool = False

    def __post_init__(self):
        backend = str(self.backend).strip().lower()
        all2all = str(self.all2all_backend or "auto").strip().lower()
        if backend not in {"standard", "mega"}:
            raise ValueError(
                "moe_backend must be one of {'standard', 'mega'}, "
                f"got {self.backend!r}"
            )
        if all2all not in {"auto", "mori", "rccl", "none"}:
            raise ValueError(
                "moe_all2all_backend must be one of "
                "{'auto', 'mori', 'rccl', 'none'}, "
                f"got {self.all2all_backend!r}"
            )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "all2all_backend", all2all)

    def validate_request(self, *, enable_ep: bool, sequence_parallel_size: int) -> None:
        # SP selects Mega over native TP weights rather than the EP backend.
        if self.backend != "mega" or sequence_parallel_size > 1:
            return
        if not enable_ep:
            raise ValueError(
                "moe_backend='mega' requires expert parallelism; pass "
                "--enable-expert-parallel, or --sequence-parallel-size for SP MegaMoE."
            )
        if self.all2all_backend == "rccl":
            raise ValueError(
                "moe_backend='mega' owns its MoRI transport and cannot use the "
                "experimental RCCL prepare/finalize backend"
            )


def get_moe_backend_config(config: "Config") -> MoEBackendConfig:
    """Normalize entry-point flags once, without capturing a rank topology.

    Config carries this snapshot through spawn and plugin setup. Runtime
    decisions read the snapshot; the CLI request fields remain input values.
    """
    cached = config.moe_backend_config
    if cached is not None:
        return cached
    from atom.utils import envs

    result = MoEBackendConfig(
        backend=config.moe_backend,
        all2all_backend=config.moe_all2all_backend,
        low_latency=config.enable_low_latency,
        mori_available=find_spec("mori") is not None,
        disable_mori=envs.ATOM_DISABLE_MORI_EP,
        mori_v2=envs.ATOM_MORI_V2,
        fp4_dispatch=envs.ATOM_MORI_FP4_DISPATCH,
        combine_quant=envs.ATOM_MORI_COMBINE_QUANT,
        pcp_merge=envs.ATOM_PCP_MOE_MERGE,
    )
    config.moe_backend_config = result
    return result


@dataclass(frozen=True)
class MoEParallelMapping:
    tp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    dp_rank: int
    ep_rank: int
    use_ep: bool
    local_ep_size: int
    dp_logical_ratio: int = 1
    pcp_merge: bool = False
    dp_is_sharded: bool = False
    enable_dp_attention: bool = False

    @classmethod
    def make(
        cls,
        *,
        tp_size: int,
        dp_size: int,
        tp_rank: int = 0,
        dp_rank: int = 0,
        local_dp_size: int = 1,
        dp_logical_size: int = 0,
        enable_ep: bool = False,
        flatten_dp: bool = False,
        pcp_size: int = 1,
        pcp_rank: int = 0,
        merge_pcp: bool = False,
        enable_dp_attention: bool = False,
    ) -> "MoEParallelMapping":
        for name, size, rank in (
            ("TP", tp_size, tp_rank),
            ("DP", dp_size, dp_rank),
            ("PCP", pcp_size, pcp_rank),
        ):
            if size < 1 or not 0 <= rank < size:
                raise ValueError(f"Invalid {name} topology: size={size}, rank={rank}")
        if not 1 <= local_dp_size <= dp_size:
            raise ValueError("local_dp_size must be between 1 and dp_size")
        dp_logical = dp_logical_size or dp_size
        if dp_logical < dp_size or dp_logical % dp_size:
            raise ValueError("dp_logical_size must be a multiple of physical DP size")

        # Preserve the pre-mapping eligibility and per-node transport width.
        use_ep = enable_ep and dp_logical * tp_size > 1
        local_shards = local_dp_size * tp_size
        shard_size, shard_rank = tp_size, tp_rank
        if flatten_dp:
            shard_size *= dp_logical
            shard_rank += dp_rank * tp_size
        pcp_merge = merge_pcp and pcp_size > 1
        if pcp_merge:
            # Checkpoint ownership keeps the existing PCP x (DP x TP) order.
            # The transport process group's order is validated separately.
            shard_rank += pcp_rank * shard_size
            shard_size *= pcp_size
        return cls(
            tp_size=1 if use_ep else shard_size,
            tp_rank=0 if use_ep else shard_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            ep_size=shard_size if use_ep else 1,
            ep_rank=shard_rank if use_ep else 0,
            use_ep=use_ep,
            local_ep_size=local_shards if use_ep else 1,
            dp_logical_ratio=dp_logical // dp_size if flatten_dp and use_ep else 1,
            pcp_merge=pcp_merge,
            dp_is_sharded=flatten_dp,
            enable_dp_attention=enable_dp_attention,
        )


@dataclass(frozen=True)
class MoEExecutionPlan:
    mapping: MoEParallelMapping
    backend_config: MoEBackendConfig
    implementation: Literal["standard", "mega", "sp_mega"]
    transport: Literal["local", "dp_gather_scatter", "mori", "mori_v2", "rccl"]
    reason: str

    @property
    def uses_routed_dispatch(self) -> bool:
        return self.transport in {"mori", "mori_v2", "rccl"}

    @property
    def needs_dp_gather_scatter(self) -> bool:
        return self.transport == "dp_gather_scatter"

    @property
    def owns_dispatch_combine(self) -> bool:
        return self.implementation == "mega"

    @property
    def needs_prepare_finalize(self) -> bool:
        return self.uses_routed_dispatch and not self.owns_dispatch_combine

    @property
    def needs_pcp_collectives(self) -> bool:
        return self.mapping.pcp_merge

    @property
    def needs_tp_all_reduce(self) -> bool:
        return self.mapping.tp_size > 1 or self.mapping.ep_size > 1

    @property
    def use_fp4_dispatch(self) -> bool:
        return self.transport == "mori" and self.backend_config.fp4_dispatch

    @property
    def supports_tbo(self) -> bool:
        return self.transport in {"mori", "mori_v2"} or (
            self.transport == "local" and self.mapping.dp_logical_ratio == 1
        )

    @property
    def selected_all2all_backend(self) -> str | None:
        if self.transport in {"mori", "mori_v2"}:
            return "mori"
        return "rccl" if self.transport == "rccl" else None


def resolve_moe_execution_plan(
    mapping: MoEParallelMapping,
    backend: MoEBackendConfig,
    *,
    sequence_parallel_size: int = 1,
) -> MoEExecutionPlan:
    """Select a single row-transport owner for the resolved expert mapping."""
    implementation = backend.backend
    if implementation == "mega" and sequence_parallel_size > 1:
        # SP Mega uses native TP weights and decides per step whether it runs.
        implementation = "sp_mega"
    # Config validates Mega's existing EP request and RCCL restrictions.
    # Mega owns its kernels; surrounding row movement keeps the existing
    # all2all request/availability decision, including singleton/simulated DP.

    fallback = (
        "dp_gather_scatter"
        if mapping.enable_dp_attention and mapping.dp_size > 1
        else "local"
    )
    reason = (
        "DP-attention collective fallback" if fallback != "local" else "replicated rows"
    )
    if mapping.dp_size <= 1 or not mapping.use_ep:
        reason = "no DP expert dispatch group"
    elif mapping.dp_logical_ratio != 1:
        reason = "simulated DP has no matching dispatch peers"
    elif backend.all2all_backend == "none":
        reason = "routed all-to-all disabled explicitly"
    elif backend.all2all_backend == "rccl":
        return MoEExecutionPlan(
            mapping, backend, implementation, "rccl", "explicit RCCL"
        )
    elif not backend.disable_mori and backend.mori_available:
        return MoEExecutionPlan(
            mapping,
            backend,
            implementation,
            "mori_v2" if backend.mori_v2 else "mori",
            "MoRI available and enabled",
        )
    else:
        reason = "MoRI disabled" if backend.disable_mori else "MoRI unavailable"
    return MoEExecutionPlan(mapping, backend, implementation, fallback, reason)


def get_moe_execution_plan(config: "Config") -> MoEExecutionPlan:
    """Return the saved worker plan, or the saved rank-zero configuration plan."""
    worker_plan = config.worker_moe_execution_plan
    if worker_plan is not None:
        return worker_plan
    cached = config.moe_execution_plan
    if cached is not None:
        return cached
    from atom.parallel_mapping import get_parallel_mapping

    plan = resolve_moe_execution_plan(
        get_parallel_mapping(config).worker_topology().moe_for_rank(0),
        get_moe_backend_config(config),
        sequence_parallel_size=config.sequence_parallel_size,
    )
    config.moe_execution_plan = plan
    return plan
