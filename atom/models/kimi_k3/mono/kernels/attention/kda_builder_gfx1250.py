# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 Kimi K1 declarations and rank-local resident builder.

The declaration binds compile-time wave32 geometry, K1 compute, mailbox
operations, and an exact LDS type specification without importing FlyDSL.
``build_local`` emits the complete rank-local pipeline used by the product
runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from importlib import import_module

from atom.models.kimi_k3.mono.kernels.attention.mailbox_contract import (
    GFX1250_MAILBOX,
    K1_REQUIRED_CAPABILITIES,
)
from atom.models.kimi_k3.mono.kernels.attention.protocol import (
    K1Protocol,
    ProgressProof,
    ResidencyAssumption,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    ALIGNMENT,
    Gfx1250KimiBuildPlan,
    KimiLdsPlan,
    LdsRegion,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import TokenTile
from atom.mono.plan.architecture import GFX1250
from atom.mono.plan.layout import pair_layout

TIMELINE_POINTS = 15
HIDDEN = 7168
HEAD_DIM = 128
VALUE_CHUNK = 16
HEADS = 12
STATE_CHUNKS = HEAD_DIM // VALUE_CHUNK
ATTENTION_REDUCTION_GROUPS = 8
LDS_PAD = 8
# The local resident pipeline is intentionally independent of gfx950's
# persistent-kernel transport arguments (ReplaySSM and peer-memory buffers).
K1_ABI_NAMES = (
    "prefix",
    "delta",
    "blocks",
    "blk_sm",
    "blk_sr",
    "ares_nw",
    "ares_qk",
    "in_nw",
    "w_in",
    "w_fb",
    "conv_w",
    "conv_st",
    "cs_seq",
    "cs_dim",
    "cs_tok",
    "a_log",
    "dt_bias",
    "on_w",
    "rstate",
    "rs_seq",
    "st_idx",
    "st_stride",
    "num_acc",
    "core_out",
    "scratch",
)


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(frozen=True)
class LdsField:
    """One relative field within a 16-byte-aligned LDS struct."""

    name: str
    offset: int
    bytes: int

    @property
    def end(self) -> int:
        return self.offset + self.bytes


@dataclass(frozen=True)
class LdsTypeField:
    """FlyDSL array annotation expressed without importing FlyDSL."""

    name: str
    dtype: str
    elements: int
    alignment: int
    offset: int

    @property
    def bytes(self) -> int:
        return self.elements * {"f32": 4, "i32": 4, "i64": 8}[self.dtype]


@dataclass(frozen=True)
class K1LdsTypeSpec:
    """Host-checkable source of truth consumed by the FlyDSL type emitter."""

    smem: tuple[LdsTypeField, ...]
    role_variants: tuple[tuple[str, tuple[LdsTypeField, ...]], ...]

    def variant(self, name: str) -> tuple[LdsTypeField, ...]:
        return next(fields for variant, fields in self.role_variants if variant == name)


def _fields(*items: tuple[str, int]) -> tuple[LdsField, ...]:
    result = []
    offset = 0
    for name, size in items:
        if size < 0:
            raise ValueError(f"{name}: negative LDS size {size}")
        offset = _align(offset)
        result.append(LdsField(name, offset, size))
        offset += size
    return tuple(result)


def _field_struct_bytes(fields: tuple[LdsField, ...]) -> int:
    return _align(max((field.end for field in fields), default=0))


@dataclass(frozen=True)
class RoleLdsVariant:
    """One member of K1's role union, with offsets relative to the union."""

    name: str
    fields: tuple[LdsField, ...]

    def __post_init__(self) -> None:
        previous_end = 0
        for field in self.fields:
            if field.offset % ALIGNMENT:
                raise ValueError(f"{self.name}.{field.name}: unaligned offset")
            if field.offset < previous_end:
                raise ValueError(f"{self.name}.{field.name}: overlaps")
            previous_end = field.end

    @property
    def bytes(self) -> int:
        return _field_struct_bytes(self.fields)


@dataclass(frozen=True)
class K1RoleLdsPlan:
    """K1's sequential Smem allocation followed by one role union."""

    arena: KimiLdsPlan
    smem_fields: tuple[LdsField, ...]
    role_variants: tuple[RoleLdsVariant, ...]
    xrow: int

    @property
    def smem(self) -> LdsRegion:
        return self.arena.regions[0]

    @property
    def role(self) -> LdsRegion:
        return self.arena.regions[1]

    def variant(self, name: str) -> RoleLdsVariant:
        try:
            return next(item for item in self.role_variants if item.name == name)
        except StopIteration as error:
            raise ValueError(f"unknown K1 LDS role {name!r}") from error

    def absolute_fields(self, name: str) -> tuple[LdsField, ...]:
        """Fields of one union member as absolute LDS byte offsets."""

        return tuple(
            LdsField(field.name, self.role.offset + field.offset, field.bytes)
            for field in self.variant(name).fields
        )


COMPUTE_BINDINGS = (
    ("kda_zero", "zero_accumulator"),
    ("kda_gemv_loads", "gemv_loads"),
    ("kda_gemv_slab", "gemv_slab"),
    ("kda_gemv_store", "store_accumulator"),
    ("kda_gemv_row_sum", "row_sum"),
    ("kda_gate_loads", "gate_loads"),
    ("kda_gate_compute", "gate_compute"),
    ("kda_state_loads", "state_loads"),
    ("kda_qk_normalize", "qk_normalize"),
    ("kda_conv_inputs", "kda_conv_inputs"),
    ("kda_conv_compute", "kda_conv_compute"),
    ("kda_head", "kda_head"),
    ("kda_gates", "kda_gates"),
    ("kda_recur", "kda_recur"),
    ("kda_norm", "kda_norm"),
    ("attention_reduce32", "attention_reduce32"),
    ("attention_gate", "attention_gate"),
    ("attention_gate_tile", "attention_gate_tile"),
    ("attention_mix", "attention_mix"),
    ("attention_mix_tile", "attention_mix_tile"),
)

DEVICE_PROTOCOL_BINDINGS = (
    ("workgroup_barrier", "workgroup_barrier"),
    ("wait_vector_stores", "wait_vector_stores"),
    ("device_fence_acquire", "device_fence_acquire"),
    ("mailbox_store_device", "mailbox_store_device"),
    ("mailbox_load_device", "mailbox_load_device"),
    ("mailbox_poll_device", "mailbox_poll_device"),
    ("mailbox_reset_device", "mailbox_reset_device"),
    ("mailbox_publish_device", "mailbox_publish_device"),
    ("epoch_store_system_release", "epoch_store_system_release"),
    ("epoch_load_system_acquire", "epoch_load_system_acquire"),
    ("epoch_fence_system_release", "epoch_fence_system_release"),
    ("epoch_fence_system_acquire", "epoch_fence_system_acquire"),
    ("epoch_arrive_system_atomic", "epoch_arrive_system_atomic"),
    ("emit_role_lds_types", "emit_role_lds_types"),
)

UNSUPPORTED_DEVICE_PROTOCOL = (
    "gfx1250 XGMI peer payload+tag vector mailbox atomicity/cache policy",
)


@dataclass(frozen=True)
class K1KernelDeclaration:
    """Static inputs sufficient to start a FlyDSL gfx1250 K1 declaration."""

    name: str
    threads: int
    waves: int
    blocks: int
    tokens: int
    qlen: int
    nblocks: int
    token_tiles: tuple[TokenTile, ...]
    xrow: int
    gemv_unit: int
    lds: K1RoleLdsPlan
    backend_module: str
    protocol: K1Protocol
    compute_bindings: tuple[tuple[str, str], ...] = COMPUTE_BINDINGS
    device_protocol_bindings: tuple[tuple[str, str], ...] = DEVICE_PROTOCOL_BINDINGS
    unsupported_device_protocol: tuple[str, ...] = UNSUPPORTED_DEVICE_PROTOCOL

    @property
    def known_block_size(self) -> tuple[int, int, int]:
        return self.threads, 1, 1

    @property
    def candidate_grid(self) -> tuple[int]:
        """Unvalidated persistent grid; never launched by this skeleton."""

        return (self.blocks,)

    @property
    def static_context(self) -> tuple[tuple[str, int], ...]:
        return (
            ("S", self.tokens),
            ("L", self.qlen),
            ("NS", self.nblocks + 1),
            ("NB", self.nblocks),
            ("THREADS", self.threads),
            ("WAVES", self.waves),
            ("WAVE_SIZE", GFX1250.wave_size),
            ("TOKEN_TILES", len(self.token_tiles)),
            ("XROW", self.xrow),
            ("GEMV_UNIT", self.gemv_unit),
            (
                "RED_ELEMENTS",
                self.lds.smem_fields[0].bytes // 4,
            ),
        )

    @property
    def lds_type_spec(self) -> K1LdsTypeSpec:
        return role_lds_type_spec(self.lds)

    def bind_compute_backend(self, backend) -> dict[str, object]:
        """Bind the proven backend API without emitting a kernel."""

        bindings = {}
        for context_name, attribute in self.compute_bindings:
            try:
                bindings[context_name] = getattr(backend, attribute)
            except AttributeError as error:
                raise ValueError(f"gfx1250 K1 backend lacks {attribute!r}") from error
        return bindings

    def bind_device_protocol(self, backend) -> dict[str, object]:
        """Bind only evidenced gfx1250 barrier/drain operations."""

        bindings = {}
        for context_name, attribute in self.device_protocol_bindings:
            try:
                bindings[context_name] = getattr(backend, attribute)
            except AttributeError as error:
                raise ValueError(
                    f"gfx1250 protocol backend lacks {attribute!r}"
                ) from error
        return bindings


@dataclass(frozen=True)
class BuildRefusal:
    code: str
    missing_contracts: tuple[str, ...]


@dataclass(frozen=True)
class K1BuildResult:
    declaration: K1KernelDeclaration
    kernel: object | None
    logical_progress: ProgressProof
    refusal: BuildRefusal
    complete: bool = False
    launch_count: int = 0

    @property
    def runnable(self) -> bool:
        return self.complete


@dataclass(frozen=True)
class K1LocalBuildSpec:
    """All scalar build-time contracts needed by rank-local K1."""

    qlen: int
    delta: bool
    state_len: int
    eps: float
    out_eps: float
    onorm_eps: float
    lower_bound: float
    write_idx: int
    debug: bool = False
    timeline: bool = False
    kda0: int = 0
    xp: int = 0
    gemv_unit: int = 2

    def validate(self, plan: Gfx1250KimiBuildPlan) -> None:
        if not 1 <= self.qlen <= 16 or plan.tokens % self.qlen:
            raise ValueError("local K1 qlen must be in [1, 16] and divide tokens")
        if type(self.delta) is not bool:
            raise TypeError("local K1 delta must be bool")
        if self.debug:
            raise ValueError("resident gfx1250 K1 does not expose debug taps")
        if self.state_len < self.qlen + 3:
            raise ValueError(
                "local K1 state_len must cover qlen plus the three-entry "
                "causal-convolution history"
            )
        for name, value in (
            ("eps", self.eps),
            ("out_eps", self.out_eps),
            ("onorm_eps", self.onorm_eps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"local K1 {name} must be finite and positive")
        if not math.isfinite(self.lower_bound):
            raise ValueError("local K1 lower_bound must be finite")
        if plan.nblocks is None:
            raise ValueError("local K1 needs explicit nblocks")
        if self.write_idx < -1 or self.write_idx > plan.nblocks:
            raise ValueError(f"local K1 write_idx must be -1 or in [0, {plan.nblocks}]")
        if not 0 <= self.kda0 < plan.blocks:
            raise ValueError("local K1 kda0 must name a block in the grid")
        if self.xp != 0:
            raise ValueError("local K1 resident kernel does not admit timing xp")
        if self.gemv_unit not in (1, 2):
            raise ValueError("local K1 gemv_unit must be 1 or 2")
        if self.timeline:
            raise ValueError(
                "local K1 resident kernel does not expose timeline taps"
            )

    @property
    def stage_order(self) -> tuple[str, ...]:
        return ("resident",)

    @property
    def launch_count(self) -> int:
        return len(self.stage_order)


def role_lds_plan(
    plan: Gfx1250KimiBuildPlan,
    *,
    qlen: int,
    gemv_unit: int = 2,
) -> K1RoleLdsPlan:
    """Construct K1's exact Smem + role-union LDS byte layout."""

    if plan.nblocks is None:
        raise ValueError("gfx1250 K1 LDS needs explicit nblocks")
    if not 1 <= qlen <= 16 or plan.tokens % qlen:
        raise ValueError(
            f"qlen must be in [1, 16] and divide {plan.tokens}, got {qlen}"
        )
    if plan.waves != 8:
        raise ValueError(f"gfx1250 K1 skeleton requires 8 waves, got {plan.waves}")
    if gemv_unit not in (1, 2):
        raise ValueError("gfx1250 K1 gemv_unit must be 1 or 2")

    sources = plan.nblocks + 1
    if 2 * ATTENTION_REDUCTION_GROUPS * sources > plan.threads:
        raise ValueError(
            f"gfx1250 K1 attention supports at most {plan.threads // 16} "
            f"sources, got {sources}"
        )
    tile_rows = plan.token_tiling.lds_rows
    plan.attention_seam_lds_bytes(sources)
    red_elements = plan.kda_accumulator_lds_elements
    smem_fields = _fields(
        ("red", red_elements * 4),
        ("tls", TIMELINE_POINTS * 8),
    )
    smem_bytes = _field_struct_bytes(smem_fields)

    contract = plan.kda_matrix_contract()
    groups = contract.projection_groups
    slabs = tuple(
        (first, min(gemv_unit, groups - first))
        for first in range(0, groups, gemv_unit)
    )
    widest_groups = max(groups for _, groups in slabs)
    xrow = widest_groups * contract.group_columns + LDS_PAD
    projection_bytes = tile_rows * xrow * 2
    if smem_bytes + projection_bytes > plan.lds_bytes:
        raise ValueError(
            f"gfx1250 K1 projection needs {smem_bytes + projection_bytes} LDS bytes, "
            f"candidate has {plan.lds_bytes}"
        )

    attention = RoleLdsVariant(
        "attention",
        _fields(("vals", tile_rows * sources * HEAD_DIM * 4)),
    )
    projection = RoleLdsVariant(
        "projection",
        _fields(("xl", tile_rows * xrow * 2)),
    )
    state_width = HEAD_DIM if qlen == 1 else VALUE_CHUNK
    recurrence = RoleLdsVariant(
        "recurrence",
        _fields(
            ("fa", qlen * HEAD_DIM * 4),
            ("qkl", qlen * 2 * HEAD_DIM * 4),
            ("decay", qlen * HEAD_DIM * 4),
            ("vloc", qlen * state_width * 4),
            ("ol", qlen * state_width * 4),
            ("beta", 16 * 4),
            ("slots", 16 * 4),
            ("onrm", qlen * STATE_CHUNKS * 4),
        ),
    )
    variants = attention, projection, recurrence
    role_bytes = max(variant.bytes for variant in variants)
    regions = (
        LdsRegion("smem", 0, smem_bytes),
        LdsRegion("role", smem_bytes, role_bytes),
    )
    arena = KimiLdsPlan(plan.lds_bytes, regions, (slabs,))
    return K1RoleLdsPlan(arena, smem_fields, variants, xrow)


def role_lds_type_spec(layout: K1RoleLdsPlan) -> K1LdsTypeSpec:
    """Translate the byte planner into exact FlyDSL scalar array annotations."""

    dtypes = {
        "red": "f32",
        "tls": "i64",
        "vals": "f32",
        "xl": "i32",
        "fa": "f32",
        "qkl": "f32",
        "decay": "f32",
        "vloc": "f32",
        "ol": "f32",
        "beta": "f32",
        "slots": "i32",
        "onrm": "f32",
    }

    def convert(fields: tuple[LdsField, ...]) -> tuple[LdsTypeField, ...]:
        result = []
        for field in fields:
            dtype = dtypes[field.name]
            item_bytes = {"f32": 4, "i32": 4, "i64": 8}[dtype]
            if field.bytes % item_bytes:
                raise ValueError(f"{field.name}: size is not a whole {dtype} array")
            item = LdsTypeField(
                field.name,
                dtype,
                field.bytes // item_bytes,
                ALIGNMENT,
                field.offset,
            )
            if item.bytes != field.bytes or item.offset % item.alignment:
                raise ValueError(f"{field.name}: LDS type does not match byte plan")
            result.append(item)
        return tuple(result)

    spec = K1LdsTypeSpec(
        convert(layout.smem_fields),
        tuple(
            (variant.name, convert(variant.fields)) for variant in layout.role_variants
        ),
    )
    smem_bytes = _field_struct_bytes(
        tuple(LdsField(item.name, item.offset, item.bytes) for item in spec.smem)
    )
    if smem_bytes != layout.smem.bytes:
        raise ValueError("FlyDSL Smem type spec does not match planned size")
    for variant in layout.role_variants:
        typed = spec.variant(variant.name)
        typed_bytes = _field_struct_bytes(
            tuple(LdsField(item.name, item.offset, item.bytes) for item in typed)
        )
        if typed_bytes != variant.bytes:
            raise ValueError(f"FlyDSL {variant.name} type does not match byte plan")
    return spec


def declare(
    plan: Gfx1250KimiBuildPlan,
    *,
    qlen: int,
    gemv_unit: int = 2,
) -> K1KernelDeclaration:
    """Create the gfx1250 K1 declaration up to kernel-body emission."""

    lds = role_lds_plan(plan, qlen=qlen, gemv_unit=gemv_unit)
    GFX1250_MAILBOX.require(*K1_REQUIRED_CAPABILITIES)
    assert plan.nblocks is not None
    name = (
        f"k3_mono_kda_pre_gfx1250_s{plan.tokens}_l{qlen}"
        f"_nb{plan.nblocks}_u{gemv_unit}"
    )
    return K1KernelDeclaration(
        name=name,
        threads=plan.threads,
        waves=plan.waves,
        blocks=plan.blocks,
        tokens=plan.tokens,
        qlen=qlen,
        nblocks=plan.nblocks,
        token_tiles=plan.token_tiling.tiles,
        xrow=lds.xrow,
        gemv_unit=gemv_unit,
        lds=lds,
        backend_module=("atom.models.kimi_k3.mono.kernels.attention.kda_gfx1250"),
        protocol=K1Protocol(
            blocks=plan.blocks,
            threads=plan.threads,
            tokens=plan.tokens,
            qlen=qlen,
            sources=plan.nblocks + 1,
        ),
    )


def declare_local(
    plan: Gfx1250KimiBuildPlan,
    *,
    qlen: int,
    gemv_unit: int = 2,
) -> K1KernelDeclaration:
    """Declare K1 for the external-collective path.

    K1's handoffs are between CTAs of this rank through ``scratch``.  It does
    not need a TP peer mapping; the historical XGMI note belongs only to the
    fully fused K2 transport and is removed from this local declaration.
    """

    return replace(
        declare(plan, qlen=qlen, gemv_unit=gemv_unit),
        unsupported_device_protocol=(),
    )


def local_scratch_layout(
    declaration: K1KernelDeclaration,
) -> dict[str, tuple[int, int]]:
    """Lay out resident K1's tagged pairs, generation, and x handoff."""

    layout = pair_layout(declaration.protocol.mailbox_pairs.items())
    end = max(offset + size for offset, size in layout.values())
    layout["sync"] = (_align(end), 8)
    end = layout["sync"][0] + layout["sync"][1]
    layout["x"] = (_align(end), declaration.tokens * HIDDEN * 2)
    return layout


def build_local(
    plan: Gfx1250KimiBuildPlan,
    *,
    spec: K1LocalBuildSpec,
) -> K1BuildResult:
    """Emit the complete rank-local K1 pipeline."""

    spec.validate(plan)
    declaration = declare_local(
        plan,
        qlen=spec.qlen,
        gemv_unit=spec.gemv_unit,
    )
    declaration = replace(
        declaration,
        protocol=replace(declaration.protocol, kda0=spec.kda0),
    )
    progress = declaration.protocol.prove_progress(
        ResidencyAssumption(all_declared_ctas_resident=True)
    )
    kernel = None
    dependency_refusal = ()
    try:
        kernels = import_module(
            "atom.models.kimi_k3.mono.kernels.attention." "kda_resident_gfx1250"
        )
    except ModuleNotFoundError as error:
        if (error.name or "").split(".", 1)[0] not in {"aiter", "flydsl"}:
            raise
        dependency_refusal = (
            f"resident K1 not emitted because {error.name} is unavailable",
        )
    except RuntimeError as error:
        if "No gfx arch found" not in str(error):
            raise
        dependency_refusal = (
            "resident K1 not emitted because no gfx target is visible",
        )
    else:
        kernel = kernels.build_resident_k1(plan, declaration, spec)

    if kernel is not None:
        return K1BuildResult(
            declaration,
            kernel,
            progress,
            BuildRefusal("none", ()),
            complete=True,
            launch_count=kernel.launch_count,
        )

    refusal = BuildRefusal(
        code="gfx1250-k1-local-target-unavailable",
        missing_contracts=dependency_refusal
        + (
            (
                "focused target compile/correctness is required before "
                "the complete resident local callable can be returned"
            ),
        ),
    )
    return K1BuildResult(declaration, kernel, progress, refusal)
