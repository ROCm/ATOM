# SPDX-License-Identifier: Apache-2.0
"""MoE on aiter's MegaMoEGfx1250 over mori's cco (gfx1250), for ``ATOM_MORI_V2=1``.

The production mori v1 (``mori.ops.EpDispatchCombineOp``) is authored for
gfx942/950 HIP kernels and does not run on gfx1250. There, aiter's
MegaMoEGfx1250 owns the whole layer (dispatch -> expert GEMM -> combine), so
MoriV2ModularKernel hands it the layer and returns its output; the
prepare()/finalize() walk is never taken. The model-wide recipe (activation,
gate mode, quant type, padding, swiglu limit) is fixed at construction and the
per-layer weights/biases go to each forward(). ATOM_MEGA_STAGE1_FUSED and
ATOM_MEGA_STAGE2_FUSED become its recipe's dispatch and combine (an older
aiter's stage1_fused / stage2_fused): stage 1 picks the dispatch (flydsl compact
plan, or mori), stage 2 the combine (gemm2 epilogue P2P-writing each weighted
(token,k) result into the peers' staging so combine only barriers + sums, or
mori's combine over fused_moe's rows).

Shared experts are NOT fused in the mori EP+DP path (ATOM disables fusion there,
see topK.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config), so
topk_ids carry only routed expert ids and mori routes them cleanly.
"""

import contextlib
import logging
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from aiter import ActivationType, QuantType, dtypes
from aiter.dist.parallel_state import get_dp_group
from aiter.ops.flydsl.moe_common import GateMode

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

try:
    import mori
    from mori.cco import Communicator

    MORI_AVAILABLE = True
except ImportError:  # pragma: no cover
    mori = None  # type: ignore
    Communicator = None  # type: ignore
    MORI_AVAILABLE = False

logger = logging.getLogger("atom")


def _import_mega():
    """The gemm2-fused transport, which lives in aiter rather than the op-layer.

    aiter used to vendor its own copy of the v2 op-layer carrying the fused
    combine (aiter.ops.flydsl.dispatch_combine_v2), since that mode is a contract
    between the op and aiter's gemm2 epilogue. That copy is gone: it was
    refactored into kernels.mega_moe_gfx1250, which widened the contract to the
    whole layer -- dispatch, expert GEMM and combine are one object now, so ATOM
    configures and calls it instead of interleaving its own steps with it.
    """
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import (  # type: ignore
        MegaMoEGfx1250,
    )

    return MegaMoEGfx1250


@lru_cache(maxsize=1)
def _import_mega_recipe() -> SimpleNamespace | None:
    """aiter's MegaMoERecipe, its ExpertComputeConfig and MegaMoEGeometry, with
    the kinds a recipe picks; None on an aiter from before them, which takes
    stage1/stage2 flags instead."""
    try:
        from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import (  # type: ignore
            CombineKind,
            DispatchKind,
            ExpertComputeConfig,
            MegaMoEGeometry,
            MegaMoERecipe,
        )
    except ImportError:
        return None
    return SimpleNamespace(
        Recipe=MegaMoERecipe,
        Compute=ExpertComputeConfig,
        Geometry=MegaMoEGeometry,
        DispatchKind=DispatchKind,
        CombineKind=CombineKind,
    )


def _init_cco_comm(
    ep_size: int,
    ep_rank: int,
    ep_src_global_rank: int,
    per_rank_vmm: int,
) -> Any:
    """Collective: create a cco Communicator over the EP group, sized for one
    MegaMoE. Never shared: _MEGA_TRANSPORTS already dedups equal MegaMoEs, and
    two different ones would overrun a communicator sized for either.

    The mori cco unique-id is generated on the EP leader and broadcast over the
    EP gloo cpu_group (mirrors mori.shmem.shmem_torch_process_group_init but for
    the cco fabric). All EP ranks must call this together.
    """
    uid = Communicator.get_unique_id() if ep_rank == 0 else None
    uid = _broadcast_from_ep_group(uid, src=ep_src_global_rank)
    comm = Communicator.init(ep_size, ep_rank, uid, per_rank_vmm=per_rank_vmm)
    comm.barrier()
    logger.info(
        "[MORI-V2] cco Communicator ready: ep_rank=%d ep_size=%d "
        "per_rank_vmm=%.2fGiB",
        ep_rank,
        ep_size,
        per_rank_vmm / (1 << 30),
    )
    return comm


def _cco_per_rank_vmm(
    ep_size: int,
    hidden_dim: int,
    max_num_inp_token_per_rank: int,
    itemsize: int,
) -> int:
    """Size the cco symmetric VMM for the worst-case all-to-all: every rank could
    send all its tokens to one peer -> ws * M recv slots, plus a 2x headroom
    (tokens + combine buffers) and a fixed slack, matching test_moe_layer_ep.py.

    Only for an aiter whose MegaMoE has no required_vmm_bytes(), which sizes
    the arena exactly: with stage1_fused its dispatch rows are one per route
    rather than one per token, which outgrows this.
    """
    tok_bytes = max_num_inp_token_per_rank * hidden_dim * itemsize
    win_bytes = ep_size * tok_bytes * 2 + (1 << 24)
    return 2 * win_bytes + (1 << 28)


# Keyed by everything MegaMoE fixes at construction -- the recipe and geometry,
# plain values -- so the MoE layers of one model share a single instance and a
# single cco symmetric arena. The Situv2 betas are model-wide and stay out of the
# key; a layer whose betas differ from the shared instance's is refused
# (_check_same_situ).
_MEGA_TRANSPORTS: dict = {}

# $ATOM_MEGA_COMBINE_WIRE, the return trip's counterpart to $MEGA_DISPATCH_WIRE
# below, spelled the same bf16|fp8|fp4 way. Unlike the dispatch wire it is a
# free choice: combine moves post-expert tokens, so nothing downstream demands
# a particular width. aiter names the same formats after their MX block layout.
#
# PREFILL only by default -- the quant/dequant pair is a fixed per-token cost
# against a saving that scales with the tokens on the wire, so decode asks for
# bf16 back unless $ATOM_MEGA_DECODE_COMBINE_QUANT is on.
# See MoriV2PrepareAndFinalize.combine_wire_for_step.
_COMBINE_WIRES = {"bf16": "none", "fp8": "mxfp8", "fp4": "mxfp4"}
_MEGA_COMBINE_WIRE = envs.ATOM_MEGA_COMBINE_WIRE
if _MEGA_COMBINE_WIRE not in _COMBINE_WIRES:
    raise RuntimeError(
        f"ATOM_MEGA_COMBINE_WIRE must be one of {sorted(_COMBINE_WIRES)}, "
        f"got {_MEGA_COMBINE_WIRE!r}"
    )
_MEGA_DECODE_COMBINE_QUANT = envs.ATOM_MEGA_DECODE_COMBINE_QUANT

# aiter's stage2_fused: the gemm2-fused combine, or mori's combine over
# fused_moe's rows. Which combine wires each offers is aiter's to check.
_MEGA_STAGE2_FUSED = envs.ATOM_MEGA_STAGE2_FUSED


# bf16 | fp8 | fp4, and it must MATCH the expert GEMM's A operand -- on gfx1250
# that is fp4 unless AITER_FORCE_A8W4=1. A mismatch is a row-width error, not a
# slow path. Read once: init_mega_transport runs per MoE layer (61x for V4-Pro).
#
# Checked here as well as in aiter: this module passes dispatch_wire= down
# explicitly, so aiter's own env read -- and its guard -- never runs for us.
if os.environ.get("MEGA_WIRE") not in (None, os.environ.get("MEGA_DISPATCH_WIRE")):
    raise RuntimeError(
        "MEGA_WIRE was renamed to MEGA_DISPATCH_WIRE; update the launch script, "
        "the old name is no longer read"
    )
_MEGA_DISPATCH_WIRE = os.environ.get("MEGA_DISPATCH_WIRE", "bf16")
# aiter's stage1_fused requires a quantizing wire and stage2_fused; without
# either, dispatch runs on mori.
_MEGA_STAGE1_FUSED = (
    envs.ATOM_MEGA_STAGE1_FUSED
    and _MEGA_STAGE2_FUSED
    and _MEGA_DISPATCH_WIRE in ("fp8", "fp4")
)


# The one VMM granule a DevComm's resource window takes; init_mega_transport
# creates one on the communicator it sizes.
_CCO_DEVCOMM_BYTES = 2 << 20

# The communicators ATOM created for MegaMoE itself (see _mega_communicator);
# close_mega_transports() destroys them.
_OWN_COMMUNICATORS: list = []
# The all2all manager's communicator, once a MegaMoE has borrowed it. It is sized
# for that MegaMoE alone, and the manager destroys it.
_BORROWED_COMMUNICATOR: Any = None
# Set once every rank has been checked to run the same MegaMoE settings.
_SETTINGS_AGREED = False


def mori_v2_all2all_manager(device_communicator: Any) -> Any:
    """The EP group's MoRI v2 all2all manager, whose cco communicator MegaMoE
    borrows.

    Asked for by name before anything reads
    ``device_communicator.all2all_manager``: that property creates the default
    ("mori", v1) manager, and the backend cannot change afterwards. An aiter
    without ``get_or_create_all2all_manager`` (before ROCm/aiter#6517) or
    without the "mori_v2" backend (before ROCm/aiter#6618) has none to lend; it
    gets the default manager, and MegaMoE a communicator of its own.
    """
    get_manager = getattr(device_communicator, "get_or_create_all2all_manager", None)
    if get_manager is None:
        return device_communicator.all2all_manager
    try:
        from aiter.dist.device_communicators.all2all import (  # noqa: F401
            MoriV2All2AllManager,
        )
    except ImportError:
        return device_communicator.all2all_manager
    return get_manager("mori_v2")


def _mega_communicator(
    all2all_manager: Any,
    *,
    ep_size: int,
    ep_rank: int,
    ep_src_global_rank: int,
    per_rank_vmm: int,
) -> Any:
    """The cco communicator a MegaMoE allocates its arena on.

    The EP group's all2all manager owns it where it offers one (aiter's
    MoriV2All2AllManager): it is created once, at the capacity asked for, and
    destroyed with the process groups -- after close_mega_transports(). It
    cannot grow, so it serves the first MegaMoE only; a second, differently
    shaped one gets a communicator of its own, as does any MegaMoE on an aiter
    whose manager lends none.
    """
    global _BORROWED_COMMUNICATOR
    borrow = getattr(all2all_manager, "get_or_create_cco_communicator", None)
    if borrow is not None and _BORROWED_COMMUNICATOR is None:
        _BORROWED_COMMUNICATOR = borrow(per_rank_vmm)
        return _BORROWED_COMMUNICATOR
    if borrow is not None:
        logger.warning(
            "[MegaMoEGfx1250] the EP group's cco communicator is sized for the "
            "first MegaMoE; this differently shaped one gets its own"
        )
    comm = _init_cco_comm(ep_size, ep_rank, ep_src_global_rank, per_rank_vmm)
    _OWN_COMMUNICATORS.append(comm)
    return comm


def close_mega_transports() -> None:
    """Close every MegaMoE this process built, newest first, then destroy the
    communicators ATOM created for them.

    The arenas live on cco communicators; the all2all manager destroys its own
    with the process groups, so this has to run first (ModelRunner.exit does).
    Collective, as building them was. Every close is attempted; an error is
    raised once all have run.
    """
    global _BORROWED_COMMUNICATOR
    transports = list(_MEGA_TRANSPORTS.values())
    communicators = list(_OWN_COMMUNICATORS)
    _MEGA_TRANSPORTS.clear()
    _OWN_COMMUNICATORS.clear()
    _BORROWED_COMMUNICATOR = None
    # Required order: each MegaMoE, newest first, then its DevComm, while their
    # communicator is alive; then the communicators ATOM created. The borrowed
    # one goes later, in destroy_dist_env(). ExitStack runs callbacks last-in
    # first-out, and runs the rest when one raises.
    with contextlib.ExitStack() as stack:
        for comm in communicators:
            stack.callback(comm.destroy)
        for mega in transports:
            dev_comm = getattr(mega, "_atom_dev_comm", None)
            if dev_comm is not None:
                stack.callback(dev_comm.close)
            stack.callback(mega.close)


def _gather_from_ep_group(value: Any) -> list:
    from aiter.dist.parallel_state import get_ep_group

    ep = get_ep_group()
    gathered = [None] * ep.world_size
    dist.all_gather_object(gathered, value, group=ep.cpu_group)
    return gathered


def _broadcast_from_ep_group(value: Any, src: int) -> Any:
    from aiter.dist.parallel_state import get_ep_group

    objs = [value]
    dist.broadcast_object_list(objs, src=src, group=get_ep_group().cpu_group)
    return objs[0]


def _launch_settings(gate_mode: Any) -> tuple:
    """What each rank reads from its own environment: the dispatch and combine
    formats, each step's combine wire, and the gate mode ($ATOM_MOE_GU_ITLV)."""
    return (
        _MEGA_DISPATCH_WIRE,
        _MEGA_STAGE1_FUSED,
        _MEGA_STAGE2_FUSED,
        _MEGA_COMBINE_WIRE,
        _MEGA_DECODE_COMBINE_QUANT,
        gate_mode,
    )


def _check_settings_agree(gate_mode: Any) -> None:
    """Refuse ranks whose MegaMoE settings differ.

    They are read from each rank's environment and never negotiated; the rest
    of what MegaMoE fixes comes from the model config, the same on every rank.
    Collective, once per process, at the first init_mega_transport and before
    anything there can fail on one rank alone (aiter's recipe checks).
    """
    global _SETTINGS_AGREED
    if _SETTINGS_AGREED:
        return
    gathered = _gather_from_ep_group(_launch_settings(gate_mode))
    if any(settings != gathered[0] for settings in gathered):
        raise RuntimeError(
            "MEGA_DISPATCH_WIRE, ATOM_MEGA_STAGE1_FUSED, ATOM_MEGA_STAGE2_FUSED, "
            "ATOM_MEGA_COMBINE_WIRE, ATOM_MEGA_DECODE_COMBINE_QUANT and "
            "ATOM_MOE_GU_ITLV must be the same on every EP rank; per rank: "
            f"{gathered}"
        )
    _SETTINGS_AGREED = True


def _check_same_situ(mega: Any, situ_beta: Any, situ_linear_beta: Any) -> None:
    """A layer reusing the shared MegaMoE must bring the Situv2 betas it was
    built with: they are fixed at construction and not part of the cache key."""
    for name, value in (
        ("situ_beta", situ_beta),
        ("situ_linear_beta", situ_linear_beta),
    ):
        built = getattr(mega, name, None)
        if value is None or built is None:
            same = value is None and built is None
        else:
            same = torch.equal(
                torch.as_tensor(value).cpu(), torch.as_tensor(built).cpu()
            )
        if not same:
            raise ValueError(
                f"this layer's {name} differs from the one the shared MegaMoE "
                "was built with"
            )


def _mega_recipe(
    api: SimpleNamespace,
    *,
    activation: Any,
    gate_mode: Any,
    quant_type: Any,
    hidden_pad: int,
    intermediate_pad: int,
    swiglu_limit: float,
    combine_wire: str,
) -> Any:
    """The MegaMoERecipe this launch asks for: ATOM picks the dispatch, the
    combine and the wires from its environment, the layer supplies the expert
    GEMM. Whether the pairing can run is aiter's to say."""
    compute = api.Compute(
        activation=activation,
        gate_mode=gate_mode,
        quant_type=quant_type,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        swiglu_limit=swiglu_limit,
    )
    return api.Recipe(
        dispatch=(
            api.DispatchKind.COMPACT if _MEGA_STAGE1_FUSED else api.DispatchKind.MORI
        ),
        combine=(
            api.CombineKind.FUSED_SCATTER
            if _MEGA_STAGE2_FUSED
            else api.CombineKind.MORI
        ),
        dispatch_wire=_MEGA_DISPATCH_WIRE,
        # bf16 is always built; a quantized wire is built alongside it.
        combine_wires=("bf16", combine_wire),
        compute=compute,
    )


def _step_combine(mega: Any, wire: str) -> dict:
    """forward()'s per-step combine choice: by wire name on an aiter that builds
    from a recipe, by the older combine_quant spelling otherwise."""
    if hasattr(mega, "recipe"):
        return {"combine_wire": wire}
    return {"combine_quant": _COMBINE_WIRES[wire]}


def _check_mega_weights(layer: torch.nn.Module, quant_method: Any) -> None:
    """Reject expert weights MegaMoEGfx1250 cannot run: it only takes MXFP4."""
    w13, w2 = layer.w13_weight, layer.w2_weight
    if not {w13.dtype, w2.dtype} <= {torch.uint8, dtypes.fp4x2}:
        raise ValueError(
            "MegaMoEGfx1250 requires MXFP4 w1/w2 weights (uint8 or fp4x2), got "
            f"{w13.dtype} and {w2.dtype} from {type(quant_method).__name__}; it "
            "runs every MoE layer under ATOM_MORI_V2=1, with either stage setting"
        )


def init_mega_transport(
    *,
    ep_rank: int,
    ep_size: int,
    ep_src_global_rank: int,
    hidden_dim: int,
    max_num_inp_token_per_rank: int,
    num_experts: int,
    num_experts_per_token: int,
    data_type_itemsize: int,
    inter_dim: int,
    activation: Any,
    gate_mode: Any,
    quant_type: Any,
    hidden_pad: int,
    intermediate_pad: int,
    swiglu_limit: float,
    situ_beta: torch.Tensor | None = None,
    situ_linear_beta: torch.Tensor | None = None,
    combine_wire: str = "bf16",
    all2all_manager: Any = None,
) -> Any:
    """Create (and share) the MegaMoE that runs every MoE layer of this model.

    Everything here is per-model: the EP geometry, the cco arena, and the expert
    GEMM recipe. Only the weights differ per layer and those are forward()
    arguments, so one instance covers the whole model. Which dispatch and
    combine it runs is set by _MEGA_STAGE1_FUSED / _MEGA_STAGE2_FUSED.

    ``combine_wire`` names the quantized return-trip format this transport is
    to BUILD ($ATOM_MEGA_COMBINE_WIRE). It is a capability, not the choice:
    MegaMoE builds a combine for it and for bf16, and every forward then names
    the one it wants -- bf16 unless it says otherwise. See combine_wire_for_step.

    ``all2all_manager`` is the EP group's; its cco communicator carries the
    arena where it lends one (see _mega_communicator).
    """
    _check_settings_agree(gate_mode)
    api = _import_mega_recipe()
    if api is not None:
        recipe = _mega_recipe(
            api,
            activation=activation,
            gate_mode=gate_mode,
            quant_type=quant_type,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            swiglu_limit=swiglu_limit,
            combine_wire=combine_wire,
        )
        geometry = api.Geometry(
            rank=ep_rank,
            world_size=ep_size,
            model_dim=hidden_dim,
            inter_dim=inter_dim,
            experts=num_experts,
            topk=num_experts_per_token,
            max_tokens_per_rank=max_num_inp_token_per_rank,
        )
        return _init_mega_from_recipe(
            recipe,
            geometry,
            ep_src_global_rank=ep_src_global_rank,
            all2all_manager=all2all_manager,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    # An aiter from before MegaMoERecipe: the stage1/stage2 flags.
    combine_quant = _COMBINE_WIRES[combine_wire]
    key = (
        ep_rank,
        ep_size,
        hidden_dim,
        max_num_inp_token_per_rank,
        num_experts,
        num_experts_per_token,
        inter_dim,
        activation,
        gate_mode,
        quant_type,
        hidden_pad,
        intermediate_pad,
        swiglu_limit,
        # Keyed on: the wire sets the payload width and whether the scale
        # region exists, and combine_quant sets which combine reduces are
        # built (the staging layout itself no longer depends on it).
        _MEGA_DISPATCH_WIRE,
        combine_quant,
        _MEGA_STAGE1_FUSED,
        _MEGA_STAGE2_FUSED,
    )
    cached = _MEGA_TRANSPORTS.get(key)
    if cached is not None:
        _check_same_situ(cached, situ_beta, situ_linear_beta)
        return cached

    MegaMoEGfx1250 = _import_mega()
    if hasattr(MegaMoEGfx1250, "required_vmm_bytes"):
        per_rank_vmm = MegaMoEGfx1250.required_vmm_bytes(
            world_size=ep_size,
            hidden_dim=hidden_dim,
            max_tokens_per_rank=max_num_inp_token_per_rank,
            experts=num_experts,
            topk=num_experts_per_token,
            stage1_fused=_MEGA_STAGE1_FUSED,
        )
    else:
        logger.warning(
            "[MegaMoEGfx1250] aiter has no required_vmm_bytes(); falling back to "
            "a per-token VMM budget, which a stage1_fused arena can outgrow"
        )
        per_rank_vmm = _cco_per_rank_vmm(
            ep_size, hidden_dim, max_num_inp_token_per_rank, data_type_itemsize
        )
    comm = _mega_communicator(
        all2all_manager,
        ep_size=ep_size,
        ep_rank=ep_rank,
        ep_src_global_rank=ep_src_global_rank,
        # Plus the DevComm created below.
        per_rank_vmm=per_rank_vmm + _CCO_DEVCOMM_BYTES,
    )
    mega = MegaMoEGfx1250(
        communicator=comm,
        rank=ep_rank,
        world_size=ep_size,
        model_dim=hidden_dim,
        inter_dim=inter_dim,
        experts=num_experts,
        topk=num_experts_per_token,
        max_tokens_per_rank=max_num_inp_token_per_rank,
        activation=activation,
        gate_mode=gate_mode,
        quant_type=quant_type,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        # Passed, not left to aiter's own read of the env, so the key and the
        # transport cannot drift.
        dispatch_wire=_MEGA_DISPATCH_WIRE,
        stage1_fused=_MEGA_STAGE1_FUSED,
        stage2_fused=_MEGA_STAGE2_FUSED,
        # Only injected when asked for: an aiter without the combine-quant
        # epilogue has no such kwarg and would raise TypeError on every run.
        **({"combine_quant": combine_quant} if combine_quant != "none" else {}),
    )
    # Peer-region stride in the flat symmetric VA. triton_mega_moe needs it to
    # address the combine staging window, and MegaMoE does not keep it.
    #
    # Read HERE -- every rank is constructing its transport and the barrier
    # below follows -- because create_dev_comm() may be COLLECTIVE while the
    # forward that consumes this is NOT rank-aligned: the Triton experts are
    # picked per step, so a lazy first-use read could have only some ranks
    # enter the collective, and hang. Kept so close_mega_transports() can close
    # it before the communicator goes.
    mega._atom_dev_comm = comm.create_dev_comm()
    mega._atom_per_rank_size = int(mega._atom_dev_comm.per_rank_size)
    comm.barrier()
    _MEGA_TRANSPORTS[key] = mega
    logger.info(
        "[MegaMoEGfx1250] Created: ep_rank=%d ep_size=%d hidden=%d inter=%d "
        "experts=%d topk=%d M=%d act=%s gate=%s quant=%s pad=(%d,%d) "
        "swiglu_limit=%s dispatch=%s wire=%s combine_quant=%s force_a8w4=%s "
        "stage1_fused=%s stage2_fused=%s per_rank_vmm=%.2fGiB",
        ep_rank,
        ep_size,
        hidden_dim,
        inter_dim,
        num_experts,
        num_experts_per_token,
        max_num_inp_token_per_rank,
        activation,
        gate_mode,
        quant_type,
        hidden_pad,
        intermediate_pad,
        swiglu_limit,
        "flydsl" if mega._config.stage1_fused else "mori",
        mega._config.dispatch_wire,
        combine_quant,
        # The other half of the pair: logged together so a mismatch is readable.
        os.environ.get("AITER_FORCE_A8W4", "0"),
        mega._config.stage1_fused,
        mega._config.stage2_fused,
        per_rank_vmm / (1 << 30),
    )
    return mega


def _init_mega_from_recipe(
    recipe: Any,
    geometry: Any,
    *,
    ep_src_global_rank: int,
    all2all_manager: Any,
    situ_beta: torch.Tensor | None,
    situ_linear_beta: torch.Tensor | None,
) -> Any:
    """init_mega_transport on an aiter that builds MegaMoE from a recipe."""
    key = (recipe, geometry)
    cached = _MEGA_TRANSPORTS.get(key)
    if cached is not None:
        _check_same_situ(cached, situ_beta, situ_linear_beta)
        return cached

    MegaMoEGfx1250 = _import_mega()
    # MegaMoE's arena -- exactly what the constructor allocates -- plus the
    # DevComm created below, asked for in full before the communicator exists.
    per_rank_vmm = (
        MegaMoEGfx1250.required_vmm_bytes(recipe, geometry) + _CCO_DEVCOMM_BYTES
    )
    comm = _mega_communicator(
        all2all_manager,
        ep_size=geometry.world_size,
        ep_rank=geometry.rank,
        ep_src_global_rank=ep_src_global_rank,
        per_rank_vmm=per_rank_vmm,
    )
    mega = MegaMoEGfx1250.from_recipe(
        recipe,
        geometry,
        communicator=comm,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    # Peer-region stride in the flat symmetric VA, for triton_mega_moe; read
    # here, while every rank constructs, for the reason the flag path gives.
    # Kept so close_mega_transports() can close it before the communicator goes.
    mega._atom_dev_comm = comm.create_dev_comm()
    mega._atom_per_rank_size = int(mega._atom_dev_comm.per_rank_size)
    comm.barrier()
    _MEGA_TRANSPORTS[key] = mega
    logger.info(
        "[MegaMoEGfx1250] Created: ep_rank=%d ep_size=%d hidden=%d inter=%d "
        "experts=%d topk=%d M=%d dispatch=%s wire=%s combine=%s "
        "combine_wires=%s act=%s gate=%s quant=%s pad=(%d,%d) swiglu_limit=%s "
        "force_a8w4=%s per_rank_vmm=%.2fGiB",
        geometry.rank,
        geometry.world_size,
        geometry.model_dim,
        geometry.inter_dim,
        geometry.experts,
        geometry.topk,
        geometry.max_tokens_per_rank,
        mega.recipe.dispatch.value,
        mega.recipe.dispatch_wire,
        mega.recipe.combine.value,
        ",".join(mega.recipe.combine_wires),
        mega.recipe.compute.activation,
        mega.recipe.compute.gate_mode,
        mega.recipe.compute.quant_type,
        mega.recipe.compute.hidden_pad,
        mega.recipe.compute.intermediate_pad,
        mega.recipe.compute.swiglu_limit,
        # The other half of the wire pair: logged together so a mismatch reads.
        os.environ.get("AITER_FORCE_A8W4", "0"),
        per_rank_vmm / (1 << 30),
    )
    return mega


class MoriV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """The MegaMoE a MoE layer runs on, and what the modular kernel asks of its
    transport. MegaMoE dispatches and combines itself, so prepare()/finalize()
    are never walked."""

    def __init__(
        self,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        mega_geometry: dict,
        all2all_manager: Any,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriV2PrepareAndFinalize but not installed."
            )
        super().__init__()
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        # MegaMoE's EP geometry; the rest of what it fixes at construction only
        # shows up on the layer -- see bind_mega_transport().
        self._mega_geometry = mega_geometry
        # The EP group's; MegaMoE borrows its cco communicator.
        self._all2all_manager = all2all_manager
        self.mega: Any = None

    def bind_mega_transport(self, layer: torch.nn.Module, quant_method: Any) -> None:
        """Build the shared MegaMoE once the layer reveals the model-wide recipe.

        Called from init_prepare_finalize, the one place that sees both the layer
        and its quant method: the EP geometry is known when this object is built,
        but the expert-GEMM recipe (activation, gate mode, quant type, padding,
        swiglu limit) lives on those two. That hook also runs after weight
        post-processing and before any cudagraph capture, which is where the cco
        arena allocation and the FlyDSL JIT belong.
        """
        if self.mega is not None:
            return
        # Before init_mega_transport, which builds the cco communicator and
        # allocates the arena.
        _check_mega_weights(layer, quant_method)
        inter_dim = getattr(quant_method, "intermediate_size", 0)
        if inter_dim <= 0:
            raise ValueError(
                "the fused transport needs the per-partition intermediate size, "
                f"got {inter_dim} from {type(quant_method).__name__}"
            )
        self.mega = init_mega_transport(
            **self._mega_geometry,
            inter_dim=inter_dim,
            activation=layer.activation,
            gate_mode=(
                GateMode.INTERLEAVE.value
                if quant_method.is_guinterleave
                else GateMode.SEPARATED.value
            ),
            quant_type=quant_method.quant_type,
            hidden_pad=quant_method.hidden_pad,
            intermediate_pad=quant_method.intermediate_pad,
            swiglu_limit=float(getattr(layer, "swiglu_limit", 0.0)),
            situ_beta=getattr(layer, "activation_situ_beta", None),
            situ_linear_beta=getattr(layer, "activation_situ_linear_beta", None),
            combine_wire=_MEGA_COMBINE_WIRE,
            all2all_manager=self._all2all_manager,
        )

    @staticmethod
    def combine_wire_for_step() -> str:
        """This step's combine wire, named outright -- and DP-agreed.

        The quantized wire is prefill-only: the quant/dequant pair is a fixed
        per-token cost against a saving that scales with the tokens on the
        wire, so it only pays off once the wire is busy.
        $ATOM_MEGA_DECODE_COMBINE_QUANT opens it on decode too, so every step
        names it.

        Every rank has to name the SAME wire, and that is why `is_prefill`
        cannot gate this on its own. Combine is P2P: gemm2's epilogue (or, with
        stage 2 unfused, mori's fp4 push) writes this rank's rows into every
        PEER's arena in the format this rank picked, and the peer reads them
        back in the format IT picked.
        `is_prefill` is the local batch's (`ForwardMode.decide` never reduces
        it), so on a ragged step a prefilling rank would scatter fp4 rows into
        a decoding peer that reduces them as bf16 -- the same rank-local trap
        select_mega documents. `running_tokens_are_unified` IS the group's
        answer to "is anybody prefilling", so it is what opens the wire;
        `is_prefill` then only speaks for dp_size==1, where it is the group.

        No context (warmup, profile runs) counts as prefill -- that is the
        shape those run at, and it exercises the quantized reduce before a
        capture rather than first reaching it mid-serve.
        """
        if _MEGA_DECODE_COMBINE_QUANT:
            return _MEGA_COMBINE_WIRE
        context = get_forward_context().context
        if context is None:
            return _MEGA_COMBINE_WIRE
        if context.is_prefill or not context.running_tokens_are_unified:
            return _MEGA_COMBINE_WIRE
        return "bf16"

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        quant_type: QuantType = QuantType.No,
        x_scale: torch.Tensor | None = None,
    ) -> mk.PrepareResultType:
        raise RuntimeError(
            "MegaMoE dispatches its own rows; MoriV2ModularKernel never walks "
            "prepare()/finalize()"
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise RuntimeError(
            "MegaMoE combines its own rows; MoriV2ModularKernel never walks "
            "prepare()/finalize()"
        )


class MoriV2ModularKernel(mk.FusedMoEModularKernel):
    """Modular kernel for the v2 path.

    It steps out of the way: MegaMoE runs the whole layer, so forward() hands it
    this layer's weights and returns its output instead of walking
    prepare -> fused_moe -> finalize.

    The dispatch arena is padded to a huge static token_num
    (ws * max_num_inp_token_per_rank) while the received tokens occupy only the
    first ``total_recv`` rows, so the static ``sum(running_tokens_across_dp)``
    bound (same as the base policy) goes down as ``recv_token_bound``: the
    grid-bound aiter kernels then launch a grid sized to what the group
    actually sent instead of the full arena.
    """

    def _recv_bound(self, arena_rows: int) -> int | None:
        """Recv-row bound for this step, or None when it does not shrink.

        Correctness / capture-safety:
          * The counts are python ints, fixed per captured graph, so the bound is
            static across capture/replay and no GPU->CPU sync is needed (unlike
            reading the device ``total_recv``).
          * mori dispatch deduplicates per destination rank, so each source
            rank contributes at most its own token count -- never that times
            topk. ``total_recv <= sum(running_tokens_across_dp)``; the bound
            therefore never drops a valid row, and the aiter kernels'
            device-side ``num_valid_routes`` guard still skips the exact
            within-buffer tail [total_recv, bound).
            Why the sum and not ``running_tokens * dp``: see the base method.
        """
        context = get_forward_context().context
        if context is None:
            return None
        across_dp = context.running_tokens_across_dp
        assert across_dp is not None, (
            "an all2all MoE needs the group's per-rank counts to bound what its "
            "dispatch delivered; this step reached it with none reduced"
        )
        bound = sum(across_dp)
        return bound if bound < arena_rows else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Bound in init_prepare_finalize; there is no path without it.
        mega = self.prepare_finalize.mega
        assert mega is not None, "MegaMoE was not bound"
        triton_experts = (kwargs.get("moe_extra_args") or {}).get("triton_experts")

        # MegaMoE as transport, with OUR experts, driven from one place.
        #
        # The alternative -- and what runs when this is off -- is the normal
        # prepare -> experts -> finalize walk, where prepare() calls MegaMoE's
        # dispatch, GEMM2 delivers into its staging window and finalize() calls
        # its combine. That works, but it spreads one layer's transport across
        # three methods and needs a public transport-only API on MegaMoE
        # (dispatch / combine_scatter_target / combine) that exists solely for
        # ATOM. triton_mega_moe does the same three steps in one function
        # against MegaMoE's internals instead, so aiter needs no such API and
        # the dispatch/combine underneath can later be swapped for our own.
        #
        # Bit-identical to the walk it replaces (mega fp4 + a4w4, mega bf16 +
        # a4w4, mega bf16 + a8w4 all match to the last bit). Off by default
        # until it has run a real serve.
        if mega is not None and triton_experts is not None:
            compact_dispatch, fused_combine = _mega_kinds(mega)
            if compact_dispatch:
                raise RuntimeError(
                    "triton_mega_moe drives MegaMoE's token-major dispatch and "
                    "cannot run on the compact stage-1 plan; set "
                    "ATOM_MEGA_STAGE1_FUSED=0 to fall back to mori dispatch"
                )
            if not fused_combine:
                raise RuntimeError(
                    "triton_mega_moe scatters GEMM2's rows into the fused "
                    "combine's staging and cannot run on mori's combine; set "
                    "ATOM_MEGA_STAGE2_FUSED=1 to use the Triton experts"
                )
            from atom.model_ops.fused_moe_triton import triton_mega_moe

            assert not kwargs.get(
                "apply_router_weight_on_input", False
            ), "mori does not support apply_router_weight_on_input=True now."
            arena_rows = (
                self.prepare_finalize.num_dispatchers() * mega.max_tokens_per_rank
            )
            bound = self._recv_bound(arena_rows)
            # The "Direction-3" shrink, reproduced from the base-class walk: the
            # bound leaves graph_bs*topk*dp rows, but mori de-duplicates per
            # destination rank, so at most graph_bs*max_seqlen_q*dp can be live.
            # Guarded by the same uniform-decode test, via `bound`: None there
            # means a mixed/prefill batch, which keeps the full arena.
            #
            # Guarded on the unified-decode test below, NOT on `bound is not
            # None`: _recv_bound also returns None when the bound would not
            # actually shrink the arena, and the base walk still applies M_eff in
            # that case. Keying off `bound` would silently skip the trim there.
            # Same test and same quantity the base walk uses. Inlined rather
            # than imported: the flag it mirrors is a local in moe.apply().
            m_eff = None
            _ctx = get_forward_context().context
            if (
                _ctx is not None
                and not _ctx.is_prefill
                and getattr(_ctx, "running_tokens_are_unified", True)
            ):
                m_eff = _ctx.running_tokens * get_dp_group().world_size
            return triton_mega_moe(
                mega,
                hidden_states,
                topk_weights,
                topk_ids,
                w1=triton_experts["w13_weight"],
                w2=triton_experts["w2_weight"],
                w1_scale=triton_experts["w13_scale"],
                w2_scale=triton_experts["w2_scale"],
                expert_map=kwargs.get("expert_map"),
                n_local_experts=w1.size(0),
                w13_swizzle_layout=triton_experts["w13_swizzle_layout"],
                w2_swizzle_layout=triton_experts["w2_swizzle_layout"],
                w1_bias=triton_experts.get("w1_bias"),
                w2_bias=triton_experts.get("w2_bias"),
                swiglu_limit=triton_experts.get("swiglu_limit", 10.0),
                recv_token_bound=bound,
                m_eff=m_eff,
            )

        assert not kwargs.get(
            "apply_router_weight_on_input", False
        ), "mori does not support apply_router_weight_on_input=True now."
        self._assert_recipe_matches(mega, kwargs)

        ids = topk_ids.to(torch.int32).contiguous()
        out = mega(
            hidden_states.contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            ids,
            w1=w1,
            w2=w2,
            w1_scale=kwargs.get("w1_scale"),
            w2_scale=kwargs.get("w2_scale"),
            bias1=kwargs.get("bias1"),
            bias2=kwargs.get("bias2"),
            a1_scale=kwargs.get("a1_scale"),
            a2_scale=kwargs.get("a2_scale"),
            recv_token_bound=self._recv_bound(
                self.prepare_finalize.num_dispatchers() * mega.max_tokens_per_rank,
            ),
            **_step_combine(mega, self.prepare_finalize.combine_wire_for_step()),
        )
        return out

    @staticmethod
    def _assert_recipe_matches(mega, kwargs: dict) -> None:
        """Fail loudly if this layer's recipe is not the one MegaMoE was built with.

        MegaMoE fixes the expert-GEMM recipe at construction, on the premise that
        every MoE layer of a model shares it, while ATOM re-sends it per layer.
        Comparing the two turns a violated premise into an error here rather than
        into silently ignored arguments (a wrong swiglu_limit or gate mode still
        produces plausible-looking logits).
        """
        extra = kwargs.get("moe_extra_args") or {}
        actual = {
            "activation": kwargs.get("activation", ActivationType.Silu),
            "quant_type": kwargs.get("quant_type", QuantType.No),
            "gate_mode": extra.get("gate_mode", GateMode.SEPARATED.value),
            "swiglu_limit": float(extra.get("swiglu_limit", 0.0) or 0.0),
            "hidden_pad": int(kwargs.get("hidden_pad", 0) or 0),
            "intermediate_pad": int(kwargs.get("intermediate_pad", 0) or 0),
        }
        # The recipe it was built from, or, on an older aiter, the same names
        # on the instance.
        recipe = getattr(mega, "recipe", None)
        source = mega if recipe is None else recipe.compute
        built = {name: getattr(source, name) for name in actual}
        differing = {
            name: (value, built[name])
            for name, value in actual.items()
            if value != built[name]
        }
        if differing:
            raise ValueError(
                "this layer's expert-GEMM recipe differs from the one MegaMoE was "
                f"built with (this layer vs built): {differing}"
            )


def _mega_kinds(mega: Any) -> tuple[bool, bool]:
    """(the compact dispatch, the fused combine): read off the recipe, or the
    flags an older aiter kept on its config."""
    recipe = getattr(mega, "recipe", None)
    if recipe is None:
        return mega._config.stage1_fused, mega._config.stage2_fused
    api = _import_mega_recipe()
    return (
        recipe.dispatch is api.DispatchKind.COMPACT,
        recipe.combine is api.CombineKind.FUSED_SCATTER,
    )


def make_mori_v2_prepare_finalize(moe, all2all_manager) -> MoriV2PrepareAndFinalize:
    """Build a MoriV2PrepareAndFinalize for the given MoE config + EP group."""
    from aiter.dist.parallel_state import get_ep_group

    ep_group = get_ep_group()
    ep_src_global_rank = ep_group.ranks[0]
    ep_size = all2all_manager.world_size

    # Geometry only; the expert-GEMM recipe comes from the layer later.
    return MoriV2PrepareAndFinalize(
        max_tokens_per_rank=moe.max_num_tokens,
        num_dispatchers=ep_size,
        mega_geometry={
            "ep_rank": all2all_manager.rank,
            "ep_size": ep_size,
            "ep_src_global_rank": ep_src_global_rank,
            "hidden_dim": moe.hidden_dim,
            "max_num_inp_token_per_rank": moe.max_num_tokens,
            "num_experts": moe.num_experts,
            "num_experts_per_token": moe.experts_per_token,
            "data_type_itemsize": moe.in_dtype.itemsize,
        },
        all2all_manager=all2all_manager,
    )
