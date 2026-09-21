# SPDX-License-Identifier: Apache-2.0
"""Prepare/Finalize using mori dispatch_combine_v2 (FlyDSL/cco, gfx1250 wave32).

The production mori v1 (``mori.ops.EpDispatchCombineOp``) is authored for
gfx942/950 HIP kernels and does not run on gfx1250. dispatch_combine_v2 is the
gfx1250-capable cco/FlyDSL implementation. This module wires it into ATOM's
FusedMoEModularKernel as a drop-in replacement for MoriPrepareAndFinalize,
gated by ``ATOM_MORI_V2=1``.

Pipeline of the gather transport (mirrors the validated standalone
test_moe_layer_ep.py):
    recv_x, recv_w, _, recv_idx, total_recv, routing = op.dispatch(
        a1, topk_weights, None, topk_ids, return_routing=True)
    dispatch_a1 = recv_x[:total_recv].clone()   # out of the cco VMM window
    fused_out = aiter.fused_moe(dispatch_a1, ...)   # driven by the modular kernel
    out, _ = op.combine(fused_out, routing=routing)

Two transports sit behind the same prepare/finalize pair:

  * ATOM_MORI_V2_FUSED=0 -- mori's own v2 op-layer, combine_mode="gather". The
    untouched upstream baseline.
  * ATOM_MORI_V2_FUSED=1 -- aiter's MegaMoEGfx1250, whose gemm2 epilogue
    P2P-writes each weighted (token,k) result straight into the peers' combine
    staging, so combine only barriers + sums. It owns the whole layer
    (dispatch -> expert GEMM -> fused combine), so MoriV2ModularKernel hands it
    the layer and returns its output; prepare()/finalize() are not reached and
    the transport is configured, not bypassed -- the model-wide recipe
    (activation, gate mode, quant type, padding, swiglu limit) is fixed at
    construction and the per-layer weights/biases go to each forward().

Shared experts are NOT fused in the mori EP+DP path (ATOM disables fusion there,
see topK.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config), so
topk_ids carry only routed expert ids and mori routes them cleanly.

``ATOM_EP_BACKEND=moonep`` adds a policy layer without replacing the transport:
prefill remaps logical ids into ``EPR+B`` virtual slots and runs the resident and
prefetched subsets through standard fused_moe, while decode keeps the owner-only
``EPR`` geometry. The policy factory uses the existing MoRI v1 transport by
default for gfx950 and selects v2 when ``ATOM_MORI_V2=1``.
"""

import logging
import os
import sys
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from aiter import ActivationType, QuantType
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

# Populated lazily by _import_v2().
EpDispatchCombineConfig = None
EpDispatchCombineOp = None
_V2_IMPORTED = False


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


def _import_v2_from_mori():
    try:
        from mori.ops.dispatch_combine_v2.dispatch_combine_op import (  # type: ignore
            EpDispatchCombineConfig as _Cfg,
        )
        from mori.ops.dispatch_combine_v2.dispatch_combine_op import (
            EpDispatchCombineOp as _Op,
        )
    except ImportError:
        # Older mori shipped dispatch_combine_v2 as loose test-only modules
        # with no __init__.py, importing each other by top-level name --
        # they only resolve with their own directory on sys.path.
        v2_dir = os.path.join(
            os.path.dirname(mori.__file__), "ops", "dispatch_combine_v2"
        )
        if v2_dir not in sys.path:
            sys.path.insert(0, v2_dir)
        from dispatch_combine_op import (  # type: ignore
            EpDispatchCombineConfig as _Cfg,
        )
        from dispatch_combine_op import (
            EpDispatchCombineOp as _Op,
        )

    return _Cfg, _Op


def _import_v2() -> None:
    """Bind mori's v2 op-layer -- the non-fused (gather) baseline.

    The gemm2-fused mode is no longer an op-layer combine_mode, so FUSED=1 does
    not come through here at all: it binds aiter's MegaMoE instead (see
    _import_mega). Only the cco communication substrate (mori.cco) is shared by
    both transports.
    """
    global EpDispatchCombineConfig, EpDispatchCombineOp, _V2_IMPORTED
    if _V2_IMPORTED:
        return
    if not MORI_AVAILABLE:
        raise ImportError("mori is required for MoriV2PrepareAndFinalize")

    EpDispatchCombineConfig, EpDispatchCombineOp = _import_v2_from_mori()
    _V2_IMPORTED = True
    logger.info("[MORI-V2] op-layer from mori (%s)", EpDispatchCombineOp.__module__)


def _resolve_transport() -> str:
    """ "mega" when ATOM_MORI_V2_FUSED is on, else mori's plain gather op-layer."""
    from atom.utils import envs as _atom_envs

    return "mega" if _atom_envs.ATOM_MORI_V2_FUSED else "gather"


@lru_cache(maxsize=1)
def _init_cco_comm(
    ep_size: int,
    ep_rank: int,
    ep_src_global_rank: int,
    per_rank_vmm: int,
) -> Any:
    """Collective: create a persistent cco Communicator over the EP group.

    The mori cco unique-id is generated on the EP leader and broadcast over the
    EP gloo cpu_group (mirrors mori.shmem.shmem_torch_process_group_init but for
    the cco fabric). All EP ranks must call this together.
    """
    from aiter.dist.parallel_state import get_ep_group

    ep = get_ep_group()
    uid = Communicator.get_unique_id() if ep_rank == 0 else None
    objs = [uid]
    dist.broadcast_object_list(objs, src=ep_src_global_rank, group=ep.cpu_group)
    uid = objs[0]
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

    MegaMoE's arena needs strictly less than this (one recv-sized token buffer
    plus an M*topk combine staging), so the same budget covers both transports.
    """
    tok_bytes = max_num_inp_token_per_rank * hidden_dim * itemsize
    win_bytes = ep_size * tok_bytes * 2 + (1 << 24)
    return 2 * win_bytes + (1 << 28)


# Keyed by everything MegaMoE fixes at construction, so the MoE layers of one
# model share a single instance -- and a single cco symmetric arena. Not an
# lru_cache because the Situv2 betas are tensors and cannot be cache keys; they
# are config-wide, so the first layer's are the model's.
_MEGA_TRANSPORTS: dict = {}

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
) -> Any:
    """Create (and share) the MegaMoE that runs every MoE layer of this model.

    Everything here is per-model: the EP geometry, the cco arena, and the expert
    GEMM recipe. Only the weights differ per layer and those are forward()
    arguments, so one instance covers the whole model. Which dispatch kernel it
    uses is aiter's own call (MEGA_DISPATCH=flydsl|mori).
    """
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
        # region exists.
        _MEGA_DISPATCH_WIRE,
    )
    cached = _MEGA_TRANSPORTS.get(key)
    if cached is not None:
        return cached

    MegaMoEGfx1250 = _import_mega()
    comm = _init_cco_comm(
        ep_size,
        ep_rank,
        ep_src_global_rank,
        _cco_per_rank_vmm(
            ep_size, hidden_dim, max_num_inp_token_per_rank, data_type_itemsize
        ),
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
        # Only mori's dispatch carries the scale row, so a quantizing wire has
        # no other backend to run on. Named here rather than left to
        # $MEGA_DISPATCH, whose default is flydsl: otherwise asking for fp4 is
        # rejected at the first MoE layer for a reason the operator did not set.
        **(
            {"dispatch_backend": "mori"}
            if _MEGA_DISPATCH_WIRE in ("fp8", "fp4")
            else {}
        ),
    )
    # Peer-region stride in the flat symmetric VA. triton_mega_moe needs it to
    # address the combine staging window, and MegaMoE does not keep it.
    #
    # Read HERE -- every rank is constructing its transport and the barrier
    # below follows -- because create_dev_comm() may be COLLECTIVE while the
    # forward that consumes this is NOT rank-aligned: the Triton experts are
    # picked per step, so a lazy first-use read could have only some ranks
    # enter the collective, and hang.
    mega._atom_per_rank_size = int(comm.create_dev_comm().per_rank_size)
    comm.barrier()
    _MEGA_TRANSPORTS[key] = mega
    logger.info(
        "[MORI-V2] Created MegaMoE: ep_rank=%d ep_size=%d hidden=%d inter=%d "
        "experts=%d topk=%d M=%d act=%s gate=%s quant=%s pad=(%d,%d) "
        "swiglu_limit=%s dispatch=%s wire=%s force_a8w4=%s",
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
        mega._config.dispatch_backend,
        mega._config.dispatch_wire,
        # The other half of the pair: logged together so a mismatch is readable.
        os.environ.get("AITER_FORCE_A8W4", "0"),
    )
    return mega


@lru_cache(maxsize=4)
def init_mori_v2_op(
    ep_rank: int,
    ep_size: int,
    ep_src_global_rank: int,
    hidden_dim: int,
    max_num_inp_token_per_rank: int,
    num_local_experts: int,
    num_experts_per_token: int,
    data_type_itemsize: int,
    combine_mode: str = "gather",
) -> Any:
    """Create (and cache) a dispatch_combine_v2 op bound to the EP cco comm."""
    _import_v2()

    data_type = torch.bfloat16
    for dt in (torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.bfloat16):
        if dt.itemsize == data_type_itemsize:
            data_type = dt
            break

    per_rank_vmm = _cco_per_rank_vmm(
        ep_size, hidden_dim, max_num_inp_token_per_rank, data_type.itemsize
    )
    comm = _init_cco_comm(ep_size, ep_rank, ep_src_global_rank, per_rank_vmm)

    cfg = EpDispatchCombineConfig(
        rank=ep_rank,
        world_size=ep_size,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_experts_per_token,
        data_type=data_type,
        combine_mode=combine_mode,
    )
    op = EpDispatchCombineOp(cfg, comm)
    comm.barrier()
    logger.info(
        "[MORI-V2] Created dispatch_combine_v2 op: ep_rank=%d ep_size=%d "
        "hidden=%d num_local_experts=%d topk=%d M=%d combine=%s",
        ep_rank,
        ep_size,
        hidden_dim,
        num_local_experts,
        num_experts_per_token,
        max_num_inp_token_per_rank,
        combine_mode,
    )
    return op


class MoriV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Prepare/Finalize backed by mori dispatch_combine_v2 (sync path only)."""

    def __init__(
        self,
        mori_v2_op: Any,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        mega_geometry: dict | None = None,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriV2PrepareAndFinalize but not installed."
            )
        super().__init__()
        self._op = mori_v2_op
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        # Routing handle stashed between prepare() and finalize() of one forward.
        self._routing = None
        # The fused transport's EP geometry; the rest of what MegaMoE fixes at
        # construction only shows up on the layer -- see bind_mega_transport().
        self._mega_geometry = mega_geometry
        self.mega: Any = None
        self.is_fused = mega_geometry is not None

    def bind_mega_transport(self, layer: torch.nn.Module, quant_method: Any) -> None:
        """Build the shared MegaMoE once the layer reveals the model-wide recipe.

        Called from init_prepare_finalize, the one place that sees both the layer
        and its quant method: the EP geometry is known when this object is built,
        but the expert-GEMM recipe (activation, gate mode, quant type, padding,
        swiglu limit) lives on those two. That hook also runs after weight
        post-processing and before any cudagraph capture, which is where the cco
        arena allocation and the FlyDSL JIT belong.
        """
        if self._mega_geometry is None or self.mega is not None:
            return
        inter_dim = getattr(quant_method, "intermediate_size", 0)
        if inter_dim <= 0:
            raise ValueError(
                "the fused transport needs the per-partition intermediate size, "
                f"got {inter_dim}; ATOM_MORI_V2_FUSED=1 requires the a8w4 "
                "(Mxfp4MoEMethod) quant path."
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
        )

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
    ) -> mk.PrepareResultType:
        assert (
            not apply_router_weight_on_input
        ), "mori does not support apply_router_weight_on_input=True now."

        # bf16 dispatch, no wire quant: scales=None. indices carry global expert
        # ids (0..global_num_experts-1); the transport routes id -> rank = id //
        # EPR.
        #
        # Gather transport only. MegaMoE never reaches prepare()/finalize() any
        # more: MoriV2ModularKernel.forward sends it to triton_mega_moe (Triton
        # experts) or to MegaMoE's own forward (flydsl experts), and both own
        # their dispatch and combine end to end.
        recv_x, recv_w, _recv_s, recv_idx, _total_recv_t, routing = self._op.dispatch(
            a1,
            topk_weights.to(torch.float32),
            None,
            topk_ids.to(torch.int32),
            return_routing=True,
        )
        self._routing = routing

        # Capture-safe: do NOT call _total_recv_t.item() (a GPU->CPU sync that is
        # illegal during cudagraph capture). fused_moe is handed the FULL
        # fixed-size arena buffers, aliased in place rather than sliced to the
        # received count, so the shapes stay static across capture/replay.
        dispatch_a1 = recv_x
        dispatch_ids = recv_idx
        dispatch_weights = recv_w

        # The received-row count, for whoever needs it as a row mask.
        #
        # flydsl fused_moe does NOT: the grouped a8w4 path derives per-expert
        # routing from the (already trimmed) global ids + expert_mask, and its
        # kernels skip the tail past the device-side count on their own -- which
        # is the whole correctness argument in _recv_bound. So it stays
        # None there, exactly as before.
        #
        # The Triton/gluon EP experts DO: ep_sort_routing hands this straight to
        # _ep_gate_prep_scan_kernel, whose row mask is skipped entirely when it
        # is None. The mori buffer always has M > R, so without it the garbage
        # rows in [R, M) fold into the histogram as LIVE gates (on the rank
        # owning global expert 0, a zeroed/stale id maps to local expert 0) and,
        # under the scatter-fused combine, get delivered into staging slots
        # belonging to real tokens. Silently wrong rather than an error.
        #
        # Capture-safe: a (1,) int32 device scalar, allocated once by the
        # transport and re-zeroed per dispatch, so the pointer is stable across
        # cudagraph capture/replay and nothing is read on the host.
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=(_total_recv_t if envs.ATOM_USE_TRITON_MOE else None),
            expert_num_tokens_cpu=None,
        )
        return (
            dispatch_a1,
            None,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        # topk_ids here is the ORIGINAL (pre-dispatch) routing, so shape[0] == ct.
        num_token = topk_ids.shape[0]
        assert self._routing is not None, "finalize() called before prepare()"
        out, _ = self._op.combine(fused_expert_output, routing=self._routing)
        self._routing = None
        return out[:num_token]


class _MoriV1PolicyTransport:
    """Normalize the production MoRI v1 API to the v2 routing-handle contract."""

    def __init__(self, prepare_finalize: Any, num_experts_per_rank: int) -> None:
        self._prepare_finalize = prepare_finalize
        native_cfg = getattr(prepare_finalize._sync_mori_op, "cfg", None)
        self.cfg = SimpleNamespace(
            num_experts_per_rank=num_experts_per_rank,
            dispatch_block_num=getattr(native_cfg, "block_num", 1024),
        )

    def dispatch(
        self,
        hidden: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor | None,
        indices: torch.Tensor,
        *,
        return_routing: bool,
    ):
        if scales is not None:
            raise ValueError("MoonEP's gfx950 closure expects BF16 MoRI dispatch")
        if not return_routing:
            raise ValueError("MoonEP requires a routing handle for combine")
        block_num, warp_per_block = self._prepare_finalize._get_dispatch_config(
            hidden.shape[0]
        )
        recv_x, recv_w, recv_s, recv_idx, total_recv = (
            self._prepare_finalize._sync_mori_op.dispatch(
                hidden,
                weights,
                None,
                indices,
                block_num,
                warp_per_block,
            )
        )
        # MoRI v1 reconstructs combine routing from the dispatched physical IDs
        # rather than returning an opaque handle as v2 does.
        return recv_x, recv_w, recv_s, recv_idx, total_recv, indices

    def combine(self, output: torch.Tensor, *, routing: torch.Tensor):
        block_num, warp_per_block = self._prepare_finalize._get_dispatch_config(
            routing.shape[0]
        )
        result = self._prepare_finalize._sync_mori_op.combine(
            output,
            None,
            routing,
            block_num,
            warp_per_block,
        )
        return result[0], None


def _make_virtual_expert_masks(
    *,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    prefetch_slots: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks mapping virtual global ids to home/prefetch weight rows."""

    physical_per_rank = experts_per_rank + prefetch_slots
    num_physical = world_size * physical_per_rank
    home = torch.zeros(num_physical, dtype=torch.int32, device=device)
    prefetched = torch.zeros_like(home)
    begin = rank * physical_per_rank
    home[begin : begin + experts_per_rank] = 1
    prefetched[begin + experts_per_rank : begin + physical_per_rank] = 1
    return home, prefetched


class MoonEPPolicyMoriPrepareAndFinalize(MoriV2PrepareAndFinalize):
    """MoonEP planning in front of normal MoRI dispatch + standard fused_moe.

    Prefill uses ``EPR + B`` virtual slots per rank. Resident experts occupy
    ``[0, EPR)`` and the selected remote experts occupy ``[EPR, EPR + B)``.
    Decode keeps the ordinary owner-only ``EPR`` geometry and never creates a
    histogram or calls the global planner.

    The VMM weight pool keeps home and prefetched rows in separate views, so
    prefill runs two ordinary fused_moe calls and adds their local partials.
    This avoids a hot-path concatenation of expert weights and deliberately
    does not depend on MoonEP's grouped-row dispatch/GEMM contract.
    """

    ADOPTED = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_bias",
        "w2_bias",
    )

    def __init__(
        self,
        *,
        prefill_op: Any,
        decode_op: Any,
        rank: int,
        world_size: int,
        num_experts: int,
        prefetch_slots: int,
        max_tokens_per_rank: int,
        num_dispatchers: int,
    ) -> None:
        super().__init__(
            decode_op,
            max_tokens_per_rank=max_tokens_per_rank,
            num_dispatchers=num_dispatchers,
        )
        from aiter.ops.flydsl.moonep import MoonEPDecodePolicy

        if num_experts % world_size:
            raise ValueError("MoonEP requires num_experts divisible by world_size")
        if prefetch_slots <= 0:
            raise ValueError("MoonEP prefetch_slots must be positive")

        self._prefill_op = prefill_op
        self._decode_op = decode_op
        self._rank = rank
        self._world_size = world_size
        self._num_experts = num_experts
        self._experts_per_rank = num_experts // world_size
        self._prefetch_slots = prefetch_slots
        self._decode_policy = MoonEPDecodePolicy(
            world_size=world_size, num_experts=num_experts
        )
        self._prefill_policies: dict[tuple[int, int, str], Any] = {}
        self._histogram_exchange = None
        self._pools = None
        self._virtual_masks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._active_op = None
        self._active_plan = None
        self._active_is_prefill = False

    def _is_prefill(self) -> bool:
        """Resolve the phase without adding a decode-side collective.

        This backend targets phase-disaggregated serving, where every rank in
        an EP group is either prefill or decode for a forward.  Consequently
        the forward context is sufficient and DecodePolicy remains completely
        free of histogram/all-reduce synchronization.
        """

        context = get_forward_context().context
        if context is None:
            return True
        return not bool(
            getattr(
                context,
                "dp_uniform_decode",
                not bool(getattr(context, "is_prefill", True)),
            )
        )

    def _prefill_policy(self, topk_ids: torch.Tensor):
        from aiter.ops.flydsl.moonep import (
            MoonEPPlanConfig,
            MoonEPPrefillPolicy,
            MoonEPSymmetricHistogramExchange,
        )

        key = (topk_ids.shape[0], topk_ids.shape[1], str(topk_ids.device))
        policy = self._prefill_policies.get(key)
        if policy is not None:
            return policy

        if self._histogram_exchange is None:
            self._histogram_exchange = MoonEPSymmetricHistogramExchange(
                rank=self._rank,
                world_size=self._world_size,
                num_experts=self._num_experts,
                device=topk_ids.device,
            )
        config = MoonEPPlanConfig(
            rank=self._rank,
            world_size=self._world_size,
            num_tokens=topk_ids.shape[0],
            top_k=topk_ids.shape[1],
            num_experts=self._num_experts,
            prefetch_slots=self._prefetch_slots,
        )
        policy = MoonEPPrefillPolicy(
            config,
            topk_ids.device,
            histogram_exchange=self._histogram_exchange,
        )
        self._prefill_policies[key] = policy
        return policy

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
    ) -> mk.PrepareResultType:
        del expert_map, quant_config, quant_type
        assert (
            not apply_router_weight_on_input
        ), "MoonEP policy + MoRI does not support router weights on input."
        if self._routing is not None:
            raise RuntimeError("prepare() called before the previous finalize()")
        if num_experts != self._num_experts:
            raise ValueError(
                f"routing width changed from {self._num_experts} to {num_experts}"
            )

        self._active_is_prefill = self._is_prefill()
        if self._active_is_prefill:
            plan = self._prefill_policy(topk_ids).plan(topk_ids)
            op = self._prefill_op
        else:
            plan = self._decode_policy.plan(topk_ids)
            op = self._decode_op

        expected_epr = op.cfg.num_experts_per_rank
        if plan.num_experts_per_rank != expected_epr:
            raise RuntimeError(
                "planner/dispatch geometry mismatch: "
                f"plan={plan.num_experts_per_rank}, mori={expected_epr}"
            )

        planned_ids = plan.planned_topk_ids
        recv_x, recv_w, _recv_s, recv_idx, total_recv, routing = op.dispatch(
            a1,
            # Preserve the router weights. They are consumed by fused_moe on
            # the destination and folded exactly once before MoRI combine.
            topk_weights.to(torch.float32),
            None,
            planned_ids,
            return_routing=True,
        )
        self._active_op = op
        self._active_plan = plan
        self._routing = routing
        return (
            recv_x,
            None,
            mk.ExpertTokensMetadata(
                expert_num_tokens=total_recv, expert_num_tokens_cpu=None
            ),
            recv_idx,
            recv_w,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        del output, topk_weights, apply_router_weight_on_input
        if self._routing is None or self._active_op is None:
            raise RuntimeError("finalize() called before prepare()")
        op = self._active_op
        routing = self._routing
        self._routing = None
        self._active_op = None
        self._active_plan = None
        out, _ = op.combine(fused_expert_output, routing=routing)
        return out[: topk_ids.shape[0]]

    def adopt_weights(self, layer: torch.nn.Module) -> None:
        """Move resident weights to P2P-readable storage and add B cache rows."""

        if self._pools is not None:
            return
        pools = []
        for name in self.ADOPTED:
            param = getattr(layer, name, None)
            entry = (None, False, False)
            if param is not None and param.data is not None:
                pool, flat = self._pool_for(param.data)
                shuffled = bool(getattr(param, "is_shuffled", False))
                param.data = pool.home.reshape(param.data.shape)
                entry = (pool, flat, shuffled)
            pools.append(entry)
        self._pools = tuple(pools)

    def _pool_for(self, tensor: torch.Tensor):
        from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

        epn = self._experts_per_rank
        flat = tensor.shape[0] != epn
        if flat:
            if tensor.shape[0] % epn:
                raise ValueError(
                    f"cannot index {tuple(tensor.shape)} by expert: leading dim "
                    f"is neither {epn} nor a multiple of it"
                )
            view = tensor.reshape(epn, tensor.shape[0] // epn, *tensor.shape[1:])
        else:
            view = tensor
        pool = MoonEPWeightPool(
            rank=self._rank,
            world_size=self._world_size,
            experts_per_rank=epn,
            prefetch_slots=self._prefetch_slots,
            weight_shape=tuple(view.shape[1:]),
            dtype=tensor.dtype,
            block_num=self._prefill_op.cfg.dispatch_block_num or 1024,
        )
        pool.stage_home(view.contiguous())
        logger.info(
            "MoonEP policy adopted %s%s: %d resident + %d prefetch slots",
            tuple(tensor.shape),
            tensor.dtype,
            epn,
            self._prefetch_slots,
        )
        return pool, flat

    @staticmethod
    def _pool_view(entry, *, prefetched: bool):
        pool, flat, shuffled = entry
        if pool is None:
            return None
        tensor = pool.prefetched if prefetched else pool.home
        if flat:
            tensor = tensor.reshape(-1, *tensor.shape[2:])
        if shuffled:
            tensor.is_shuffled = True
        return tensor

    def _masks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        masks = self._virtual_masks.get(key)
        if masks is None:
            masks = _make_virtual_expert_masks(
                rank=self._rank,
                world_size=self._world_size,
                experts_per_rank=self._experts_per_rank,
                prefetch_slots=self._prefetch_slots,
                device=device,
            )
            self._virtual_masks[key] = masks
        return masks

    def run_dispatched_experts(
        self,
        rows: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_mask: torch.Tensor | None,
        num_local_tokens: torch.Tensor | None,
        activation=None,
        quant_type=None,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
        dtype=None,
        extra_kwargs: dict | None = None,
    ) -> torch.Tensor:
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe

        if self._pools is None:
            raise RuntimeError("adopt_weights() must run before the first MoE call")
        if w1.data_ptr() != self._pools[0][0].home.data_ptr():
            raise RuntimeError("the layer weights no longer alias the MoonEP pool")
        if self._active_plan is None:
            raise RuntimeError("experts called before policy planning")

        act = activation if activation is not None else ActivationType.Silu
        qt = quant_type if quant_type is not None else QuantType.No
        common = dict(
            activation=act,
            quant_type=qt,
            num_local_tokens=num_local_tokens,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            dtype=dtype if dtype is not None else rows.dtype,
            **(extra_kwargs or {}),
        )

        if not self._active_is_prefill:
            return fused_moe(
                rows,
                self._pool_view(self._pools[0], prefetched=False),
                self._pool_view(self._pools[1], prefetched=False),
                topk_weights,
                topk_ids,
                expert_mask,
                w1_scale=self._pool_view(self._pools[2], prefetched=False),
                w2_scale=self._pool_view(self._pools[3], prefetched=False),
                bias1=(
                    self._pool_view(self._pools[4], prefetched=False)
                    if self._pools[4][0] is not None
                    else bias1
                ),
                bias2=(
                    self._pool_view(self._pools[5], prefetched=False)
                    if self._pools[5][0] is not None
                    else bias2
                ),
                **common,
            )

        selected = self._active_plan.experts_to_copy[self._rank].contiguous()
        for pool, _flat, _shuffled in self._pools:
            if pool is not None:
                pool.prefetch(selected)

        home_mask, prefetch_mask = self._masks(rows.device)
        home_out = fused_moe(
            rows,
            self._pool_view(self._pools[0], prefetched=False),
            self._pool_view(self._pools[1], prefetched=False),
            topk_weights,
            topk_ids,
            home_mask,
            w1_scale=self._pool_view(self._pools[2], prefetched=False),
            w2_scale=self._pool_view(self._pools[3], prefetched=False),
            bias1=(
                self._pool_view(self._pools[4], prefetched=False)
                if self._pools[4][0] is not None
                else bias1
            ),
            bias2=(
                self._pool_view(self._pools[5], prefetched=False)
                if self._pools[5][0] is not None
                else bias2
            ),
            **common,
        )
        prefetch_out = fused_moe(
            rows,
            self._pool_view(self._pools[0], prefetched=True),
            self._pool_view(self._pools[1], prefetched=True),
            topk_weights,
            topk_ids,
            prefetch_mask,
            w1_scale=self._pool_view(self._pools[2], prefetched=True),
            w2_scale=self._pool_view(self._pools[3], prefetched=True),
            bias1=(
                self._pool_view(self._pools[4], prefetched=True)
                if self._pools[4][0] is not None
                else bias1
            ),
            bias2=(
                self._pool_view(self._pools[5], prefetched=True)
                if self._pools[5][0] is not None
                else bias2
            ),
            **common,
        )
        return home_out.add_(prefetch_out)


class MoriV2ModularKernel(mk.FusedMoEModularKernel):
    """Modular kernel for the v2 path.

    On the fused transport it steps out of the way: MegaMoE runs the whole layer,
    so forward() hands it this layer's weights and returns its output instead of
    walking prepare -> fused_moe -> finalize.

    Both transports get the same grid shrink. The dispatch arena is padded to a
    huge static token_num (ws * max_num_inp_token_per_rank) while the received
    tokens occupy only the first ``total_recv`` rows, so it is capped at the
    static ``sum(running_tokens_across_dp)`` bound (same as the base policy):
    the grid-bound aiter kernels (route-ksplit preshuffle, gather-reduce) then
    launch a grid sized to what the group actually sent instead of the full
    arena, and the single-block route/psum kernels shrink too.
    Gather slices the buffers here; the fused path passes the bound down as
    ``recv_token_bound`` because it never sees them.
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

    def _maybe_trim_dispatch_output(
        self,
        dispatch_a1: torch.Tensor,
        dispatch_scale: torch.Tensor | None,
        dispatch_ids: torch.Tensor,
        dispatch_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_tokens_meta,
    ):
        bound = self._recv_bound(dispatch_a1.shape[0])
        if bound is not None:
            dispatch_a1 = dispatch_a1[:bound]
            dispatch_ids = dispatch_ids[:bound]
            dispatch_weights = dispatch_weights[:bound]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:bound]
        return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        mega = self.prepare_finalize.mega
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
            from atom.model_ops.fused_moe_triton import triton_mega_moe

            assert not kwargs.get(
                "apply_router_weight_on_input", False
            ), "mori does not support apply_router_weight_on_input=True now."
            arena_rows = (
                self.prepare_finalize.num_dispatchers() * mega.max_tokens_per_rank
            )
            bound = self._recv_bound(topk_ids, arena_rows)
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

        if mega is None or triton_experts:
            return super().forward(
                hidden_states, w1, w2, topk_weights, topk_ids, **kwargs
            )

        assert not kwargs.get(
            "apply_router_weight_on_input", False
        ), "mori does not support apply_router_weight_on_input=True now."
        self._assert_recipe_matches(mega, kwargs)

        return mega(
            hidden_states.contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.to(torch.int32).contiguous(),
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
        )

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
        built = {
            "activation": mega.activation,
            "quant_type": mega.quant_type,
            "gate_mode": mega.gate_mode,
            "swiglu_limit": mega.swiglu_limit,
            "hidden_pad": mega.hidden_pad,
            "intermediate_pad": mega.intermediate_pad,
        }
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


def make_mori_v2_prepare_finalize(moe, all2all_manager) -> MoriV2PrepareAndFinalize:
    """Build a MoriV2PrepareAndFinalize for the given MoE config + EP group."""
    from aiter.dist.parallel_state import get_ep_group

    ep_group = get_ep_group()
    ep_src_global_rank = ep_group.ranks[0]
    ep_size = all2all_manager.world_size

    if _resolve_transport() == "mega":
        # Geometry only; the expert-GEMM recipe comes from the layer later.
        return MoriV2PrepareAndFinalize(
            None,
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
        )

    op = init_mori_v2_op(
        ep_rank=all2all_manager.rank,
        ep_size=ep_size,
        ep_src_global_rank=ep_src_global_rank,
        hidden_dim=moe.hidden_dim,
        max_num_inp_token_per_rank=moe.max_num_tokens,
        num_local_experts=moe.num_experts // ep_size,
        num_experts_per_token=moe.experts_per_token,
        data_type_itemsize=moe.in_dtype.itemsize,
        combine_mode="gather",
    )
    return MoriV2PrepareAndFinalize(
        op,
        max_tokens_per_rank=moe.max_num_tokens,
        num_dispatchers=ep_size,
    )


def make_moonep_policy_prepare_finalize(
    moe,
    all2all_manager,
    *,
    prefetch_slots: int,
    quant_config: FusedMoEQuantConfig | None,
) -> MoonEPPolicyMoriPrepareAndFinalize:
    """Build the two MoRI geometries used by the disaggregated MoonEP policy."""

    from aiter.dist.parallel_state import get_ep_group

    ep_group = get_ep_group()
    ep_src_global_rank = ep_group.ranks[0]
    ep_size = all2all_manager.world_size
    experts_per_rank = moe.num_experts // ep_size

    if envs.ATOM_MORI_V2_FUSED:
        raise ValueError(
            "ATOM_EP_BACKEND=moonep uses standard fused_moe and cannot be "
            "combined with ATOM_MORI_V2_FUSED=1"
        )

    if envs.ATOM_MORI_V2:
        common = {
            "ep_rank": all2all_manager.rank,
            "ep_size": ep_size,
            "ep_src_global_rank": ep_src_global_rank,
            "hidden_dim": moe.hidden_dim,
            "max_num_inp_token_per_rank": moe.max_num_tokens,
            "num_experts_per_token": moe.experts_per_token,
            "data_type_itemsize": moe.in_dtype.itemsize,
            "combine_mode": "gather",
        }
        decode_op = init_mori_v2_op(
            num_local_experts=experts_per_rank,
            **common,
        )
        prefill_op = init_mori_v2_op(
            num_local_experts=experts_per_rank + prefetch_slots,
            **common,
        )
        transport = "v2"
    else:
        decode_op = _make_mori_v1_policy_transport(
            moe,
            all2all_manager,
            num_local_experts=experts_per_rank,
            quant_config=quant_config,
        )
        prefill_op = _make_mori_v1_policy_transport(
            moe,
            all2all_manager,
            num_local_experts=experts_per_rank + prefetch_slots,
            quant_config=quant_config,
        )
        transport = "v1"
    logger.info(
        "MoonEP policy over MoRI %s: rank=%d world=%d home=%d prefetch=%d "
        "decode_epr=%d prefill_epr=%d",
        transport,
        all2all_manager.rank,
        ep_size,
        experts_per_rank,
        prefetch_slots,
        experts_per_rank,
        experts_per_rank + prefetch_slots,
    )
    return MoonEPPolicyMoriPrepareAndFinalize(
        prefill_op=prefill_op,
        decode_op=decode_op,
        rank=all2all_manager.rank,
        world_size=ep_size,
        num_experts=moe.num_experts,
        prefetch_slots=prefetch_slots,
        max_tokens_per_rank=moe.max_num_tokens,
        num_dispatchers=ep_size,
    )


def _make_mori_v1_policy_transport(
    moe,
    all2all_manager,
    *,
    num_local_experts: int,
    quant_config: FusedMoEQuantConfig | None,
) -> _MoriV1PolicyTransport:
    """Create the existing gfx950 MoRI transport with a policy-specific EPR."""

    from atom.model_ops.fused_moe.mori_prepare_finalize import (
        MoriPrepareAndFinalize,
        resolve_mori_dispatch,
    )

    dispatch_format = resolve_mori_dispatch(
        in_dtype=moe.in_dtype,
        hidden_dim=moe.hidden_dim,
        quant_config=quant_config,
    )
    if dispatch_format.is_fp4 or dispatch_format.is_fp8:
        raise ValueError(
            "ATOM_EP_BACKEND=moonep currently requires BF16 MoRI dispatch; "
            "disable ATOM_MORI_FP4_DISPATCH for the gfx950 closure"
        )

    all_to_all_args = {
        "rank": all2all_manager.rank,
        "num_ep_ranks": all2all_manager.world_size,
        "quant_dtype": dispatch_format.dtype,
        "token_hidden_size": moe.hidden_dim,
        "scale_dim": dispatch_format.scale_dim,
        "scale_type_size": dispatch_format.scale_type_size,
        "max_num_tokens_per_dp_rank": moe.max_num_tokens,
        "input_dtype": moe.in_dtype,
        "num_local_experts": num_local_experts,
        "num_experts_per_token": moe.experts_per_token,
        "gpu_per_node": moe.moe_parallel_config.local_ep_size,
    }
    if envs.ATOM_MORI_COMBINE_QUANT != "none":
        all_to_all_args["quant_type"] = envs.ATOM_MORI_COMBINE_QUANT

    handle = all2all_manager.get_handle(all_to_all_args)
    prepare_finalize = MoriPrepareAndFinalize(
        handle,
        max_tokens_per_rank=moe.max_num_tokens,
        num_dispatchers=all2all_manager.world_size,
        dispatch_format=dispatch_format,
        is_async=False,
    )
    return _MoriV1PolicyTransport(prepare_finalize, num_local_experts)
