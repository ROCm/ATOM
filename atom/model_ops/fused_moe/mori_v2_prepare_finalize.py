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
"""

import logging
import os
import sys
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from aiter import ActivationType, QuantType
from aiter.dist.parallel_state import get_dp_group
from aiter.ops.flydsl.moe_common import GateMode

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
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
    )
    comm.barrier()
    _MEGA_TRANSPORTS[key] = mega
    logger.info(
        "[MORI-V2] Created MegaMoE: ep_rank=%d ep_size=%d hidden=%d inter=%d "
        "experts=%d topk=%d M=%d act=%s gate=%s quant=%s pad=(%d,%d) "
        "swiglu_limit=%s dispatch=%s",
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
        assert (
            self.mega is None
        ), "the fused transport runs the layer in MoriV2ModularKernel.forward()"

        # bf16 dispatch, no wire quant: scales=None. indices carry global expert
        # ids (0..global_num_experts-1); mori routes id -> rank = id // EPR.
        recv_x, recv_w, _recv_s, recv_idx, _total_recv_t, routing = self._op.dispatch(
            a1,
            topk_weights.to(torch.float32),
            None,
            topk_ids.to(torch.int32),
            return_routing=True,
        )
        self._routing = routing

        # Capture-safe: do NOT call total_recv_t.item() (a GPU->CPU sync that is
        # illegal during cudagraph capture). fused_moe is handed the FULL
        # fixed-size arena buffers, aliased in place rather than sliced to the
        # received count, so the shapes stay static across capture/replay.
        dispatch_a1 = recv_x
        dispatch_ids = recv_idx
        dispatch_weights = recv_w

        # num_local_tokens is left unset (expert_num_tokens=None): the grouped
        # a8w4 path derives per-expert routing from the (already trimmed) global
        # ids + expert_mask, exactly as test_moe_layer_ep.py does.
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=None, expert_num_tokens_cpu=None
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


class MoriV2ModularKernel(mk.FusedMoEModularKernel):
    """Modular kernel for the v2 path.

    On the fused transport it steps out of the way: MegaMoE runs the whole layer,
    so forward() hands it this layer's weights and returns its output instead of
    walking prepare -> fused_moe -> finalize.

    Both transports get the same grid shrink. The dispatch arena is padded to a
    huge static token_num (ws * max_num_inp_token_per_rank) while the received
    tokens occupy only the first ``total_recv`` rows, so under a uniform
    all-ranks-decode batch it is capped at the static ``running_tokens*topk*dp``
    bound (the V1/base policy): the grid-bound aiter kernels (route-ksplit
    preshuffle, gather-reduce) then launch a grid sized to the decode bucket
    instead of the full arena, and the single-block route/psum kernels shrink too.
    Gather slices the buffers here; the fused path passes the bound down as
    ``recv_token_bound`` because it never sees them.
    """

    def _decode_recv_bound(self, topk_ids: torch.Tensor, arena_rows: int) -> int | None:
        """Static recv-row bound for a uniform decode batch, else None (no shrink).

        Correctness / capture-safety:
          * ``running_tokens`` is a python int, fixed per captured graph, so the
            bound is static across capture/replay and no GPU->CPU sync is needed
            (unlike reading the device ``total_recv``).
          * Under uniform decode each of ``dp`` ranks holds ``running_tokens``,
            each routed to ``topk`` experts; worst case every route lands on this
            rank, so ``total_recv <= running_tokens*topk*dp``. The bound therefore
            never drops a valid row, and the aiter kernels' device-side
            ``num_valid_routes`` guard still skips the exact within-buffer tail
            [total_recv, bound).
          * Mixed/prefill batches keep the full arena, matching the base-class
            guard.
        """
        context = get_forward_context().context
        if context is None:
            return None
        tokens_unified = getattr(
            context, "running_tokens_are_unified", not context.is_prefill
        )
        if not tokens_unified:
            return None
        bound = context.running_tokens * topk_ids.shape[1] * get_dp_group().world_size
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
        bound = self._decode_recv_bound(topk_ids, dispatch_a1.shape[0])
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
        if mega is None:
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
            recv_token_bound=self._decode_recv_bound(
                topk_ids,
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
