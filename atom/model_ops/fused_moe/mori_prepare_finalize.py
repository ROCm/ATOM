# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import bisect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any

import torch
from aiter import QuantType, dtypes
from aiter.jit.utils.chip_info import get_cu_num

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.plugin import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import (
    enable_pad_rows_device,
    get_forward_context,
    get_pad_rows_device,
)

try:
    import mori

    MORI_AVAILABLE = True
except ImportError:
    mori = None  # type: ignore
    MORI_AVAILABLE = False

logger = logging.getLogger("atom")


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)

# Read once at import: the wire format is fixed for the life of the process, and
# make_prepare_finalize runs per MoE layer (58x for DSR1). envs.__getattr__ does
# not cache, so binding here also keeps it to a single getenv.
_FP4_DISPATCH = envs.ATOM_MORI_FP4_DISPATCH
# Same reasoning: the launch policy is consulted on every dispatch and combine.
_LAUNCH_POLICY = envs.ATOM_MORI_LAUNCH_POLICY

# IntraNode kernels launch warp_per_block * 64 threads per block; 16 warps is
# the 1024-thread block limit.
_MAX_WARP_PER_BLOCK = 16

# Launch tables already logged; every MoE layer has its own prepare/finalize.
_LOGGED_LAUNCH_TABLES: set[tuple] = set()


@cache
def mori_tuned_launch_table(
    phase: str,
    ep_size: int,
    dtype: torch.dtype,
    hidden_dim: int,
    topk: int,
    zero_copy: bool | None = None,
    quant_type: str | None = None,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None:
    """MoRI's shipped IntraNode tuning rules for one call shape, as a step table.

    Returns ``(ceilings, geometries)``: the ``(block_num, warp_per_block)`` for
    a per-rank token count ``n`` is ``geometries[bisect_left(ceilings, n)]``,
    and the extra last entry covers counts past the largest rule. It is built
    by asking mori's own ``TuningConfigManager.lookup`` at every rule boundary
    (and one past the last), so it reproduces what mori's AUTO mode would pick
    -- tightest ceiling, clamp above, exact-then-relaxed hidden/topk -- without
    paying that walk on every launch. ``zero_copy``/``quant_type`` filter the
    combine rules and are left None for dispatch, as mori does.

    block_num is capped at the CU count: the IntraNode combine's cross-device
    barrier spins until every block has arrived, so all blocks must be
    co-resident. None when mori's tuning module or a matching rule is missing.
    """
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from mori.ops.tuning_config import (
            TuningConfigManager,
            detect_gpu_model,
            quant_type_to_config_str,
        )

        manager = TuningConfigManager.get_instance(
            get_gfx_runtime(), "IntraNode", ep_size, detect_gpu_model()
        )
        if quant_type is not None:
            quant_type = quant_type_to_config_str(quant_type)
    except Exception as exc:  # noqa: BLE001 -- the caller falls back to legacy
        logger.warning(f"[MORI] tuned launch table unavailable ({exc!r})")
        return None
    rules = manager.dispatch_rules if phase == "dispatch" else manager.combine_rules
    if not rules:
        return None
    ceilings = tuple(sorted({int(rule["num_tokens"]) for rule in rules}))
    cu_num = get_cu_num()
    geometries = []
    for num_tokens in ceilings + (ceilings[-1] + 1,):
        params = TuningConfigManager.lookup(
            rules,
            dtype,
            num_tokens,
            hidden_dim,
            zero_copy=zero_copy,
            quant_type=quant_type,
            topk=topk,
        )
        if params is None:
            return None
        geometries.append(
            (
                min(params.block_num, cu_num),
                min(params.warp_per_block, _MAX_WARP_PER_BLOCK),
            )
        )
    return ceilings, tuple(geometries)


_MXFP8_DISPATCH = envs.ATOM_MORI_FP8_DISPATCH
_MASK_PAD_ROWS = envs.ATOM_MORI_MASK_PAD_ROWS
_ZERO_COPY_COMBINE = envs.ATOM_MORI_ZERO_COPY_COMBINE

# ATOM_MORI_ZERO_COPY_COMBINE value -> how the expert output reaches MoRI's
# registered combine input buffer (None: push combine, as before).
_ZERO_COPY_COMBINE_MODES = {"0": None, "1": "direct", "copy": "copy"}


@cache
def _log_pad_masking(active: bool) -> None:
    """Say once, not per MoE layer, whether ATOM_MORI_MASK_PAD_ROWS took."""
    if active:
        logger.info("[MORI] DP pad rows are routed to expert -1 before dispatch")
    else:
        logger.warning(
            "[MORI] ATOM_MORI_MASK_PAD_ROWS=1 ignored: needs the IntraNode "
            "kernel and native ATOM serving"
        )


def _drops_negative_ids(mori_op: Any) -> bool:
    """Whether this op's dispatch skips a (token, k) whose expert id is < 0.

    Read in intranode.hpp: IntraNode dispatch writes such a pair's destination
    as `worldSize`, sends nothing for it, and its combine then treats that
    destination as absent. Other kernels are unverified, so they keep the ids.
    """
    kernel_type = getattr(getattr(mori_op, "config", None), "kernel_type", None)
    return kernel_type == mori.ops.EpDispatchCombineKernelType.IntraNode


def resolve_zero_copy_combine(mode: str, mori_op: Any) -> str | None:
    """How the sync combine gets its input: None (push), "direct" or "copy".

    MoRI's IntraNode combine has two transports over the same symmetric
    buffer. Push (use_external_inp_buf, what aiter's handle defaults to) has
    every rank copy each received row into the sender's staging slot over
    XGMI, then barrier, then sum local staging. Zero-copy (pull) skips that
    copy: after the barrier each rank reads its tokens' rows straight out of
    the peers' registered combine input buffers, so the expert output has to
    already be there, in dispatch-receive row order. The buffer, its size and
    the kernels are the same for both, and combine() takes the transport per
    call, so aiter's cached handle serves either -- no second handle on the
    shmem heap.

    Only the IntraNode kernel's pull path was checked, and it has no combine
    codec: MoRI raises for blockwise and would silently drop fp8_direct_cast,
    so those keep the push combine. Decided once per layer from the env and
    the handle, so every rank picks the same transport.
    """
    if mode not in _ZERO_COPY_COMBINE_MODES:
        raise ValueError(
            f"ATOM_MORI_ZERO_COPY_COMBINE={mode!r}: expected one of "
            f"{sorted(_ZERO_COPY_COMBINE_MODES)}"
        )
    zero_copy = _ZERO_COPY_COMBINE_MODES[mode]
    if zero_copy is None:
        return None
    config = mori_op.config
    usable = (
        config.kernel_type == mori.ops.EpDispatchCombineKernelType.IntraNode
        and config.quant_type == "none"
    )
    _log_zero_copy_combine(zero_copy if usable else None)
    return zero_copy if usable else None


@cache
def _log_zero_copy_combine(zero_copy: str | None) -> None:
    """Say once, not per MoE layer, whether ATOM_MORI_ZERO_COPY_COMBINE took."""
    if zero_copy is None:
        logger.warning(
            "[MORI] ATOM_MORI_ZERO_COPY_COMBINE ignored: needs the IntraNode "
            "kernel and ATOM_MORI_COMBINE_QUANT=none"
        )
    else:
        logger.info(
            "[MORI] zero-copy combine: peers pull the expert output from the "
            "registered combine input buffer "
            + (
                "(fused_moe writes it there)"
                if zero_copy == "direct"
                else "(copied in before combine)"
            )
        )


@dataclass(frozen=True)
class MoriDispatchFormat:
    dtype: torch.dtype  # what dispatch() receives -> picks the MoRI kernel
    quant_type: Any | None  # aiter QuantType for the pre-dispatch quantizer
    scale_dim: int
    scale_type_size: int

    @property
    def is_fp4(self) -> bool:
        return self.dtype == dtypes.fp4x2

    @property
    def is_fp8(self) -> bool:
        return self.dtype in _FP8_DTYPES

    @property
    def is_mxfp8(self) -> bool:
        return self.is_fp8 and self.quant_type == QuantType.per_1x32


def resolve_mori_dispatch(
    in_dtype: torch.dtype,
    hidden_dim: int,
    quant_config: FusedMoEQuantConfig | None = None,
) -> MoriDispatchFormat:
    """Decide the MoRI wire format. Call once per layer construction."""
    if _FP4_DISPATCH and _MXFP8_DISPATCH:
        raise ValueError(
            "ATOM_MORI_FP4_DISPATCH and ATOM_MORI_FP8_DISPATCH both pick the "
            "dispatch wire format; set at most one"
        )
    if _MXFP8_DISPATCH:
        # Only an MXFP4-weight layer runs aiter's a8w4 fused_moe, whose own
        # activation quant this moves ahead of the dispatch; any other layer
        # would take these fp8 rows as data. Refuse rather than fall back to
        # bf16: a second wire format is a second MoRI handle, i.e. another set
        # of 131072-row staging buffers on the shmem heap.
        weight_dtype = None if quant_config is None else quant_config._w1.dtype
        if weight_dtype != "mxfp4":
            raise ValueError(
                "ATOM_MORI_FP8_DISPATCH=1 needs MXFP4 routed experts (aiter "
                f"a8w4 fused_moe); this MoE layer has weight dtype {weight_dtype!r}"
            )
        _log_dispatch_format_once(
            f"mxfp8 (fp8 e4m3 + e8m0 per 1x32): scale_dim={hidden_dim // 32}, "
            "scale_type_size=1"
        )
        # Per 1x32, e8m0 scales, unshuffled -- what aiter's fused_moe would
        # compute from the received bf16 rows; it then only sorts the scales.
        return MoriDispatchFormat(
            dtype=dtypes.fp8,
            quant_type=QuantType.per_1x32,
            scale_dim=hidden_dim // 32,
            scale_type_size=dtypes.fp8_e8m0.itemsize,
        )
    if _FP4_DISPATCH:
        # fp4 blockwise is one scale per 32 elements, not per 128 as fp8 uses,
        # and get_hip_quant(per_1x32) emits e8m0 scales (1 byte), not fp32.
        return MoriDispatchFormat(
            dtype=dtypes.fp4x2,
            quant_type=QuantType.per_1x32,
            scale_dim=hidden_dim // 32,
            scale_type_size=torch.float8_e8m0fnu.itemsize,
        )
    return MoriDispatchFormat(
        dtype=in_dtype,
        quant_type=None,
        scale_dim=0,
        scale_type_size=torch.float32.itemsize,
    )


@lru_cache(maxsize=16)
def _log_dispatch_format_once(desc: str) -> None:
    logger.info(f"[MORI] dispatch wire format: {desc}")


@lru_cache(maxsize=16)
def check_mxfp8_dispatch_consumable(
    quant_type: QuantType,
    w1_dtype: torch.dtype,
    activation: Any,
    gate_mode: str,
    hidden_pad: int,
) -> None:
    """Refuse an MXFP8 dispatch that aiter's fused_moe would not consume as-is.

    fused_moe skips its own activation quant only on its a8w4/a8w8 branch
    (fp8 input plus a1_scale: the scales are just sorted); on any other branch
    the fp8 rows are read as data -- its bf16 path casts them and drops the
    scales -- which is wrong numerics, not an error. The branch comes from
    resolve_activation_dtype, per call from M unless AITER_BF16_FP8_MOE_BOUND=0
    takes M out of it; asking with M=None returns fp8 only when every batch
    size lands there. Cached, so the eager path pays a dict lookup per layer.
    """
    from aiter.fused_moe import resolve_activation_dtype

    q_dtype_a = resolve_activation_dtype(
        quant_type, w1_dtype, activation=activation, gate_mode=gate_mode, M=None
    )
    if (
        QuantType(quant_type) != QuantType.per_1x32
        or w1_dtype not in (dtypes.fp4x2, dtypes.fp8)
        or q_dtype_a != dtypes.fp8
        or hidden_pad
    ):
        raise RuntimeError(
            "ATOM_MORI_FP8_DISPATCH=1 sends MXFP8 rows, but aiter fused_moe "
            f"would not take them as-is here ({quant_type=}, {w1_dtype=}, "
            f"{activation=}, {gate_mode=}, {hidden_pad=}, resolved activation "
            f"dtype {q_dtype_a}). On gfx950 it needs ATOM_MOE_GU_ITLV=1 and "
            "AITER_BF16_FP8_MOE_BOUND=0; otherwise unset ATOM_MORI_FP8_DISPATCH."
        )


def _mxfp8_quant(a1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-1x32 fp8 with e8m0 scales, row-major and unshuffled.

    The kernel aiter's fused_moe runs on the received rows for its split path
    (more than 8*256/topk rows); below that it runs a fused quant+sort kernel
    with the same e8m0 RoundUp scale. Given these scales, fused_moe only
    sorts them into the GEMM layout (mxfp4_moe_sort_fwd).
    """
    if a1.shape[0] == 0:
        # e8m0 scales are one byte each -- must match the scale_type_size
        # handed to MoRI in moe.py.
        return (
            torch.empty(a1.shape, dtype=dtypes.fp8, device=a1.device),
            torch.empty(
                (0, a1.shape[-1] // 32), dtype=dtypes.fp8_e8m0, device=a1.device
            ),
        )
    from aiter import get_hip_quant

    return get_hip_quant(QuantType.per_1x32)(
        a1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using MoRI kernels.
    """

    # "direct" / "copy" when the sync combine pulls (see
    # resolve_zero_copy_combine); None pushes.
    _zero_copy_combine: str | None = None

    def __init__(
        self,
        mori_ops: list[Any],
        max_tokens_per_rank: int,
        num_dispatchers: int,
        dispatch_format: MoriDispatchFormat,
        *,
        low_latency: bool,
        internode: bool,
        quant_dtype: torch.dtype = None,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriPrepareAndFinalize but not installed. "
                "Please install mori to use this feature."
            )
        if not mori_ops:
            raise ValueError("mori_ops must contain at least one handle")
        super().__init__()
        self._mori_ops = tuple(mori_ops)
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dispatch_format = dispatch_format
        self.quant_dtype = quant_dtype
        self._use_split_send_recv = low_latency and not internode
        # (phase, op, dtype, hidden_dim) -> step table or None; see
        # _get_launch_config.
        self._launch_tables: dict[tuple, Any] = {}
        # See mask_pad_topk_ids. Plugin frontends own their own step shapes
        # and never publish the pad-row mask, so they are left out.
        self._mask_pad_rows = False
        if _MASK_PAD_ROWS:
            self._mask_pad_rows = not is_plugin_mode() and _drops_negative_ids(
                self._mori_ops[0]
            )
            _log_pad_masking(self._mask_pad_rows)
            if self._mask_pad_rows:
                enable_pad_rows_device(
                    max_tokens_per_rank,
                    torch.device("cuda", torch.cuda.current_device()),
                )
        # Sync path only. Slot 0 also serves TBO ubatch 0, but the ubatch
        # combines stay on push: the transport is chosen per combine() call.
        self._zero_copy_combine = resolve_zero_copy_combine(
            _ZERO_COPY_COMBINE, self._mori_ops[0]
        )
        # (dtype, hidden_dim) -> view of the sync op's registered combine
        # input buffer; see _registered_combine_input.
        self._combine_inputs: dict[tuple, torch.Tensor] = {}

    # Derived from the resolved format so there is no second copy to keep in
    # sync with the staging config.
    @property
    def use_fp4_dispatch(self) -> bool:
        return self.dispatch_format.is_fp4

    @property
    def use_fp8_dispatch(self) -> bool:
        return self.dispatch_format.is_fp8

    @property
    def use_mxfp8_dispatch(self) -> bool:
        return self.dispatch_format.is_mxfp8

    @property
    def quant_type(self):
        return self.dispatch_format.quant_type

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        from atom.utils.tbo.ubatching import tbo_active

        return len(self._mori_ops) > 1 and tbo_active()

    def _current_tbo_mori_op(self):
        from atom.utils.tbo.ubatching import tbo_current_ubatch_id

        return self._mori_ops[tbo_current_ubatch_id()]

    def adapt_routing_for_fused_moe(
        self,
        dispatch_ids: torch.Tensor,
        dispatch_weights: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # MoRI delivers only the real top-k columns, and AITER drops the last
        # one from its tuned-kernel key under EP, so without the masked
        # sentinel column a top-6 model looks up top-5 rows and misses every
        # tuned kernel. Sits after the trim (see FusedMoEModularKernel.forward)
        # so it costs the group's rows, not the receive arena's; the sentinel
        # routes to expert_mask[num_experts] == 0, so moe_sorting drops it. The
        # TBO receivers feed the same forward, so they get it too.
        sentinel_expert_id = mk.aiter_ep_sentinel_expert_id(num_experts, expert_map)
        if sentinel_expert_id is None:
            return dispatch_ids, dispatch_weights
        return mk.append_aiter_ep_sentinel(
            dispatch_ids, dispatch_weights, sentinel_expert_id
        )

    def _tuned_launch_table(
        self,
        phase: str,
        mori_op: Any,
        dtype: torch.dtype,
        hidden_dim: int,
        pull_combine: bool = False,
    ):
        config = mori_op.config
        # MoRI tunes the pull combine separately. The transport is per call,
        # not per op: slot 0 pulls on the sync path and pushes as TBO ubatch 0.
        zero_copy = phase == "combine" and (
            pull_combine or not config.use_external_inp_buf
        )
        key = (phase, id(mori_op), dtype, hidden_dim, zero_copy)
        if key in self._launch_tables:
            return self._launch_tables[key]
        kernel_type = getattr(config.kernel_type, "name", str(config.kernel_type))
        table = None
        # Only the IntraNode kernels' tables and co-residency limits are
        # accounted for here; other kernel types keep the legacy grid.
        if kernel_type == "IntraNode":
            table = mori_tuned_launch_table(
                phase,
                config.world_size,
                dtype,
                hidden_dim,
                config.num_experts_per_token,
                **(
                    {}
                    if phase == "dispatch"
                    else {
                        "zero_copy": zero_copy,
                        "quant_type": config.quant_type,
                    }
                ),
            )
        self._launch_tables[key] = table
        log_key = (phase, kernel_type, dtype, hidden_dim, zero_copy)
        if config.rank == 0 and log_key not in _LOGGED_LAUNCH_TABLES:
            _LOGGED_LAUNCH_TABLES.add(log_key)
            if table is None:
                logger.info(
                    f"[MORI] {phase} launch: no tuned table for {kernel_type} "
                    f"{dtype} hidden={hidden_dim}; using the legacy grid"
                )
            else:
                ceilings, geometries = table
                steps = ", ".join(
                    f"<={c}:{b}x{w}" for c, (b, w) in zip(ceilings, geometries)
                )
                transport = (
                    "" if phase == "dispatch" else (" pull" if zero_copy else " push")
                )
                logger.info(
                    f"[MORI] {phase}{transport} launch (tokens/rank:blocks x warps) "
                    f"for {dtype} hidden={hidden_dim}: {steps}, "
                    f">{ceilings[-1]}:{geometries[-1][0]}x{geometries[-1][1]}"
                )
        return table

    def _get_launch_config(
        self,
        phase: str,
        mori_op: Any,
        num_tokens: int,
        dtype: torch.dtype,
        hidden_dim: int,
        *,
        pull_combine: bool = False,
    ) -> tuple[int, int]:
        """Return (block_num, warp_per_block) for one dispatch or combine launch.

        Both kernels move what the whole group sends, not what this rank has:
        a decoding rank's combine pushes back every row a prefilling peer
        dispatched to it. So the tuned policy keys on the group's largest
        per-rank count -- the shape mori's EP8 tables were measured at, every
        rank sending the same count -- and falls back to this rank's own count
        only where no DP reduction produced one. The count is a host value
        fixed per captured graph, so the choice is graph safe. Dispatch and
        combine are tuned separately; their best grids differ.

        atom-vllm has no ATOM forward context at this call site and overrides
        this method via a plugin patch, so keep the signature stable.
        """
        context = get_forward_context().context
        if _LAUNCH_POLICY != "legacy":
            table = self._tuned_launch_table(
                phase, mori_op, dtype, hidden_dim, pull_combine
            )
            if table is not None:
                across_dp = (
                    None if context is None else context.running_tokens_across_dp
                )
                group_tokens = max(across_dp) if across_dp else num_tokens
                ceilings, geometries = table
                return geometries[bisect.bisect_left(ceilings, group_tokens)]
        # The legacy grid: what upstream launches without a tuned table -- 128
        # blocks x 16 warps on a prefilling rank, 64 x 4 otherwise, keyed on
        # this rank's own prefill flag. Also the grid for kernel types the
        # tuned tables do not cover. block_num is capped at the CU count for
        # the same co-residency reason as the tuned table (e.g. 128 blocks on
        # the 80-CU MI308X deadlocked warmup).
        mp = get_cu_num()
        if context is not None and context.is_prefill:
            return min(128, mp), _MAX_WARP_PER_BLOCK
        return min(64, mp), 4

    def mask_pad_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Route this rank's DP pad rows to expert -1.

        A padded decode step runs `running_tokens` rows, of which only the
        leading `scheduled_tokens` carry a request; the tail is there to match
        the group's graph width. Its routing is real, so it pays dispatch, GEMM
        and combine like any token, and being identical rows it piles onto the
        same experts. IntraNode dispatch sends nothing for a negative id, and
        combine sums nothing for a token sent nowhere, so a masked row costs
        neither and comes back as zeros that are sliced off with the padding.

        The pad-row mask is published on the device before every forward and
        draft pass, so a capture records one select against it and each replay
        follows its own step; an eager step checks the host count first and
        skips the select when nothing is padded. Only rows that ARE the step's
        `running_tokens` are touched: nothing else has a padded tail. TBO
        ubatches are left alone, a ubatch's rows not being the step's prefix.
        """
        if not self._mask_pad_rows or self.supports_async():
            return topk_ids
        context = get_forward_context().context
        num_rows = topk_ids.shape[0]
        if context is None or num_rows != context.running_tokens:
            return topk_ids
        if (
            not torch.cuda.is_current_stream_capturing()
            and context.scheduled_tokens >= num_rows
        ):
            return topk_ids
        pad_rows = get_pad_rows_device()
        if pad_rows is None or pad_rows.shape[0] < num_rows:
            return topk_ids
        return torch.where(pad_rows[:num_rows], -1, topk_ids)

    # ---- Synchronous (non-TBO) path ----

    def _registered_combine_input(
        self, dtype: torch.dtype, hidden_dim: int
    ) -> torch.Tensor:
        """The sync op's registered combine input buffer as [max_recv, hidden].

        Row i is what the pull combine reads for dispatch-receive row i. A
        cached view of a fixed symmetric allocation, so it is the same
        pointer in every CUDA graph and costs no device work to fetch.
        """
        key = (dtype, hidden_dim)
        buf = self._combine_inputs.get(key)
        if buf is None:
            config = self._mori_ops[0].config
            if (
                dtype.itemsize > config.max_token_type_size
                or hidden_dim > config.hidden_dim
            ):
                raise RuntimeError(
                    f"zero-copy combine: a {dtype} x {hidden_dim} expert output "
                    "does not fit MoRI's combine input rows "
                    f"({config.max_token_type_size} B x {config.hidden_dim}); "
                    "unset ATOM_MORI_ZERO_COPY_COMBINE"
                )
            buf = self._mori_ops[0].get_registered_combine_input_buffer(
                dtype, hidden_dim=hidden_dim
            )
            self._combine_inputs[key] = buf
        return buf

    def expert_output_buffer(
        self, num_rows: int, hidden_dim: int, dtype: torch.dtype
    ) -> torch.Tensor | None:
        # "direct": fused_moe zeroes and accumulates the received rows' expert
        # output in place in the buffer peers pull from, so the combine input
        # costs neither a push nor a copy. Its rows are the dispatch-receive
        # rows (the trim keeps a prefix; the sentinel adds a column, not a
        # row). The buffer is only rewritten after this rank's next dispatch
        # has completed, and that completion waits for every peer to reach
        # the same dispatch -- i.e. to have finished pulling from it.
        if self._zero_copy_combine != "direct" or self.supports_async():
            return None
        return self._registered_combine_input(dtype, hidden_dim)[:num_rows]

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
        """
        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        assert (
            not apply_router_weight_on_input
        ), "mori does not support apply_router_weight_on_input=True now."
        scale = None
        if self.use_fp4_dispatch:
            from aiter import get_hip_quant

            quant_func = get_hip_quant(self.quant_type or quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp4x2)
        elif self.use_mxfp8_dispatch:
            a1, scale = _mxfp8_quant(a1)
        elif self.use_fp8_dispatch:
            from aiter import get_hip_quant

            quant_func = get_hip_quant(quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)

        block_num, warp_per_block = self._get_launch_config(
            "dispatch", self._mori_ops[0], a1.shape[0], a1.dtype, a1.shape[1]
        )

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self._mori_ops[0].dispatch(
            a1,
            topk_weights,
            scale,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
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
        num_token = topk_ids.shape[0]

        combine_kwargs = {}
        if self._zero_copy_combine is not None:
            num_rows, hidden_dim = fused_expert_output.shape
            combine_input = self._registered_combine_input(
                fused_expert_output.dtype, hidden_dim
            )
            if fused_expert_output.data_ptr() != combine_input.data_ptr():
                # Not written in place ("copy", or an expert path that took no
                # output buffer): one local copy of the rows peers read. Plain
                # stores into the symmetric buffer, as MoRI's own zero-copy
                # tests stage it; the combine's entry barrier orders them
                # before any peer reads.
                combine_input[:num_rows].copy_(fused_expert_output)
            fused_expert_output = combine_input[:num_rows]
            combine_kwargs["use_external_inp_buf"] = 0

        block_num, warp_per_block = self._get_launch_config(
            "combine",
            self._mori_ops[0],
            num_token,
            fused_expert_output.dtype,
            fused_expert_output.shape[1],
            pull_combine=self._zero_copy_combine is not None,
        )

        result = self._mori_ops[0].combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
            **combine_kwargs,
        )[0]
        return result[:num_token]

    # 1. comm-stream (IntraNode / InterNodeV1 / InterNodeV1LL): dispatch() and
    #    combine() on comm_stream. `_use_split_send_recv` is False for these;
    #    mori only exposes the split send/recv API on AsyncLL.
    # 2. AsyncLL (--low-latency, intra-node): dispatch_send/recv,
    #    combine_send/recv (CU-free)
    # Both keep the push combine whatever ATOM_MORI_ZERO_COPY_COMBINE says.
    # Ubatch 0 shares slot 0 with the sync path, but its combine is called
    # without use_external_inp_buf=0, and expert_output_buffer never hands
    # fused_moe the registered combine input while a ubatch is running.
    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> mk.ReceiverType:
        assert (
            not apply_router_weight_on_input
        ), "mori does not support apply_router_weight_on_input=True now."

        scale = None
        if self.use_fp4_dispatch:
            from aiter import get_hip_quant

            num_tokens = a1.shape[0]
            if num_tokens > 0:
                quant_func = get_hip_quant(self.quant_type or QuantType.per_1x32)
                a1, scale = quant_func(a1, quant_dtype=dtypes.fp4x2)
            else:
                hidden_size = a1.shape[1] if a1.dim() > 1 else 0
                a1 = torch.empty(a1.shape, dtype=dtypes.fp4x2, device=a1.device)
                # per_1x32 emits e8m0 scales, one byte each -- must match the
                # scale_type_size handed to MoRI in moe.py.
                scale = torch.empty(
                    (0, hidden_size // 32),
                    dtype=torch.float8_e8m0fnu,
                    device=a1.device,
                )
        elif self.use_mxfp8_dispatch:
            a1, scale = _mxfp8_quant(a1)
        elif self.use_fp8_dispatch:
            from aiter import get_hip_quant

            num_tokens = a1.shape[0]
            if num_tokens > 0:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)
            else:
                hidden_size = a1.shape[1] if a1.dim() > 1 else 0
                a1 = torch.empty(a1.shape, dtype=dtypes.fp8, device=a1.device)
                scale = torch.empty(
                    (0, hidden_size // 128),
                    dtype=torch.float32,
                    device=a1.device,
                )

        if self._use_split_send_recv:
            return self._prepare_async_ll(a1, topk_weights, topk_ids, scale)
        return self._prepare_async_comm_stream(a1, topk_weights, topk_ids, scale)

    def _prepare_async_ll(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> tuple[Callable, mk.ReceiverType]:
        """AsyncLL path: dispatch_send (CU-free) → yield → dispatch_recv."""
        mori_op = self._current_tbo_mori_op()

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = mori_op.dispatch_send(a1, topk_weights, scale, topk_ids)

        def hook():
            mori_op.dispatch_recv()

        def receiver() -> mk.PrepareResultType:
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=dispatch_recv_token_num,
                expert_num_tokens_cpu=None,
            )
            return (
                dispatch_a1,
                dispatch_scale,
                expert_tokens_meta,
                dispatch_ids,
                dispatch_weights,
            )

        return (hook, receiver)

    def _prepare_async_comm_stream(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> mk.ReceiverType:
        from atom.utils.tbo.ubatching import (
            tbo_switch_to_compute_sync,
            tbo_yield_and_switch_from_compute_to_comm,
        )

        mori_op = self._current_tbo_mori_op()

        block_num, warp_per_block = self._get_launch_config(
            "dispatch", mori_op, a1.shape[0], a1.dtype, a1.shape[1]
        )

        tbo_yield_and_switch_from_compute_to_comm()

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = mori_op.dispatch(
            a1,
            topk_weights,
            scale,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

        tbo_switch_to_compute_sync()

        def receiver() -> mk.PrepareResultType:
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=dispatch_recv_token_num,
                expert_num_tokens_cpu=None,
            )
            return (
                dispatch_a1,
                dispatch_scale,
                expert_tokens_meta,
                dispatch_ids,
                dispatch_weights,
            )

        return receiver

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> Callable:
        num_token = topk_ids.shape[0]
        if self._use_split_send_recv:
            return self._finalize_async_ll(num_token, fused_expert_output, topk_ids)
        return self._finalize_async_comm_stream(
            num_token,
            fused_expert_output,
            topk_ids,
        )

    def _finalize_async_ll(
        self,
        num_token: int,
        fused_expert_output: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[Callable, Callable]:
        """AsyncLL path: combine_send (CU-free) → yield → combine_recv."""
        mori_op = self._current_tbo_mori_op()

        combined_hidden_states = mori_op.combine_send(
            fused_expert_output, None, topk_ids
        )

        def hook():
            mori_op.combine_recv()

        def receiver():
            return combined_hidden_states[0][:num_token]

        return (hook, receiver)

    def _finalize_async_comm_stream(
        self,
        num_token: int,
        fused_expert_output: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Callable:
        from atom.utils.tbo.ubatching import (
            tbo_switch_to_compute_sync,
            tbo_yield_and_switch_from_compute_to_comm,
        )

        mori_op = self._current_tbo_mori_op()

        block_num, warp_per_block = self._get_launch_config(
            "combine",
            mori_op,
            num_token,
            fused_expert_output.dtype,
            fused_expert_output.shape[1],
        )

        # Yield to other thread FIRST, then switch to comm stream.
        tbo_yield_and_switch_from_compute_to_comm()

        result = mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )[0]

        tbo_switch_to_compute_sync()

        def receiver():
            return result[:num_token]

        return receiver
