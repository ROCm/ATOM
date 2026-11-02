# SPDX-License-Identifier: MIT
"""Ulysses SP integration for AITER's MegaMoeTP ag_rs kernel.

`FusedMoE.forward_impl` hands it the row counts it serves, on every MoE layer
built with `allow_sp_mega_moe`. One runtime is shared by the sequential MoE
layers; its collective buffers and launch epochs must survive layer/graph
boundaries.
Large token shards are tiled locally, with the same tiles on every SP rank.
"""

from __future__ import annotations

import logging
import math

import torch

from atom.distributed.group_agreement import group_agree, group_max
from atom.distributed.ulysses_sp import get_sp_world_size
from atom.utils import envs

logger = logging.getLogger("atom")


# AITER's largest big-M tier, in local tokens.
MAX_LOCAL_TOKENS = 8192
# AITER compares int32 launch epochs wrap-safe; reset well before they wrap.
RESET_LAUNCHES = 1 << 30


def local_token_tile(max_num_batched_tokens, world_size):
    """Rows per launch: the largest local shard, within AITER's supported range."""
    return min(math.ceil(max_num_batched_tokens / world_size), MAX_LOCAL_TOKENS)


def _weights(layer):
    return {
        "w1": layer.w13_weight,
        "w1_scale": layer.w13_weight_scale,
        "w2": layer.w2_weight,
        "w2_scale": layer.w2_weight_scale,
    }


def _unsupported_layer(layer):
    """Why MegaMoeTP cannot serve this FusedMoE layer, or None."""
    from aiter import ActivationType, QuantType, dtypes

    from atom.model_ops.moe import Mxfp4MoEMethod

    method = layer.quant_method
    if (
        type(method) is not Mxfp4MoEMethod
        or method.quant_type != QuantType.per_1x32
        or method.quant_dtype != dtypes.fp4x2
        or method.is_guinterleave
        or method.use_triton
        or method.use_triton_decode
        or method.use_triton_ep
        or method.fused_experts is not None
        or method.hidden_pad != 0
        or method.intermediate_pad != 0
    ):
        return "needs MXFP4 FlyDSL experts without Triton MoE, GU interleave or padding"
    if (
        layer.reduce_results
        or layer.expert_layout.uses_dispatch_remap
        or layer.expert_layout.shared_is_routed
    ):
        return "needs unreduced experts without dispatch remapping"
    # AITER counts the last expert apart, as the one shared expert.
    if (
        layer.num_fused_shared_experts != 1
        or layer.apply_router_weight_on_input
        or layer.activation != ActivationType.Swiglu
        or layer.w13_bias is not None
        or layer.w2_bias is not None
        or layer.w13_input_scale is not None
        or layer.w2_input_scale is not None
    ):
        return "needs SwiGLU experts with one fused shared expert and no biases"
    hidden = layer.hidden_size
    inter = layer.intermediate_size_per_partition
    experts = layer.local_num_experts
    for name, shape in (
        ("w13_weight", (experts, 2 * inter, hidden // 2)),
        ("w2_weight", (experts, hidden, inter // 2)),
        ("w13_weight_scale", (experts * 2 * inter, hidden // 32)),
        ("w2_weight_scale", (experts * hidden, inter // 32)),
    ):
        value = getattr(layer, name)
        if (
            tuple(value.shape) != shape
            or value.element_size() != 1
            or not value.is_contiguous()
            or (name.endswith("weight") and not getattr(value, "is_shuffled", False))
        ):
            return f"{name} has an incompatible layout"
    return None


class SPMegaMoE:
    """Shared collective runtime; outputs own their storage across layer calls."""

    def __init__(self, runtime, layer, num_layers, group):
        self.runtime = runtime
        # The layers share one expert format, so one of them decides for all.
        self.layer = layer
        self.num_layers = num_layers
        self.launches = 0
        # The SP CPU group, to agree on timeouts before the collective reset().
        self.group = group
        # Host copy of the runtime's sticky timeout bits, and the event after it.
        self._errors = self._errors_copied = None
        # A copy not read yet; launches enqueued after the last copy.
        self._copy_pending = self._uncopied = False
        self._runs = {}

    def _timed_out(self, bits):
        raise RuntimeError(
            f"SP MegaMoE rank {self.runtime.cfg.rank}: a MegaMoeTP wait timed out "
            f"(error bits {bits:#x}); the output of a recent step is invalid."
        )

    def _raise_on_timeout(self):
        """Fail on a timed-out launch of an earlier step, without a host sync.

        A timeout invalidates that step's output and the KV written from it, and
        its bits are only cleared by the collective reset(), so it is fatal.
        """
        if self._copy_pending:
            if not self._errors_copied.query():
                return
            self._copy_pending = False
            if self._errors.item():
                self._timed_out(self._errors.item())
        if not self._uncopied:
            return
        flag = self.runtime.error_flag()
        if self._errors is None:
            self._errors = torch.empty_like(flag, device="cpu").pin_memory()
            self._errors_copied = torch.cuda.Event()
        self._errors.copy_(flag, non_blocking=True)
        self._errors_copied.record()
        self._copy_pending, self._uncopied = True, False

    def count_step(self, rows):
        """Count a model step's launches up front, so graph replays count too.

        ``rows`` is this rank's share of the step's padded tokens. A step that
        keeps the native MoE launches nothing and adds no copy. Decode graphs
        from 256 global rows up replay MegaMoE launches without running Python.
        """
        self._raise_on_timeout()
        if not self.runs(rows):
            return
        self._uncopied = True
        self.launches += self.num_layers * math.ceil(
            rows / self.runtime.max_local_tokens
        )
        if self.launches >= RESET_LAUNCHES:
            # reset() clears bits the copy above may not have read yet: read
            # them now, on every SP rank, which all run the same steps.
            bits = group_max(self.runtime.poll_errors(), self.group)
            if bits:
                self._timed_out(bits)
            self.runtime.reset()
            self.launches = 0

    def runs(self, rows):
        """Keep the native path whenever MegaMoE would change activation precision.

        Use the padded global row count, exactly as the native SP gather does.
        Small decode batches use BF16 activations, which this MegaMoE kernel
        cannot produce. All ranks make the same shape decision.
        """
        runs = self._runs.get(rows)
        if runs is None:
            from aiter import dtypes

            method = self.layer.quant_method
            tokens = rows * self.runtime.cfg.world_size
            runs = method.activation_dtype(self.layer, tokens) == dtypes.fp4x2
            self._runs[rows] = runs
        return runs

    @staticmethod
    def _route(layer, hidden_states, router_logits):
        method = layer.quant_method
        weights, ids = method.select_experts(layer, hidden_states, router_logits)
        return method.place_experts(layer, weights, ids)

    def forward(self, layer, hidden_states, router_logits):
        """The layer's output, shared expert included, for rows `runs` takes."""
        rows = hidden_states.shape[0]
        weights, ids = self._route(layer, hidden_states, router_logits)
        self.runtime.set_weights(**_weights(layer))
        out = torch.empty_like(hidden_states)
        tile = self.runtime.max_local_tokens
        for start in range(0, rows, tile):
            sl = slice(start, min(start + tile, rows))
            self.runtime.forward(hidden_states[sl], weights[sl], ids[sl], out=out[sl])
        return out


def _unsupported_config(config):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    # Config._init_sequence_parallel refuses the rest.
    if config.eplb_enable or config.fake_eplb:
        return "needs static expert placement, without EPLB"
    if config.torch_dtype != torch.bfloat16:
        return "needs BF16 activations"
    if get_gfx_runtime() != "gfx950":
        return "needs gfx950"
    return None


def _model_layers(model):
    """The MoE layers their model allows SP MegaMoE on.

    Not `isinstance(module, FusedMoE)`: that name is the plugin-mode lazy
    wrapper, a subclass the built layers are not instances of.
    """
    return [
        module
        for module in model.modules()
        if getattr(module, "allow_sp_mega_moe", False)
    ]


# MegaMoeTP setup failures that AITER gathers and raises on every rank.
_AGREED_SETUP_FAILURES = (
    "SdmaComm: MORI SDMA setup failed on ranks",
    "SymmetricArena: mapping failed on ranks",
    "MegaMoeTP.prepare: warmup launches timed out",
)


def _plan(config, layers):
    """`(hidden, inter, experts, top_k, activation, swiglu_limit)` when this rank
    can run every layer through MegaMoeTP, else why not."""
    reason = _unsupported_config(config)
    if reason is not None:
        return reason
    for layer in layers:
        reason = _unsupported_layer(layer)
        if reason is not None:
            return f"{layer.prefix}: {reason}"
    shapes = {
        (
            layer.hidden_size,
            layer.intermediate_size_per_partition,
            layer.local_num_experts,
            layer.top_k,
            layer.activation,
            float(getattr(layer, "swiglu_limit", 0.0)),
        )
        for layer in layers
    }
    if len(shapes) != 1:
        return "the shared runtime needs one expert shape across layers"
    try:
        from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP  # noqa: F401
    except ImportError as exc:
        return f"MegaMoeTP unavailable: {exc}"
    return shapes.pop()


def _probe_mori():
    """Why MORI cannot be imported, or None."""
    try:
        from aiter.ops.communication import _mori

        _mori()
    except ImportError as exc:
        return f"MORI unavailable for the big-M kernel: {exc}"
    return None


def initialize_sp_mega_moe(model, config):
    """After weight processing, before warmup/capture.

    Returns the shared backend, or None when the native MoE is kept. It is
    all or nothing: one layer it cannot serve keeps every layer native.
    """
    from aiter.dist.parallel_state import get_pcp_group

    if not envs.ATOM_SP_MOE_MEGA or get_sp_world_size() < 2:
        return

    def skip(reason, log=logger.warning):
        log("SP MegaMoE disabled, native MoE kept on every layer: %s", reason)

    layers = _model_layers(model)
    if not layers:
        return skip("the model has no MoE layers it can run through it", logger.info)
    group = get_pcp_group()
    plan = _plan(config, layers)
    # A rank-local verdict (an import, a layout) must not split the ranks before
    # the collective MegaMoeTP setup.
    if not group_agree(not isinstance(plan, str), group.cpu_group):
        return skip(plan if isinstance(plan, str) else "another SP rank refused it")
    hidden, inter, experts, top_k, activation, swiglu_limit = plan
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    max_local_rows = local_token_tile(config.max_num_batched_tokens, group.world_size)
    # Prefill needs larger row tiles than AITER's default.
    lb_mt = "6"
    runtime = None
    try:
        runtime = MegaMoeTP(
            MegaMoeTPConfig(
                rank=group.rank_in_group,
                world_size=group.world_size,
                model_dim=hidden,
                inter_dim=inter,
                experts=experts,
                topk=top_k + 1,
                max_local_tokens=max_local_rows,
                activation=activation,
                swiglu_limit=swiglu_limit,
                comm_mode="ag_rs",
                comm_dtype="fp8",
                lb_mt=lb_mt,
                fit_max_local=True,
            ),
            **_weights(layers[0]),
            group=group.cpu_group,
            device=layers[0].w13_weight.device,
        )
        tile = runtime.max_local_tokens
        # MORI's SDMA queues serve the big-M kernel only.
        if 0 < runtime.bigm_m <= tile:
            mori = _probe_mori()
            if not group_agree(mori is None, group.cpu_group):
                runtime.close()
                return skip(mori or "another SP rank cannot import MORI")
        # Prepare each distinct launch variant once. Cover every possible
        # remainder of a local tile so an unseen graph bucket cannot compile
        # during capture.
        representatives = {}
        for rows in range(1, tile + 1):
            representatives.setdefault(runtime.launch_config(rows), rows)
        runtime.prepare(sorted(representatives.values()))
    except RuntimeError as exc:
        # Any other failure (OOM, a compile error) may hit one rank while its
        # peers wait in a collective; falling back there alone desyncs them.
        if not str(exc).startswith(_AGREED_SETUP_FAILURES):
            raise
        if runtime is not None:
            runtime.close()
        return skip(f"MegaMoeTP setup failed on every rank: {exc}")
    # The big-M warmup leaves a timed-out wait in the sticky bits only.
    bits = group_max(runtime.poll_errors(), group.cpu_group)
    if bits:
        runtime.close()
        return skip(f"a MegaMoeTP warmup wait timed out (error bits {bits:#x})")
    backend = SPMegaMoE(runtime, layers[0], len(layers), group.cpu_group)
    for layer in layers:
        layer.sp_mega_moe = backend
    logger.info(
        "SP MegaMoE enabled: %d layers, one shared runtime, tile=%d local tokens, "
        "lb_mt=%s.",
        len(layers),
        tile,
        lb_mt,
    )
    return backend
