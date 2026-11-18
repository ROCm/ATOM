# SPDX-License-Identifier: MIT
"""TP MegaMoE: AITER's MegaMoeTP "ar" kernel as the experts of a TP MoE layer.

`--moe-backend mega` on a pure-TP deployment (no EP, no SP; those run the other
MegaMoE paths) for the models whose MoE layers serve it. Pure TP
replicates the layer input on every rank and shards each expert's intermediate
dim, so the production path is ``fused_moe`` (this rank's partial sum) followed
by an all-reduce that M3's next layer fuses with its add + RMSNorm. MegaMoeTP's
``comm_mode="ar"`` is that whole pipeline in one persistent launch: it routes the
replicated rows, runs GEMM1 + SwiGLU + GEMM2 on this rank's intermediate slice
and all-reduces the output in-kernel.

Contract with the model: the
output of a layer that carries ``tp_mega_moe_requested`` is *already
all-reduced*, on every step: the steps this backend serves get it from the
kernel, the others from the production path plus ``FusedMoE.reduce_results``.
The layer that consumes it therefore must not all-reduce it again.

* One runtime is shared by the sequential MoE layers (``set_weights`` per layer);
  its symmetric buffers and launch epochs must survive layer / graph boundaries.
* The runtime is built and every launch config compiled / armed / warmed
  (``prepare``) after weight processing and before warmup / graph capture, so no
  decode or prefill bucket compiles or allocates inside a captured region.
* ``ar`` takes a multiple of tp rows (small engine): other row counts are padded
  with zero rows routed nowhere through preallocated staging buffers.
* Every TP rank must call it the same number of times with the same rows, input
  and routing (same replicated activations): true for pure TP.
"""

from __future__ import annotations

import logging

import torch

from atom.distributed.group_agreement import group_max
from atom.model_ops.fused_moe.sp_mega_moe import RESET_LAUNCHES, SPMegaMoE, _weights

logger = logging.getLogger("atom")

# AITER's small-batch engine holds this many local tokens on 256 CUs; beyond it
# the big-M path (MORI SDMA, not graph-capturable) would be needed, so the steps
# above are left to the production MoE.
MAX_SMALL_ENGINE_LOCAL_TOKENS = 2048
# Largest step (padded tokens) served; bigger prefill chunks keep the production
# MoE + all-reduce. It sizes the kernel's symmetric buffers.
MAX_ROWS = 4096


def tp_mega_moe_selected(config) -> bool:
    """Whether ``--moe-backend mega`` means TP MegaMoE here: EP runs the MoRI
    MegaMoE and SP the Ulysses one."""
    return (
        config.moe_backend == "mega"
        and not config.enable_expert_parallel
        and config.sequence_parallel_size == 1
    )


def tp_mega_moe_max_rows(config, tp_size: int) -> int:
    """The step size (rows) the backend is asked to serve, before AITER's own
    capacity limit (``fit_max_local``) lowers it."""
    limit = max(min(MAX_ROWS, int(config.max_num_batched_tokens)), tp_size)
    return min(limit, MAX_SMALL_ENGINE_LOCAL_TOKENS * tp_size)


def _unsupported_config(config) -> str | None:
    """Why the deployment cannot use TP MegaMoE, or None."""
    for field, ok, why in (
        ("tensor_parallel_size", config.tensor_parallel_size >= 2, "needs TP >= 2"),
        ("pipeline_parallel_size", config.pipeline_parallel_size == 1, "needs PP 1"),
        (
            "tp_world_size",
            config.tp_world_size == config.tensor_parallel_size,
            "needs one process per TP shard",
        ),
        (
            "data_parallel_size",
            config.parallel_config.data_parallel_size == 1,
            "needs DP 1",
        ),
        (
            "prefill_context_parallel_size",
            config.prefill_context_parallel_size == 1,
            "needs PCP 1",
        ),
        (
            "eplb",
            not (config.eplb_enable or config.fake_eplb),
            "needs static expert placement, without EPLB",
        ),
    ):
        if not ok:
            return f"{field}: {why}"
    return None


def _unsupported_layer(layer) -> str | None:
    """Why MegaMoeTP cannot serve this FusedMoE layer, or None. The expert
    weights are the production FlyDSL ones, read in place."""
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
        not layer.reduce_results
        or layer.expert_layout.uses_dispatch_remap
        or layer.expert_layout.shared_is_routed
    ):
        return "needs the all-reduced contract and no dispatch remapping"
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
            or (name.endswith("weight") and not getattr(value, "is_shuffled", False))
        ):
            return f"{name} has an incompatible layout"
    return None


class TPMegaMoE(SPMegaMoE):
    """The shared TP runtime. It reuses SPMegaMoE's timeout watchdog (async copy
    of the sticky error bits, collective ``reset`` before the launch epochs wrap)
    and ``close``; routing, row counts and the launch differ."""

    def __init__(self, runtime, layer, num_layers, group, max_rows, staging=None):
        super().__init__(runtime, layer, num_layers, group)
        self.max_rows = max_rows
        self.tp = runtime.cfg.world_size
        # (x, topk_weights, topk_ids, y) staging for row counts that are not a
        # multiple of tp, allocated before graph capture.
        self._staging = staging
        # Steps seen by before_step (not graph warmups) and the ones served.
        self.steps = 0
        self.steps_served = 0

    def runs(self, rows):
        """A step of ``rows`` (padded) tokens, the same on every TP rank."""
        return 0 < rows <= self.max_rows

    def close(self):
        logger.info(
            "TP MegaMoE: %d of %d steps ran the kernel, the rest the production MoE.",
            self.steps_served,
            self.steps,
        )
        super().close()

    def _timed_out(self, bits):
        self.failed = True
        raise RuntimeError(
            f"TP MegaMoE rank {self.runtime.cfg.rank}: a MegaMoeTP wait timed out "
            f"(error bits {bits:#x}); the output of a recent step is invalid."
        )

    def before_step(self, rows):
        """Host bookkeeping before a model step, eager or graph replay: raise if
        an earlier step's wait timed out (async copy, no host sync), count this
        step's launches (one per MoE layer) and, every RESET_LAUNCHES, agree over
        the TP group on the error bits and run the collective ``reset``."""
        self._raise_on_timeout()
        self.steps += 1
        if not self.runs(rows):
            return
        self.steps_served += 1
        self._uncopied = True
        self.launches += self.num_layers
        if self.launches >= RESET_LAUNCHES:
            bits = group_max(self.runtime.poll_errors(), self.group)
            if bits:
                self._timed_out(bits)
            self.runtime.reset()
            self.launches = 0

    def forward(self, layer, hidden_states, router_logits):
        """The layer's all-reduced output, shared expert included."""
        rows = hidden_states.shape[0]
        weights, ids = self._route(layer, hidden_states, router_logits)
        self.runtime.set_weights(**_weights(layer))
        out = torch.empty_like(hidden_states)
        padded = -(-rows // self.tp) * self.tp
        if padded == rows:
            self.runtime.forward(hidden_states, weights, ids, out=out)
            return out
        sx, sw, si, sy = self._staging
        sx[:rows].copy_(hidden_states)
        sw[:rows].copy_(weights)
        si[:rows].copy_(ids)
        # The pad rows: zero activations routed to no expert.
        sx[rows:padded].zero_()
        sw[rows:padded].zero_()
        si[rows:padded].fill_(-1)
        self.runtime.forward(sx[:padded], sw[:padded], si[:padded], out=sy[:padded])
        out.copy_(sy[:rows])
        return out


def _model_layers(model):
    """The FusedMoE layers that asked for TP MegaMoE (``tp_mega_moe_requested``),
    which is also where their model switched them to the all-reduced contract."""
    return [m for m in model.modules() if getattr(m, "tp_mega_moe_requested", False)]


def _plan(config, layers):
    """``(hidden, inter, experts, top_k, activation, swiglu_limit)`` when MegaMoeTP
    can run every layer, else why not."""
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
    return shapes.pop()


def _refuse(reason):
    return RuntimeError(f"--moe-backend mega with TP MegaMoE: {reason}")


def build_tp_mega_moe(
    layers, *, rank, tp, cpu_group, max_rows_hint, comm="bf16", act="auto"
):
    """The shared TP MegaMoE backend over ``layers`` (FusedMoE-like, one expert
    shape): builds the AITER runtime on the first layer's weights, compiles / arms
    / warms every launch config a step up to the runtime's capacity can use, and
    allocates the staging buffers. Collective over the TP group. Raises
    ``RuntimeError`` (see ``_refuse``) on a setup error."""
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    first = layers[0]
    hidden = first.hidden_size
    top_k = first.top_k
    device = first.w13_weight.device
    runtime = None
    try:
        runtime = MegaMoeTP(
            MegaMoeTPConfig(
                rank=rank,
                world_size=tp,
                model_dim=hidden,
                inter_dim=first.intermediate_size_per_partition,
                experts=first.local_num_experts,
                topk=top_k + 1,
                max_local_tokens=-(-max_rows_hint // tp),
                activation=first.activation,
                swiglu_limit=float(getattr(first, "swiglu_limit", 0.0)),
                comm_mode="ar",
                comm_dtype=comm,
                ar_gather="bf16",
                act_dtype=act,
                # The small engine alone: the big-M path needs MORI SDMA and is
                # not graph-capturable; bigger steps keep the production MoE.
                bigm_min_tokens=0,
                fit_max_local=True,
            ),
            **_weights(first),
            group=cpu_group,
            device=device,
        )
        # Every local row count a step up to max_rows can map to (rows are
        # padded up to a multiple of tp): compile, arm and run each launch
        # config once, so none compiles during graph capture.
        runtime.prepare(range(1, runtime.max_local_tokens + 1))
    except (ValueError, RuntimeError) as exc:
        # AITER's setup errors are config errors, raised on every rank alike, so
        # the collective close() is safe; a timed-out warmup raised on every rank
        # too (prepare() agrees over the group), but peers may not have arrived.
        if runtime is not None and isinstance(exc, ValueError):
            runtime.close()
        raise _refuse(f"MegaMoeTP setup failed: {exc}") from exc
    bits = group_max(runtime.poll_errors(), cpu_group)
    if bits:
        runtime.close()
        raise _refuse(f"a MegaMoeTP warmup wait timed out (error bits {bits:#x})")

    max_rows = runtime.max_local_tokens * tp
    staging = (
        torch.zeros((max_rows, hidden), dtype=torch.bfloat16, device=device),
        torch.zeros((max_rows, top_k + 1), dtype=torch.float32, device=device),
        torch.full((max_rows, top_k + 1), -1, dtype=torch.int32, device=device),
        torch.zeros((max_rows, hidden), dtype=torch.bfloat16, device=device),
    )
    return TPMegaMoE(runtime, first, len(layers), cpu_group, max_rows, staging)


def initialize_tp_mega_moe(model, config):
    """After weight processing, before warmup / capture (next to
    ``initialize_sp_mega_moe``). Returns the shared backend, or None unless
    ``--moe-backend mega`` runs TP MegaMoE. Raises on every rank when a layer or
    setting is outside what it serves: the choice is explicit, so there is no
    silent fallback to a layout whose outputs would no longer be all-reduced."""
    if not tp_mega_moe_selected(config):
        return None
    from aiter.dist.parallel_state import get_tp_group

    reason = _unsupported_config(config)
    if reason is not None:
        raise _refuse(reason)
    layers = _model_layers(model)
    if not layers:
        raise _refuse(
            "no MoE layer of this model serves it; use --enable-expert-parallel "
            "or --sequence-parallel-size with --moe-backend mega"
        )
    plan = _plan(config, layers)
    if isinstance(plan, str):
        raise _refuse(plan)
    group = get_tp_group()
    backend = build_tp_mega_moe(
        layers,
        rank=group.rank_in_group,
        tp=group.world_size,
        cpu_group=group.cpu_group,
        max_rows_hint=tp_mega_moe_max_rows(config, group.world_size),
    )
    for layer in layers:
        layer.tp_mega_moe = backend
    logger.info(
        "TP MegaMoE enabled: %d layers, one shared runtime, up to %d rows/step; "
        "larger steps keep fused_moe + all-reduce.",
        len(layers),
        backend.max_rows,
    )
    return backend


__all__ = [
    "TPMegaMoE",
    "initialize_tp_mega_moe",
    "tp_mega_moe_max_rows",
    "tp_mega_moe_selected",
]
