# SPDX-License-Identifier: MIT
"""Ulysses SP integration for AITER's MegaMoeTP ag_rs kernel.

Enabled by `--moe-backend mega` with `--sequence-parallel-size` > 1. On pure TP
(no EP, no SP) it runs the same kernel in its `ar` mode (`TPMegaMoE`), with
the MXFP8 shared expert computed inside the kernel (AITER's ``w1_shared``).
`FusedMoE.forward_impl` hands it the row counts it serves, on every MoE layer.
One runtime is shared by the sequential MoE layers; its collective buffers and
launch epochs must survive layer/graph boundaries.
Large token shards are tiled locally, with the same tiles on every SP rank.
"""

from __future__ import annotations

import importlib
import logging
import math

import torch

from atom.distributed.group_agreement import group_max

logger = logging.getLogger("atom")


# AITER's largest big-M tier, in local tokens.
MAX_LOCAL_TOKENS = 8192
# AITER compares int32 launch epochs wrap-safe; reset well before they wrap.
RESET_LAUNCHES = 1 << 30
# Pure TP: the steps (rows) the small-batch engine serves, measured per layer
# against the standard MoE on MiniMax-M3.1 TP4 (routed + shared expert +
# all-reduce + the next norm, real layer-3 weights, inputs and routing; the
# table is in ROCm/ATOM#3512): its small-M tier (a16w4, < 256 rows) and LB
# tier are faster up to 6144 rows (6144: parity), except at 897..1024 rows,
# where AITER's tuned `ar_sh8` 1024-row config is ~8% slower (the 1024-row
# graph bucket). From 6145 rows up to AITER's big-M threshold (8192 on
# M3.1 TP4) the LB tier is slower (7168: +5%), so the standard MoE keeps those.
TP_SMALL_MAX_ROWS = 6144
TP_STANDARD_ROWS = (897, 1024)
SP_CONFIG = {"comm_mode": "ag_rs", "comm_dtype": "fp8"}
# bf16 collectives (lossless, as the standard MoE's all-reduce), a16w4 below
# 256 rows; the big-M path from AITER's smallest tuned big-M batch.
TP_CONFIG = {
    "comm_mode": "ar",
    "comm_dtype": "bf16",
    "ar_gather": "bf16",
    "act_dtype": "auto",
}


def local_token_tile(max_num_batched_tokens, world_size):
    """Rows per launch: the largest local shard, within AITER's supported range."""
    return min(math.ceil(max_num_batched_tokens / world_size), MAX_LOCAL_TOKENS)


def _weights(layer):
    weights = {
        "w1": layer.w13_weight,
        "w1_scale": layer.w13_weight_scale,
        "w2": layer.w2_weight,
        "w2_scale": layer.w2_weight_scale,
    }
    # TP: the MXFP8 shared expert, in the layout of AITER's kernel (the model
    # sets it, see `tp_shared_expert_weights`).
    shared = getattr(layer, "tp_mega_shared", None)
    if shared is not None:
        weights.update(shared)
    return weights


def tp_shared_expert_weights(w1, w1_scale, w2, w2_scale):
    """A TP MegaMoE layer's MXFP8 shared expert (this rank's shard, row-major
    as loaded: ``w1`` [2 I, H] gate rows then up rows, ``w2`` [H, I], e8m0
    scales per 32) in the layout AITER's MegaMoeTP takes as ``w1_shared`` ...
    ``w2_shared_scale``."""
    from aiter.ops.flydsl.mega_moe_tp import shuffle_weight_mega_fp8
    from aiter.utility.fp4_utils import e8m0_shuffle

    for name, weight, scale in (("w1", w1, w1_scale), ("w2", w2, w2_scale)):
        if (
            weight.dim() != 2
            or weight.dtype != torch.float8_e4m3fn
            or scale.element_size() != 1
            or tuple(scale.shape) != (weight.shape[0], weight.shape[1] // 32)
        ):
            raise ValueError(
                f"TP MegaMoE: the shared expert's {name} is not a row-major MXFP8 "
                f"weight ({tuple(weight.shape)} {weight.dtype}, scale "
                f"{tuple(scale.shape)} {scale.dtype})"
            )
    e8m0 = torch.float8_e8m0fnu
    return {
        "w1_shared": shuffle_weight_mega_fp8(w1),
        "w1_shared_scale": e8m0_shuffle(w1_scale.view(e8m0)),
        "w2_shared": shuffle_weight_mega_fp8(w2),
        "w2_shared_scale": e8m0_shuffle(w2_scale.view(e8m0)),
    }


def _unsupported_layer(layer, tp=False):
    """Why MegaMoeTP cannot serve this FusedMoE layer, or None. ``tp``: TP
    MegaMoE, whose shared expert is the model's separate MXFP8 one
    (``tp_mega_shared``), not one fused into the MXFP4 experts."""
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
    # TP: a step the kernel does not take leaves the output TP-partial for the
    # next fused all-reduce + norm, like the standard MoE.
    if (
        layer.reduce_results
        or layer.expert_layout.uses_dispatch_remap
        or layer.expert_layout.shared_is_routed
    ):
        return "needs unreduced experts without dispatch remapping"
    # AITER counts the last expert apart, as the one shared expert.
    if tp:
        if layer.num_fused_shared_experts != 0:
            return (
                "needs the MXFP8 shared expert kept apart, not fused into the "
                "MXFP4 experts (unset ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4)"
            )
        if getattr(layer, "tp_mega_shared", None) is None:
            return "needs the model's MXFP8 shared expert (tp_mega_shared)"
        # (TPMegaMoE._routes runs this routing itself)
        if (
            layer.scoring_func != "sigmoid"
            or layer.use_grouped_topk
            or layer.custom_routing_function is not None
        ):
            return "needs sigmoid top-k routing without groups or a custom function"
    elif layer.num_fused_shared_experts != 1:
        return "needs SwiGLU experts with one fused shared expert and no biases"
    if (
        layer.apply_router_weight_on_input
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


class SPMegaMoE:
    """Shared collective runtime; outputs own their storage across layer calls."""

    name = "SP MegaMoE"
    # FusedMoE.forward_impl dispatches the steps `runs` takes.
    serves_block = False

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
        self.failed = False

    def _timed_out(self, bits):
        self.failed = True
        raise RuntimeError(
            f"{self.name} rank {self.runtime.cfg.rank}: a MegaMoeTP wait timed out "
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

    def before_step(self, rows):
        """Host bookkeeping before a model step, eager or graph replay.

        Raises if an earlier step's wait timed out (read through an async
        GPU->CPU copy, no host sync), then counts this step's launches. Every
        RESET_LAUNCHES launches it syncs the host on the error bits, agreed
        over the SP group, and runs the collective reset(). ``rows`` is this
        rank's share of the step's padded tokens; a step that keeps the native
        MoE launches nothing. Capture warmups call the model directly and go
        uncounted, which RESET_LAUNCHES' margin absorbs.
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

    def close(self):
        """Collective over the SP group: every rank calls it before the group
        is destroyed. Skipped after a timeout, which peers may not have seen:
        its CPU barrier would block this rank's exit instead."""
        if not self.failed:
            self.runtime.close()

    def runs(self, rows):
        """Whether MegaMoE takes a step: only where the native MoE quantizes
        activations to FP4 too. Small decode batches keep BF16 activations,
        which this MegaMoE kernel cannot produce.

        ``rows`` is this rank's share of the padded tokens; the decision is
        on the global count, as the native SP gather sees it, so every rank
        makes the same one.
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


class TPMegaMoE(SPMegaMoE):
    """Pure TP: every rank holds the same rows. On the steps it takes, one
    launch a layer runs the routed experts, the MXFP8 shared expert and the
    all-reduce (the kernel's `ar` mode), so the layer's output is all-reduced;
    the model asks `runs(rows)` per step and leaves every other step's output
    TP-partial for its next fused all-reduce + norm, as the standard MoE does.
    """

    name = "TP MegaMoE"
    # The model's MoE block calls `forward` (it covers the shared expert), not
    # FusedMoE.forward_impl.
    serves_block = True
    # Rows per tier for the coverage log: small-M (a16w4), LB, big-M.
    _TIERS = ((256, "small-M"), (TP_SMALL_MAX_ROWS, "LB"))

    def __init__(self, runtime, layer, num_layers, group):
        super().__init__(runtime, layer, num_layers, group)
        tp = runtime.cfg.world_size
        self.max_rows = runtime.max_local_tokens * tp
        # Global rows from which AITER runs its big-M kernel (0: never, also
        # when the step budget stays below it).
        bigm_rows = runtime.bigm_m * tp if runtime.bigm_m else 0
        self.bigm_rows = bigm_rows if bigm_rows <= self.max_rows else 0
        # Coverage: steps / rows per tier, and the steps that keep the
        # standard MoE.
        self.coverage: dict[str, list[int]] = {}
        self._steps = 0
        # The routing with the shared expert appended (last expert, weight 1),
        # per row count: written in place so graph replays keep their inputs.
        self._shared_routes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _big(self, rows):
        return bool(self.bigm_rows) and self.bigm_rows <= rows <= self.max_rows

    def runs(self, rows):
        """Whether the kernel takes a `rows`-row step: the small-batch engine
        up to TP_SMALL_MAX_ROWS (but TP_STANDARD_ROWS), the big-M path from
        AITER's threshold. Big-M launches are not graph-capturable, so a step
        recorded into a CUDA graph keeps the standard MoE there (consistently
        for the whole step: the decision is made per row count while
        recording)."""
        low, high = TP_STANDARD_ROWS
        if 0 < rows <= min(TP_SMALL_MAX_ROWS, self.max_rows):
            return not low <= rows <= high
        return self._big(rows) and not torch.cuda.is_current_stream_capturing()

    def tier(self, rows):
        if not self.runs(rows):
            return "standard"
        if self._big(rows):
            return "big-M"
        return next(name for top, name in self._TIERS if rows <= top)

    def before_step(self, rows):
        if rows > 0:
            entry = self.coverage.setdefault(self.tier(rows), [0, 0])
            entry[0] += 1
            entry[1] += rows
            self._steps += 1
            if self._steps % 5000 == 0 and self.runtime.cfg.rank == 0:
                logger.info("%s coverage: %s", self.name, self.coverage_summary())
        # One launch a layer, none on a step that keeps the standard MoE.
        super().before_step(1 if self.runs(rows) else 0)

    def coverage_summary(self):
        """Per tier: steps / rows (the host counts every step, replays
        included; a mono decode step, which bypasses the MoE layers, counts
        under the tier of its row count)."""
        return "; ".join(
            f"{tier} {steps} steps / {rows} rows"
            for tier, (steps, rows) in sorted(self.coverage.items())
        )

    def _routes(self, layer, router_logits):
        """The routing as AITER takes it: the top-k routed experts, weights
        scaled by the routed scaling factor (the standard MoE scales their sum
        instead), then the shared expert, AITER's last expert, at weight 1.
        One routing launch, as the standard MoE's: `topk_gating` writes the
        routed columns of a buffer kept per row count, whose shared column is
        filled once (a graph replay reads the same buffer)."""
        from aiter import topk_gating

        from atom.utils.forward_context import pass_scheduled_tokens_gpu

        rows, top_k = router_logits.shape[0], layer.top_k
        buffers = self._shared_routes.get(rows)
        if buffers is None:
            device = router_logits.device
            buffers = self._shared_routes[rows] = (
                torch.ones((rows, top_k + 1), dtype=torch.float32, device=device),
                torch.full(
                    (rows, top_k + 1),
                    layer.local_num_experts,
                    dtype=torch.int32,
                    device=device,
                ),
            )
        weights, ids = buffers
        # (as FusedMoE.select_experts: rows past the step's real count -> -1)
        topk_gating(
            weights[:, :top_k],
            ids[:, :top_k],
            router_logits,
            layer.e_score_correction_bias,
            layer.renormalize,
            layer.routed_scaling_factor,
            score_func=layer.scoring_func,
            num_valid_rows=pass_scheduled_tokens_gpu(rows),
        )
        return weights, ids

    def forward(self, layer, hidden_states, router_logits):
        """The MoE block's all-reduced output (routed + shared expert)."""
        weights, ids = self._routes(layer, router_logits)
        self.runtime.set_weights(**_weights(layer))
        # `ar` pads a row count that is not a multiple of tp itself.
        out = torch.empty_like(hidden_states)
        self.runtime.forward(hidden_states, weights, ids, out=out)
        return out

    def close(self):
        if self.coverage and self.runtime.cfg.rank == 0:
            logger.info("%s coverage: %s", self.name, self.coverage_summary())
        super().close()


def tp_mega_moe_selected(config):
    """`--moe-backend mega` on pure TP; EP runs the MoRI MegaMoE."""
    return (
        config.moe_backend == "mega"
        and config.sequence_parallel_size == 1
        and not config.enable_expert_parallel
    )


def tp_local_token_tile(max_num_batched_tokens, world_size):
    """Local rows of the largest TP step (AITER's big-M path covers up to
    MAX_LOCAL_TOKENS a rank)."""
    rows = max(max_num_batched_tokens, world_size)
    return min(math.ceil(rows / world_size), MAX_LOCAL_TOKENS)


def _unsupported_config(config):
    # Config._init_sequence_parallel and _init_moe_backend refuse the rest.
    if config.eplb_enable or config.fake_eplb:
        return "needs static expert placement, without EPLB"
    return None


def _model_layers(model):
    """The FusedMoE layers, by the slot each one has for the backend.

    Not `isinstance(module, FusedMoE)`: that name is the plugin-mode lazy
    wrapper, a subclass the built layers are not instances of.
    """
    return [module for module in model.modules() if hasattr(module, "sp_mega_moe")]


def _plan(config, layers, tp=False):
    """`(hidden, inter, experts, top_k, activation, swiglu_limit)` when MegaMoeTP
    can run every layer, else why not."""
    reason = _unsupported_config(config)
    if reason is not None:
        return reason
    for layer in layers:
        reason = _unsupported_layer(layer, tp)
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


def _refuse(reason, mode="Ulysses SP"):
    return RuntimeError(f"--moe-backend mega with {mode}: {reason}")


def initialize_sp_mega_moe(model, config):
    """After weight processing, before warmup/capture.

    Returns the shared backend, or None without `--moe-backend mega` under SP or
    on pure TP. Raises on every rank when any layer or setting is outside what
    it serves.
    """
    from aiter.dist.parallel_state import get_pcp_group, get_tp_group

    tp = tp_mega_moe_selected(config)
    if config.moe_backend != "mega" or not (tp or config.sequence_parallel_size > 1):
        return
    mode = "TP" if tp else "Ulysses SP"
    layers = _model_layers(model)
    if not layers:
        raise _refuse("the model has no MoE layers", mode)
    group = get_tp_group() if tp else get_pcp_group()
    plan = _plan(config, layers, tp)
    if isinstance(plan, str):
        raise _refuse(plan, mode)
    hidden, inter, experts, top_k, activation, swiglu_limit = plan
    mega_moe_tp = importlib.import_module("aiter.ops.flydsl.mega_moe_tp")
    MegaMoeTP, MegaMoeTPConfig = mega_moe_tp.MegaMoeTP, mega_moe_tp.MegaMoeTPConfig
    # (an AITER without MegaMoeSetupError: nothing to catch then)
    setup_error = getattr(mega_moe_tp, "MegaMoeSetupError", ())
    # TP: the separate MXFP8 shared expert is one more expert, and one more
    # route per token, in the kernel.
    if tp:
        experts += 1

    tile = tp_local_token_tile if tp else local_token_tile
    max_local_rows = tile(config.max_num_batched_tokens, group.world_size)
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
                fit_max_local=True,
                **(TP_CONFIG if tp else SP_CONFIG),
            ),
            **_weights(layers[0]),
            group=group.cpu_group,
            device=layers[0].w13_weight.device,
        )
        # Every local row count of the small-batch engine, so no graph bucket
        # compiles during capture; it also sets up and warms the big-M path.
        if tp:
            small = min(TP_SMALL_MAX_ROWS, max_local_rows * group.world_size)
            runtime.prepare(range(1, math.ceil(small / group.world_size) + 1))
        else:
            runtime.prepare_all()
    except setup_error as exc:
        # AITER raises it on every rank alike, so the collective close() is safe.
        if runtime is not None:
            runtime.close()
        raise _refuse(f"MegaMoeTP setup failed: {exc}", mode) from exc
    # The big-M warmup leaves a timed-out wait in the sticky bits only.
    bits = group_max(runtime.poll_errors(), group.cpu_group)
    if bits:
        runtime.close()
        raise _refuse(f"a MegaMoeTP warmup wait timed out (error bits {bits:#x})", mode)
    backend = (TPMegaMoE if tp else SPMegaMoE)(
        runtime, layers[0], len(layers), group.cpu_group
    )
    for layer in layers:
        layer.sp_mega_moe = backend
    logger.info(
        "%s enabled: %d layers, one shared runtime, tile=%d local tokens.",
        backend.name,
        len(layers),
        runtime.max_local_tokens,
    )
    return backend
