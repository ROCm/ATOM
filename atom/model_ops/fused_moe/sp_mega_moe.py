# SPDX-License-Identifier: MIT
"""Ulysses SP integration for AITER's MegaMoeTP ag_rs kernel.

Enabled by `--moe-backend mega` with `--sequence-parallel-size` > 1.
`FusedMoE.forward_impl` hands it the row counts it serves, on every MoE layer.
One runtime is shared by the sequential MoE layers; its collective buffers and
launch epochs must survive layer/graph boundaries.
Large token shards are tiled locally, with the same tiles on every SP rank.
"""

from __future__ import annotations

import logging
import math

import torch

from atom.distributed.group_agreement import group_max

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
        self.failed = False

    def _timed_out(self, bits):
        self.failed = True
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


def _unsupported_config(config):
    # Config._init_sequence_parallel refuses the rest.
    if config.eplb_enable or config.fake_eplb:
        return "needs static expert placement, without EPLB"
    return None


def _model_layers(model):
    """The FusedMoE layers, by the slot each one has for the backend.

    Not `isinstance(module, FusedMoE)`: that name is the plugin-mode lazy
    wrapper, a subclass the built layers are not instances of.
    """
    return [module for module in model.modules() if hasattr(module, "sp_mega_moe")]


def _plan(config, layers):
    """`(hidden, inter, experts, top_k, activation, swiglu_limit)` when MegaMoeTP
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
    return RuntimeError(f"--moe-backend mega with Ulysses SP: {reason}")


def initialize_sp_mega_moe(model, config):
    """After weight processing, before warmup/capture.

    Returns the shared backend, or None without `--moe-backend mega` under SP.
    Raises on every rank when any layer or setting is outside what it serves.
    """
    from aiter.dist.parallel_state import get_pcp_group

    if config.moe_backend != "mega" or config.sequence_parallel_size < 2:
        return
    layers = _model_layers(model)
    if not layers:
        raise _refuse("the model has no MoE layers")
    group = get_pcp_group()
    plan = _plan(config, layers)
    if isinstance(plan, str):
        raise _refuse(plan)
    hidden, inter, experts, top_k, activation, swiglu_limit = plan
    from aiter.ops.flydsl.mega_moe_tp import (
        MegaMoeSetupError,
        MegaMoeTP,
        MegaMoeTPConfig,
    )

    max_local_rows = local_token_tile(config.max_num_batched_tokens, group.world_size)
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
                fit_max_local=True,
            ),
            **_weights(layers[0]),
            group=group.cpu_group,
            device=layers[0].w13_weight.device,
        )
        # Every remainder of a local tile, so no graph bucket compiles during
        # capture.
        runtime.prepare_all()
    except MegaMoeSetupError as exc:
        # AITER raises it on every rank alike, so the collective close() is safe.
        if runtime is not None:
            runtime.close()
        raise _refuse(f"MegaMoeTP setup failed: {exc}") from exc
    # The big-M warmup leaves a timed-out wait in the sticky bits only.
    bits = group_max(runtime.poll_errors(), group.cpu_group)
    if bits:
        runtime.close()
        raise _refuse(f"a MegaMoeTP warmup wait timed out (error bits {bits:#x})")
    backend = SPMegaMoE(runtime, layers[0], len(layers), group.cpu_group)
    for layer in layers:
        layer.sp_mega_moe = backend
    logger.info(
        "SP MegaMoE enabled: %d layers, one shared runtime, tile=%d local tokens.",
        len(layers),
        runtime.max_local_tokens,
    )
    return backend
