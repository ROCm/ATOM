# SPDX-License-Identifier: MIT
"""Ulysses SP integration for AITER's MegaMoeTP ag_rs kernel.

The MoE tail also fuses the next input norm; the residual add, norm and router
before the MoE run the native kernels. One runtime is shared by the sequential
MoE layers; its collective buffers and launch epochs must survive layer/graph
boundaries.
Large token shards are tiled locally, with the same tiles on every SP rank.
"""

from __future__ import annotations

import logging
import math

import torch

from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")


def copy_tail(source, source_scale, out, out_scale):
    # Keep dispatch and settings importable without the GPU kernel dependencies.
    from atom.model_ops.sp_mega_buffers import copy_tail as copy

    copy(source, source_scale, out, out_scale)


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
        layer.use_ep
        or layer.reduce_results
        or layer.moe_parallel_config.use_all2all_kernels
        or layer.moe_parallel_config.dp_logical_ratio != 1
        or layer.expert_layout.uses_dispatch_remap
        or layer.expert_layout.shared_is_routed
    ):
        return "needs unreduced experts without EP, all2all or dispatch remapping"
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

    def __init__(self, runtime, launches_per_step):
        self.runtime = runtime
        self.launches_per_step = launches_per_step
        self.launches = 0
        # Host copy of the runtime's sticky timeout bits, and the event after it.
        self._errors = self._errors_copied = None

    def _raise_on_timeout(self):
        """Fail on a timed-out launch of an earlier step, without a host sync.

        A timeout invalidates that step's output and the KV written from it, and
        its bits are only cleared by the collective reset(), so it is fatal.
        """
        if self._errors_copied is not None:
            if not self._errors_copied.query():
                return
            if self._errors.item():
                raise RuntimeError(
                    f"SP MegaMoE rank {self.runtime.cfg.rank}: a MegaMoeTP wait "
                    f"timed out (error bits {self._errors.item():#x}); the output "
                    "of a recent step is invalid."
                )
        flag = self.runtime.error_flag()
        if self._errors is None:
            self._errors = torch.empty_like(flag, device="cpu").pin_memory()
            self._errors_copied = torch.cuda.Event()
        self._errors.copy_(flag, non_blocking=True)
        self._errors_copied.record()

    def count_step(self):
        """Count a model step's launches up front, so graph replays count too."""
        # Before the reset below, which would clear a pending timeout.
        self._raise_on_timeout()
        self.launches += self.launches_per_step
        if self.launches >= RESET_LAUNCHES:
            # Collective; every SP rank runs the same steps.
            self.runtime.reset()
            self.launches = 0

    def supports(self, layer, rows):
        """Keep the native path whenever MegaMoE would change activation precision.

        Use the padded global row count, exactly as the native SP gather does.
        Decode typically uses BF16 activations, which this MegaMoE kernel
        cannot produce. All ranks make the same shape decision.
        """
        from aiter import dtypes
        from aiter.fused_moe import resolve_activation_dtype
        from aiter.ops.flydsl.moe_common import GateMode

        native_dtype = resolve_activation_dtype(
            layer.quant_method.quant_type,
            layer.w13_weight.dtype,
            activation=layer.activation,
            gate_mode=GateMode.SEPARATED,
            M=rows * self.runtime.cfg.world_size,
            hidden_dtype=torch.bfloat16,
            gfx="gfx950",
        )
        return native_dtype == dtypes.fp4x2

    @staticmethod
    def _route(layer, hidden_states, router_logits):
        method = layer.quant_method
        weights, ids = method.select_experts(layer, hidden_states, router_logits)
        return method.place_experts(layer, weights, ids)

    def forward_block(
        self,
        layer,
        attention,
        residual,
        post_weight,
        gate_weight,
        next_weight,
        next_eps,
        fp8_output,
    ):
        """Complete SP-local FFN region, returning the NEXT normalized input.

        Attention is already reduced and token-sharded; the MoE's intermediate
        shards communicate over SP4. Tail rows are replicated scratch in AITER:
        copy only this rank's rows before another tile/layer reuses them. No
        returned tensor aliases an input.
        """
        from aiter import dtypes

        rows = attention.shape[0]
        hidden, _, post_res = _norm_input(
            attention, residual, post_weight, self.runtime.cfg.tail_eps, False
        )
        logits = torch.nn.functional.linear(hidden, gate_weight)
        if not self.supports(layer, rows):
            # Small batches (decode) keep the native A16W4 MoE.
            output = layer.forward_impl(hidden, logits)
            return _norm_input(output, post_res, next_weight, next_eps, fp8_output)
        weights, ids = self._route(layer, hidden, logits)

        tile = self.runtime.max_local_tokens
        self.runtime.set_weights(**_weights(layer))
        normalized = torch.empty(
            hidden.shape,
            dtype=dtypes.fp8 if fp8_output else hidden.dtype,
            device=hidden.device,
        )
        scale = torch.empty((rows, 1), dtype=torch.float32, device=hidden.device)
        next_res = torch.empty_like(post_res)
        for start in range(0, rows, tile):
            end = min(start + tile, rows)
            sl = slice(start, end)
            m = end - start
            if self.runtime.tail_ok(m) and next_eps == self.runtime.cfg.tail_eps:
                result = self.runtime.forward(
                    hidden[sl],
                    weights[sl],
                    ids[sl],
                    tail=(post_res[sl], next_res[sl], next_weight),
                    bf16=not fp8_output,
                )
                own = slice(self.runtime.cfg.rank * m, (self.runtime.cfg.rank + 1) * m)
                copy_tail(
                    result[1 if fp8_output else 3][own],
                    result[2][own] if fp8_output else None,
                    normalized[sl],
                    scale[sl],
                )
            else:
                # Dynamic schedule / unsupported tail eps: same output
                # contract, with the ordinary fused residual+norm kernel.
                y = self.runtime.forward(hidden[sl], weights[sl], ids[sl])
                x, s, r = _norm_input(
                    y, post_res[sl], next_weight, next_eps, fp8_output
                )
                normalized[sl].copy_(x)
                next_res[sl].copy_(r)
                if fp8_output:
                    scale[sl].copy_(s)
        return normalized, scale, next_res


def _norm_input(x, residual, weight, eps, fp8):
    if fp8:
        from atom.model_ops.gemma_norm_fp8 import gemma_rms_norm_fp8

        return gemma_rms_norm_fp8(x, weight, eps, residual)
    # GemmaRMSNorm's own kernel, so these tiles match the native next layer.
    from atom.model_ops.layernorm import gemma_rms_norm_unquantized

    out, res_out = gemma_rms_norm_unquantized(x, weight, eps, residual)
    return (
        out,
        torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
        res_out,
    )


def _block_fake(
    attention: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    next_weight: torch.Tensor,
    layer_name: str,
    next_eps: float,
    fp8_output: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter import dtypes

    return (
        torch.empty_like(
            attention, dtype=dtypes.fp8 if fp8_output else attention.dtype
        ),
        torch.empty(
            (attention.shape[0], 1), device=attention.device, dtype=torch.float32
        ),
        torch.empty_like(residual),
    )


def sp_mega_moe_block(
    attention: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    next_weight: torch.Tensor,
    layer_name: str,
    next_eps: float,
    fp8_output: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from atom.config import get_current_atom_config

    layer = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return layer.sp_mega_moe.forward_block(
        layer,
        attention,
        residual,
        post_weight,
        gate_weight,
        next_weight,
        next_eps,
        fp8_output,
    )


# Keep the whole FFN region in one capturable opaque op. Dynamic row decisions
# and shared runtime state must not escape into torch.compile's traced graph.
direct_register_custom_op(
    op_name="sp_mega_moe_block",
    op_func=sp_mega_moe_block,
    mutates_args=[],
    fake_impl=_block_fake,
)


def _unsupported_config(config):
    from aiter.dist.parallel_state import get_dp_group, get_tp_group
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.plugin.prepare import is_plugin_mode

    if (
        get_tp_group().world_size != 1
        or get_dp_group().world_size != 1
        or config.pipeline_parallel_size != 1
        or config.moe_backend != "standard"
        or config.enable_tbo
        or config.enable_tbo_decode
        or config.enable_rapidserve
        or config.enable_expert_parallel
        or config.eplb_enable
        or config.fake_eplb
        or config.speculative_config is not None
        or config.torch_dtype != torch.bfloat16
        or is_plugin_mode()
    ):
        return (
            "needs native BF16 SP with TP1/DP1/PP1, without TBO, RapidServe, "
            "speculative decoding, EP or EPLB"
        )
    if get_gfx_runtime() != "gfx950":
        return "needs gfx950"
    return None


def _model_layers(model, config):
    """The registered `(layers, norm_eps)` of the model, or None.

    The model runs each layer's FFN region through `sp_mega_moe_block`: the
    post-attention GemmaRMSNorm, the router, the experts, and the next layer's
    input GemmaRMSNorm, all with `norm_eps`.
    """
    from atom.models.registry import sp_mega_moe_layers
    from atom.utils import resolve_obj_by_qualname

    find = sp_mega_moe_layers.get(config.hf_config.architectures[0])
    return None if find is None else resolve_obj_by_qualname(find)(model)


# MegaMoeTP setup failures that AITER gathers and raises on every rank.
_AGREED_SETUP_FAILURES = (
    "SdmaComm: MORI SDMA setup failed on ranks",
    "SymmetricArena: mapping failed on ranks",
    "MegaMoeTP.prepare: warmup launches timed out",
)


def initialize_sp_mega_moe(model, config):
    """After weight processing, before warmup/capture.

    Returns the shared backend, or None when the native MoE is kept.
    """
    from atom.distributed.ulysses_sp import get_sp_group, get_sp_world_size

    if config is None or not config.sp_mega_moe_enabled or get_sp_world_size() < 2:
        return

    def skip(reason):
        # The compile hash reads this flag; native graphs must not share it.
        config.sp_mega_moe_enabled = False
        logger.info("SP MegaMoE disabled, native MoE kept: %s", reason)

    reason = _unsupported_config(config)
    if reason is not None:
        return skip(reason)
    found = _model_layers(model, config)
    if not found or not found[0]:
        return skip("the model has no MoE layers it can run through it")
    layers, tail_eps = found
    for layer in layers:
        reason = _unsupported_layer(layer)
        if reason is not None:
            return skip(f"{layer.prefix}: {reason}")
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
        return skip("the shared runtime needs one expert shape across layers")
    hidden, inter, experts, top_k, activation, swiglu_limit = shapes.pop()
    from atom.model_ops.gemma_norm_fp8 import gemma_rms_norm_fp8_supported

    # Rows the fused tail does not cover are normalized with the same rule.
    if not gemma_rms_norm_fp8_supported(hidden):
        return skip(f"no fused Gemma RMSNorm FP8 kernel for hidden size {hidden}")
    try:
        from aiter.ops.communication import _mori
        from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

        # Probe MORI before any collective, so every rank falls back together.
        _mori()
    except ImportError as exc:
        return skip(f"MegaMoeTP unavailable: {exc}")
    group = get_sp_group()
    local_rows = math.ceil(config.max_num_batched_tokens / group.world_size)
    max_local_rows = local_token_tile(config.max_num_batched_tokens, group.world_size)
    # Prefill needs larger row tiles than AITER's default.
    lb_mt = "6"
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
                tail_eps=tail_eps,
                fit_max_local=True,
            ),
            **_weights(layers[0]),
            group=group.cpu_group,
            device=layers[0].w13_weight.device,
        )
        tile = runtime.max_local_tokens
        # Prepare each distinct launch variant once. Cover every possible
        # remainder of a local tile so an unseen graph bucket cannot compile
        # during capture.
        representatives = {}
        for rows in range(1, tile + 1):
            # Tail eligibility can change inside a launch bucket (TN_MAX).
            key = (runtime.launch_config(rows), runtime.tail_ok(rows))
            representatives.setdefault(key, rows)
        runtime.prepare(sorted(representatives.values()), tail=True, tail_bf16=True)
    except RuntimeError as exc:
        # Any other failure (OOM, a compile error) may hit one rank while its
        # peers wait in a collective; falling back there alone desyncs them.
        if not str(exc).startswith(_AGREED_SETUP_FAILURES):
            raise
        return skip(f"MegaMoeTP setup failed on every rank: {exc}")
    backend = SPMegaMoE(runtime, len(layers) * math.ceil(local_rows / tile))
    for layer in layers:
        layer.sp_mega_moe = backend
    logger.info(
        "SP MegaMoE enabled: %d layers, one shared runtime, tile=%d local tokens, "
        "lb_mt=%s. Each tail supplies the next GemmaNorm/FP8 (final layer BF16).",
        len(layers),
        tile,
        lb_mt,
    )
    return backend
