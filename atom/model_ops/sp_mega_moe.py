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

from atom.distributed.group_agreement import group_agree, group_max
from atom.utils import envs
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


def _router_logits(hidden, gate_weight):
    # The native gate's GEMM (ReplicatedLinear), so both paths pick the same top-k.
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(hidden, gate_weight, None, otype=hidden.dtype)


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
        keeps the native MoE (decode) launches nothing and adds no copy.
        """
        self._raise_on_timeout()
        if not self.supports(rows):
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

    def supports(self, rows):
        """Keep the native path whenever MegaMoE would change activation precision.

        Use the padded global row count, exactly as the native SP gather does.
        Decode typically uses BF16 activations, which this MegaMoE kernel
        cannot produce. All ranks make the same shape decision.
        """
        from aiter import dtypes

        method = self.layer.quant_method
        tokens = rows * self.runtime.cfg.world_size
        return method.activation_dtype(self.layer, tokens) == dtypes.fp4x2

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
        post_eps,
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
            attention, residual, post_weight, post_eps, False
        )
        logits = _router_logits(hidden, gate_weight)
        if not self.supports(rows):
            # Small batches (decode) keep the native A16W4 MoE.
            output = layer.forward_impl(hidden, logits)
            return _norm_input(output, post_res, next_weight, next_eps, fp8_output)
        weights, ids = self._route(layer, hidden, logits)

        def tailed(m):
            return self.runtime.tail_ok(m) and next_eps == self.runtime.cfg.tail_eps

        tile = self.runtime.max_local_tokens
        self.runtime.set_weights(**_weights(layer))
        if 0 < rows <= tile and not tailed(rows):
            # One untailed launch: its norm outputs are already fresh.
            y = self.runtime.forward(hidden, weights, ids)
            return _norm_input(y, post_res, next_weight, next_eps, fp8_output)
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
            if tailed(m):
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
    post_eps: float,
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
    post_eps: float,
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
        post_eps,
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
    # Forward hooks on the decoder layers and their MoE never see a fused block.
    if envs.ATOM_FWD_DUMP_DIR:
        return "ATOM_FWD_DUMP_DIR hooks every layer's forward"
    return None


def _model_layers(model, config):
    """The registered `(layers, norm_eps)` of the model, or None.

    The model runs each layer's FFN region through `sp_mega_moe_block`: the
    post-attention GemmaRMSNorm, the router, the experts, and the next layer's
    input GemmaRMSNorm, each with its own eps. The fused tail is built for
    `norm_eps`; a next norm with another eps runs unfused.
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
    shape = shapes.pop()
    from atom.model_ops.gemma_norm_fp8 import gemma_rms_norm_fp8_supported

    # Rows the fused tail does not cover are normalized with the same rule.
    if not gemma_rms_norm_fp8_supported(shape[0]):
        return f"no fused Gemma RMSNorm FP8 kernel for hidden size {shape[0]}"
    try:
        from aiter.ops.communication import _mori
        from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP  # noqa: F401

        _mori()
    except ImportError as exc:
        return f"MegaMoeTP unavailable: {exc}"
    return shape


def initialize_sp_mega_moe(model, config):
    """After weight processing, before warmup/capture.

    Returns the shared backend, or None when the native MoE is kept. It is
    all or nothing: one layer it cannot serve keeps every layer native.
    """
    from atom.distributed.ulysses_sp import get_sp_group, get_sp_world_size

    if config is None or not config.sp_mega_moe_enabled or get_sp_world_size() < 2:
        return

    def skip(reason, log=logger.warning):
        # The compile hash reads this flag; native graphs must not share it.
        config.sp_mega_moe_enabled = False
        log("SP MegaMoE disabled, native MoE kept on every layer: %s", reason)

    found = _model_layers(model, config)
    if not found or not found[0]:
        return skip("the model has no MoE layers it can run through it", logger.info)
    layers, tail_eps = found
    group = get_sp_group()
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
        "lb_mt=%s. Each tail supplies the next GemmaNorm/FP8 (final layer BF16).",
        len(layers),
        tile,
        lb_mt,
    )
    return backend
