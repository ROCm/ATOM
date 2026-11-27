import contextlib
import logging
import os
from typing import ClassVar

import torch
import torch.nn.functional as F
from aiter import QuantType
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.rotary_embedding import get_rope
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_config.qwen3_next import Qwen3NextConfig

# registers torch.ops.aiter.maybe_dual_stream_forward
from atom.model_ops import (
    module_dispatch_ops,  # noqa: F401
    mxfp8_1x32,
)
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.attn_output_epilogue import AttentionOutputEpilogue
from atom.model_ops.base_attention import LinearAttention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import (
    DualRMSNorm,
    GemmaRMSNorm,
    RMSNormGated,
    gluon_gemm_allreduce_gemma_rmsnorm,
)
from atom.model_ops.layernorm import GemmaRMSNorm as Qwen3NextRMSNorm
from atom.model_ops.linear import (
    BAParallelLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVGParallelLinear,
    QKVParallelLinear,
    QKVZBAParallelLinear,
    QKVZParallelLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.moe_tail_norm import (
    CombineTailRequest,
    GatheredRouted,
    MoEGatedSum,
    MoENormed,
    can_fuse_gather_tail_mx32,
    can_fuse_moe_tail_mx32,
    combine_tail_request,
    fused_router_shared_available,
    fused_router_shared_prefill_available,
    gated_gemm_available,
    interleave_gate_up,
    moe_gather_tail_rmsnorm_mx32,
    moe_tail_rmsnorm_mx32,
    pad_router_weight,
    router_logits_gemm,
    router_shared_expert,
    shared_expert_gemms,
    silu_gate_up_gemm,
)
from atom.model_ops.qwen_mono_moe.integration import mono_moe_forward
from atom.model_ops.sigmoid_gate import (
    can_fuse_output_gate_mx32,
    output_gate_mul,
    output_gate_mul_mx32,
    output_gate_rows,
    shared_expert_gate_add,
)
from atom.model_ops.topK import (
    init_aiter_topK_meta_data,
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
    rocm_aiter_topk_softmax,
    topk_meta_rows,
)
from atom.model_ops.triton_mrope import (
    gfx1250_fused_qk_norm_mrope_cache_applies,
    try_fused_qk_norm_mrope,
    try_mrope_qk_fused,
)
from atom.model_ops.utils import atom_parameter
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.plugin.prepare import is_plugin_mode, is_vllm
from atom.utils import envs
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import pass_scheduled_tokens_gpu
from atom.utils.region_graph import token_graph

ENABLE_ALLREDUCE_RMSNORM_FUSION = envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION
ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION = (
    envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION
)
ATOM_FUSED_SIGMOID_GATE = envs.ATOM_FUSED_SIGMOID_GATE

logger = logging.getLogger("atom")


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
):
    """Create a weight loader for mamba v2. This ensures that the projections
    are correctly sharded so that they can be split into x, B, C. It also
    ensures that all the groups corresponding to a head shard is placed
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (loaded_start_idx + take)  # type: ignore[misc]
            ]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


class Qwen3NextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate
        # ATOM_SHARED_EXPERT_GATED_GEMM: gate_up_proj's weight rows interleaved
        # gate / up after load, read by one GEMM whose epilogue is the SiLU.
        self.gated_gate_up = False

    def process_weights_after_loading(self):
        proj = self.gate_up_proj
        if (
            (envs.ATOM_SHARED_EXPERT_GATED_GEMM or envs.ATOM_FUSED_ROUTER_SHARED_EXPERT)
            and not self.gated_gate_up
            and proj.quant_type.value == QuantType.No.value
            and proj.weight.dtype == torch.bfloat16
            and proj.bias is None
            and gated_gemm_available()
        ):
            proj.weight.data = interleave_gate_up(proj.weight.data)
            self.gated_gate_up = True

    def forward(self, x):
        if self.gated_gate_up:
            out = silu_gate_up_gemm(x, self.gate_up_proj.weight)
        else:
            out = self.act_fn(self.gate_up_proj(x))
        out = self.down_proj(out)

        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)) * out

        return out


_SHARED_EXPERT_STREAM: torch.cuda.Stream | None = None
# ATOM_QWEN_FHMOE: one zero [tokens, hidden] buffer every block's MoEGatedSum
# hands the next norm as its (already summed) shared part, and whether the
# top-k buffer with the prefilled shared column is set up.
_FHMOE_ZEROS: torch.Tensor | None = None
_FHMOE_TOPK_READY = False


def _fhmoe_setup(block) -> None:
    """The shared-column top-k buffer (topk_softmax writes the routed columns
    and scores the shared one) and the zero shared rows, sized to the MoE's
    token capacity. Once for every block."""
    global _FHMOE_ZEROS, _FHMOE_TOPK_READY
    from atom.model_ops.moe import moe_kernel_token_capacity

    cfg = block.experts.moe_parallel_config
    capacity = moe_kernel_token_capacity(
        get_current_atom_config(),
        dp_size=cfg.dp_size,
        use_all2all=cfg.use_all2all_kernels,
        dp_logical_ratio=cfg.dp_logical_ratio,
    )
    if not _FHMOE_TOPK_READY:
        init_aiter_topK_meta_data(
            n_routed_experts=block.n_routed_experts,
            n_shared_experts=1,
            top_k=block.experts.top_k,
            tp_rank=0,
            tp_size=1,
            max_num_tokens=capacity,
        )
        _FHMOE_TOPK_READY = True
    hidden = block.experts.hidden_size
    if _FHMOE_ZEROS is None or _FHMOE_ZEROS.shape[1] != hidden:
        _FHMOE_ZEROS = torch.zeros(
            (capacity, hidden), dtype=torch.bfloat16, device="cuda"
        )


_FHMOE_DEFER: bool | None = None


def _fhmoe_defer_topk() -> bool:
    """Whether FHMoE leaves its top-k to aiter's fused top-k + route + quant
    (ATOM_MOE_FUSED_TOPK_ROUTE_QUANT and an aiter whose deferral takes the
    shared column); never under micro-batching (one deferral at a time)."""
    global _FHMOE_DEFER
    if _FHMOE_DEFER is None:
        ok = False
        if envs.ATOM_MOE_FUSED_TOPK_ROUTE_QUANT:
            try:
                import inspect

                from aiter.ops.flydsl.grouped_moe_gfx1250 import defer_topk_softmax

                ok = (
                    "num_fused_shared_experts"
                    in inspect.signature(defer_topk_softmax).parameters
                )
            except ImportError:
                ok = False
        _FHMOE_DEFER = ok
    if not _FHMOE_DEFER:
        return False
    from atom.utils.tbo.ubatching import tbo_active

    return not tbo_active()


def _shared_expert_stream() -> torch.cuda.Stream:
    """One side stream per process, shared by every layer's MoE block."""
    global _SHARED_EXPERT_STREAM
    if _SHARED_EXPERT_STREAM is None:
        _SHARED_EXPERT_STREAM = torch.cuda.Stream()
    return _SHARED_EXPERT_STREAM


class Qwen3NextSparseMoeBlock(nn.Module):
    # set in __init__ (a block built without it runs single-stream)
    _use_dual_stream = False
    # Set by `defer_moe_gated_sums` once the norm that reads this block's output
    # is known to fold the gated shared-expert add in (ATOM_FUSED_MOE_TAIL_NORM):
    # forward then returns a MoEGatedSum instead of the summed tensor.
    defer_gated_sum = False
    # ATOM_MOE_GATHER_TAIL_NORM (set with defer_gated_sum, TP1): the routed
    # experts' top-k gather-reduce is left to the norm's kernel too.
    defer_gather = False
    # ATOM_MEGA_COMBINE_TAIL (set with defer_gated_sum): the norm that reads this
    # block's output (not a registered submodule: it is the next layer's), run
    # by the MoE combine itself, and whether its new residual is read.
    combine_tail_norm = None
    combine_tail_writes_residual = True
    # ATOM_QWEN_MONO_MOE (`enable_mono_moe`): the norm that reads this block's
    # output (not a registered submodule), which the fused MoE kernel runs on
    # decode steps of up to 128 rows; the alternating output buffer and the
    # layer index.
    mono_norm = None
    mono_parity = 0
    mono_layer = -1
    # ATOM_FUSED_ROUTER_SHARED_EXPERT: the router weight padded to whole 16-row
    # groups (gate.weight is a view of its first rows), set at load.
    router_padded = None
    # ATOM_QWEN_FHMOE: set in __init__ (a block built without it runs the base
    # path).
    fhmoe = False

    def __init__(
        self, config, quant_config, prefix: str = "", reduce_results: bool = True
    ):
        super().__init__()
        # parallel_config = atom_config.parallel_config
        self.prefix = prefix
        # False when the next norm folds the TP all-reduce in.
        self.reduce_results = reduce_results

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        # self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        # eplb_config = atom_config.parallel_config.eplb_config
        # # self.enable_eplb = parallel_config.enable_eplb

        # self.n_logical_experts = self.n_routed_experts
        # self.n_redundant_experts = eplb_config.num_redundant_experts
        # self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        # self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        # self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        # self.physical_expert_end = (
        #     self.physical_expert_start + self.n_local_physical_experts
        # )

        self.gate = MergedReplicatedLinear(
            config.hidden_size,
            [config.num_experts, config.n_shared_experts],
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.is_rocm_aiter_fusion_shared_expert_enabled = (
            is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                quant_config,
                shared_expert_prefix=f"{prefix}.shared_expert",
                routed_expert_prefix=f"{prefix}.experts",
            )
        )

        if (
            config.shared_expert_intermediate_size > 0
            and not self.is_rocm_aiter_fusion_shared_expert_enabled
        ):
            # When shared expert fusion is disabled (e.g. MXFP4 where shared
            # experts are BF16 while routed experts are FP4), run the shared
            # expert MLP separately.  The sigmoid gating is applied in
            # forward() using the shared-expert portion of the merged gate
            # output — no separate nn.Linear needed here.
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=None,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        # The shared expert reads only this rank's tokens, so it can run beside
        # the routed experts -- their dispatch / combine under EP, their small
        # kernels at decode under TP -- instead of after them.
        self._use_dual_stream = (
            self.shared_expert is not None
            and envs.ATOM_QWEN_SHARED_EXPERT_STREAM
            and envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0
            and not get_current_atom_config().enable_tbo
        )
        if self._use_dual_stream:
            self.alt_stream = _shared_expert_stream()
            get_current_atom_config().compilation_config.static_forward_context[
                prefix
            ] = self

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            use_grouped_topk=False,
            has_bias=False,
            shared_expert_scoring_func=(
                "sigmoid" if self.is_rocm_aiter_fusion_shared_expert_enabled else None
            ),
            prefix=f"{prefix}.experts",
            config=config,
            shared_expert_prefix=f"{prefix}.shared_expert",
            # ATOM_QWEN_MOE_A8W4: MXFP8 activations into the MXFP4 experts.
            mxfp4_fp8_activations=envs.ATOM_QWEN_MOE_A8W4,
            # The MXFP4 default pads each rank's experts to 256 columns: at TP4
            # (512 / 4 = 128) half of every expert the grouped GEMMs stream is zeros.
            pad_align=(
                128
                if envs.ATOM_QWEN_MOE_PAD_ALIGN_128
                and (config.moe_intermediate_size // self.tp_size) % 128 == 0
                else None
            ),
        )
        if envs.ATOM_QWEN_MOE_A8W4:
            m = self.experts.quant_method
            logger.info(
                "ATOM_QWEN_MOE_A8W4: %s experts %s fp8_activations=%s",
                prefix,
                type(m).__name__,
                getattr(m, "fp8_activations", None),
            )
            asked = [
                f
                for f in ("ATOM_FOLD_TOPK_ROUTER", "ATOM_FOLD_POSTNORM")
                if os.environ.get(f, "0") == "1"
            ]
            if asked:
                logger.warning(
                    "ATOM_QWEN_MOE_A8W4: %s set but off (the folds hand MXFP4 rows to the experts)",
                    " / ".join(asked),
                )
        # ATOM_QWEN_FHMOE (gfx1250, TP1): decode-size steps run the shared expert
        # inside the routed grouped MoE (see `fhmoe_forward`); the experts' quant
        # method builds its operands at load from the shared expert handed here.
        self.fhmoe = bool(
            envs.ATOM_QWEN_FHMOE
            and self.shared_expert is not None
            and self.tp_size == 1
            and not self.is_rocm_aiter_fusion_shared_expert_enabled
            and get_gfx() == "gfx1250"
        )
        if self.fhmoe:
            # Not a submodule: the shared expert stays this block's.
            object.__setattr__(self.experts, "fhmoe_shared_mlp", self.shared_expert)
            _fhmoe_setup(self)
        # The same padding for experts quantized online at load: the MTP layer's
        # experts are unquantized in the checkpoint and become MXFP4 there.
        # ATOM_QWEN_MONO_MOE's kernel needs 128 (TP4) either way.
        if (
            envs.ATOM_QWEN_MOE_PAD_ALIGN_128
            and (config.moe_intermediate_size // self.tp_size) % 128 == 0
        ):
            self.experts.online_quant_pad_align = 128

    def process_weights_after_loading(self):
        w = self.gate.weight
        if (
            envs.ATOM_FUSED_ROUTER_SHARED_EXPERT
            and self.router_padded is None
            and w.dtype == torch.bfloat16
            and w.is_contiguous()
        ):
            padded = pad_router_weight(w.data)
            w.data = padded[: w.shape[0]]
            self.router_padded = padded

    def _router_shared_operands_ok(self, hidden_states: torch.Tensor) -> bool:
        """Whether the router GEMM and the shared expert can run as the aiter
        a16w16 kernels: TP1, a gated (interleaved) bf16 shared expert without
        its own gate, bf16 rows and router weight, eager / graph capture."""
        mlp = self.shared_expert
        return not (
            mlp is None
            or self.tp_size != 1
            or not mlp.gated_gate_up
            or mlp.expert_gate is not None
            or hidden_states.shape[0] <= 0
            or torch.compiler.is_compiling()
            or hidden_states.dtype != torch.bfloat16
            or self.gate.bias is not None
            or self.gate.weight.dtype != torch.bfloat16
            or not fused_router_shared_available()
        )

    def _router_weight_padded(self) -> torch.Tensor:
        w_router = self.gate.weight
        padded = self.router_padded
        if padded is None or padded.data_ptr() != w_router.data_ptr():
            padded = w_router  # correct, only slower
        return padded

    def fused_router_shared(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """ATOM_FUSED_ROUTER_SHARED_EXPERT: (router logits, shared expert
        output) from one kernel (decode sizes) or from the router + gate_up
        GEMM and the down GEMM (prefill sizes), or None where neither
        applies."""
        mlp = self.shared_expert
        m = hidden_states.shape[0]
        if (
            not envs.ATOM_FUSED_ROUTER_SHARED_EXPERT
            or (
                m > envs.ATOM_FUSED_ROUTER_SHARED_EXPERT_MAX_TOKENS
                and not (
                    envs.ATOM_FUSED_ROUTER_SHARED_EXPERT_PREFILL
                    and m >= envs.ATOM_FUSED_ROUTER_SHARED_EXPERT_PREFILL_MIN_TOKENS
                    and fused_router_shared_prefill_available()
                )
            )
            or not self._router_shared_operands_ok(hidden_states)
        ):
            return None
        w_router = self.gate.weight
        padded = self._router_weight_padded()
        if envs.ATOM_FOLD_TOPK_ROUTER:
            folded = self._fold_topk_router(hidden_states, padded, w_router.shape[0])
            if folded is not None:
                return folded
        return router_shared_expert(
            hidden_states,
            padded,
            w_router.shape[0],
            mlp.gate_up_proj.weight,
            mlp.down_proj.weight,
            prefill=m > envs.ATOM_FUSED_ROUTER_SHARED_EXPERT_MAX_TOKENS,
        )

    def fhmoe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor | MoEGatedSum:
        """ATOM_QWEN_FHMOE: the router GEMM, the top-k that also scores the
        shared column (route id E, weight sigmoid of its logit; deferred to the
        MoE's fused top-k + route + quant where it can) and one aiter FHMoE
        grouped MoE over the routed experts and the shared expert. The next
        norm still takes the sum (MoEGatedSum, its shared part zero)."""
        ex = self.experts
        logits = self.gate(hidden_states)
        nvt = pass_scheduled_tokens_gpu(hidden_states.shape[0])
        if _fhmoe_defer_topk():
            # ATOM_MOE_FUSED_TOPK_ROUTE_QUANT: the top-k (shared column too) is
            # left to aiter's grouped MoE, which computes it with the routing
            # and the routed activation quant in one fused kernel (two above
            # 512 rows). The buffers are the prefilled shared-column rows.
            from aiter.ops.flydsl.grouped_moe_gfx1250 import defer_topk_softmax

            weights, ids = topk_meta_rows(hidden_states.shape[0])
            defer_topk_softmax(
                weights, ids, logits, ex.renormalize, nvt, num_fused_shared_experts=1
            )
        else:
            weights, ids = rocm_aiter_topk_softmax(
                logits,
                ex.top_k,
                ex.renormalize,
                num_fused_shared_experts=1,
                num_routing_experts=self.n_routed_experts,
                fused_shared_experts_scoring_func="sigmoid",
                num_valid_rows=nvt,
            )
        apply = ex.quant_method.apply_fhmoe
        if self.defer_gather and not torch.compiler.is_compiling():
            from aiter.ops.flydsl.grouped_moe_gfx1250 import defer_gather_reduce

            with defer_gather_reduce() as req:
                out = apply(ex, hidden_states, weights, ids, nvt)
            if req.taken:
                out = GatheredRouted(
                    req.grouped, req.route_rows, req.weights, req.num_valid_tokens
                )
        else:
            out = apply(ex, hidden_states, weights, ids, nvt)
        if self.defer_gated_sum:
            zeros = _FHMOE_ZEROS[: hidden_states.shape[0]]
            return MoEGatedSum(out, zeros, None)
        if isinstance(out, GatheredRouted):
            out = out.materialize()
        return out

    def _fold_topk_router(
        self,
        hidden_states: torch.Tensor,
        w_router_padded: torch.Tensor,
        n_router: int,
        prenorm=None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """ATOM_FOLD_TOPK_ROUTER: (router logits, shared expert output) from the
        router + shared expert kernel that also computes this block's MoE top-k,
        routing and MXFP4 activation quant, handed to the experts' deferred top-k
        (the grouped MoE then skips moe_topk_route_quant). None where it does not
        apply: then the plain kernel runs, and the routing as usual. The geometry
        of the routing is aiter's, learned from an unfused call at this row count
        (the first layer of the first step at it, e.g. the warmup)."""
        try:
            from aiter.ops.flydsl.grouped_moe_gfx1250 import (
                topk_route_quant_decode_geometry,
            )
            from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_router_shared_expert_topk import (
                fused_gemm_a16w16_router_shared_expert_topk,
                fused_gemm_a16w16_router_shared_expert_topk_supported,
            )
        except ImportError:
            return None
        from atom.model_ops.fused_moe.fused_topk_route_quant import (
            set_prepared_routing,
        )
        from atom.utils.forward_context import pass_scheduled_tokens_gpu

        m, k = hidden_states.shape
        if (
            not envs.ATOM_MOE_FUSED_TOPK_ROUTE_QUANT
            or self.is_rocm_aiter_fusion_shared_expert_enabled
            or self.n_routed_experts != 256
        ):
            return None
        geometry = topk_route_quant_decode_geometry(m, k, 8)
        mlp = self.shared_expert
        if envs.ATOM_ROUTER_SHARED_V2 and geometry is not None:
            # the decode rewrite of the same kernel (bitwise the same outputs), where it applies
            try:
                from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_router_shared_expert_topk_v2 import (
                    fused_gemm_a16w16_router_shared_expert_topk_v2,
                    fused_gemm_a16w16_router_shared_expert_topk_v2_supported,
                )
            except ImportError:
                fused_gemm_a16w16_router_shared_expert_topk_v2_supported = None
            if (
                fused_gemm_a16w16_router_shared_expert_topk_v2_supported is not None
                and fused_gemm_a16w16_router_shared_expert_topk_v2_supported(
                    hidden_states,
                    w_router_padded,
                    mlp.gate_up_proj.weight,
                    mlp.down_proj.weight,
                    n_router,
                    geometry[0],
                )
            ):
                fused_gemm_a16w16_router_shared_expert_topk = (
                    fused_gemm_a16w16_router_shared_expert_topk_v2
                )
                fused_gemm_a16w16_router_shared_expert_topk_supported = (
                    fused_gemm_a16w16_router_shared_expert_topk_v2_supported
                )
        prenorm_kw = {}
        w_gate_up = mlp.gate_up_proj.weight
        if prenorm is not None:
            # ATOM_FOLD_POSTNORM: only the v2 kernel's narrow variant takes the norm
            from atom.model_ops import fold_postnorm

            if (
                fused_gemm_a16w16_router_shared_expert_topk.__name__
                != "fused_gemm_a16w16_router_shared_expert_topk_v2"
                or m > 16
            ):
                return None
            pw = fold_postnorm.prescaled(self, prenorm.norm)
            if pw is None:
                return None
            w_router_padded, w_gate_up = pw
            prenorm_kw = {
                "prenorm": (
                    prenorm.ssq,
                    prenorm.norm.weight,
                    prenorm.norm.variance_epsilon,
                )
            }
        if geometry is None or not fused_gemm_a16w16_router_shared_expert_topk_supported(
            hidden_states,
            w_router_padded,
            mlp.gate_up_proj.weight,
            mlp.down_proj.weight,
            n_router,
            geometry[0],
        ):
            return None
        nvr = pass_scheduled_tokens_gpu(m)
        logits, shared, routed = fused_gemm_a16w16_router_shared_expert_topk(
            hidden_states,
            w_router_padded,
            w_gate_up,
            mlp.down_proj.weight,
            n_router,
            align_m=geometry[0],
            contiguous_m=geometry[1],
            wmma_rep=geometry[2],
            num_valid_rows=nvr,
            **prenorm_kw,
        )
        set_prepared_routing(logits, geometry, routed, nvr, required=prenorm is not None)
        return logits, shared

    def _prenorm_forward(self, pn) -> MoEGatedSum | None:
        """ATOM_FOLD_POSTNORM: `single_stream_moe_forward` on the un-normalized residual sum `pn.x`, the norm
        applied inside the fold D v2 router kernel; None where that kernel cannot take it (the caller then
        materializes the norm)."""
        x = pn.x
        if (
            getattr(self, "fhmoe", False)
            or self._shared_side_stream_mode(x) is not None
            or self.is_rocm_aiter_fusion_shared_expert_enabled
            or not self.defer_gated_sum
            or not envs.ATOM_FUSED_ROUTER_SHARED_EXPERT
            or not self._router_shared_operands_ok(x)
        ):
            return None
        fused = self._fold_topk_router(
            x, self._router_weight_padded(), self.gate.weight.shape[0], prenorm=pn
        )
        if fused is None:
            return None
        from atom.model_ops import fold_postnorm

        fold_postnorm._stats["folded"] += 1
        logits, shared_output = fused
        router_logits = logits[:, : self.n_routed_experts]
        # x is not the experts' input: the routing prepared above (MXFP4 rows of the normalized x) must be the
        # one consumed (try_defer_topk_route_quant raises otherwise)
        if self.defer_gather and not torch.compiler.is_compiling():
            routed_output = self._experts_deferred_gather(x, router_logits)
        else:
            routed_output = self.experts(hidden_states=x, router_logits=router_logits)
        return MoEGatedSum(routed_output, shared_output, logits[:, self.n_routed_experts :])

    def combine_tail_forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> MoENormed | MoEGatedSum:
        """ATOM_MEGA_COMBINE_TAIL: the shared expert first, then the routed
        experts with the gated add, the residual add and the next norm handed to
        the MoE combine. A combine that cannot take them (another transport, a
        quantized combine wire) leaves the request unanswered, and the block
        returns the unsummed parts for the norm to finish."""
        fused = self.fused_router_shared(hidden_states)
        if fused is not None:
            logits, shared_output = fused
        else:
            logits = self.gate(hidden_states)
            shared_output = self.shared_expert(hidden_states)
        gate_logits = logits[:, self.n_routed_experts :]
        norm = self.combine_tail_norm
        req = CombineTailRequest(
            shared_output,
            gate_logits,
            residual,
            norm.weight,
            norm.variance_epsilon,
            self.combine_tail_writes_residual,
        )
        with combine_tail_request(req):
            routed_output = self.experts(
                hidden_states=hidden_states,
                router_logits=logits[:, : self.n_routed_experts],
            )
        if req.result is not None:
            return MoENormed(*req.result)
        return MoEGatedSum(routed_output, shared_output, gate_logits)

    def _experts_deferred_gather(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | GatheredRouted:
        """The routed experts with their top-k gather-reduce left to the next
        norm (ATOM_MOE_GATHER_TAIL_NORM): MoE2's rows and the route map as a
        `GatheredRouted` where aiter's grouped MoE deferred it, else the sum."""
        from aiter.ops.flydsl.grouped_moe_gfx1250 import defer_gather_reduce

        with defer_gather_reduce() as req:
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        if not req.taken:
            return routed_output
        # aiter's unwritten stand-in must come back untouched: no gather,
        # collective or slice between it and here (see `gather_deferrable`).
        if (
            routed_output.data_ptr() != req.placeholder.data_ptr()
            or req.route_rows.shape[0] != hidden_states.shape[0]
        ):
            raise RuntimeError(
                f"{self.prefix}: the MoE call moved the rows of a deferred gather"
            )
        return GatheredRouted(
            req.grouped, req.route_rows, req.weights, req.num_valid_tokens
        )

    @property
    def gather_deferrable(self) -> bool:
        """Whether the routed sum comes straight out of the grouped MoE's
        gather-reduce on this rank's own rows: TP1, no all-to-all, no DP
        gather / reduce-scatter, no simulated DP ranks, no PCP merge; and an
        aiter that can defer the reduce."""
        try:
            from aiter.ops.flydsl.grouped_moe_gfx1250 import (  # noqa: F401
                defer_gather_reduce,
            )
        except ImportError:
            return False
        cfg = self.experts.moe_parallel_config
        return (
            self.tp_size == 1
            and self.shared_expert is not None
            and not self.is_rocm_aiter_fusion_shared_expert_enabled
            and not cfg.use_all2all_kernels
            and (cfg.dp_size <= 1 or cfg.replicate_experts)
            and cfg.dp_logical_ratio == 1
            and not getattr(cfg, "pcp_merge", False)
        )

    def _shared_side_stream_mode(self, hidden_states: torch.Tensor) -> str | None:
        """ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM: "early" (fork the shared expert
        right after the router GEMM), "late" (fork it after the routing /
        quant launch, beside the expert GEMMs only) or None."""
        m = hidden_states.shape[0]
        if not (
            envs.ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM
            and envs.ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM_MIN_TOKENS
            <= m
            <= envs.ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM_MAX_TOKENS
            and not self.is_rocm_aiter_fusion_shared_expert_enabled
            and self._router_shared_operands_ok(hidden_states)
        ):
            return None
        if m >= envs.ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM_LATE_MIN_TOKENS:
            try:
                from aiter.ops.flydsl.grouped_moe_gfx1250 import (  # noqa: F401
                    after_routing_launched,
                )

                return "late"
            except ImportError:  # an aiter without the routing hook
                pass
        return "early"

    def side_stream_moe_forward(
        self, hidden_states: torch.Tensor, late: bool
    ) -> torch.Tensor | MoEGatedSum:
        """`single_stream_moe_forward` with the router GEMM alone on the main
        stream and the shared expert on a side stream beside the routed
        experts (memory-bound at prefill sizes, so the shared expert's bf16
        GEMMs run in their shadow); joined before the sum."""
        mlp = self.shared_expert
        logits = router_logits_gemm(
            hidden_states, self._router_weight_padded(), self.gate.weight.shape[0]
        )
        router_logits = logits[:, : self.n_routed_experts]
        current_stream = torch.cuda.current_stream()
        side = _shared_expert_stream()
        shared = []

        def fork():
            side.wait_stream(current_stream)
            with torch.cuda.stream(side):
                shared.append(
                    shared_expert_gemms(
                        hidden_states, mlp.gate_up_proj.weight, mlp.down_proj.weight
                    )
                )

        hook = contextlib.nullcontext()
        if late:
            from aiter.ops.flydsl.grouped_moe_gfx1250 import after_routing_launched

            hook = after_routing_launched(fork)
        else:
            fork()
        with hook:
            if self.defer_gather and not torch.compiler.is_compiling():
                routed_output = self._experts_deferred_gather(
                    hidden_states, router_logits
                )
            else:
                routed_output = self.experts(
                    hidden_states=hidden_states, router_logits=router_logits
                )
        if shared:
            current_stream.wait_stream(side)
        else:  # the routed path never reached its expert GEMMs (no fork)
            shared.append(
                shared_expert_gemms(
                    hidden_states, mlp.gate_up_proj.weight, mlp.down_proj.weight
                )
            )
        shared_gate_logits = logits[:, self.n_routed_experts :]
        if self.defer_gated_sum:
            return MoEGatedSum(routed_output, shared[0], shared_gate_logits)
        return shared_expert_gate_add(
            shared[0], shared_gate_logits, routed_output, ATOM_FUSED_SIGMOID_GATE
        )

    def single_stream_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        side_mode = self._shared_side_stream_mode(hidden_states)
        if side_mode is not None:
            return self.side_stream_moe_forward(hidden_states, side_mode == "late")
        # router_logits: (num_tokens, n_experts + 1)
        fused = self.fused_router_shared(hidden_states)
        shared_output = None
        if fused is not None:
            # The shared expert runs before the routed experts here.
            logits, shared_output = fused
        else:
            logits = self.gate(hidden_states)
        if not self.is_rocm_aiter_fusion_shared_expert_enabled:
            router_logits = logits[:, : self.n_routed_experts]
        else:
            router_logits = logits
        if self.defer_gather and not torch.compiler.is_compiling():
            routed_output = self._experts_deferred_gather(hidden_states, router_logits)
        else:
            routed_output = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        if not self.is_rocm_aiter_fusion_shared_expert_enabled:
            if shared_output is None:
                shared_output = self.shared_expert(hidden_states)
            # Apply shared expert gate: the merged gate output contains
            # [routed_logits, shared_expert_gate_logits], extract the tail
            shared_gate_logits = logits[:, self.n_routed_experts :]
            if self.defer_gated_sum:
                return MoEGatedSum(routed_output, shared_output, shared_gate_logits)
            return shared_expert_gate_add(
                shared_output,
                shared_gate_logits,
                routed_output,
                ATOM_FUSED_SIGMOID_GATE,
            )
        return routed_output

    def dual_stream_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`single_stream_moe_forward` with the shared expert on `alt_stream`;
        the gated combine runs on the main stream after the join."""
        logits = self.gate(hidden_states)
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            shared_output = self.shared_expert(hidden_states)
        routed_output = self.experts(
            hidden_states=hidden_states,
            router_logits=logits[:, : self.n_routed_experts],
        )
        current_stream.wait_stream(self.alt_stream)
        return shared_expert_gate_add(
            shared_output,
            logits[:, self.n_routed_experts :],
            routed_output,
            ATOM_FUSED_SIGMOID_GATE,
        )

    def native_moe_forward(self, hidden_states: torch.Tensor):
        """Routed + gated shared experts of 2-D rows, before any all-reduce."""
        if self._use_dual_stream:
            return torch.ops.aiter.maybe_dual_stream_forward(hidden_states, self.prefix)
        return self.single_stream_moe_forward(hidden_states)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        from atom.model_ops import fold_postnorm

        if isinstance(hidden_states, fold_postnorm.PreNormed):
            out = self._prenorm_forward(hidden_states)
            if out is not None:
                return out
            hidden_states = hidden_states.materialize()
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        _num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if (
            self.fhmoe
            and residual is None
            and hidden_states.shape[0] <= envs.ATOM_QWEN_FHMOE_MAX_TOKENS
            and not torch.compiler.is_compiling()
        ):
            out = self.fhmoe_forward(hidden_states)
            return out if isinstance(out, MoEGatedSum) else out.view(orig_shape)
        if (
            residual is not None
            and self.combine_tail_norm is not None
            and hidden_states.shape[0] <= envs.ATOM_MEGA_COMBINE_TAIL_MAX_TOKENS
            and not torch.compiler.is_compiling()
        ):
            return self.combine_tail_forward(hidden_states, residual)
        if residual is not None and self.mono_norm is not None:
            normed = mono_moe_forward(self, hidden_states, residual)
            if normed is not None:
                return normed
        final_hidden_states = self.native_moe_forward(hidden_states)
        if isinstance(final_hidden_states, MoEGatedSum):
            # TP1 (see `defer_moe_gated_sums`): nothing to reduce, and the rows
            # stay 2-D for the norm that sums them.
            return final_hidden_states

        if self.tp_size > 1 and self.reduce_results:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


def defer_moe_gated_sums(
    layers, final_norm, combine_tail: bool = True, tp_partial: bool = False
) -> int:
    """ATOM_FUSED_MOE_TAIL_NORM: let each sparse MoE block of `layers` hand its
    output unsummed (`MoEGatedSum`) to the norm that reads it -- the next
    layer's input_layernorm, or `final_norm` after the last layer -- which then
    adds `routed + sigmoid(gate) * shared` and the residual in its own kernel.

    Only where that norm can (GemmaRMSNorm, neither all-reduce-fused nor
    FP8-quantizing), the block runs single-stream with a separate shared expert
    at TP1 (a TP-partial sum would need its all-reduce first) and the layer has
    no layer_scale. With `tp_partial` (ATOM_FUSED_MOE_TAIL_AR_NORM), also at TP
    > 1 where the norm all-reduces the block's partial output and its kernel
    can add the gated sum while it reads it. Returns how many blocks defer.
    With ATOM_MEGA_COMBINE_TAIL (and `combine_tail`), the TP1 blocks also hand
    the norm to the MoE combine.
    """
    # Plugin frontends patch the decoder layer's forward (rtpllm), which knows
    # nothing of a MoEGatedSum.
    if not envs.ATOM_FUSED_MOE_TAIL_NORM or is_plugin_mode():
        return 0
    deferred = 0
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, Qwen3NextSparseMoeBlock):
            continue
        consumer = (
            getattr(layers[i + 1], "input_layernorm", None)
            if i + 1 < len(layers)
            else final_norm
        )
        if not isinstance(consumer, GemmaRMSNorm):
            continue
        if mlp.tp_size == 1:
            norm_takes_it = consumer.takes_moe_gated_sum
        else:
            # The norm all-reduces the partial sums; the block must not.
            norm_takes_it = (
                tp_partial
                and not mlp.reduce_results
                and consumer.takes_tp_partial_moe_gated_sum
            )
        if (
            norm_takes_it
            and mlp.shared_expert is not None
            and not mlp._use_dual_stream
            and not getattr(layer, "layer_scale", False)
        ):
            mlp.defer_gated_sum = True
            # The norm's own kernel, not the all-reduce one, and aiter's TP
            # grouped path (no EP scatter) reduces the routed rows.
            mlp.defer_gather = (
                envs.ATOM_MOE_GATHER_TAIL_NORM
                and mlp.gather_deferrable
                and not (envs.ATOM_MEGA_COMBINE_TAIL and combine_tail)
            )
            deferred += 1
            if envs.ATOM_MEGA_COMBINE_TAIL and combine_tail and mlp.tp_size == 1:
                # Not a submodule: the norm belongs to the next layer / model.
                object.__setattr__(mlp, "combine_tail_norm", consumer)
                mlp.combine_tail_writes_residual = consumer is not final_norm
    moe_blocks = sum(
        isinstance(getattr(ly, "mlp", None), Qwen3NextSparseMoeBlock) for ly in layers
    )
    if moe_blocks:
        logger.info(
            "ATOM_FUSED_MOE_TAIL_NORM: %d of %d MoE blocks fold their gated "
            "shared-expert add into the next norm (%d also their top-k "
            "gather-reduce)",
            deferred,
            moe_blocks,
            sum(
                getattr(ly.mlp, "defer_gather", False)
                for ly in layers
                if hasattr(ly, "mlp")
            ),
        )
    return deferred


class Qwen3NextAttention(nn.Module):
    # Set by the MTP drafter (ATOM_MTP_ATTN_GATE_IN_REDUCE): the output gate goes
    # to the attention's own epilogue where it can take it.
    gate_in_attention = False
    # ATOM_UA_FUSED_OUTPUT_GATE, set in __init__ (off for a layer built without it).
    fused_output_gate = False
    # ATOM_MXFP8_1X32's gate + quant into o_proj, set in __init__ (likewise).
    mx32_output_gate = False
    # ATOM_UA_REDUCE_GATE_MX32: that gate + quant in the attention's split reduce.
    reduce_gate_mx32 = False

    def __init__(
        self,
        atom_config,
        quant_config=None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if hasattr(atom_config.hf_config, "text_config"):
            config = atom_config.hf_config.text_config
        else:
            config = atom_config.hf_config
        self.atom_config = atom_config
        self.config = config
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        qkv_cls = QKVGParallelLinear if self.attn_output_gate else QKVParallelLinear
        self.qkv_proj = qkv_cls(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.o_proj",
        )
        # Set by the decoder layer: it runs o_proj in one kernel with its next norm.
        self.skip_out_proj = False
        if is_vllm():
            # print("hf_config: ", atom_config.plugin_config.vllm_config.model_config.hf_config, flush=True)
            model_config = atom_config.plugin_config.vllm_config.model_config
            text_hf_config = (
                model_config.hf_text_config
                if hasattr(model_config, "hf_text_config")
                else model_config
            )
            rope_parameters = getattr(
                text_hf_config,
                "rope_parameters",
                None,
            )
        else:
            rope_parameters = getattr(config, "rope_parameters", None)
        rope_parameters = rope_parameters or {}
        rope_theta = rope_parameters.get("rope_theta", 10000)
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)

        rotary_dim = int(self.head_dim * partial_rotary_factor)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.qk_norm = DualRMSNorm(
            self.q_norm,
            self.k_norm,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            prefix=f"{prefix}.qk_norm",
        )

        from atom.model_ops.base_attention import Attention

        # The attention runs the Q/K norm, the RoPE and the KV cache write. On
        # gfx1250, ATOM's own attention does this for Qwen3.5 by default, in one
        # aiter kernel (try_fused_qk_norm_mrope_cache). Plugin mode keeps the
        # separate kernels (not validated there).
        self.fuse_qk_norm_rope_cache = ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION or (
            not is_plugin_mode()
            and gfx1250_fused_qk_norm_mrope_cache_applies(
                self.rotary_emb, self.head_dim, atom_config.kv_cache_dtype
            )
        )
        fusion_kwargs = {}
        if self.fuse_qk_norm_rope_cache:
            fusion_kwargs = {
                "rotary_emb": self.rotary_emb,
                "q_norm": self.q_norm,
                "k_norm": self.k_norm,
            }

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            quant_config=quant_config,
            use_mla=False,
            layer_num=extract_layer_index(prefix),
            config=atom_config,
            prefix=f"{prefix}",
            **fusion_kwargs,
        )

        self.use_fused_sigmoid_mul_quant = (
            ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION
            and self.attn_output_gate
            and self.o_proj.quant_type == QuantType.per_1x128
        )
        # ATOM_MXFP8_1X32 with its fused act quant: the output gate kernel writes the 1x32
        # o_proj's MXFP8 input (gate + quant in one launch), so the attention returns the
        # ungated output instead of gating in its epilogue.
        self.mx32_output_gate = (
            self.attn_output_gate
            and not self.use_fused_sigmoid_mul_quant
            and mxfp8_1x32.fused_act_quant()
            and mxfp8_1x32.takes_mx32(self.o_proj)
        )
        # ATOM_UA_FUSED_OUTPUT_GATE: the attention returns o * sigmoid(gate), the
        # gate in the unified attention's epilogue (attention_mha: the paths that
        # end on another kernel gate after it). ATOM native attention only. Not
        # `gate_in_attention`, the MTP drafter's epilogue request (_attend).
        self.fused_output_gate = (
            envs.ATOM_UA_FUSED_OUTPUT_GATE
            and self.attn_output_gate
            and not self.use_fused_sigmoid_mul_quant
            and not self.mx32_output_gate
            and not is_plugin_mode()
            and getattr(getattr(self.attn, "impl", None), "accepts_output_gate", False)
        )
        # ATOM_UA_REDUCE_GATE_MX32: the mx32 gate + quant done by the unified
        # attention's split-KV reduce (it writes o_proj's 1x32 MXFP8 input), on the
        # paths that end on it (decode / verify); `_gated_mx32_kernel` elsewhere.
        self.reduce_gate_mx32 = (
            envs.ATOM_UA_REDUCE_GATE_MX32
            and self.mx32_output_gate
            and not is_plugin_mode()
        )

    def _attend(
        self, gate, select_rows, *args, mx32: bool = False, **kwargs
    ) -> tuple[torch.Tensor, bool]:
        """`self.attn(*args, **kwargs)`, handing it the output gate and the row
        selection (ATOM_MTP_ATTN_GATE_IN_REDUCE) where they go. Returns the
        output and whether the attention applied both. With `mx32`
        (ATOM_UA_REDUCE_GATE_MX32) the request is the gate and o_proj's 1x32
        MXFP8 input instead: the output is then `(x_fp8, x_scale)` when applied."""
        if mx32:
            with AttentionOutputEpilogue(gate, None, mx32=True) as req:
                out = self.attn(*args, **kwargs)
            return (req.mx32_out, True) if req.applied else (out, False)
        if not self.gate_in_attention or gate is None:
            return self.attn(*args, **kwargs), False
        with AttentionOutputEpilogue(gate, select_rows) as req:
            out = self.attn(*args, **kwargs)
        return out, req.applied

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        x_scale=None,
        select_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`select_rows` (the MTP drafter, ATOM_MTP_SELECT_ROWS_BEFORE_O_PROJ):
        every row attends (and writes its KV), and only these rows go on to the
        output gate and o_proj: the output has `len(select_rows)` rows."""
        # ATOM_FOLD_QKNORM_QKV: the Q/K norm, MRoPE and KV write in the qkv GEMM's epilogue (decode)
        from atom.model_ops.fold_qkv_qknorm import fold_enabled, try_fold_qkv

        qkv = (
            try_fold_qkv(self, positions, hidden_states, x_scale)
            if fold_enabled()
            else None
        )
        if qkv is None and x_scale is None:
            # ATOM_FOLD_QKNORM_QKV_BF16: the same fold for a bf16 qkv_proj (the MTP drafter's)
            from atom.model_ops import fold_qkv_qknorm_bf16

            if fold_qkv_qknorm_bf16.ENABLED:
                qkv = fold_qkv_qknorm_bf16.try_fold_qkv_bf16(
                    self, positions, hidden_states
                )
        if qkv is None:
            qkv = self.qkv_proj(hidden_states, x_scale=x_scale)
        gate = None

        if self.attn_output_gate:
            gate, qkv = torch.split(
                qkv, [self.q_size, self.q_size + self.kv_size + self.kv_size], dim=-1
            )
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
        else:
            q, k, v = torch.split(
                qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
        # The drafter's epilogue request (gate_in_attention) takes the gate when it
        # is on, so the gate is applied once. The 1x32 MXFP8 o_proj path keeps its own
        # gate + quant kernel (mx32_output_gate, excluded from fused_output_gate).
        fused_output_gate = self.fused_output_gate and not self.gate_in_attention
        attn_gate = {"output_gate": gate} if fused_output_gate else {}
        # ATOM_UA_REDUCE_GATE_MX32: every row to o_proj (no selection), eager / graphs.
        mx32_reduce = (
            self.reduce_gate_mx32
            and gate is not None
            and select_rows is None
            and not self.skip_out_proj
            and not self.gate_in_attention
            and gate.dtype == torch.bfloat16
            and gate.stride(-1) == 1
            and not torch.compiler.is_compiling()
        )
        if mx32_reduce:
            attn_gate["mx32"] = True
        # ATOM_FOLD_ATTN_GATE: the output gate and o_proj's 1x32 MXFP8 quant in the epilogue of the
        # attention's split-KV reduce (no separate gate + quant launch)
        from atom.model_ops import fold_attn_gate as _fag

        fold_req = (
            _fag.open_request(gate)
            if not mx32_reduce and self.mx32_output_gate and _fag.can_fold(self, gate, select_rows)
            else None
        )
        if fold_req is not None:
            attn_gate = {"output_gate": gate}
        if self.fuse_qk_norm_rope_cache:
            # Pass the packed [q, k, v] tensor (gate excluded) for smuggling
            # through vLLM's typed custom op which requires Tensor args.
            # qkv_packed = qkv[:, self.q_size :] if self.attn_output_gate else qkv
            attn_output, epilogue_done = self._attend(
                gate,
                select_rows,
                query=q,
                key=k,
                value=v,
                positions=positions,
                qkv=qkv,
                **attn_gate,
            )
        else:
            # gfx1250: the norm and the MRoPE in one kernel. Else the norm, then the MRoPE.
            fused_qk = try_fused_qk_norm_mrope(
                self.qk_norm,
                self.rotary_emb,
                positions,
                q,
                k,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            if fused_qk is None:
                q, k = self.qk_norm(q, k)
                fused_qk = try_mrope_qk_fused(
                    self.rotary_emb,
                    positions,
                    q,
                    k,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )
            if fused_qk is None:
                q, k = self.rotary_emb(positions, q, k)
            else:
                q, k = fused_qk
            attn_output, epilogue_done = self._attend(
                gate, select_rows, q, k, v, **attn_gate
            )

        if epilogue_done and mx32_reduce:
            # The reduce wrote the gated output as o_proj's 1x32 MXFP8 input.
            x_fp8, x_scale = attn_output
            return self.o_proj(x_fp8, x_scale=x_scale)
        if fold_req is not None:
            x_fp8, x_scale = _fag.finish(fold_req, attn_output)
            return self.o_proj(x_fp8, x_scale=x_scale)
        if epilogue_done:
            # The attention gated (and selected) its output itself.
            return attn_output if self.skip_out_proj else self.o_proj(attn_output)
        if select_rows is not None:
            if (
                self.attn_output_gate
                and not fused_output_gate
                and not self.use_fused_sigmoid_mul_quant
            ):
                # The gate and the row gather in one launch.
                attn_output = output_gate_rows(attn_output, gate, select_rows)
                return attn_output if self.skip_out_proj else self.o_proj(attn_output)
            attn_output = torch.index_select(attn_output, 0, select_rows)
            if gate is not None:
                gate = torch.index_select(gate, 0, select_rows)

        if self.use_fused_sigmoid_mul_quant:
            from atom.model_ops.triton_fused_sigmoid_mul_quant import (
                sigmoid_mul_fp8_group_quant,
            )

            attn_output, attn_scale = sigmoid_mul_fp8_group_quant(attn_output, gate)
            output = self.o_proj(attn_output, x_scale=attn_scale)
        elif (
            self.mx32_output_gate
            and not self.skip_out_proj
            and can_fuse_output_gate_mx32(attn_output, gate)
        ):
            # the gate writes o_proj's 1x32 MXFP8 input itself (fp8 + e8m0 scales)
            x_fp8, x_scale = output_gate_mul_mx32(attn_output, gate)
            output = self.o_proj(x_fp8, x_scale=x_scale)
        elif self.attn_output_gate:
            if not fused_output_gate:
                attn_output = output_gate_mul(
                    attn_output, gate, ATOM_FUSED_SIGMOID_GATE
                )
            output = attn_output if self.skip_out_proj else self.o_proj(attn_output)
        else:
            output = attn_output if self.skip_out_proj else self.o_proj(attn_output)

        return output


class Qwen3NextGatedDeltaNet(nn.Module):
    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def __init__(
        self,
        atom_config: Qwen3NextConfig,
        quant_config=None,
        speculative_config=None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.atom_config = atom_config
        if hasattr(atom_config.hf_config, "text_config"):
            config = atom_config.hf_config.text_config
        else:
            config = atom_config.hf_config
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.quant_config = quant_config

        self.speculative_config = speculative_config or atom_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.projection_size_ba = self.num_v_heads * 2
        self.create_qkvzba_proj(quant_config, prefix)

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [query_key_settings, query_key_settings, value_settings],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = atom_parameter(torch.ones(self.num_v_heads // self.tp_size))
        self.A_log = atom_parameter(
            torch.empty(
                (self.num_v_heads // self.tp_size),
            )
        )

        # Get downstream out_proj quant_config for norm
        norm_quant_config = (
            quant_config.get_layer_quant_config(f"{prefix}.out_proj")
            if quant_config is not None
            else None
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            dtype=config.dtype,
            quant_config=norm_quant_config,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.out_proj",
        )

        self.attn = LinearAttention(
            self.hidden_size,
            self.num_v_heads,
            self.num_k_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.key_dim,
            self.value_dim,
            dt_bias=self.dt_bias,
            A_log=self.A_log,
            conv1d=self.conv1d,
            activation=self.activation,
            layer_num=extract_layer_index(self.prefix),
            prefix=self.prefix,
        )

    def create_qkvzba_proj(self, quant_config, prefix):
        # This projection fusion should only opened when model type is bfloat16
        if self.quant_config.global_quant_config.quant_dtype == torch.bfloat16:
            self.in_proj_qkvzba = QKVZBAParallelLinear(
                input_size=self.hidden_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{self.prefix}.in_proj_qkvzba",
            )
        else:
            # Non-fused path (FP8): deinterleave weights + scales at load
            # time so the forward can use zero-copy torch.split.
            self.in_proj_qkvz = QKVZParallelLinear(
                input_size=self.hidden_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
            )
            self.in_proj_ba = BAParallelLinear(
                input_size=self.hidden_size,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_ba",
            )

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = (
            rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim) for x in (query, key)
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        x_fp8=None,
        x_scale=None,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """

        num_tokens = hidden_states.shape[0]
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        v_heads_tp = self.num_v_heads // self.tp_size
        qkv_size = self.conv_dim // self.tp_size
        z_size = v_heads_tp * self.head_v_dim
        b_size = v_heads_tp
        a_size = v_heads_tp

        if hasattr(self, "in_proj_qkvzba"):
            projected = self.in_proj_qkvzba(hidden_states)
            # Output layout is [q|k|v|z|b|a] contiguous (deinterleaved at load)
            mixed_qkv, z_flat, b, a = torch.split(
                projected, [qkv_size, z_size, b_size, a_size], dim=-1
            )
            z = z_flat.view(num_tokens, v_heads_tp, self.head_v_dim)
        else:
            if x_fp8 is not None:
                projected_qkvz = self.in_proj_qkvz(x_fp8, x_scale=x_scale)
            else:
                projected_qkvz = self.in_proj_qkvz(hidden_states)
            projected_ba = self.in_proj_ba(hidden_states)  # always BF16
            # Weights deinterleaved at load → output is [q|k|v|z] and [b|a]
            mixed_qkv = projected_qkvz[:, :qkv_size]
            z = projected_qkvz[:, qkv_size:].view(
                num_tokens, v_heads_tp, self.head_v_dim
            )
            b = projected_ba[:, :b_size]
            a = projected_ba[:, b_size:]

        core_attn_out = torch.empty(z.shape, dtype=z.dtype, device=z.device)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================

        core_attn_out = self.attn(mixed_qkv, b, a, core_attn_out)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================

        core_attn_out, maybe_scale = self.norm(core_attn_out, z)
        output = self.out_proj(core_attn_out, x_scale=maybe_scale)
        return output


class Qwen3NextDecoderLayer(nn.Module):
    # Set by Qwen3_5DecoderLayer: the attention skips its last projection, and
    # forward runs it with post_attention_layernorm in one aiter kernel.
    fuse_gemm_ar = False
    # Set by the MTP drafter (ATOM_MTP_SELECT_ROWS_BEFORE_O_PROJ): `select_rows`
    # is applied right after the attention, before o_proj (all row-wise).
    select_rows_before_o_proj = False

    def __init__(
        self,
        atom_config,
        layer_type: str,
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()

        if hasattr(atom_config.hf_config, "text_config"):
            config = atom_config.hf_config.text_config
        else:
            config = atom_config.hf_config
        quant_config = atom_config.quant_config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
                atom_config,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                atom_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        self.hidden_size = config.hidden_size
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        self.layer_idx = layer_num

        # `mlp_only_layers` in the config.
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                atom_config.hf_config,
                atom_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
                prefix=f"{prefix}.mlp",
            )

        if self.layer_type == "full_attention":
            input_norm_quant = (
                quant_config.get_layer_quant_config(f"{prefix}.self_attn.qkv_proj")
                if quant_config is not None
                else None
            )
            input_norm_write_bf16 = False
        elif self.layer_type == "linear_attention":
            input_norm_quant = (
                quant_config.get_layer_quant_config(
                    f"{prefix}.linear_attn.in_proj_qkvz"
                )
                if quant_config is not None
                else None
            )
            input_norm_write_bf16 = True  # in_proj_ba needs BF16
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            quant_config=input_norm_quant,
            write_bf16=input_norm_write_bf16,
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = atom_parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                )
            )
            self.ffn_layer_scale = atom_parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.dtype,
                )
            )

    def _fold_postnorm_proj(self, hidden_states, residual, select_rows):
        """ATOM_FOLD_POSTNORM: the projection (GDN out_proj / attention o_proj) to ask for the residual + ssq
        epilogue on this call, or None."""
        from atom.model_ops import fold_postnorm

        if (
            not fold_postnorm.ENABLED
            or self.fuse_gemm_ar
            or self.layer_scale
            or not isinstance(self.mlp, Qwen3NextSparseMoeBlock)
            or getattr(self.mlp, "shared_expert", None) is None
            or "mtp" in getattr(self.mlp, "prefix", "")  # the verifier only (not the MTP drafter's layer)
            or not isinstance(self.post_attention_layernorm, GemmaRMSNorm)
            or getattr(self.mlp, "combine_tail_norm", None) is not None
            or getattr(self.mlp, "mono_norm", None) is not None
            or not envs.ATOM_FOLD_TOPK_ROUTER
            or not envs.ATOM_ROUTER_SHARED_V2
            or torch.compiler.is_compiling()
        ):
            return None
        # the pre-scaled router / gate_up weights must exist before a graph captures the fold: built on the
        # first eager call of any size (the startup dummy / warmup runs), never inside a capture
        if fold_postnorm.prescaled(self.mlp, self.post_attention_layernorm) is None:
            return None
        if (
            residual is None
            or select_rows is not None
            or residual.dim() != 2
            or residual.shape[0] > fold_postnorm.MAX_ROWS
        ):
            return None
        if self.layer_type == "linear_attention":
            proj = getattr(self.linear_attn, "out_proj", None)
        elif self.layer_type == "full_attention":
            if getattr(self.self_attn, "skip_out_proj", False):
                return None
            proj = getattr(self.self_attn, "o_proj", None)
        else:
            return None
        if proj is None or not mxfp8_1x32.takes_mx32(proj):
            return None
        return proj

    def _input_norm_mx32(self, hidden_states, residual):
        """ATOM_MXFP8_1X32 with its fused act quant: the MoE-tail RMSNorm writing this layer's
        first projection's (qkv_proj / fused GDN in_proj) 1x32 MXFP8 input --
        `(x_fp8, x_scale, residual)` -- or None where it does not apply (then the norm runs in
        bf16 and the Linear quantizes: the same bytes). A deferred gather (`GatheredRouted`,
        ATOM_MOE_GATHER_TAIL_NORM) goes to the gather + tail kernel's 1x32 epilogue."""
        from atom.model_ops.embed_norm_mx32 import EmbedNormMx32

        if isinstance(hidden_states, EmbedNormMx32):
            # ATOM_EMBED_NORM0_MX32: layer 0's embedding + norm + quant ran as one launch
            return tuple(hidden_states)
        if residual is None or not isinstance(hidden_states, MoEGatedSum):
            return None
        if not mxfp8_1x32.fused_act_quant():
            return None
        norm = self.input_layernorm
        if not isinstance(norm, GemmaRMSNorm) or not norm.takes_moe_gated_sum:
            return None
        if self.layer_type == "linear_attention":
            # only a GDN whose forward feeds x_fp8 / x_scale to its fused in_proj (Qwen3.5)
            if not getattr(self.linear_attn, "takes_mx32_input", False):
                return None
            proj = getattr(self.linear_attn, "in_proj_qkvzba", None)
        elif self.layer_type == "full_attention":
            proj = getattr(self.self_attn, "qkv_proj", None)
        else:
            return None
        if not mxfp8_1x32.takes_mx32(proj):
            return None
        if isinstance(hidden_states.routed, GatheredRouted):
            if not can_fuse_gather_tail_mx32(hidden_states, residual, norm.weight):
                return None
            return moe_gather_tail_rmsnorm_mx32(
                hidden_states, residual, norm.weight, norm.variance_epsilon
            )
        if not can_fuse_moe_tail_mx32(hidden_states, residual, norm.weight):
            return None
        routed, shared, gate = hidden_states
        return moe_tail_rmsnorm_mx32(
            routed, shared, gate, residual, norm.weight, norm.variance_epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        select_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`select_rows`: after the attention (which every row needs: it writes
        their KV), keep only these rows for the MLP; the layer then returns
        `len(select_rows)` rows (the MTP drafter's step 0). With
        `select_rows_before_o_proj` they are kept before o_proj already."""
        rows_before_o_proj = (
            select_rows
            if select_rows is not None
            and self.select_rows_before_o_proj
            and self.layer_type == "full_attention"
            else None
        )
        # Self Attention
        mx32_in = self._input_norm_mx32(hidden_states, residual)
        if mx32_in is not None:
            hidden_states, x_scale, residual = mx32_in
            hidden_bf16 = None
        elif self.input_layernorm.use_fused_quant:
            if residual is None:
                residual = hidden_states
                hidden_states, x_scale, hidden_bf16 = self.input_layernorm(
                    hidden_states
                )
            else:
                hidden_states, x_scale, hidden_bf16, residual = self.input_layernorm(
                    hidden_states, residual
                )
        else:
            x_scale = hidden_bf16 = None
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        pn_proj = self._fold_postnorm_proj(hidden_states, residual, select_rows)
        if pn_proj is not None:
            from atom.model_ops import fold_postnorm

            fold_postnorm.request(pn_proj, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=(
                    hidden_bf16 if hidden_bf16 is not None else hidden_states
                ),
                x_fp8=hidden_states if x_scale is not None else None,
                x_scale=x_scale,
            )
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                x_scale=x_scale,
                **(
                    {"select_rows": rows_before_o_proj}
                    if rows_before_o_proj is not None
                    else {}
                ),
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        pn_ssq = None
        if pn_proj is not None:
            from atom.model_ops import fold_postnorm

            pn_ssq = fold_postnorm.take(pn_proj)
        # Fully Connected
        if pn_ssq is not None:
            # ATOM_FOLD_POSTNORM: the projection added the residual; the norm goes to the MoE block's router
            residual = hidden_states
            hidden_states = fold_postnorm.PreNormed(
                hidden_states, pn_ssq, self.post_attention_layernorm
            )
        elif self.fuse_gemm_ar:
            hidden_states, residual = gluon_gemm_allreduce_gemma_rmsnorm(
                hidden_states.contiguous(),
                self.gemm_ar_proj.weight,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
                # The kernel gathers the residual's rows.
                *(() if rows_before_o_proj is None else (rows_before_o_proj,)),
            )
        else:
            if rows_before_o_proj is not None:
                residual = torch.index_select(residual, 0, rows_before_o_proj)
            # ATOM_MTP_ROWS_IN_POST_ATTN_NORM: the step-0 row selection in the norm's launch
            rows_norm = None
            if select_rows is not None and rows_before_o_proj is None:
                from atom.model_ops import draft_rows_norm

                if draft_rows_norm.ENABLED:
                    rows_norm = draft_rows_norm.try_post_attn_norm_rows(
                        self.post_attention_layernorm,
                        hidden_states,
                        residual,
                        select_rows,
                    )
            if rows_norm is not None:
                hidden_states, residual = rows_norm
                select_rows = None  # already selected
            else:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )

        if select_rows is not None and rows_before_o_proj is None:
            hidden_states = torch.index_select(hidden_states, 0, select_rows)
            residual = torch.index_select(residual, 0, select_rows)
        if (
            getattr(self.mlp, "combine_tail_norm", None) is not None
            or getattr(self.mlp, "mono_norm", None) is not None
        ):
            # The MoE combine (ATOM_MEGA_COMBINE_TAIL) or the fused MoE kernel
            # (ATOM_QWEN_MONO_MOE) runs the next norm too.
            hidden_states = self.mlp(hidden_states, residual)
        else:
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@token_graph
@support_torch_compile
class Qwen3NextModel(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = atom_config.hf_config
        self.config = config
        self.config.n_shared_experts = 1
        self.config.n_routed_experts = self.config.num_experts

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Qwen3NextDecoderLayer(
                atom_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )
        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        defer_moe_gated_sums(self.layers, self.norm)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    # For vllm compatibility
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _embed_first(self, input_ids: torch.Tensor):
        """Layer 0's input: the embedding rows, or (ATOM_EMBED_NORM0_MX32=1, default off) the
        embedding + layer 0's input norm + its 1x32 MXFP8 quant from one launch
        (embed_norm_mx32.EmbedNormMx32, taken by layer 0's `_input_norm_mx32`). Same bytes."""
        from atom.model_ops import embed_norm_mx32

        if embed_norm_mx32.enabled():
            fused = self._embed_norm0_mx32(input_ids)
            if fused is not None:
                return fused
        return self.get_input_embeddings(input_ids)

    def _embed_norm0_mx32(self, input_ids: torch.Tensor):
        from atom.model_ops import embed_norm_mx32
        from atom.model_ops.layernorm import gluon_gemma_rmsnorm

        embed = self.embed_tokens
        if (
            self.start_layer != 0
            or not isinstance(embed, VocabParallelEmbedding)
            or embed.tp_size != 1
            or not mxfp8_1x32.fused_act_quant()
            or gluon_gemma_rmsnorm() is None
        ):
            return None
        layer = self.layers[0]
        norm = layer.input_layernorm
        if (
            not isinstance(norm, GemmaRMSNorm)
            or norm.use_fused_quant
            or norm.fused_allreduce
        ):
            return None
        if layer.layer_type == "linear_attention":
            if not getattr(layer.linear_attn, "takes_mx32_input", False):
                return None
            proj = getattr(layer.linear_attn, "in_proj_qkvzba", None)
        elif layer.layer_type == "full_attention":
            proj = getattr(layer.self_attn, "qkv_proj", None)
        else:
            return None
        if not mxfp8_1x32.takes_mx32(proj):
            return None
        return embed_norm_mx32.embed_norm_mx32(
            input_ids, embed.weight, norm.weight.data, norm.variance_epsilon
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self._embed_first(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if isinstance(hidden_states, MoENormed):
            return hidden_states.out  # the MoE combine ran the final norm
        if isinstance(hidden_states, MoEGatedSum):
            # The last MoE block's gated add folded into the final norm, which
            # skips the residual store nobody reads.
            from atom.model_ops import model_out_buffer

            return self.norm.forward_moe_tail(
                hidden_states,
                residual,
                write_residual=False,
                out=model_out_buffer.take(
                    residual.shape[0],
                    residual.shape[1],
                    residual.dtype,
                    residual.device,
                ),
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.config.n_shared_experts,
        )


class Qwen3NextForCausalLM(nn.Module):
    packed_modules_mapping: ClassVar[dict[str, tuple[str, str | int]]] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        ".gate.": (".gate.", 0),
        "shared_expert_gate": ("gate", 1),
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        config = atom_config.hf_config
        quant_config = atom_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if self.quant_config.global_quant_config.quant_dtype == torch.bfloat16:
            self.packed_modules_mapping["in_proj_qkvz"] = ("in_proj_qkvzba", "qkvz")
            self.packed_modules_mapping["in_proj_ba"] = ("in_proj_qkvzba", "ba")
        # Only perform the following mapping when Qwen3NextMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = Qwen3NextModel(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Expose embed_tokens at this level for vLLM MTP embedding sharing.
        # vLLM's proposer accesses target_wrapper.model.embed_tokens, where
        # target_wrapper.model = this class (Qwen3NextForCausalLM).
        self.embed_tokens = self.model.embed_tokens

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.lm_head(hidden_states)
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
