# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""V4.1 DSpark math and checkpoint layout; context storage is caller-owned."""

import logging
from copy import copy

import torch
import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.model_loader.weight_names import WeightsMapper
from atom.model_ops.blockscale import (
    native_quant_linear,
    quantize_fp8,
    quantize_fp8_taps,
)
from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE
from atom.model_ops.deepseek_v41.draft_block import DraftStep, draft_step, rotate_rows
from atom.model_ops.deepseek_v41.dspark import (
    DraftSrcBuffers,
    build_block_state,
    build_peer_block_state,
    draft_attention,
    draft_attention_fp8,
    draft_src_indices,
    draft_step_indices,
    fused_draft_kv_tail,
)
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.model_ops.deepseek_v41.rope_window import rope_quant_window
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.dspark_markov_argmax import (
    dspark_markov_argmax_tp,
    markov_weights_unsupported,
)
from atom.model_ops.embed_head import VocabParallelEmbedding, masked_embedding
from atom.model_ops.layernorm import RMSNorm, rmsnorm2d_fwd_
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.model_ops.peer_embedding import PeerEmbedding, shared_peer_embedding
from atom.models.deepseek_v4 import make_v4_quant_config
from atom.models.deepseek_v4_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    _DSparkInner,
)
from atom.models.dspark_draft import DSparkDraftModel, VocabShard
from atom.mono.runtime.consensus import MonoUnsupported
from atom.spec_decode.drafter import AuxCaptureSpec
from atom.utils import envs

from .attention import Attention
from .config import FP4PTPC_RECIPE, build_attention_topology
from .layers import (
    converted_linear_spec,
    native_quant_config,
    routed_experts_are_ptpc,
)
from .model import Block, DeepseekV41ForCausalLM

logger = logging.getLogger("atom")


class DraftAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wkv_shard = None  # built by `wkv_shard` on first use

    def context_keys(self, kv_normed, positions, rope, *, packed=False):
        """The context row alone, for a caller with no query to rotate with it.

        Takes the latent already normed, because its one caller normed it. A
        draft step's own rows go through `forward`, which rotates and quantizes
        them beside the query in a single launch instead.
        """
        keys = rotate_rows(rope, kv_normed, positions)
        # Unlike V4's mixed NoPE/RoPE layout, V4.1 QAT covers all head lanes.
        return quantize_fp8(keys, dequantize=not packed)

    @property
    def wkv_shard(self):
        """`wqkv_a` narrowed to the KV rows, for the fused context-KV GEMM."""
        if self._wkv_shard is None:
            # A view of `wqkv_a`, not a submodule: kept out of `_modules`.
            object.__setattr__(self, "_wkv_shard", self.wqkv_a.shard_view(1))
        return self._wkv_shard

    def project_context(self, hidden, positions, rope, *, packed=False):
        """Target keys alone, only on the unfused path -- when the stages
        fuse, `write_context_kv` reads `wkv_shard`. A BF16 `wqkv_a` projects
        only the KV rows; an FP8 one computes the query half and drops it.
        """
        if self.wqkv_a.runs_bf16:
            kv_pre = self.wkv_shard(hidden)
        else:
            _, kv_pre = self.project_qkv(hidden)
        return self.context_keys(self.kv_norm(kv_pre), positions, rope, packed=packed)

    def forward(self, hidden, hidden_scale, context_kv, step, rope, *, hidden_fp8=None):
        q_lora, kv_pre = self.project_qkv(
            hidden if hidden_fp8 is None else hidden_fp8, hidden_scale
        )
        qr, qr_scale, kv_normed = self.qk_norm(q_lora, kv_pre)
        query = self.wq_b(qr, x_scale=qr_scale).unflatten(
            -1, (self.heads, self.head_dim)
        )
        context = context_kv[self.spec.layer_id]
        if isinstance(context, tuple):
            if context[0].shape[-1] != self.head_dim:
                raise ValueError("A8W8 draft history must contain MXFP8 rows")
            if not getattr(self, "_draft_attention_announced", False):
                logger.info(
                    "V4.1 draft layer %d: MXFP8 history, MXFP8 current KV, "
                    "opus A8W8 attention",
                    self.spec.layer_id,
                )
                self._draft_attention_announced = True
            rows = kv_normed.shape[:-1].numel()
            keys = (
                torch.empty_like(kv_normed, dtype=FP8_DTYPE),
                torch.empty(
                    (*kv_normed.shape[:-1], self.head_dim // 32),
                    dtype=torch.float8_e8m0fnu,
                    device=query.device,
                ),
            )
            query_fp8 = (
                torch.empty_like(query, dtype=FP8_DTYPE),
                torch.empty(
                    (*query.shape[:-1], self.head_dim // 32),
                    dtype=torch.float8_e8m0fnu,
                    device=query.device,
                ),
            )
            rope_quant_window(
                query.view(rows, self.heads, self.head_dim),
                kv_normed.view(rows, self.head_dim),
                rope.cos_cache,
                rope.sin_cache,
                step.positions.flatten(),
                rope_dim=rope.rope_dim,
                values=keys[0].view(rows, self.head_dim),
                scales=keys[1].view(rows, self.head_dim // 32),
                query_fp8=query_fp8,
            )
            output = draft_attention_fp8(
                query_fp8, context, keys, self.attn_sink, step, self.softmax_scale
            )
            return self._project_out(output, rope, step.positions)
        # Both rotations and the quantized row in one launch.
        keys = torch.empty_like(kv_normed)
        rows = kv_normed.shape[:-1].numel()
        rope_quant_window(
            query.view(rows, self.heads, self.head_dim),
            kv_normed.view(rows, self.head_dim),
            rope.cos_cache,
            rope.sin_cache,
            step.positions.flatten(),
            rope_dim=rope.rope_dim,
            qat=keys.view(rows, self.head_dim),
        )
        output = draft_attention(
            query,
            context_kv[self.spec.layer_id],
            keys,
            self.attn_sink,
            step,
            self.softmax_scale,
        )
        return self._project_out(output, rope, step.positions)


class DraftBlock(Block):
    attention_cls = DraftAttention

    def __init__(
        self,
        config,
        spec,
        stage,
        prefix: str = "",
        *,
        moe_quant_config,
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__(
            config,
            spec,
            prefix=prefix,
            moe_quant_config=moe_quant_config,
            alt_stream=alt_stream,
        )
        if stage == 0:
            self.main_proj = ReplicatedLinear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                quant_config=native_quant_config(),
            )
            self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if stage == config.num_nextn_predict_layers - 1:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(
                config.vocab_size, config.dspark_markov_rank
            )
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size, config.dspark_markov_rank
            )


class DeepseekV41DSpark(DSparkDraftModel):
    # Same checkpoint and the same layer types as the backbone, so its rules are
    # referenced rather than restated and the two cannot drift apart. The rule
    # on top is the draft's alone: this checkpoint names the Markov tables after
    # the modules that once held them, V4's head calls them markov_w1 / w2.
    weights_mapper = DeepseekV41ForCausalLM.weights_mapper | WeightsMapper(
        orig_to_new_substr={
            ".markov_head.embed.": ".markov_head.markov_w1.",
            ".markov_head.head.": ".markov_head.markov_w2.",
        }
    )
    weights_mapping = DeepseekV41ForCausalLM.weights_mapping
    packed_modules_mapping = DeepseekV41ForCausalLM.packed_modules_mapping
    disable_fused_shared_loading = DeepseekV41ForCausalLM.disable_fused_shared_loading

    def __init__(self, config, *, max_length=None, alt_stream=None):
        super().__init__()
        args = config.hf_config
        if args.num_nextn_predict_layers < 1 or args.dspark_block_size < 1:
            raise ValueError(
                "V4.1 DSpark needs draft stages and a positive block width"
            )
        self.config = args
        self.draft_kv_dtype = config.dspark.draft_kv_dtype
        self.block_size = args.dspark_block_size
        self.window_size, self.vocab_size = args.sliding_window, args.vocab_size
        # Sized at the engine's batch bound, as the backbone's index buffers are.
        self.draft_src_buffers = DraftSrcBuffers(config.max_num_seqs, self.block_size)
        draft = copy(args)
        draft.n_routed_experts = args.dspark_n_routed_experts
        draft.num_experts_per_tok = args.dspark_num_experts_per_tok
        # An FP4PTPC checkpoint converts only the backbone's linears; the
        # draft's stay native.
        draft.fp4ptpc = False
        # Same rules as the backbone.
        self.moe_quant_config = make_v4_quant_config(
            draft,
            online_quant_config=getattr(config, "online_quant_config", None),
            shared_expert_spec=converted_linear_spec(draft),
            online_routed_experts=True,
        )
        # Read by the loader's shared-expert rename (`_fuse_shared_expert`).
        self.quant_config = self.moe_quant_config
        topology = build_attention_topology(args)[args.num_hidden_layers :]
        if args.fp4ptpc and not all(
            routed_experts_are_ptpc(self.moe_quant_config, f"mtp.{i}.ffn.experts")
            for i in range(len(topology))
        ):
            # The draft's FP4 experts run only converted, as the backbone's.
            raise ValueError(f"mtp: {FP4PTPC_RECIPE}")
        # The backbone's, handed down: the two models never run at once, so a
        # stream of its own would buy nothing. Absent when built offline,
        # which is also where nothing is fast enough to care.
        self.alt_stream = alt_stream
        self.mtp = nn.ModuleList(
            # The stage index, not the topology's layer id: this prefix names
            # the module's own parameters, and `nn.ModuleList` numbers them
            # from zero -- which is also how the checkpoint numbers them.
            DraftBlock(
                draft,
                spec,
                i,
                prefix=f"mtp.{i}",
                moe_quant_config=self.moe_quant_config,
                alt_stream=self.alt_stream,
            )
            for i, spec in enumerate(topology)
        )
        capacity = max_length or getattr(
            config, "max_model_len", args.max_position_embeddings
        )
        self.rope = RotaryEmbedding(
            args.qk_rope_head_dim, capacity + self.block_size, base=args.rope_theta
        )
        self.embed = self.head = None
        self._unbind_embedding()

    @property
    def model(self):
        """The one indirection the inherited loader attributes reach through."""
        return self

    def remap_mtp_weight_name(self, name: str) -> str | None:
        """Keep the draft's own stages; drop the rest of the checkpoint.

        The drafter is loaded from the same file as the target, in a second
        pass over every tensor in it. Stage names survive `weights_mapper`
        untouched, so the remap is identity -- what this is really for is the
        `None`, which is how the loader is told a tensor belongs to somebody
        else rather than that it failed to route.
        """
        return name if name.startswith("mtp.") else None

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Same mapping as the backbone, over the draft's smaller expert set."""
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.dspark_n_routed_experts,
        )

    def share_with_target(self, target_base, loaded=None):
        self.embed, self.head = target_base.embed, target_base.head
        self._unbind_embedding()

    def _unbind_embedding(self) -> None:
        self._embedding_bound = False
        self._embed_peer: PeerEmbedding | None = None
        self._noise_row: torch.Tensor | None = None

    @torch.no_grad()
    def _bind_embedding(self) -> None:
        """Where the draft reads the shared embedding's rows, and the noise
        row, resolved once on the first draft step outside a graph capture
        (a warmup forward), as `LazyRunner` binds mono.

        A vocab-parallel bf16 table is read from its owner ranks' shards
        (`shared_peer_embedding`, the mapping mono's draft also reads):
        the anchors in `build_peer_block_state`, no lookup and no all-reduce.
        The noise row is one fixed row every draft path reuses (this block,
        hi's, mono's without the peer read); it is never taken through the
        custom all-reduce, which returns zeros during a graph warmup.
        """
        if self._embedding_bound:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "V4.1 DSpark binds its embedding on an eager draft step, "
                "not under graph capture"
            )
        embed = self.embed
        sharded = isinstance(embed, VocabParallelEmbedding) and embed.tp_size > 1
        if sharded and embed.weight.dtype == torch.bfloat16:
            try:
                self._embed_peer = shared_peer_embedding(
                    embed, get_tp_group().cpu_group
                )
            except MonoUnsupported as why:
                logger.warning("V4.1 DSpark: peer embedding off: %s", why)
        ids = torch.full(
            (1,),
            self.config.dspark_noise_token_id,
            dtype=torch.long,
            device=embed.weight.device,
        )
        if self._embed_peer is not None:
            row = self._embed_peer(
                ids, embed.weight.new_empty(1, embed.weight.shape[1])
            )
        elif sharded:
            row = masked_embedding(
                ids, embed.weight, embed.vocab_start_idx, embed.vocab_end_idx
            )
            dist.all_reduce(row, group=get_tp_group().device_group)
        else:
            row = embed(ids)
        self._noise_row = row.flatten()
        self._embedding_bound = True

    def target_aux_capture_spec(self, layer_ids, hidden_size, _target_layers=None):
        # Published taps (37, 38, 39) have no Engram. A block pre-hook at an
        # Engram layer would read before its injection, violating the checkpoint.
        if set(layer_ids).intersection(self.config.engram_layer_ids):
            raise ValueError("V4.1 DSpark input taps cannot precede Engram injection")
        return AuxCaptureSpec(
            layer_ids,
            hidden_size,
            self._target_layer_input,
            capture="input",
            writes_into=True,
        )

    @staticmethod
    def _target_layer_input(inputs, block, out_for=None):
        # Settled, because a pre-hook runs before the fold: the block ahead of
        # this one leaves its FFN post owed, and `residual` is by definition
        # the value before it. The block still pays that post into its own pre;
        # the tap reads it with the mean (`stream_mean`), in one launch where
        # `post_reduce_supported` holds, without writing the settled residual,
        # and writes the capture's own rows when the hook offers them (`out_for`).
        state = inputs[0]
        hidden = state.residual.shape[-1]
        out = None
        if out_for is not None:
            out = out_for(state.residual.numel() // (state.residual.shape[-2] * hidden))
        return state.stream_mean(out=out).reshape(-1, hidden)

    def project_context(self, taps):
        """`main_norm(main_proj(concat(taps)))` as `[1, N, hidden]`."""
        return self.mtp[0].main_norm(self._project_taps(taps))

    def _project_taps(self, taps):
        """`main_proj(concat(taps))` as `[1, N, hidden]`, before the norm.

        An FP8 `main_proj` quantizes its input with `quantize_fp8`; the taps
        are quantized where they lie instead, to the same fp8 rows and scales,
        minus the concatenated copy. `quantize_fp8_taps` joins them itself for
        a shape its kernel does not take, and passes a single joined block
        (a PP relay) through as it is. A BF16 `main_proj` (no group32 FP8
        GEMM) takes the joined taps unquantized.
        """
        main_proj = self.mtp[0].main_proj
        if main_proj.runs_bf16:
            return main_proj(self.concat_taps(taps).unsqueeze(0))
        x, x_scale = quantize_fp8_taps(taps)
        return main_proj(x.unsqueeze(0), x_scale=x_scale.unsqueeze(0))

    @property
    def context_layers(self):
        return tuple(layer.attn for layer in self.mtp)

    def write_context_kv(self, taps, positions, slot_mapping=None):
        from atom.utils.forward_context import get_forward_context

        forward = get_forward_context()
        if forward.context.is_dummy_run:
            return
        metadata = forward.attn_metadata
        cache, step = metadata.cache, metadata.step
        plan = step.replay
        if plan is not None:
            # A replayed forward ran the decoder, whose rows these come from,
            # on its tail alone: the draft's window takes the same rows.
            if not plan.tail.requests:
                return
            rows, step = plan.tail_rows, plan.tail
            taps, positions = [tap[rows] for tap in taps], positions[rows]
        # The forward's own width, padding included, because that is what the
        # target just ran and what these buffers therefore hold. A padding row
        # belongs to a zero-length request in `cu_seqlens_q`, so it is
        # projected and then written nowhere. The rows are cut here, as views,
        # and `project_context` takes every row it is given.
        tokens = step.running_tokens
        taps = [tap[:tokens] for tap in taps]
        positions = positions[:tokens][None]
        layers = self.context_layers
        fused = self._context_kv_fusion()
        if fused is None:
            hidden = self.project_context(taps)
            for attention in layers:
                keys = attention.project_context(
                    hidden, positions, self.rope, packed=cache.window_fp8
                )
                cache.write_window(attention.spec.layer_id, keys, step)
            return
        weight, weight_scale, group_rows, width = fused
        # main_norm lands in bf16 and the GEMM quantizes that, as the original
        # model rounds.
        kv = native_quant_linear(
            self.project_context(taps),
            weight,
            weight_scale,
            weight_group_rows=group_rows,
            weight_preshuffled=weight.is_shuffled,
        ).unflatten(-1, (len(layers), width))
        tail = self._draft_kv_tail_fusion()
        if tail is None:
            # Concatenating the per-stage norms puts the stage axis in front,
            # which buys two things: each stage's rows become contiguous again
            # (both window writers address rows by width, so a strided stage
            # would be written wrong), and the stage axis is then exactly what
            # RoPE treats as its batch -- `_rotate_cuda` repeats `positions`
            # once per batch row, which is the per-stage rotation, unrolled.
            kv = torch.cat(
                [
                    rmsnorm2d_fwd_(
                        kv[..., i, :], a.kv_norm.weight, a.kv_norm.eps, width
                    )
                    for i, a in enumerate(layers)
                ]
            )
            keys = quantize_fp8(
                self.rope(kv, positions.flatten()), dequantize=not cache.window_fp8
            )
        else:
            # Same stage-major result, one kernel: the cat is not fused but
            # deleted, each program writing its row straight to its stage slot,
            # or, when the stages' fp8 windows share a ring, to its window.
            norm_weight, eps = tail
            window = cache.window_target([a.spec.layer_id for a in layers], step)
            keys = fused_draft_kv_tail(
                kv,
                norm_weight,
                positions.flatten(),
                self.rope.cos_cache,
                self.rope.sin_cache,
                eps,
                packed=cache.window_fp8,
                window=window,
                step=step,
            )
            if window is not None:
                return
        for i, attention in enumerate(layers):
            cache.write_window(
                attention.spec.layer_id,
                (
                    tuple(t[i : i + 1] for t in keys)
                    if cache.window_fp8
                    else keys[i : i + 1]
                ),
                step,
            )

    def _context_kv_fusion(self):
        """The stages' `wkv` weights concatenated once, or None to stay per stage.

        Every stage's `wkv` reads the same projected hidden and emits the same
        `head_dim` width, so the weights concatenate along the output dim and
        one GEMM produces all of them. The norms stay per stage -- their
        weights differ, and hoisting them out by normalizing with a unit weight
        and scaling after costs a second bf16 rounding that measurably loses to
        the single fused one (a quarter of the elements move, every one of them
        away from an fp64 reference). So this fuses only what is bit-exact:
        measured against the per-stage chain, both cache layouts reproduce it
        exactly, stage for stage.

        This removes host work, not FLOPs. `write_context_kv` runs eager -- the
        draft KV write is not in the graph -- and at decode widths most of its
        wall time is the Python between the ops rather than the ops: one GEMM,
        one RoPE and one quantize replace three of each, and the shared
        activation is quantized once instead of once per stage.

        Measured on MI355X (TP=4, DSpark, 5 spec tokens): the eager region's
        median host span falls 1485us -> 993us and its top-level op count
        169 -> 96, which moves median TPOT 3.35ms -> 3.20ms. Mind that gap.
        This decode loop is GPU-bound -- compute kernels cover ~91% of it and
        collectives ~1% -- so cutting host work pays only where it was exposed,
        and a launch-count fix here is worth a few percent, not more.
        """
        fused = self.__dict__.get("_context_kv_fused", False)
        if fused is not False:
            return fused
        layers = self.context_layers
        # A caller may hand this model stand-in layers that carry only
        # `project_context` -- the per-stage path needs nothing else, so a
        # missing projection means "do not fuse", not "crash". The KV half of
        # `wqkv_a` is what a stage contributes; `shard_view` owns where its
        # rows and its scale rows start.
        if any(getattr(a, "wkv_shard", None) is None for a in layers):
            self._context_kv_fused = None
            return None
        shards = [a.wkv_shard for a in layers]
        # One GEMM needs one layout. A (16, 16)-preshuffled weight permutes
        # only within 16-row groups, so stages that share it concatenate into
        # one in that layout.
        layouts = {
            (
                s.native_a8_group_rows,
                s.input_size,
                s.output_size,
                getattr(s.weight, "is_shuffled", False),
            )
            for s in shards
        }
        (group_rows, _, width, preshuffled), *others = layouts
        # BF16 shards have no FP8 weights to fuse: per-stage path.
        bf16 = any(s.runs_bf16 for s in shards)
        if (
            others
            or group_rows is None
            or bf16
            or any(s.bias is not None for s in shards)
        ):
            fused = None
        else:
            weight = torch.cat([s.weight for s in shards])
            # the concatenation is a new tensor in the stages' layout: it
            # carries their layout flag, as every loaded weight does
            weight.is_shuffled = preshuffled
            fused = (
                weight,
                torch.cat([s.weight_scale for s in shards]),
                group_rows,
                width,
            )
        self._context_kv_fused = fused
        return fused

    def _draft_kv_tail_fusion(self):
        """The stages' norm weights stacked once, or None to keep the op chain.

        What the tail kernel needs that the GEMM fusion above does not: norms
        whose math it inlines (ATOM's RMSNorm, not a Gemma-style `x * (1 + w)`),
        one shared eps, a rope whose cached frequencies cover the row's tail
        lanes, and a row width its group-32 quantizer and `tl.arange` can take.
        Each stage keeps its own weight -- that is the axis the kernel indexes,
        and hoisting the weights out is the rounding the sibling fusion already
        measured as a loss.

        Every one of those is structural for V4.1, so a None means a model this
        file was not written for rather than anything transient -- which is why
        there is no switch: the op chain below is the fallback, not debug code,
        and it stays reachable on its own terms.
        """
        tail = self.__dict__.get("_draft_kv_tail", False)
        if tail is not False:
            return tail
        layers = self.context_layers
        norms = [a.kv_norm for a in layers]
        width = norms[0].weight.shape[-1]
        pe_dim = self.rope.rope_dim
        eps = {n.eps for n in norms}
        if (
            not all(isinstance(n, RMSNorm) for n in norms)
            or len(eps) != 1
            # Stacking would raise rather than fall back, and the stage axis is
            # only an axis if every stage is the same width.
            or any(n.weight.shape[-1] != width for n in norms)
            or width % 32
            or width & (width - 1)
            or pe_dim > width
        ):
            tail = None
        else:
            tail = (torch.stack([n.weight for n in norms]), eps.pop())
        # Resolved once per process and then cached, so one line. Without it a
        # model this file does not recognise would leave the fusion inert with
        # nothing in the log to say so.
        logger.info(
            "DSpark draft KV tail: %s (%d stages x %d, rope %d)",
            "per-op" if tail is None else "FUSED",
            len(norms),
            width,
            pe_dim,
        )
        self._draft_kv_tail = tail
        return tail

    def _context_layer_ids(self, device):
        """The stages' layer ids as one device tensor, built once.

        It indexes the batched window read and never changes, so rebuilding it
        per step would put a host-to-device copy on the drafting path for
        three integers.
        """
        ids = self.__dict__.get("_context_ids")
        if ids is None or ids.device != device:
            ids = torch.tensor(
                [layer.spec.layer_id for layer in self.context_layers],
                dtype=torch.long,
                device=device,
            )
            self._context_ids = ids
        return ids

    def _noise_embedding(self) -> torch.Tensor:
        """The noise token's embedding row (`_bind_embedding`)."""
        self._bind_embedding()
        return self._noise_row

    def block_backbone(self, input_ids, positions, num_draft):
        from atom.utils.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
        cache = metadata.cache
        # Published at running_bs, the width DraftGraph stages anchors at, so
        # window addressing holds no captured Python object.
        slots = metadata.state_slot_out[: input_ids.numel()]
        width = self.block_size if num_draft is None else num_draft
        # One gather for every stage: they read the same rows and differ only
        # in the layer, so indexing them one at a time spent a gather and a
        # `.long()` cast apiece. The slices below are views.
        layer_ids = self._context_layer_ids(slots.device)
        if self.draft_kv_dtype == "fp8":
            windows = cache.read_window_planes(layer_ids, slots)
            context = {
                layer.spec.layer_id: tuple(p[i] for p in windows)
                for i, layer in enumerate(self.context_layers)
            }
        else:
            windows = cache.read_windows(layer_ids, slots)
            context = {
                layer.spec.layer_id: windows[i]
                for i, layer in enumerate(self.context_layers)
            }
        # The slot -> position map and the block's index list are one closed
        # form over (positions, ring, window, width), so they come from a
        # single launch rather than the ~16 torch ops that spelled them out --
        # launch-floor work on [B, ring] rows, four of them an `arange`
        # rebuilding a constant.
        ring = cache.geometry.ring_slots
        step_positions, indices, context_positions = draft_step_indices(
            positions, ring, self.window_size, width
        )
        # One build for whichever draft runs: the FP8 draft always reads these
        # lists (`DraftAttention.forward` picks it by the window's type); the
        # BF16 one only where `two_source_draft()` says, None keeping its
        # concatenating path.
        src = draft_src_indices(
            positions,
            ring,
            self.window_size,
            width,
            self.draft_src_buffers,
            fp8=self.draft_kv_dtype == "fp8",
        )
        return self.draft_hidden(
            input_ids,
            positions,
            context,
            context_positions,
            num_draft=num_draft,
            step=DraftStep(step_positions, indices, src_indices=src),
        )

    def forward_spec(self, input_ids, positions, num_draft=None):
        width = self.block_size if num_draft is None else num_draft
        return self.head_and_sample(
            self.block_backbone(input_ids, positions, width), input_ids, width
        )

    def draft_hidden(
        self,
        anchor_ids,
        anchors,
        context_kv,
        context_positions,
        *,
        num_draft=None,
        step=None,
    ):
        """Pure block math; the caller supplies a committed per-request window.

        `step` lets a caller that already built the block's positions and
        indices hand them over -- `block_backbone` does, from one kernel. An
        FP8 draft reads only the step's two source lists, so it needs a step
        that carries them; `block_backbone` builds them once a round.
        """
        if self.embed is None or self.head is None:
            raise RuntimeError("DSpark must share the loaded target embedding and head")
        width = self.block_size if num_draft is None else num_draft
        if not 1 <= width <= self.block_size:
            raise ValueError("Draft width must fit the published DSpark block")
        # Every column past the anchor is the same noise token, whose row is
        # kept (`_bind_embedding`); only the anchors are read per step.
        noise = self._noise_embedding()
        hc_mult = self.config.hc_mult
        if self._embed_peer is not None:
            block = build_peer_block_state(
                self._embed_peer, anchor_ids.reshape(-1), noise, width, hc_mult
            )
        else:
            block = build_block_state(self.embed(anchor_ids), noise, width, hc_mult)
        state = SinglePassHCState(*block)
        if step is None:
            step = draft_step(context_positions, anchors, width, self.window_size)
        for layer in self.mtp:
            state = layer(state, context_kv, step, self.rope)
        hidden = state.collapse()
        # V4's seam: post-norm flat for `get_logits`, pre-norm still [B, T, dim]
        # because it is the confidence head's h_k and carries the block width.
        return self.mtp[-1].norm(hidden).flatten(0, 1), hidden

    # Adopted whole from V4. `_DSparkInner` is `@support_torch_compile`, so the
    # half it shares with this class cannot move into a common base.
    _head_and_sample = _DSparkInner.head_and_sample
    forward_head = _DSparkInner.forward_head

    # `head_and_sample` draws the block through the bound `draft_sampling`, so
    # `DSparkProposer` may arm lossless sampling here; its drafts are verified
    # as one block.
    supports_lossless_draft = True
    lossless_block_verify = True

    @property
    def markov_head(self):
        """The last stage's Markov head, the one the block is sampled with."""
        return self.mtp[-1].markov_head

    def head_and_sample(self, out, anchor_ids, num_draft, *, greedy=False):
        # `num_draft` is the shared surface's; the width rides `out[1]`.
        if self.draft_sampling is not None and not greedy:
            return self._head_and_sample_q(*out, anchor_ids)
        if self._sharded_markov_sample():
            return self._head_and_sample_sharded(*out, anchor_ids)
        return self._head_and_sample(*out, anchor_ids)

    def _head_and_sample_q(self, normed, hc_hidden, anchor_ids):
        """The lossless block: each x_k drawn from, and its q published to,
        the bound `draft_sampling` (`DSparkMarkovHead.sample_block`). Where the
        greedy block keeps the LM head sharded (`_sharded_markov_sample`), so
        does this one: the ranks exchange tile partials, histograms and their
        share of the stored entries rather than the logits, and store and
        draw what the whole vocab would (`dspark_markov_sample_tp`)."""
        B, T, _ = hc_hidden.shape
        head, markov = self.head, self.mtp[-1].markov_head
        shard = None
        if self._sharded_markov_sample():
            base_logits = tgemm.mm(normed, head.weight).view(B, T, -1)  # [B,T,V/tp]
            shard = VocabShard(
                head.vocab_start_idx, get_tp_group(), envs.ATOM_USE_CUSTOM_ALL_GATHER
            )
        else:
            base_logits = head.get_logits(normed).view(B, T, -1)  # [B, T, V]
        out_ids = markov.sample_block(
            base_logits, anchor_ids, self.draft_sampling, shard
        )
        # The confidence head reads W1[x_{k-1}], which the greedy op hands over
        # and the sampled one does not.
        prev_ids = torch.cat([anchor_ids[:, None], out_ids[:, :-1]], dim=1)
        confidence = self.mtp[-1].confidence_head(hc_hidden, markov.markov_w1(prev_ids))
        return out_ids, confidence

    def _sharded_markov_sample(self) -> bool:
        """Whether the block is sampled off this rank's vocab shard.

        V4's head all-gathers the ``[B * T, V]`` logits for a sampler that only
        wants each position's argmax. With the fused sampler the argmax of a
        shard is exact, so the ranks exchange ``(max, id)`` per position
        instead: T small gathers in place of one of the whole vocab.
        """
        head = self.head
        markov = self.mtp[-1].markov_head
        return (
            head is not None
            and head.tp_size > 1
            and head.bias is None
            and markov.fused_sample
            and markov_weights_unsupported(
                markov.markov_w1.weight, markov.markov_w2.weight
            )
            is None
        )

    def _head_and_sample_sharded(self, normed, hc_hidden, anchor_ids):
        """``_head_and_sample`` with the LM head left sharded (see
        ``_sharded_markov_sample``); the same ids and confidences."""
        B, T, _ = hc_hidden.shape
        head, last = self.head, self.mtp[-1]
        base_logits = tgemm.mm(normed, head.weight).view(B, T, -1)  # [B, T, V/tp]
        tp_group = get_tp_group()
        use_custom = envs.ATOM_USE_CUSTOM_ALL_GATHER
        # Position k's id lands in column k and is position k + 1's previous
        # id; position 0 reads the anchors where they are, not a copy of them.
        out_ids = anchor_ids.new_empty(B, T)
        # Each step writes its W1[x] rows into its column of this block, so
        # the confidence head reads them without a stack.
        markov_w1 = last.markov_head.markov_w1.weight
        markov_embeds = markov_w1.new_empty(B, T, markov_w1.shape[1])
        for k in range(T):
            dspark_markov_argmax_tp(
                base_logits[:, k],
                anchor_ids if k == 0 else out_ids[:, k - 1],
                last.markov_head.markov_w1.weight,
                last.markov_head.markov_w2.weight,
                out_ids[:, k],
                head.vocab_start_idx,
                tp_group,
                use_custom=use_custom,
                embed_out=markov_embeds[:, k],
            )
        confidence = last.confidence_head(hc_hidden, markov_embeds)
        return out_ids, confidence
