# SPDX-License-Identifier: MIT
"""V4.1 DSpark math and checkpoint layout; context storage is caller-owned."""

import logging
from copy import copy

import torch
from torch import nn

from atom.model_loader.weight_names import WeightsMapper
from atom.model_ops.blockscale import native_quant_linear, quantize_fp8
from atom.model_ops.deepseek_v41.draft_block import DraftStep, draft_step, rotate_rows
from atom.model_ops.deepseek_v41.dspark import (
    build_block_state,
    draft_attention,
    draft_step_indices,
    fused_draft_kv_tail,
)
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.model_ops.deepseek_v41.projections import grouped_output_projection
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.layernorm import RMSNorm, rmsnorm2d_fwd_
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.models.deepseek_v4 import make_v4_quant_config
from atom.models.deepseek_v4_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    _DSparkInner,
)
from atom.models.dspark_draft import DSparkDraftModel

from .attention import Attention
from .config import build_attention_topology
from .layers import native_quant_config
from .model import Block, DeepseekV41ForCausalLM

logger = logging.getLogger("atom")


class DraftAttention(Attention):
    def context_keys(self, kv_normed, positions, rope, *, packed=False):
        """Takes the latent already normed: `forward` gets it beside the query
        out of one launch, and the standalone caller norms it itself."""
        keys = rotate_rows(rope, kv_normed, positions)
        # Unlike V4's mixed NoPE/RoPE layout, V4.1 QAT covers all head lanes.
        return quantize_fp8(keys, dequantize=not packed)

    @property
    def wkv_shard(self):
        """`wqkv_a` narrowed to the KV rows, for the fused context-KV GEMM."""
        view = self.__dict__.get("_wkv_shard")
        if view is None:
            view = self._wkv_shard = self.wqkv_a.shard_view(1)
        return view

    def project_context(self, hidden, positions, rope, *, packed=False):
        """Target keys alone; the query half is computed and dropped.

        As V4's draft does, and only on the unfused path -- when the stages
        fuse, `write_context_kv` reads `wkv_shard` and no query is projected.
        """
        _, kv_pre = self.project_qkv(hidden)
        return self.context_keys(self.kv_norm(kv_pre), positions, rope, packed=packed)

    def forward(self, hidden, context_kv, step, rope):
        q_lora, kv_pre = self.project_qkv(hidden)
        qr, qr_scale, kv_normed = self.qk_norm(q_lora, kv_pre)
        query = self.wq_b(qr, x_scale=qr_scale).unflatten(
            -1, (self.heads, self.head_dim)
        )
        query = rotate_rows(rope, query, step.positions)
        keys = self.context_keys(kv_normed, step.positions, rope)
        output = draft_attention(
            query,
            context_kv[self.spec.layer_id],
            keys,
            self.attn_sink,
            step,
            self.softmax_scale,
        )
        output = rotate_rows(rope, output, step.positions, inverse=True)
        output = output.unflatten(-2, (self.groups, -1)).flatten(-2)
        weight = self.wo_a.weight.view(self.groups, self.o_rank, -1)
        return self.wo_b(grouped_output_projection(output, weight).flatten(-2))


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
        args = getattr(config, "hf_config", config)
        if args.num_nextn_predict_layers < 1 or args.dspark_block_size < 1:
            raise ValueError(
                "V4.1 DSpark needs draft stages and a positive block width"
            )
        self.config = args
        self.block_size = args.dspark_block_size
        self.window_size, self.vocab_size = args.sliding_window, args.vocab_size
        draft = copy(args)
        draft.n_routed_experts = args.dspark_n_routed_experts
        draft.num_experts_per_tok = args.dspark_num_experts_per_tok
        self.moe_quant_config = make_v4_quant_config(
            draft, online_quant_config=getattr(config, "online_quant_config", None)
        )
        topology = build_attention_topology(args)[args.num_hidden_layers :]
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

    def target_aux_capture_spec(self, layer_ids, hidden_size):
        from atom.spec_decode.drafter import AuxCaptureSpec

        # Published taps (37, 38, 39) have no Engram. A block pre-hook at an
        # Engram layer would read before its injection, violating the checkpoint.
        if set(layer_ids).intersection(self.config.engram_layer_ids):
            raise ValueError("V4.1 DSpark input taps cannot precede Engram injection")
        return AuxCaptureSpec(
            layer_ids, hidden_size, self._target_layer_input, capture="input"
        )

    @staticmethod
    def _target_layer_input(inputs, block):
        residual = inputs[0].residual
        return residual.mean(dim=-2).reshape(-1, residual.shape[-1])

    def project_context(self, aux_concat):
        first = self.mtp[0]
        return first.main_norm(first.main_proj(aux_concat))

    @property
    def context_layers(self):
        return tuple(layer.attn for layer in self.mtp)

    def write_context_kv(self, aux_concat, positions):
        from atom.utils.forward_context import get_forward_context

        forward = get_forward_context()
        if forward.context.is_dummy_run:
            return
        metadata = forward.attn_metadata
        cache, step = metadata.cache, metadata.step
        # The forward's own width, padding included, because that is what the
        # target just ran and what these buffers therefore hold. A padding row
        # belongs to a zero-length request in `cu_seqlens_q`, so it is
        # projected and then written nowhere.
        hidden = self.project_context(aux_concat[: step.width].unsqueeze(0))
        positions = positions[: step.width][None]
        layers = self.context_layers
        fused = self._context_kv_fusion()
        if fused is None:
            for attention in layers:
                keys = attention.project_context(
                    hidden, positions, self.rope, packed=cache.packed
                )
                cache.write_window(attention.spec.layer_id, keys, step)
            return
        weight, weight_scale, group_rows, width = fused
        kv = native_quant_linear(
            hidden, weight, weight_scale, weight_group_rows=group_rows
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
                self.rope(kv, positions.flatten()), dequantize=not cache.packed
            )
        else:
            # Same stage-major result, one kernel: the cat is not fused but
            # deleted, each program writing its row straight to its stage slot.
            norm_weight, eps = tail
            keys = fused_draft_kv_tail(
                kv,
                norm_weight,
                positions.flatten(),
                self.rope.cos_cache,
                self.rope.sin_cache,
                eps,
                packed=cache.packed,
            )
        for i, attention in enumerate(layers):
            cache.write_window(
                attention.spec.layer_id,
                tuple(t[i : i + 1] for t in keys) if cache.packed else keys[i : i + 1],
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
        first = shards[0]
        shape = {(s.native_a8_group_rows, s.input_size, s.output_size) for s in shards}
        if (
            len(shape) != 1
            or first.native_a8_group_rows is None
            or any(s.bias is not None for s in shards)
        ):
            fused = None
        else:
            fused = (
                torch.cat([s.weight for s in shards]),
                torch.cat([s.weight_scale for s in shards]),
                first.native_a8_group_rows,
                first.output_size,
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
        pe_dim = self.rope.cos_cache.shape[-1] * 2
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

    def _noise_embedding(self):
        """The noise token's embedding row, embedded once and kept.

        Every draft column past the anchor carries `dspark_noise_token_id`, and
        the embedding is a TP all-reduce: embedding it per step per column put
        `width` identical rows through the collective for no reason. The table
        is fixed once the checkpoint is loaded, so the row is too -- resolved on
        the first drafting step rather than at construction, which is when the
        shared target embedding has actually been attached.
        """
        row = self.__dict__.get("_noise_row")
        if row is None:
            ids = torch.full(
                (1,),
                self.config.dspark_noise_token_id,
                dtype=torch.long,
                device=self.embed.weight.device,
            )
            row = self.embed(ids).flatten()
            self._noise_row = row
        return row

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
        step_positions, indices, context_positions = draft_step_indices(
            positions, cache.geometry.ring_slots, self.window_size, width
        )
        return self.draft_hidden(
            input_ids,
            positions,
            context,
            context_positions,
            num_draft=num_draft,
            step=DraftStep(step_positions, indices),
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
        indices hand them over -- `block_backbone` does, from one kernel.
        """
        if self.embed is None or self.head is None:
            raise RuntimeError("DSpark must share the loaded target embedding and head")
        width = self.block_size if num_draft is None else num_draft
        if not 1 <= width <= self.block_size:
            raise ValueError("Draft width must fit the published DSpark block")
        # Only the anchors are embedded: every other column of the block is the
        # same noise token, whose row never changes, so it is embedded once and
        # kept. That takes `width` out of the embedding's TP all-reduce, and
        # the block state is then written straight out rather than broadcast
        # and copied. See `build_block_state`.
        state = SinglePassHCState(
            *build_block_state(
                self.embed(anchor_ids),
                self._noise_embedding(),
                width,
                self.config.hc_mult,
            )
        )
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

    def head_and_sample(self, out, anchor_ids, num_draft):
        # `num_draft` is the shared surface's; the width rides `out[1]`.
        return self._head_and_sample(*out, anchor_ids)
