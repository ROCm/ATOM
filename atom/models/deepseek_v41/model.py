# SPDX-License-Identifier: MIT
"""Full-layer eager text backbone. Checkpoint I/O and request preparation live outside."""

from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_loader.weight_names import WeightsMapper
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.model_ops.deepseek_v41.mhc_pre_delayed import pre_delayed
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.engram.device.layer import EngramOp
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.moe import FusedMoE
from atom.model_ops.utils import atom_parameter
from atom.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    ParallelHead,
    make_v4_quant_config,
)
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

from .attention import Attention
from .config import build_attention_topology
from .layers import native_quant_config
from .moe import MoE


class Block(nn.Module):
    attention_cls = Attention

    def __init__(
        self,
        config,
        spec,
        prefix: str = "",
        *,
        moe_quant_config,
        alt_stream: torch.cuda.Stream | None = None,
        compress_stream: torch.cuda.Stream | None = None,
        index_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.layer_name = f"v41.layers.{spec.layer_id}"
        self.attn = self.attention_cls(
            config, spec, compress_stream=compress_stream, index_stream=index_stream
        )
        # FusedMoE names its parameters from this prefix, so it has to match the
        # module layout used by the shared loader: `layers.N` / `mtp.N`.
        self.ffn = MoE(
            config,
            spec.layer_id,
            prefix=f"{prefix}.ffn",
            quant_config=moe_quant_config,
            alt_stream=alt_stream,
        )
        # Where `wqkv_a` is the norm's only reader, the norm emits the
        # `(e4m3, e8m0 group-32)` pair that GEMM would otherwise have made for
        # itself -- one launch instead of two. A layer whose compressor or
        # indexer also projects this tensor keeps the BF16 it needs.
        self.attn_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            **(
                {}
                if spec.shares_attention_input
                else {"fused_quant": True, "quant_config": native_quant_config()}
            ),
        )
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # `post_mult` is the 2.0 in the post gate's `2 * sigmoid(...)`, which
        # the AITER stages take as a parameter where the torch body has it
        # written in.
        self.hc_options = {
            "rms_eps": config.rms_norm_eps,
            "hc_eps": config.hc_eps,
            "sinkhorn_iters": config.hc_sinkhorn_iters,
            "post_mult": 2.0,
        }
        hc = config.hc_mult
        for sublayer in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", (hc * (hc + 2), hc * config.hidden_size)),
                ("base", (hc * (hc + 2),)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{sublayer}_{suffix}",
                    atom_parameter(torch.empty(shape, dtype=torch.float32)),
                )
        self.engram = None
        if spec.layer_id in config.engram_layer_ids:
            width = (
                (config.engram_max_ngram_size - 1)
                * config.engram_n_heads
                * config.engram_head_dim
            )
            self.engram = EngramOp(
                spec.layer_id,
                config.hidden_size,
                width,
                hc,
                config.rms_norm_eps,
                quant_config=native_quant_config(),
            )

    def attention_forward(self, hidden, cache, step, rope):
        """Norm this sublayer's input, then run it.

        The norm stays on this side of the guarded op rather than in
        `prepare_attention` so its quantized pair never has to cross one: the
        op is declared over a single BF16 tensor, and what leaves it is the
        attention output rather than anything shaped like its input.
        """
        normed = self.attn_norm(hidden)
        if isinstance(normed, tuple):
            return self.attn(*normed, cache, step, rope)
        return self.attn(normed, None, cache, step, rope)

    def engram_forward(self, residual, embeddings, image_mask):
        if embeddings is None:
            raise ValueError("Engram rows must be prepared before model execution")
        return self.engram(
            residual, embeddings, None if image_mask is None else ~image_mask
        )

    def prepare_attention(self, state, embeddings, image_mask):
        if self.engram is not None:
            # Engram reads the residual between the post and the pre, so this
            # is the one seam that cannot fold: settling leaves nothing owed
            # and the projection below runs plain.
            state = state.settle()
            state = replace(
                state,
                residual=self.engram_forward(state.residual, embeddings, image_mask),
            )
        residual, hidden, pre, post, comb = pre_delayed(
            state.residual,
            state.pre_mix,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            **self.hc_options,
            sublayer_output=state.pending,
            post_mix=state.post_mix,
            combination=state.combination,
        )
        return hidden, residual, pre, post, comb

    def prepare_ffn(self, output, residual, pre, post, comb):
        # The attention post folds into this pre, which is the shape the seam
        # has: AITER computes the new residual and projects it in one kernel,
        # and drops back to the two when its own heuristic says to.
        residual, hidden, pre, post, comb = pre_delayed(
            residual,
            pre,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            **self.hc_options,
            sublayer_output=output,
            post_mix=post,
            combination=comb,
        )
        return self.ffn_norm(hidden), residual, pre, post, comb

    def finish_ffn(self, output, residual, pre, post, comb):
        # Owed, not applied: the next block's pre folds this post into its own
        # projection.
        return SinglePassHCState(residual, pre, output, post, comb)

    def forward(self, state, cache, step, rope, embeddings=None, image_mask=None):
        hidden, residual, pre, post, comb = self.prepare_attention(
            state, embeddings, image_mask
        )
        output = self.attention_forward(hidden, cache, step, rope)
        hidden, residual, pre, post, comb = self.prepare_ffn(
            output, residual, pre, post, comb
        )
        output = self.ffn(hidden)
        return self.finish_ffn(output, residual, pre, post, comb)


class DeepseekV41ForCausalLM(nn.Module):
    """Text backbone and offline interface; TP and EP share the same rank group.

    RuntimeModel adapts this math to ModelRunner using prepared Engram values
    and a paged cache. The offline caller supplies its own private cache.
    """

    block_cls = Block

    # Disk-name -> param-name rules for `atom.model_loader.loader.load_model`.
    # V4's two tables carry over as they are; V4.1 needs one rename V4's
    # substring dict cannot express safely, and one tensor class that is not a
    # parameter at all:
    # - `.gate.bias` must be suffix-anchored. V4.1 ships a second routing bias
    #   `.gate.bias_vl` for image sentinel tokens, and a substring rule renames
    #   it to a parameter that does not exist.
    # - Engram embedding tables are host-owned mmap resources loaded by
    #   `model_loader.deepseek_v41.engram_tables`; mapping them to None drops
    #   them here instead of reporting them as unroutable.
    weights_mapper = WeightsMapper(
        orig_to_new_substr={".engram.embed.": None},
        orig_to_new_suffix={".gate.bias": ".gate.e_score_correction_bias"},
    )
    weights_mapping: ClassVar[dict[str, str]] = {".scale": ".weight_scale_inv"}
    # Substring keys, so the dots carry the meaning: `attn.wkv` must not
    # reach `attn.compressor.wkv`, nor `attn.wq_a` reach `attn.wq_b`.
    packed_modules_mapping: ClassVar[dict[str, tuple[str, int]]] = {
        "attn.wq_a": ("attn.wqkv_a", 0),
        "attn.wkv": ("attn.wqkv_a", 1),
        "compressor.wkv": ("compressor.wkv_gate", 0),
        "compressor.wgate": ("compressor.wkv_gate", 1),
        "shared_experts.w1": ("shared_experts.gate_up_proj", 0),
        "shared_experts.w3": ("shared_experts.gate_up_proj", 1),
    }

    def __init__(self, config, *, max_length, online_quant_config=None):
        super().__init__()
        group = get_tp_group()
        config.validate_parallelism(group.world_size, group.world_size)
        if not 1 <= max_length <= config.max_position_embeddings:
            raise ValueError("Invalid offline context capacity")
        self.config, self.max_length = config, max_length
        self.topology = build_attention_topology(config)[: config.num_hidden_layers]
        self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # One shared configuration owns all expert source/online quantization
        # rules. Native attention and Engram projections own their A8 layouts.
        self.moe_quant_config = make_v4_quant_config(
            config, online_quant_config=online_quant_config
        )
        # A stream earns a hardware queue only where two branches are live at
        # once -- overlapping lifetimes, not independent data. One per model
        # rather than per layer, because layers do not overlap.
        # `ATOM_DSV41_SIDE_STREAMS` picks the level and the environment doc
        # carries what each one measured.
        #
        # The compressor borrows the MoE's wherever it forks: its lifetime ends
        # at the scorer, a sublayer before the shared expert is issued, and the
        # MoE joins this stream inside its own forward. Only the indexer is
        # ever live beside both, so only it costs a queue.
        on_device = torch.cuda.is_available()
        level = envs.ATOM_DSV41_SIDE_STREAMS
        if level not in (0, 1, 2):
            raise ValueError(f"ATOM_DSV41_SIDE_STREAMS must be 0, 1 or 2, not {level}")
        if not on_device:
            level = 0
        self.alt_stream = torch.cuda.Stream() if on_device else None
        self.compress_stream = self.alt_stream if level >= 1 else None
        self.index_stream = torch.cuda.Stream() if level == 2 else None
        self.layers = nn.ModuleList(
            self.block_cls(
                config,
                spec,
                prefix=f"layers.{spec.layer_id}",
                moe_quant_config=self.moe_quant_config,
                alt_stream=self.alt_stream,
                compress_stream=self.compress_stream,
                index_stream=self.index_stream,
            )
            for spec in self.topology
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = ParallelHead(config.vocab_size, config.hidden_size)
        self.window_rope = RotaryEmbedding(
            config.qk_rope_head_dim, max_length, base=config.rope_theta
        )
        scaling = config.rope_scaling
        self.global_rope = RotaryEmbedding(
            config.qk_rope_head_dim,
            max_length,
            base=config.compress_rope_theta,
            original_length=scaling["original_max_position_embeddings"],
            factor=scaling["factor"],
            beta_fast=scaling["beta_fast"],
            beta_slow=scaling["beta_slow"],
        )

    @property
    def model(self):
        """V4 keeps its backbone under `self.model`; here the class is it.

        The only indirection the borrowed V4 attributes reach through, which
        is what lets them be used verbatim rather than copied. A property, not
        a submodule, so parameter traversal does not recurse.
        """
        return self

    load_weights = DeepseekV4ForCausalLM.load_weights

    # Whether the shared expert went into the routed buffer is a per-layer
    # fact, so V4 reads it off a built layer rather than off a global flag.
    # That reaches the layers through `self.model`, so it applies here as is.
    disable_fused_shared_loading = DeepseekV4ForCausalLM.disable_fused_shared_loading

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """(param_name, weight_name, expert_id, shard_id) for FusedMoE.

        V4.1 names its routed experts as V4 does, `ffn.experts.{e}.w{1,2,3}`.
        The count is the one thing to get right: a fused shared expert takes a
        slot of its own at `n_routed_experts`, and a mapping that is one short
        leaves it uninitialized while one that is too long mis-loads every
        expert. Both this and the rename above answer that from the same
        property, so the mapping cannot disagree with the names it is given.
        """
        shared = (
            0 if self.disable_fused_shared_loading else self.config.n_shared_experts
        )
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts + shared,
        )

    def begin_forward(self, hidden, engram_embeddings):
        stage = getattr(engram_embeddings, "stage", None)
        if stage is not None:
            stage()

    def end_forward(self, hidden, engram_embeddings):
        if getattr(engram_embeddings, "stage", None) is not None:
            engram_embeddings.join()

    def forward_hidden(
        self,
        token_ids,
        cache,
        step,
        engram_embeddings=None,
        *,
        inputs_embeds=None,
        image_mask=None,
    ):
        # ATOM's sharded embedding consumes flat tokens; restore this offline
        # interface's batch/sequence dimensions before entering model math.
        hidden = (
            self.embed(token_ids.flatten()).view(
                *token_ids.shape, self.config.hidden_size
            )
            if inputs_embeds is None
            else inputs_embeds
        )
        engram_embeddings = {} if engram_embeddings is None else engram_embeddings
        # Fork immediately before layer 0, after embedding and its TP reduce.
        # This also makes offline forwards obey the lazy staging contract.
        self.begin_forward(hidden, engram_embeddings)
        state = SinglePassHCState.from_embeddings(hidden, self.config.hc_mult)
        for spec, layer in zip(self.topology, self.layers):
            rope = self.global_rope if spec.ratio else self.window_rope
            state = layer(
                state,
                cache,
                step,
                rope,
                engram_embeddings.get(spec.layer_id),
                image_mask=image_mask,
            )
        hidden = state.collapse()
        self.end_forward(hidden, engram_embeddings)
        return hidden

    @torch.inference_mode()
    def forward(
        self,
        token_ids,
        cache,
        engram_embeddings=None,
        *,
        full_logits=False,
        logits_start=0,
        inputs_embeds=None,
        image_mask=None,
    ):
        """Execute every input token; optionally project only a logit suffix."""
        if token_ids.ndim != 2:
            raise ValueError("Offline token IDs must have shape [batch, tokens]")
        if not 0 <= logits_start < token_ids.shape[1] or (
            logits_start and not full_logits
        ):
            raise ValueError("logits_start requires a valid full-logits suffix")
        step = cache.begin_step(cache.position, token_ids.shape[1], token_ids.shape[0])
        # Serving publishes this metadata in the attention backend. The offline
        # entry owns it for one call too, so every MoE uses the same fixed hook.
        context = get_forward_context()
        previous = context.attn_metadata
        context.attn_metadata = SimpleNamespace(image_mask=image_mask)
        try:
            hidden = self.forward_hidden(
                token_ids,
                cache,
                step,
                engram_embeddings,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
            )
        finally:
            context.attn_metadata = previous
        hidden = hidden[:, logits_start:] if full_logits else hidden[:, -1]
        logits = self.head.get_logits(self.norm(hidden).flatten(0, -2)).unflatten(
            0, hidden.shape[:-1]
        )
        cache.finish_step(step)
        return logits
