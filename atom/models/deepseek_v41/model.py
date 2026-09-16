# SPDX-License-Identifier: MIT
"""Full-layer eager text backbone. Checkpoint I/O and request preparation live outside."""

from typing import ClassVar

import torch
import torch.nn.functional as F
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_loader.weight_names import WeightsMapper
from atom.model_ops.attentions.deepseek_v41_state import EagerAttentionCache
from atom.model_ops.deepseek_v41.mhc import (
    SinglePassHCState,
    expand_residual,
)
from atom.model_ops.deepseek_v41.mhc_pre_delayed import pre_delayed
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.engram_layer import EngramOp
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.models.deepseek_v4 import DeepseekV4ForCausalLM, make_v4_quant_config

from .attention import Attention
from .config import build_attention_topology
from .layers import native_quant_config
from .moe import MoE


class LogitsHead(ParallelLMHead):
    """Vocab-parallel head, projecting in FP32.

    Inherits from `ParallelLMHead` for the same reason V4's `ParallelHead` does:
    the vocab-axis sharding and its `weight_loader` come with it. Hand-rolling
    the parameter leaves the loader with no way to shard it -- the flat
    rank-major slice holds the right values but is one-dimensional, so it cannot
    be copied into a `[vocab/tp, hidden]` destination.
    """

    def __init__(self, hidden_size, vocab_size):
        super().__init__(vocab_size, hidden_size, bias=False)
        self.group = get_tp_group()
        self.register_buffer("fp32_weight", None, persistent=False)

    def process_weights_after_loading(self):
        self.fp32_weight = self.weight.float()

    def forward(self, hidden):
        if self.fp32_weight is None:
            raise RuntimeError("Logits weights must be processed after loading")
        logits = F.linear(hidden.float(), self.fp32_weight)
        return (
            self.group.all_gather(logits, dim=-1)
            if self.group.world_size > 1
            else logits
        )


class Block(nn.Module):
    attention_cls = Attention

    def __init__(self, config, spec, prefix: str = "", *, moe_quant_config):
        super().__init__()
        self.attn = self.attention_cls(config, spec)
        # FusedMoE names its parameters from this prefix, so it has to match the
        # module layout used by the shared loader: `layers.N` / `mtp.N`.
        self.ffn = MoE(
            config, spec.layer_id, prefix=f"{prefix}.ffn", quant_config=moe_quant_config
        )
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
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
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )
        self.engram = None
        if spec.layer_id in config.engram_layer_ids:
            width = (
                (config.engram_max_ngram_size - 1)
                * config.engram_n_heads
                * config.engram_head_dim
            )
            projection = ReplicatedLinear(
                width, (hc + 1) * config.hidden_size, quant_config=native_quant_config()
            )
            self.engram = EngramOp(
                spec.layer_id,
                config.hidden_size,
                width,
                hc,
                config.rms_norm_eps,
                projection=projection,
            )

    def prepare_attention(self, residual, pre_mix, embeddings, image_mask):
        if self.engram is not None:
            if embeddings is None:
                raise ValueError("Engram rows must be prepared before model execution")
            residual = self.engram(
                residual,
                embeddings,
                None if image_mask is None else ~image_mask,
            )
        residual, hidden, pre, post, comb = pre_delayed(
            residual,
            pre_mix,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            **self.hc_options,
        )
        return self.attn_norm(hidden), residual, pre, post, comb

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

    def decode_ffn(self, hidden, image_mask):
        return (self.ffn(hidden, image_mask),)

    def finish_ffn(self, output, residual, pre, post, comb):
        return expand_residual(output, residual, post, comb), pre

    def forward(
        self,
        state,
        cache,
        step,
        rope,
        embeddings=None,
        image_mask=None,
        *,
        execution=None,
    ):
        # Execution policy may capture pure tensor stages. Attention, request
        # state and expert dispatch keep their own execution/lifetime contracts.
        run = (
            (lambda function, *args: function(*args))
            if execution is None
            else execution
        )
        hidden, residual, pre, post, comb = run(
            self.prepare_attention,
            state.residual,
            state.pre_mix,
            embeddings,
            image_mask,
        )
        output = self.attn(hidden, cache, step, rope)
        hidden, residual, pre, post, comb = run(
            self.prepare_ffn,
            output,
            residual,
            pre,
            post,
            comb,
        )
        if step.decode:
            (output,) = run(self.decode_ffn, hidden, image_mask)
        else:
            output = self.ffn(hidden, image_mask)
        return SinglePassHCState(
            *run(self.finish_ffn, output, residual, pre, post, comb)
        )


class DeepseekV41ForCausalLM(nn.Module):
    """Text backbone and offline interface; TP and EP share the same rank group.

    RuntimeModel adapts this math to ModelRunner using prepared Engram values
    and a paged cache. The offline caller supplies its own private cache.
    """

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
    packed_modules_mapping: ClassVar[dict[str, tuple[str, int]]] = {
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
        self.layers = nn.ModuleList(
            Block(
                config,
                spec,
                prefix=f"layers.{spec.layer_id}",
                moe_quant_config=self.moe_quant_config,
            )
            for spec in self.topology
        )
        # Final normalization feeds the FP32 logits projection, with no further
        # activation quantization. Reuse V4's fused RMSNorm at this boundary.
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = LogitsHead(config.hidden_size, config.vocab_size)
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

    def new_cache(self, batch_size):
        return EagerAttentionCache(
            self.config,
            self.topology,
            batch_size,
            self.max_length,
            self.embed.weight.device,
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

    def forward_hidden(
        self,
        token_ids,
        cache,
        step,
        engram_embeddings=None,
        *,
        execution=None,
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
        state = SinglePassHCState.from_embeddings(hidden, self.config.hc_mult)
        engram_embeddings = {} if engram_embeddings is None else engram_embeddings
        for spec, layer in zip(self.topology, self.layers):
            rope = self.global_rope if spec.ratio else self.window_rope
            state = layer(
                state,
                cache,
                step,
                rope,
                engram_embeddings.get(spec.layer_id),
                image_mask=image_mask,
                execution=execution,
            )
        hidden = state.collapse()
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
        hidden = self.forward_hidden(
            token_ids,
            cache,
            step,
            engram_embeddings,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
        )
        hidden = hidden[:, logits_start:] if full_logits else hidden[:, -1]
        logits = self.head(self.norm(hidden))
        cache.finish_step(step)
        return logits
