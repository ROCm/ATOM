# SPDX-License-Identifier: MIT
"""Full-layer eager text backbone. Checkpoint I/O and request preparation live outside."""

import torch
import torch.nn.functional as F
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_ops.attentions.deepseek_v41_state import EagerAttentionCache
from atom.model_ops.deepseek_v41.mhc import (
    SinglePassHCState,
    expand_residual,
    predict_mixes,
)
from atom.model_ops.deepseek_v41.normalization import RMSNorm
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.engram_layer import EngramOp
from atom.model_ops.layernorm import RMSNorm as FusedRMSNorm
from atom.model_ops.linear import ReplicatedLinear

from .attention import Attention
from .config import build_attention_topology
from .layers import native_quant_config
from .moe import MoE


class LogitsHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.group = get_tp_group()
        self.weight = nn.Parameter(
            torch.empty(
                vocab_size // self.group.world_size, hidden_size, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
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
    def __init__(self, config, spec):
        super().__init__()
        self.attn = Attention(config, spec)
        self.ffn = MoE(config)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_options = {
            "norm_eps": config.rms_norm_eps,
            "sinkhorn_eps": config.hc_eps,
            "sinkhorn_iters": config.hc_sinkhorn_iters,
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
        state = SinglePassHCState(residual, pre_mix)
        pre, post, comb = predict_mixes(
            residual,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            **self.hc_options,
        )
        return self.attn_norm(state.collapse()), residual, pre, post, comb

    def prepare_ffn(self, output, residual, pre, post, comb):
        residual = expand_residual(output, residual, post, comb)
        state = SinglePassHCState(residual, pre)
        pre, post, comb = predict_mixes(
            residual,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            **self.hc_options,
        )
        return self.ffn_norm(state.collapse()), residual, pre, post, comb

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
        if step.decode and self.ffn.can_capture(hidden.shape[0] * hidden.shape[1]):
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

    def __init__(self, config, *, max_length):
        super().__init__()
        group = get_tp_group()
        config.validate_parallelism(group.world_size, group.world_size)
        if not 1 <= max_length <= config.max_position_embeddings:
            raise ValueError("Invalid offline context capacity")
        self.config, self.max_length = config, max_length
        self.topology = build_attention_topology(config)[: config.num_hidden_layers]
        self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config, spec) for spec in self.topology)
        # Final normalization feeds the FP32 logits projection, with no further
        # activation quantization. Reuse V4's fused RMSNorm at this boundary.
        self.norm = FusedRMSNorm(config.hidden_size, config.rms_norm_eps)
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

    @torch.no_grad()
    def process_weights_after_loading(self):
        for module in self.modules():
            if module is not self and hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading()

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
