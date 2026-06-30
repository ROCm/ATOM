# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Eagle3 draft model (Llama full-attention) for speculative decoding.

Implements the Eagle3 draft model matching the lightseekorg/kimi-k2.5-eagle3
checkpoint layout:

    embed_tokens.weight   — independent embedding
    fc.weight             — aux fusion projection (hidden*3 -> hidden)
    midlayer.*            — single decoder layer (dual-norm, wide QKV)
    norm.weight           — final RMSNorm
    lm_head.weight        — independent lm_head

Weight keys map directly to model attribute paths; no key rewriting needed.
"""

import torch
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from atom.config import Config
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import (
    ParallelLMHead,
    ReplicatedEmbedding,
    VocabParallelEmbedding,
)
from atom.model_ops.fused_aux_rmsnorm import (
    fused_dual_rmsnorm_cat,
    fused_group_rmsnorm,
)
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.utils import envs
from atom.utils.decorators import support_torch_compile
from torch import nn

# AR+RMSNorm fusion: when on (default), RowParallel o_proj/down_proj skip their
# own all-reduce (reduce_results=False) and the downstream RMSNorm fuses
# all-reduce + residual-add + norm into one kernel. Only active at TP>1; the
# RMSNorm/RowParallel paths fall back to plain behavior at TP1. Same env and
# kernel as ATOM's mainline TP models (deepseek_v2, qwen3_moe, ...).
ENABLE_ALLREDUCE_RMSNORM_FUSION = envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION


class Eagle3LlamaAttention(nn.Module):
    """Llama full-attention with input_size = hidden_size * 2.

    The QKV projection accepts the concatenation of normalized embeddings
    and fc output, hence input_size is doubled compared to standard Llama.
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # QKV input_size = hidden_size * 2 (concat of embed + fc_output)
        attn_input_size = hidden_size * 2
        self.qkv_proj = QKVParallelLinear(
            hidden_size=attn_input_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            reduce_results=reduce_results,
            prefix=f"{prefix}.o_proj",
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
        )

        sliding_window = -1
        if getattr(config, "use_sliding_window", False) and getattr(
            config, "sliding_window", None
        ):
            sliding_window = config.sliding_window
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            prefix=f"{prefix}.attn",
            rotary_emb=self.rotary_emb,
            per_layer_sliding_window=sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, positions)
        output = self.o_proj(attn_output)
        return output


class Eagle3LlamaDecoderLayer(nn.Module):
    """Single decoder layer for Eagle3 with dual-norm input.

    Unlike standard LlamaDecoderLayer, this layer has:
    - input_layernorm: normalizes the embedding input
    - hidden_norm: normalizes the fc output (projected aux hidden states)
    - Attention input is concat(normed_embed, normed_hidden) -> [N, hidden*2]
    """

    def __init__(
        self,
        config,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
        norm_output: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # Point 1 (always): o_proj skips its all-reduce so post_attention_layernorm
        # fuses all-reduce + residual-add + norm. Point 2 (norm_output only):
        # down_proj skips its all-reduce so the model's final self.norm fuses it;
        # for the legacy (norm_output=False) path the output norm is deferred to
        # compute_logits with no adjacent residual-add, so down_proj all-reduces
        # normally.
        attn_reduce = not ENABLE_ALLREDUCE_RMSNORM_FUSION
        mlp_reduce = not (ENABLE_ALLREDUCE_RMSNORM_FUSION and norm_output)

        self.self_attn = Eagle3LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
            reduce_results=attn_reduce,
        )

        self.mlp = Eagle3LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            prefix=f"{prefix}.mlp",
            reduce_results=mlp_reduce,
        )

        # Dual norms matching checkpoint keys: midlayer.input_layernorm, midlayer.hidden_norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )

    def _dual_norm_cat(
        self, embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """RMS-norm embeds and the carried hidden by their own weights and concat
        into the [N, 2*hidden] QKV input.

        Single fused Triton launch (one [N, 2H] write) instead of two RMSNorm
        launches + a concat. Falls back to the aiter RMSNorm + torch.cat path
        when the kernel's preconditions don't hold (non-CUDA / non-contiguous /
        shape mismatch). input_layernorm and hidden_norm share rms_norm_eps.
        """
        if (
            embeds.is_cuda
            and embeds.is_contiguous()
            and hidden_states.is_contiguous()
            and embeds.shape == hidden_states.shape
        ):
            return fused_dual_rmsnorm_cat(
                embeds,
                hidden_states,
                self.input_layernorm.weight,
                self.hidden_norm.weight,
                self.input_layernorm.eps,
            )
        normed_embeds = self.input_layernorm(embeds)
        normed_hidden = self.hidden_norm(hidden_states)
        return torch.cat([normed_embeds, normed_hidden], dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self._dual_norm_cat(embeds, hidden_states)
        attn_output = self.self_attn(positions, attn_input)
        # Fused (all-reduce +) residual-add + pre-MLP norm in one kernel:
        #   residual      = [all_reduce(attn_output)] + hidden_states
        #   hidden_states = post_attention_layernorm(residual)
        hidden_states, residual = self.post_attention_layernorm(
            attn_output, hidden_states
        )
        hidden_states = self.mlp(hidden_states)
        # Return the MLP output and its residual; the model fuses the final
        # residual-add with the output norm (norm_output) or adds plainly.
        return hidden_states, residual


class Eagle3LlamaMLP(nn.Module):
    """Simple Llama MLP (gate+up fused, silu activation, down projection)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


@support_torch_compile
class Eagle3LlamaModel(nn.Module):
    """Eagle3 draft model (Llama full-attention, single decoder layer).

    Matches the lightseekorg/kimi-k2.5-eagle3 checkpoint layout:
        embed_tokens.weight   [163840, 7168]  independent embedding
        fc.weight             [7168, 21504]   aux fusion (hidden*3 -> hidden)
        midlayer.*            single decoder layer
        norm.weight           final RMSNorm
        lm_head.weight        [163840, 7168]  independent lm_head
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # The single decoder layer is named `midlayer` here, but some EAGLE3
    # checkpoints ship it as `layers.0.*` (e.g. the torchspec-format
    # Inferact/MiniMax-M3-EAGLE3) instead of the kimi-k2.5 `midlayer.*` layout.
    # Translate that prefix on load. No-op for `midlayer.*` checkpoints (the
    # substring is absent), so both naming conventions load correctly.
    weights_mapping = {"layers.0.": "midlayer."}

    def __init__(self, atom_config: Config, prefix: str = "", layer_offset: int = 0):
        super().__init__()
        config = atom_config.hf_config
        cache_config = atom_config.kv_cache_dtype
        self.config = config

        # EAGLE 3.1 toggles (backward-compatible defaults match EAGLE 3).
        # target_hidden_size: aux chunk width. Defaults to hidden_size
        # (i.e. target hidden == drafter hidden, as in K2.5).
        # num_aux_hidden_states: how many target layers feed the FC.
        # Prefer explicit, else infer from eagle_config layer ids, else 3.
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        num_aux = getattr(config, "num_aux_hidden_states", None)
        if num_aux is None:
            eagle_cfg = getattr(config, "eagle_config", None)
            if eagle_cfg:
                aux_ids = eagle_cfg.get("eagle_aux_hidden_state_layer_ids", [])
                num_aux = len(aux_ids) if aux_ids else 3
            else:
                num_aux = 3
        self.target_hidden_size = target_hidden_size
        self.num_aux_hidden_states = num_aux
        self.norm_output = getattr(config, "norm_output", False)

        # Independent embedding (vocab matches target model). The draft embed is
        # NOT shared with the (still TP-sharded) lm_head, so it can be replicated
        # full on every rank — a local lookup with no post-embedding all-reduce.
        # Bit-identical to the sharded path; on by default (trades memory for one
        # fewer collective per draft step). Falls back to the sharded embedding
        # when ATOM_EAGLE_REPLICATE_EMBED=0.
        if envs.ATOM_EAGLE_REPLICATE_EMBED:
            self.embed_tokens = ReplicatedEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )

        # Aux fusion: [N, target_hidden_size * num_aux] -> [N, hidden_size]
        self.fc = ReplicatedLinear(
            target_hidden_size * num_aux, config.hidden_size, bias=False
        )

        # EAGLE 3.1: optional per-chunk RMSNorm applied to each aux chunk
        # before fc. When absent, identity (matches EAGLE 3 / K2.5 path).
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [
                    RMSNorm(target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(num_aux)
                ]
            )
        else:
            self.fc_norm = None

        # Draft attention layer_num must start from the target model's layer
        # count so kv_cache_data["layer_N"] maps to the correct cache entry.
        self.midlayer = Eagle3LlamaDecoderLayer(
            config=config,
            cache_config=cache_config,
            prefix="midlayer",
            layer_num=layer_offset,
            norm_output=self.norm_output,
        )

        # Final norm. Point 2: on the norm_output path it fuses down_proj's
        # all-reduce + residual-add + norm. On the legacy path it stays plain
        # (called without residual in compute_logits), so no fusion here.
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION and self.norm_output,
        )

        # Independent lm_head (not shared with target model)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def combine_hidden_states(self, aux_hidden_states) -> torch.Tensor:
        """Project the per-layer aux hidden states through fc.

        Args:
            aux_hidden_states: either a list/tuple of per-layer aux tensors
                ([N, target_hidden_size] each) — preferred, skips an extra
                concat — or a single pre-concatenated
                [N, target_hidden_size * num_aux_hidden_states] tensor
                (back-compat).

        Returns:
            [N, hidden_size] projected hidden states
        """
        is_list = isinstance(aux_hidden_states, (list, tuple))
        if self.fc_norm is None:
            if is_list:
                fc_in = (
                    aux_hidden_states[0]
                    if len(aux_hidden_states) == 1
                    else torch.cat(aux_hidden_states, dim=-1)
                )
            else:
                fc_in = aux_hidden_states
            return self.fc(fc_in)

        # fc_norm path: per-group RMSNorm, then fc. Use the single-launch fused
        # kernel (one RMSNorm over all aux chunks) instead of per-chunk RMSNorm
        # + concat; fall back to the torch path only when the fused kernel's
        # preconditions don't hold (non-CUDA / non-contiguous / shape mismatch).
        x = torch.cat(aux_hidden_states, dim=-1) if is_list else aux_hidden_states
        if (
            x.is_cuda
            and x.is_contiguous()
            and x.shape[-1] == self.num_aux_hidden_states * self.fc_norm[0].dim
        ):
            fc_in = fused_group_rmsnorm(
                x,
                self._fc_norm_weight_stacked(),
                self.fc_norm[0].eps,
                self.num_aux_hidden_states,
            )
        else:
            chunks = (
                aux_hidden_states
                if is_list
                else x.chunk(self.num_aux_hidden_states, dim=-1)
            )
            fc_in = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )
        return self.fc(fc_in)

    def _fc_norm_weight_stacked(self) -> torch.Tensor:
        """Per-group fc_norm weights stacked to [num_aux, H] (cached)."""
        ref = self.fc_norm[0].weight
        w = getattr(self, "_fc_norm_w_cache", None)
        if w is None or w.device != ref.device or w.dtype != ref.dtype:
            w = torch.stack([m.weight for m in self.fc_norm], dim=0).contiguous()
            self._fc_norm_w_cache = w
        return w

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return the single hidden state carried to the next speculative step.

        EAGLE 3.1 (norm_output=True): post-norm hidden state.
        EAGLE 3   (default):          pre-norm hidden state (legacy behavior).
        compute_logits() is norm-aware, so EagleProposer only sees one tensor.
        """
        embeds = self.embed_tokens(input_ids)
        hidden_states, residual = self.midlayer(positions, embeds, hidden_states)
        if self.norm_output:
            # EAGLE 3.1: fused final residual-add + output RMSNorm (one kernel).
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            # EAGLE 3 / K2.5: carry the pre-norm hidden forward; the norm is
            # deferred to compute_logits, so the add stays standalone here
            # (byte-equivalent to the legacy path).
            hidden_states = residual + hidden_states
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Only norm the legacy pre-norm path; norm_output already normed in
        # forward(). Avoids double-norm and stays byte-equivalent to EAGLE 3.
        if not self.norm_output:
            hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def compute_draft_token(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft token via distributed argmax — avoids all-gathering the
        full [N, vocab] logits every draft step. Token-identical to
        compute_logits(...).argmax(-1); norm handling mirrors compute_logits.
        """
        if not self.norm_output:
            hidden_states = self.norm(hidden_states)
        return self.lm_head.compute_argmax_token(hidden_states)
