# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Eagle3.1 draft model with DeepSeek-V3 MLA attention.

Built specifically for the Kimi-K2.6 EAGLE 3.1 draft
(`lightseekorg/kimi-k2.6-eagle3.1-mla`) so it can share the K2.6 target's
MLA KV cache pool (matching q_lora_rank / kv_lora_rank / head dims).

Checkpoint layout (single decoder layer):

    embed_tokens.weight                            [vocab,   hidden]
    fc.weight                                      [hidden,  target_hidden*num_aux]
    fc_norm.{0..N-1}.weight                        [target_hidden]   EAGLE 3.1 per-chunk norm
    layers.0.input_layernorm.weight                [hidden]          embeds norm (dual-input)
    layers.0.hidden_norm.weight                    [hidden]          fc-output norm (dual-input)
    layers.0.post_attention_layernorm.weight       [hidden]
    layers.0.self_attn.q_a_proj.weight             [q_lora,  hidden*2]   doubled input!
    layers.0.self_attn.q_a_layernorm.weight        [q_lora]
    layers.0.self_attn.q_b_proj.weight             [heads*qk_head, q_lora]
    layers.0.self_attn.kv_a_proj_with_mqa.weight   [kv_lora+qk_rope, hidden*2]   doubled input!
    layers.0.self_attn.kv_a_layernorm.weight       [kv_lora]
    layers.0.self_attn.kv_b_proj.weight            [heads*(qk_nope+v), kv_lora]
    layers.0.self_attn.o_proj.weight               [hidden, heads*v_head_dim]
    layers.0.mlp.{gate,up,down}_proj.weight        standard SwiGLU
    norm.weight                                    [hidden]          final RMSNorm
    lm_head.weight                                 [vocab, hidden]

EAGLE 3.1 toggles (`fc_norm`, `norm_output`) follow the same getattr-default
pattern as `eagle3_llama.Eagle3LlamaModel` for backward compat.
"""

import torch
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from atom.config import Config
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.fused_aux_rmsnorm import fused_group_rmsnorm
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.models.deepseek_v2 import yarn_get_mscale
from torch import nn


class Eagle3DeepseekMLAAttention(nn.Module):
    """MLA attention block sized for an EAGLE 3.1 drafter.

    Differs from `DeepseekV2MLAAttention` in two ways:
    1. q_a_proj / kv_a_proj_with_mqa accept `hidden_size * 2` input
       (concat of normed embeds + normed fc output), not `hidden_size`.
    2. q_a_proj and kv_a_proj_with_mqa are NOT fused into `fused_qkv_a_proj`
       (matches the checkpoint, which ships them separate); no quant fusion
       paths; bf16 only.
    """

    def __init__(
        self,
        config,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5

        attn_input_size = self.hidden_size * 2  # dual input: cat(embed, fc_out)

        self.q_a_proj = ReplicatedLinear(
            attn_input_size,
            self.q_lora_rank,
            bias=False,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            prefix=f"{prefix}.q_b_proj",
        )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            attn_input_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE with optional YaRN scaling (matches DeepseekV2MLAAttention).
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        rope_params = getattr(config, "rope_parameters", None)
        if rope_params is None:
            # Fall back to flat fields on the draft config.
            rope_params = {
                "rope_theta": getattr(config, "rope_theta", 10000),
            }
            rope_scaling_raw = getattr(config, "rope_scaling", None) or {}
            rope_params.update(rope_scaling_raw)
        rope_theta = rope_params.get("rope_theta") or 10000
        use_yarn = (
            rope_params.get("factor", 1.0) not in (1.0, None)
            or rope_params.get("type") in ("yarn", "deepseek_yarn")
            or rope_params.get("rope_type") in ("yarn", "deepseek_yarn")
        )
        if use_yarn:
            rope_scaling = dict(rope_params)
            rope_scaling.pop("rope_theta", None)
            rope_scaling["rope_type"] = "deepseek_yarn"
            if "original_max_position_embeddings" not in rope_scaling:
                factor = float(rope_scaling.get("factor", 1.0))
                rope_scaling["original_max_position_embeddings"] = (
                    int(max_position_embeddings / factor)
                    if factor > 0
                    else max_position_embeddings
                )
        else:
            rope_scaling = None
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        mla_modules = MLAModules(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            indexer=None,
        )
        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            use_mla=True,
            mla_modules=mla_modules,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_c = self.q_a_proj(hidden_states)
        q_c = self.q_a_layernorm(q_c)
        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        return self.mla_attn(q_c, kv_c_normed, k_pe, positions, None)


class Eagle3DeepseekMLAMLP(nn.Module):
    """SwiGLU MLP; gate/up are loaded separately and merged at load time
    via the model-level `packed_modules_mapping`."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
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
            prefix=f"{prefix}.down_proj",
        )
        from atom.model_ops.activation import SiluAndMul

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)


class Eagle3DeepseekMLADecoderLayer(nn.Module):
    """Single MLA decoder layer with dual-input concat.

    Mirrors Eagle3LlamaDecoderLayer's shape but with MLA attention and
    the EAGLE 3.1 checkpoint's `layers.0.*` naming.
    """

    def __init__(
        self,
        config,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        self.self_attn = Eagle3DeepseekMLAAttention(
            config=config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
        )
        self.mlp = Eagle3DeepseekMLAMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        normed_embeds = self.input_layernorm(embeds)
        normed_hidden = self.hidden_norm(hidden_states)
        attn_input = torch.cat([normed_embeds, normed_hidden], dim=-1)
        attn_output = self.self_attn(positions, attn_input)
        hidden_states = hidden_states + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Eagle3DeepseekMLAModel(nn.Module):
    """EAGLE 3.1 MLA drafter for Kimi-K2.6.

    Matches `lightseekorg/kimi-k2.6-eagle3.1-mla` weight layout. Shares
    the target's MLA KV cache pool (same q_lora_rank / kv_lora_rank /
    head dims); the runner accounts for the +1 draft layer via the
    standard MTP-style layer count path (no separate draft builder).
    """

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, atom_config: Config, prefix: str = "", layer_offset: int = 0):
        super().__init__()
        config = atom_config.hf_config
        cache_config = atom_config.kv_cache_dtype
        self.config = config

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

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        self.fc = ReplicatedLinear(
            target_hidden_size * num_aux, config.hidden_size, bias=False
        )

        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [
                    RMSNorm(target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(num_aux)
                ]
            )
        else:
            self.fc_norm = None

        # ModuleList (not midlayer) to match `layers.0.*` checkpoint keys.
        self.layers = nn.ModuleList(
            [
                Eagle3DeepseekMLADecoderLayer(
                    config=config,
                    cache_config=cache_config,
                    prefix="layers.0",
                    layer_num=layer_offset,
                )
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def combine_hidden_states(self, aux_hidden_states) -> torch.Tensor:
        # Accepts a list/tuple of per-layer aux tensors (preferred — skips an
        # extra concat) or a single pre-concatenated tensor (back-compat).
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
        EAGLE 3   (norm_output=False): pre-norm hidden state (legacy behavior).
        compute_logits() is norm-aware and consumes whichever this returns, so
        EagleProposer only ever sees one tensor (no pre/post-norm bookkeeping).
        """
        embeds = self.embed_tokens(input_ids)
        hidden_states = self.layers[0](positions, embeds, hidden_states)
        return self.norm(hidden_states) if self.norm_output else hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # forward() already applied the final norm when norm_output is set;
        # only norm here for the legacy pre-norm path, so logits are never
        # double-normed and stay byte-equivalent to EAGLE 3.
        if not self.norm_output:
            hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
