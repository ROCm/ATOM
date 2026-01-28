# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Engram Language Model for ATOM Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union

from aiter.dist.parallel_state import get_pp_group
from atom.config import Config, QuantizationConfig
from atom.model_ops.engram import EngramOp, EngramConfig
from atom.models.utils import (
    IntermediateTensors,
    make_empty_intermediate_tensors_factory,
    PPMissingLayer,
)
from atom.utils.decorators import support_torch_compile

EngramModuleConfig = EngramConfig

class EngramAttention(nn.Module):
    """Causal self-attention for Engram model."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class EngramMLP(nn.Module):
    """Feed-forward network for Engram model."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class EngramDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int,
        engram_config: EngramModuleConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.attn = EngramAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        
        self.ffn = EngramMLP(
            hidden_size=hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        
        self.engram = None
        if layer_id in engram_config.layer_ids:
            self.engram = EngramOp(layer_id, hidden_size, engram_config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Match original TransformerBlock logic exactly:
        - Engram augmentation (before attention)
        - Attention with residual
        - FFN with residual
        """
        x = hidden_states
        
        if self.engram is not None:
            engram_out = self.engram(x, input_ids)
            x = x + engram_out
        
        x = x + self.attn(self.norm1(x))
        
        x = x + self.ffn(self.norm2(x))
        return x, None


@support_torch_compile
class EngramModel(nn.Module):
    """Engram language model backbone."""
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        config = atom_config.hf_config
        self.config = config
        
        self.hidden_size = getattr(config, 'hidden_size', 128)
        self.num_layers = getattr(config, 'num_hidden_layers', 4)
        self.num_heads = getattr(config, 'num_attention_heads', 4)
        self.vocab_size = getattr(config, 'vocab_size', 128)
        self.max_seq_len = getattr(config, 'max_position_embeddings', 256)
        
        engram_config_dict = getattr(config, 'engram_config', None)
        if engram_config_dict:
            self.engram_config = EngramModuleConfig.from_dict(engram_config_dict)
        else:
            self.engram_config = EngramModuleConfig()
        
        if get_pp_group().is_first_rank:
            self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        else:
            self.token_embedding = PPMissingLayer()
        
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        
        self.blocks = nn.ModuleList([
            EngramDecoderLayer(
                layer_id=i,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                max_seq_len=self.max_seq_len,
                engram_config=self.engram_config,
                prefix=f"{prefix}.blocks.{i}",
            )
            for i in range(self.num_layers)
        ])
        self.start_layer = 0
        self.end_layer = self.num_layers
        
        if get_pp_group().is_last_rank:
            self.norm = nn.LayerNorm(self.hidden_size)
        else:
            self.norm = PPMissingLayer()
        
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if input_ids.dim() == 2:
            input_ids = input_ids.flatten()
        if positions.dim() == 2:
            positions = positions.flatten()
        
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                # Use clamped values for embedding lookup without modifying original tensor
                vocab_size = self.token_embedding.num_embeddings
                input_ids_clamped = torch.clamp(input_ids, 0, vocab_size - 1)
                hidden_states = self.get_input_embeddings(input_ids_clamped)  # [T, D]
            
            max_pos = self.position_embedding.num_embeddings
            positions_clamped = torch.clamp(positions, 0, max_pos - 1)
            pos_emb = self.position_embedding(positions_clamped)  # [T, D]
            hidden_states = hidden_states + pos_emb
            
            hidden_states = hidden_states.unsqueeze(0)
            input_ids = input_ids.unsqueeze(0)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        
        for layer in self.blocks[self.start_layer:self.end_layer]:
            hidden_states, _ = layer(hidden_states, input_ids)
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": None
            })
        
        hidden_states = self.norm(hidden_states)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        
        return hidden_states

class EngramForCausalLM(nn.Module):
    """Engram model for causal language modeling."""
    packed_modules_mapping = {}
    
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        config = atom_config.hf_config
        self.config = config
        
        hidden_size = getattr(config, 'hidden_size', 128)
        num_layers = getattr(config, 'num_hidden_layers', 4)
        num_heads = getattr(config, 'num_attention_heads', 4)
        vocab_size = getattr(config, 'vocab_size', 128)
        max_seq_len = getattr(config, 'max_position_embeddings', 256)
        
        # Load engram config
        engram_config_dict = getattr(config, 'engram_config', None)
        if engram_config_dict:
            engram_config = EngramModuleConfig.from_dict(engram_config_dict)
        else:
            engram_config = EngramModuleConfig()
        
        # Token embedding (directly on this class, no 'model.' prefix)
        if get_pp_group().is_first_rank:
            self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        else:
            self.token_embedding = PPMissingLayer()
        
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        self.blocks = nn.ModuleList([
            EngramDecoderLayer(
                layer_id=i,
                hidden_size=hidden_size,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                engram_config=engram_config,
                prefix=f"{prefix}.blocks.{i}" if prefix else f"blocks.{i}",
            )
            for i in range(num_layers)
        ])
        self.start_layer = 0
        self.end_layer = num_layers
        
        if get_pp_group().is_last_rank:
            self.norm = nn.LayerNorm(hidden_size)
        else:
            self.norm = PPMissingLayer()
        
        if get_pp_group().is_last_rank:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            
            tie_weights = getattr(config, 'tie_word_embeddings', True)
            if tie_weights and hasattr(self, 'token_embedding') and not isinstance(self.token_embedding, PPMissingLayer):
                self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = PPMissingLayer()
        
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], hidden_size
        )
        
        self._hidden_size = hidden_size
        self._max_seq_len = max_seq_len
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # # Flatten to 1D if needed
        if input_ids.dim() == 2:
            input_ids = input_ids.flatten()
        if positions.dim() == 2:
            positions = positions.flatten()
        
        T = input_ids.shape[0]
        
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                vocab_size = self.token_embedding.num_embeddings
                input_ids_clamped = torch.clamp(input_ids, 0, vocab_size - 1)
                hidden_states = self.get_input_embeddings(input_ids_clamped)  # [T, D]
            
            max_pos = self.position_embedding.num_embeddings
            positions_clamped = torch.clamp(positions, 0, max_pos - 1)
            pos_emb = self.position_embedding(positions_clamped)  # [T, D]
            hidden_states = hidden_states + pos_emb
            
            hidden_states = hidden_states.unsqueeze(0)
            input_ids = input_ids.unsqueeze(0)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.blocks[self.start_layer:self.end_layer]:
            hidden_states, _ = layer(hidden_states, input_ids)
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": None
            })
        
        hidden_states = self.norm(hidden_states)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from atom.utils.forward_context import get_forward_context
        forward_ctx = get_forward_context()
        
        if forward_ctx and forward_ctx.context and forward_ctx.context.is_prefill:
            attn_meta = forward_ctx.attn_metadata
            if attn_meta and hasattr(attn_meta, 'cu_seqlens_q') and attn_meta.cu_seqlens_q is not None:
                last_indices = attn_meta.cu_seqlens_q[1:] - 1
                # Clamp indices to valid range
                last_indices = last_indices.clamp(0, hidden_states.shape[0] - 1)
                hidden_states = hidden_states[last_indices].contiguous()
        
        logits = self.lm_head(hidden_states)
        return logits
