# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from atom.utils.forward_context import get_forward_context
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import isprime


# Global prefetch cache for engram hash
# {seq_id: {layer_id: np.ndarray}}
_global_prefetch_cache: Dict[int, Dict[int, np.ndarray]] = {}
_global_cache_lock = threading.Lock()


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class CompressedTokenizer:
    def __init__(self, tokenizer_name_or_path: Optional[str] = None):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.lookup_table = None
        self.num_new_token = 0
        if tokenizer_name_or_path is not None:
            self._build_lookup_table()
    
    def _build_lookup_table(self):
        try:
            from transformers import AutoTokenizer
            from tokenizers import normalizers
            from tokenizers.normalizers import Regex
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path, 
                trust_remote_code=True
            )
            
            SENTINEL = "\uE000"
            self.normalizer = normalizers.Sequence([
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ])
            
            old2new = {}
            key2new = {}
            new_tokens = []
            
            vocab_size = len(self.tokenizer)
            for tid in range(vocab_size):
                text = self.tokenizer.decode([tid], skip_special_tokens=False)
                
                # Handle special tokens
                if "�" in text:
                    key = self.tokenizer.convert_ids_to_tokens(tid)
                else:
                    norm = self.normalizer.normalize_str(text)
                    key = norm if norm else text
                
                nid = key2new.get(key)
                if nid is None:
                    nid = len(new_tokens)
                    key2new[key] = nid
                    new_tokens.append(key)
                old2new[tid] = nid
            
            # Create numpy lookup array
            lookup = np.empty(vocab_size, dtype=np.int64)
            for tid in range(vocab_size):
                lookup[tid] = old2new[tid]
            
            self.lookup_table = lookup
            self.num_new_token = len(new_tokens)
            
        except Exception as e:
            print(f"Warning: Failed to build compressed tokenizer lookup table: {e}")
            self.lookup_table = None
            self.num_new_token = 128000  # Default
    
    def __len__(self):
        return self.num_new_token
    
    def _compress(self, input_ids: np.ndarray) -> np.ndarray:
        if self.lookup_table is None:
            return input_ids
        
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out
    
    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        return self._compress(input_ids)


@dataclass
class EngramConfig:
    """Configuration for Engram module."""
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 3])
    pad_id: int = 0
    seed: int = 42
    kernel_size: int = 4
    tokenizer_name_or_path: Optional[str] = None  # For CompressedTokenizer
    kernel_size: int = 4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngramConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered_dict)

@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30

backbone_config = BackBoneConfig()

class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size: List[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        tokenizer_name_or_path: Optional[str],
        pad_id: int,
        seed: int,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // max(self.tokenizer_vocab_size, 1))
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(
                x, ((0, 0), (k, 0)),
                mode='constant', 
                constant_values=self.pad_id
            )[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids: np.ndarray) -> Dict[int, np.ndarray]:
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
                input_ids, layer_id=layer_id
            )
        return hash_ids_for_all_layers
    
    def hash_single_layer(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        input_ids = self.compressed_tokenizer(input_ids)
        return self._get_ngram_hashes(input_ids, layer_id)
    
    # def cache_batch_hash_results(self, hash_results: Dict[int, np.ndarray], seq_ids: List[int]):
    #     global _global_prefetch_cache, _global_cache_lock
    #     with _global_cache_lock:
    #         for i, seq_id in enumerate(seq_ids):
    #             if seq_id not in _global_prefetch_cache:
    #                 _global_prefetch_cache[seq_id] = {}
    #             for layer_id in self.layer_ids:
    #                 # Extract hash for token i: [1, 1, num_heads]
    #                 token_hash = hash_results[layer_id][:, i:i+1, :]
    #                 _global_prefetch_cache[seq_id][layer_id] = token_hash
    
    # def get_cached_hash(self, layer_id: int, seq_id: int) -> Optional[np.ndarray]:
    #     global _global_prefetch_cache, _global_cache_lock
    #     with _global_cache_lock:
    #         if seq_id in _global_prefetch_cache and layer_id in _global_prefetch_cache[seq_id]:
    #             hash_result = _global_prefetch_cache[seq_id][layer_id]
    #             # Remove from cache after use
    #             del _global_prefetch_cache[seq_id][layer_id]
    #             if not _global_prefetch_cache[seq_id]:
    #                 del _global_prefetch_cache[seq_id]
    #             return hash_result
    #     return None
    
    # def clear_cache(self, seq_id: Optional[int] = None):
    #     """Clear global cache"""
    #     global _global_prefetch_cache, _global_cache_lock
    #     with _global_cache_lock:
    #         if seq_id is not None:
    #             _global_prefetch_cache.pop(seq_id, None)
    #         else:
    #             _global_prefetch_cache.clear()


class MultiHeadEmbedding(nn.Module):

    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.D = D
        
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_vocab = sum(list_of_N)
        self.embedding = nn.Embedding(total_vocab, D)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def init_cpu_embedding(self):
        self.cpu_embedding = self.embedding.weight.detach().cpu().to(torch.float32).numpy()
        self.cpu_offsets = self.offsets.cpu().numpy()
    
    def forward_on_cpu(self, hash_ids: np.ndarray) -> np.ndarray:
        shifted_ids = hash_ids + self.cpu_offsets[None, None, :]
        # shifted_ids = np.clip(shifted_ids, 0, self.embedding.num_embeddings - 1)
        embeddings = self.cpu_embedding[shifted_ids]
        return embeddings.astype(np.float32)
    
    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        shifted_ids = hash_ids + self.offsets
        # TODO: remove clamp in real model
        # shifted_ids = shifted_ids.clamp(0, self.embedding.num_embeddings - 1)
        return self.embedding(shifted_ids)
    
    def save_embedding_results(self, embeddings: np.ndarray, seq_ids: List[int], layer_ids: List[int]):
        global _global_prefetch_cache, _global_cache_lock
        with _global_cache_lock:
            for i, seq_id in enumerate(seq_ids):
                if seq_id not in _global_prefetch_cache:
                    _global_prefetch_cache[seq_id] = {}
                for layer_id in layer_ids:
                    # embeddings[layer_id] shape: [B, T, num_heads, D]
                    # (B, T, 16, 64), 16 = 8 + 8 for 2 gram + 3 gram
                    # Extract i-th sequence: [1, T, num_heads, D]
                    seq_embedding = embeddings[layer_id][i:i+1]
                    _global_prefetch_cache[seq_id][layer_id] = seq_embedding
    
    def get_cached_embedding(self, layer_id: int, seq_id: int) -> Optional[np.ndarray]:
        global _global_prefetch_cache, _global_cache_lock
        with _global_cache_lock:
            if seq_id in _global_prefetch_cache and layer_id in _global_prefetch_cache[seq_id]:
                embedding = _global_prefetch_cache[seq_id][layer_id]
                # Remove batch dimension: [1, T, num_heads, D] -> [T, num_heads, D]
                if embedding.ndim == 4 and embedding.shape[0] == 1:
                    embedding = embedding[0]
                return embedding
        return None

class ShortConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,  # Depthwise
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_norm = self.norm(x)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]
        y = self.act(y_bct.transpose(1, 2))
        return y


# class ShortConv(nn.Module):
#     def __init__(
#         self, 
#         hidden_size: int, 
#         kernel_size: int = 4, 
#         dilation: int = 1, 
#         norm_eps: float = 1e-5,
#         hc_mult: int = 4,
#         activation: bool = True,
#     ):
#         super().__init__()
#         self.hc_mult = hc_mult
#         self.activation = activation
        
#         total_channels = hidden_size * hc_mult
#         self.conv = nn.Conv1d(
#             in_channels=total_channels,
#             out_channels=total_channels,
#             kernel_size=kernel_size,
#             groups=total_channels,
#             bias=False,
#             padding=(kernel_size - 1) * dilation,
#             dilation=dilation,
#         )

#         self.norms = nn.ModuleList([
#             nn.LayerNorm(hidden_size)
#             for _ in range(hc_mult)
#         ])
        
#         if self.activation:
#             self.act_fn = nn.SiLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Input:  (B,L,HC_MULT,D)
#         Output: (B,L,HC_MULT,D)
#         """
#         B, T, G, C = x.shape
        
#         assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

#         normed_chunks = []
#         for i in range(G):
#             chunk = x[:, :, i, :]
#             normed_chunks.append(self.norms[i](chunk))
        
#         x_norm = torch.cat(normed_chunks, dim=-1)
#         x_bct = x_norm.transpose(1, 2)
#         y_bct = self.conv(x_bct)
#         y_bct = y_bct[..., :T]

#         if self.activation:
#             y_bct = self.act_fn(y_bct)
#         y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        
#         return y


class EngramOp(nn.Module):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        config: EngramConfig,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.config = config
        
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=config.engram_vocab_size,
            max_ngram_size=config.max_ngram_size,
            n_embed_per_ngram=config.n_embed_per_ngram,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.layer_ids,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            pad_id=config.pad_id,
            seed=config.seed,
        )
        
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=[x for y in self.hash_mapping.vocab_size_across_layers[layer_id] for x in y],
            D=config.n_embed_per_ngram // config.n_head_per_ngram,
        )
        
        self.short_conv = ShortConv(
            hidden_size=hidden_size,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
        )

        # self.short_conv = ShortConv(
        #     hidden_size = backbone_config.hidden_size,
        #     kernel_size = config.kernel_size,
        #     dilation    = config.max_ngram_size,
        #     hc_mult     = backbone_config.hc_mult,
        # )

        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)
        self.key_proj = nn.Linear(engram_hidden_size, hidden_size)
        
        self.key_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D]
            input_ids: [B, T]
        Returns:
            output: [B, T, D]
        """
        B, T = input_ids.shape
        num_tokens = B * T
                
        if (hasattr(self, 'embedding_buffer') and self.embedding_buffer is not None):
            if hasattr(self, 'embedding_stream') and self.embedding_stream is not None:
                current_stream = torch.cuda.current_stream()
                current_stream.wait_stream(self.embedding_stream)
            
            # embedding_buffer shape: [max_tokens, num_heads, D]
            embeddings_from_buffer = self.embedding_buffer[:num_tokens]
            num_heads = embeddings_from_buffer.shape[1]
            embed_dim = embeddings_from_buffer.shape[2]
            # Reshape to [B, T, num_heads, D]
            embeddings_from_buffer = embeddings_from_buffer.reshape(B, T, num_heads, embed_dim)
            # Flatten to [B, T, num_heads * D]
            embeddings = embeddings_from_buffer.flatten(start_dim=-2)
            embeddings = embeddings.to(hidden_states.dtype)
            
            # print("input_ids in engram: ", input_ids)
            # input_ids_np = input_ids.cpu().numpy()
            # hash_ids_np = self.hash_mapping.hash(input_ids_np)[self.layer_id]
            # hash_ids = torch.from_numpy(hash_ids_np).to(hidden_states.device)
            # embeddings2 = self.multi_head_embedding(hash_ids).flatten(start_dim=-2)
            
            # print(f"hash_ids: {hash_ids}")
            # print(f"if same: {torch.allclose(embeddings, embeddings2)}")
            # print(f"diff: {embeddings - embeddings2}")

        else:
            input_ids_np = input_ids.cpu().numpy()
            hash_ids_np = self.hash_mapping.hash(input_ids_np)[self.layer_id]
            hash_ids = torch.from_numpy(hash_ids_np).to(hidden_states.device)
            embeddings = self.multi_head_embedding(hash_ids).flatten(start_dim=-2)
        
        key = self.key_norm(self.key_proj(embeddings))
        value = self.value_proj(embeddings)
        
        query = self.query_norm(hidden_states)
        
        gate = (key * query).sum(dim=-1) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)
        
        gated_value = gate * value
        output = gated_value + self.short_conv(gated_value)
        
        return output
