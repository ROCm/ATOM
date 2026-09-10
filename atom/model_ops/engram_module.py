"""Self-contained Engram attachment.

A model gets engram by holding one of these and calling it in two places. It owns
everything engram needs -- config, compressed tokenizer, hash layout, the host
tables, and the per-layer device modules -- so nothing about n-gram hashing or
98 GB embedding tables leaks into the model definition.

    class DeepseekV41ForCausalLM(nn.Module):
        def __init__(self, config):
            ...
            self.engram = EngramModules.from_checkpoint(config.model, config.hf_config)

        # the model runner looks for exactly this name
        def build_engram_runtime(self, device, max_num_tokens):
            return self.engram.build_engram_runtime(device, max_num_tokens)

    # and in the decoder stack, for the layers that have one:
    if layer_id in self.engram:
        hidden = hidden + self.engram[layer_id](hidden, runtime.embeddings(layer_id))
"""

from __future__ import annotations

import json
import logging
import os

import torch
from torch import nn

from atom.model_ops.engram import CompressedTokenizer, EngramConfig, NgramHashMapping
from atom.model_ops.engram_host import (
    EngramPrefetcher,
    EngramRuntime,
    HostEmbeddingTable,
)
from atom.model_ops.engram_layer import EngramOp

logger = logging.getLogger(__name__)

_EMBED = "layers.{}.engram.embed.weight"
_EMBED_SCALE = "layers.{}.engram.embed.scale"
_WKV = "layers.{}.engram.wkv.weight"
_WKV_SCALE = "layers.{}.engram.wkv.scale"
_K_WEIGHT = "layers.{}.engram.k_weight"
_Q_WEIGHT = "layers.{}.engram.q_weight"


class EngramModules(nn.Module):
    """The engram layers of one model, plus the host state they read from."""

    def __init__(
        self,
        config: EngramConfig,
        hash_mapping: NgramHashMapping,
        ops: dict[int, EngramOp],
        tables: dict[int, HostEmbeddingTable],
        handles: list = (),
    ):
        super().__init__()
        self.config = config
        self.hash_mapping = hash_mapping
        self.ops = nn.ModuleDict({str(k): v for k, v in ops.items()})
        self._tables = tables
        # The tables are memory-mapped views into the checkpoint. Dropping these
        # handles unmaps them, so they are held for the model's lifetime.
        self._handles = list(handles)

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self.config.layer_ids

    def __contains__(self, layer_id: int) -> bool:
        return str(layer_id) in self.ops

    def __getitem__(self, layer_id: int) -> EngramOp:
        return self.ops[str(layer_id)]

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        hf_config: dict | None = None,
        tokenizer=None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> EngramModules | None:
        """Build from a checkpoint directory. None when the model has no engram.

        The embedding tables are memory-mapped, never materialized: they are
        about 98 GB each here, and a host that can hold them in page cache still
        cannot afford to copy them into the process.
        """
        if hf_config is None:
            with open(os.path.join(model_path, "config.json")) as fh:
                hf_config = json.load(fh)
        text_config = hf_config.get("text_config", hf_config)
        config = EngramConfig.from_hf(text_config)
        if config is None:
            return None

        hidden_size = int(text_config["hidden_size"])
        hc_mult = int(text_config.get("hc_mult", 4))
        norm_eps = float(text_config.get("rms_norm_eps", 1e-6))
        engram_hidden = config.num_hash_heads * config.head_dim

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        hash_mapping = NgramHashMapping(
            config,
            CompressedTokenizer(tokenizer, expected_size=config.compressed_vocab_size),
        )

        from safetensors import safe_open

        with open(os.path.join(model_path, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]

        ops: dict[int, EngramOp] = {}
        tables: dict[int, HostEmbeddingTable] = {}
        handles = []
        for layer_id, num_rows in zip(config.layer_ids, config.num_embeddings):
            shard = weight_map[_EMBED.format(layer_id)]
            handle = safe_open(os.path.join(model_path, shard), framework="pt")
            handles.append(handle)
            tables[layer_id] = HostEmbeddingTable(
                handle.get_tensor(_EMBED.format(layer_id)),
                num_rows=num_rows,
                head_dim=config.head_dim,
                scale=handle.get_tensor(_EMBED_SCALE.format(layer_id)),
            )
            op = EngramOp(
                layer_id,
                hidden_size=hidden_size,
                engram_hidden_size=engram_hidden,
                hc_mult=hc_mult,
                norm_eps=norm_eps,
            ).to(dtype)
            op.load_checkpoint_weights(
                handle.get_tensor(_WKV.format(layer_id)),
                handle.get_tensor(_K_WEIGHT.format(layer_id)),
                handle.get_tensor(_Q_WEIGHT.format(layer_id)),
                wkv_scale=handle.get_tensor(_WKV_SCALE.format(layer_id)),
            )
            ops[layer_id] = op

        logger.info(
            "engram: %d modules on layers %s, %d hash heads, tables mapped from %s",
            len(ops),
            list(config.layer_ids),
            config.num_hash_heads,
            model_path,
        )
        return cls(config, hash_mapping, ops, tables, handles)

    def build_engram_runtime(
        self,
        device: torch.device,
        max_num_tokens: int,
        dtype: torch.dtype | None = None,
    ) -> EngramRuntime:
        """The contract ModelRunner looks for by name.

        The staging buffers default to the dtype the engram layers actually
        compute in, so the embeddings arrive ready to feed `wkv` -- staging in
        float32 against bf16 weights is a dtype error at the first matmul.
        """
        if dtype is None:
            dtype = next(self.ops.parameters()).dtype
        return EngramRuntime(
            EngramPrefetcher(self.hash_mapping, self._tables),
            max_num_tokens=max_num_tokens,
            num_hash_heads=self.config.num_hash_heads,
            head_dim=self.config.head_dim,
            device=device,
            dtype=dtype,
        )
