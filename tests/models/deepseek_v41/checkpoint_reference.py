# SPDX-License-Identifier: MIT
"""Pinned real-checkpoint oracle, isolated from the serving model and kernels."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_ops.blockscale import dequantize_fp8_weight
from atom.models.deepseek_v41.weights import (
    CheckpointReader,
    build_weight_manifest,
    checkpoint_schema,
)

from .reference import load_reference


@contextmanager
def checkpoint_reference(directory, config, max_length, *, vision=False):
    """Load upstream text math with independent PyTorch quant/GEMM/attention.

    Engram table gathers are supplied by the caller (their hash and table
    contracts have separate differential tests). No target kernel, residual,
    reduction, attention output or logits are substituted into this model.
    """
    directory = Path(directory)
    group = get_tp_group()
    raw = json.loads((directory / "inference/config.json").read_text())
    raw.update(
        max_batch_size=1,
        max_seq_len=max_length,
        temperature=0,
        n_mtp_layers=0,
        dspark_block_size=0,
        engram_layer_ids=(),
        engram_num_embeddings=(),
        vision_n_layers=raw.get("vision_n_layers", 0) if vision else 0,
    )
    if vision:
        vision_config = config._multimodal_config.vision_config
        for native, hf in {
            "vision_n_layers": "num_hidden_layers",
            "vision_dim": "hidden_size",
            "vision_n_heads": "num_attention_heads",
            "vision_inter_dim": "intermediate_size",
            "vision_patch_size": "patch_size",
            "vision_rope_theta": "rope_theta",
            "vision_downsample_ratio": "downsample_ratio",
        }.items():
            raw[native] = getattr(vision_config, hf)
    with load_reference(directory) as reference, reference.set_dtype(torch.bfloat16):
        with torch.device("cuda"):
            source = reference.Transformer(reference.ModelArgs(**raw))
            for layer in source.layers:
                layer.ffn.gate.bias_vl = nn.Parameter(
                    torch.empty(config.n_routed_experts, dtype=torch.float32),
                    requires_grad=False,
                )
            for index, layer_id in enumerate(config.engram_layer_ids):
                engram = reference.Engram.__new__(reference.Engram)
                nn.Module.__init__(engram)
                engram.dim = config.hidden_size
                engram.hc_mult = config.hc_mult
                engram.eps = config.rms_norm_eps
                engram.clamp_value = 1e-6
                engram.layer_hash_index = index
                engram.embed = nn.Identity()
                width = (
                    (config.engram_max_ngram_size - 1)
                    * config.engram_n_heads
                    * config.engram_head_dim
                )
                engram.wkv = reference.Linear(
                    width,
                    (config.hc_mult + 1) * config.hidden_size,
                    dtype=torch.float8_e4m3fn,
                )
                for name in ("q_weight", "k_weight"):
                    engram.register_parameter(
                        name,
                        nn.Parameter(
                            torch.empty(config.hc_mult, config.hidden_size),
                            requires_grad=False,
                        ),
                    )
                source.layers[layer_id].engram = engram
        schema = checkpoint_schema(config)
        source_schema = {}
        for name, spec in schema.items():
            if ".ffn.shared_experts." in name:
                spec = replace(spec, tp_axis=None)
            elif ".attn.indexer.wq_b." in name or ".attn.indexer.weights_proj." in name:
                spec = replace(spec, tp_axis=0)
            source_schema[name] = spec
        manifest = build_weight_manifest(
            source_schema,
            scopes=("backbone", "vision") if vision else ("backbone",),
            tp_rank=group.rank_in_group,
            tp_size=group.world_size,
            ep_rank=group.rank_in_group,
            ep_size=group.world_size,
        )
        entries = {entry.source.name: entry for entry in manifest}
        params, loaded = dict(source.named_parameters()), set()
        with CheckpointReader(directory, schema) as reader, torch.no_grad():
            for entry in manifest:
                if entry.action != "load":
                    continue
                name = entry.source.name
                value = reader.read(entry)
                if entry.source.dequantize:
                    scale = reader.read(
                        entries[name.removesuffix(".weight") + ".scale"]
                    )
                    value = dequantize_fp8_weight(value, scale)
                parameter = params[name]
                if parameter.shape != value.shape:
                    raise ValueError(f"Reference weight shape mismatch: {name}")
                if value.dtype == torch.float4_e2m1fn_x2:
                    parameter.view(torch.uint8).copy_(value.view(torch.uint8))
                else:
                    parameter.copy_(value)
                loaded.add(name)
        if loaded != set(params):
            raise ValueError(f"Unloaded reference parameters: {set(params) - loaded}")

        @torch.inference_mode()
        def forward(
            tokens,
            position,
            embeddings,
            *,
            full_logits=False,
            logits_start=0,
            images=None,
            token_types=None,
        ):
            # generate.py also installs this default device; upstream index
            # masks otherwise default to CPU.
            with torch.device(tokens.device):
                hidden = source.embed(tokens)
                if images is not None:
                    source.merge_image_embeddings(images, hidden)
                image_mask = None if token_types is None else token_types >= 0
                hidden = hidden.unsqueeze(2).repeat(1, 1, config.hc_mult, 1)
                mix = reference.make_identity_pre_mix(hidden, config.hc_mult)
                for layer_id, layer in enumerate(source.layers):
                    if layer.engram is not None:
                        rows = embeddings[layer_id].unflatten(
                            -1, (-1, config.engram_head_dim)
                        )
                        hidden = layer.engram(
                            hidden, rows, None if image_mask is None else ~image_mask
                        )
                    hidden, mix = layer(hidden, position, mix, image_mask)
                hidden = (
                    (mix.unsqueeze(-1) * hidden.float()).sum(dim=2).to(hidden.dtype)
                )
                return source.head(
                    source.norm(hidden[:, logits_start:]), full_logits=full_logits
                )

        yield forward


@contextmanager
def reference_resources(directory, max_length, *, vision=False):
    """Reference model and bounded host Engram staging without a target model."""
    from transformers import AutoTokenizer

    from atom.config import get_hf_config
    from atom.model_engine.engram_runtime import EngramHost, EngramPrefetcher
    from atom.model_ops.engram import (
        CompressedTokenizer,
        EngramConfig,
        NgramHashMapping,
    )

    config = get_hf_config(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    engram = EngramConfig.from_hf(config.to_dict())
    mapping = NgramHashMapping(
        engram,
        CompressedTokenizer(tokenizer, expected_size=engram.compressed_vocab_size),
    )
    with (
        CheckpointReader(directory, checkpoint_schema(config)) as reader,
        checkpoint_reference(directory, config, max_length, vision=vision) as reference,
    ):
        host = EngramHost(
            EngramPrefetcher(mapping, reader.engram_tables(config)),
            max_length,
            engram.num_hash_heads,
            engram.head_dim,
            torch.device("cuda"),
        )
        try:
            yield reference, tokenizer, mapping, host
        finally:
            host.shutdown()
