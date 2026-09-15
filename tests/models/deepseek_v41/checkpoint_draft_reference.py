# SPDX-License-Identifier: MIT
"""Independently loaded pinned V4.1 drafter, without target runtime kernels."""

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
def checkpoint_draft_reference(directory, config, max_length):
    group = get_tp_group()
    raw = json.loads((Path(directory) / "inference/config.json").read_text())
    raw.update(
        max_batch_size=1,
        max_seq_len=max_length,
        temperature=0,
        engram_layer_ids=(),
        engram_num_embeddings=(),
        vision_n_layers=0,
    )
    with load_reference(directory) as reference, reference.set_dtype(torch.bfloat16):
        reference.world_size, reference.rank = group.world_size, group.rank_in_group
        args = reference.ModelArgs(**raw)
        with torch.device("cuda"):
            source = nn.Module()
            source.embed = reference.ParallelEmbedding(args.vocab_size, args.dim)
            source.head = reference.ParallelHead(args.vocab_size, args.dim)
            source.mtp = nn.ModuleList(
                reference.DSparkBlock(args.n_layers + i, args)
                for i in range(args.n_mtp_layers)
            )
            for layer in source.mtp:
                layer.ffn.gate.bias_vl = nn.Parameter(
                    torch.empty(args.dspark_n_routed_experts, dtype=torch.float32),
                    requires_grad=False,
                )
        schema = checkpoint_schema(config)
        reference_schema = {
            name: (
                replace(spec, tp_axis=None) if ".ffn.shared_experts." in name else spec
            )
            for name, spec in schema.items()
        }
        manifest = build_weight_manifest(
            reference_schema,
            scopes=("backbone", "draft"),
            tp_rank=group.rank_in_group,
            tp_size=group.world_size,
            ep_rank=group.rank_in_group,
            ep_size=group.world_size,
        )
        manifest = [
            entry
            for entry in manifest
            if entry.source.scope == "draft"
            or entry.source.name in ("embed.weight", "head.weight")
        ]
        entries = {entry.source.name: entry for entry in manifest}
        params = dict(source.named_parameters())
        loaded = set()
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
                    raise ValueError(f"Reference draft shape mismatch: {name}")
                if value.dtype == torch.float4_e2m1fn_x2:
                    parameter.view(torch.uint8).copy_(value.view(torch.uint8))
                else:
                    parameter.copy_(value)
                loaded.add(name)
        if loaded != set(params):
            raise ValueError(f"Unloaded draft parameters: {set(params) - loaded}")
        for layer in source.mtp:
            layer.embed, layer.head = source.embed, source.head
        yield source, reference
