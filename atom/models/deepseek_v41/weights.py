# SPDX-License-Identifier: MIT
"""Native checkpoint schema, tensor ownership and bounded TP/EP loading.

Model math does not open checkpoints. This module accounts for every published
tensor, including explicit draft/vision exclusions and host-owned Engram tables.
"""

import json
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from atom.model_ops.engram.tables import HostEmbeddingTable

from .config import AttentionMode, build_attention_topology


@dataclass(frozen=True)
class WeightSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    scope: str = "backbone"
    tp_axis: int | None = None
    expert_id: int | None = None
    expert_count: int = 0
    block: tuple[int, int] | None = None
    dequantize: bool = False


@dataclass(frozen=True)
class LocalWeight:
    source: WeightSpec
    target: str | None
    action: str
    reason: str
    axis: int | None = None
    start: int = 0
    length: int = 0

    @property
    def shape(self):
        shape = list(self.source.shape)
        if self.axis is not None:
            shape[self.axis] = self.length
        return tuple(shape)


def checkpoint_schema(config) -> dict[str, WeightSpec]:
    """Build the exact parameter set from topology, independently of shard names."""
    specs = {}

    def add(name, shape, dtype="BF16", **kwargs):
        if name in specs:
            raise ValueError(f"Duplicate checkpoint parameter {name}")
        specs[name] = WeightSpec(name, tuple(shape), dtype, **kwargs)

    def linear(name, n, k, *, quantized=True, fp4=False, **kwargs):
        block = (1, 32) if fp4 else (32, 32)
        add(
            name + ".weight",
            (n, k // 2 if fp4 else k),
            "I8" if fp4 else ("F8_E4M3" if quantized else "BF16"),
            block=block if quantized else None,
            **kwargs,
        )
        if quantized:
            add(
                name + ".scale",
                ((n + block[0] - 1) // block[0], (k + 31) // 32),
                "F8_E8M0",
                block=block,
                **kwargs,
            )

    d, h, hd = config.hidden_size, config.num_attention_heads, config.head_dim
    hc, mid = config.hc_mult, config.moe_intermediate_size
    for name in ("embed", "head"):
        linear(name, config.vocab_size, d, quantized=False, tp_axis=0)
    add("norm.weight", (d,))
    for layer in build_attention_topology(config):
        draft = layer.layer_id >= config.num_hidden_layers
        scope = "draft" if draft else "backbone"
        layer_id = (
            layer.layer_id - config.num_hidden_layers if draft else layer.layer_id
        )
        prefix = f"{'mtp' if draft else 'layers'}.{layer_id}"
        for sublayer in ("attn", "ffn"):
            add(f"{prefix}.{sublayer}_norm.weight", (d,), scope=scope)
            mixes = hc * (hc + 2)
            for suffix, shape in (
                ("fn", (mixes, hc * d)),
                ("base", (mixes,)),
                ("scale", (3,)),
            ):
                add(f"{prefix}.hc_{sublayer}_{suffix}", shape, "F32", scope=scope)
        attn = prefix + ".attn"
        add(attn + ".attn_sink", (h,), "F32", scope=scope, tp_axis=0)
        add(attn + ".q_norm.weight", (config.q_lora_rank,), scope=scope)
        add(attn + ".kv_norm.weight", (hd,), scope=scope)
        linear(attn + ".wq_a", config.q_lora_rank, d, scope=scope)
        linear(attn + ".wq_b", h * hd, config.q_lora_rank, scope=scope, tp_axis=0)
        linear(attn + ".wkv", hd, d, scope=scope)
        linear(
            attn + ".wo_a",
            config.o_groups * config.o_lora_rank,
            h * hd // config.o_groups,
            scope=scope,
            tp_axis=0,
            dequantize=True,
        )
        linear(
            attn + ".wo_b",
            d,
            config.o_groups * config.o_lora_rank,
            scope=scope,
            tp_axis=1,
        )
        if layer.mode == AttentionMode.FULL:
            linear(attn + ".compressor.wkv", hd, d, quantized=False, scope=scope)
            if layer.ratio == 2:
                linear(attn + ".compressor.wgate", hd, d, quantized=False, scope=scope)
            add(attn + ".compressor.norm.weight", (hd,), scope=scope)
            linear(
                attn + ".indexer.wk",
                config.index_head_dim,
                hd,
                quantized=False,
                scope=scope,
            )
            add(attn + ".indexer.k_norm.weight", (config.index_head_dim,), scope=scope)
        if layer.mode in (AttentionMode.FULL, AttentionMode.REINDEX):
            # Replicate the small indexer: scores must reduce all its heads
            # before top-k, without a context-sized all-reduce on every layer.
            linear(
                attn + ".indexer.wq_b",
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
                scope=scope,
            )
            linear(
                attn + ".indexer.weights_proj",
                config.index_n_heads,
                d,
                quantized=False,
                scope=scope,
            )
        experts = config.dspark_n_routed_experts if draft else config.n_routed_experts
        ffn = prefix + ".ffn"
        linear(ffn + ".gate", experts, d, quantized=False, scope=scope)
        for bias in ("bias", "bias_vl"):
            add(ffn + ".gate." + bias, (experts,), "F32", scope=scope)
        for expert in range(experts):
            for proj, n, k, axis in (
                ("w1", mid, d, 0),
                ("w3", mid, d, 0),
                ("w2", d, mid, 1),
            ):
                linear(
                    f"{ffn}.experts.{expert}.{proj}",
                    n,
                    k,
                    fp4=True,
                    scope=scope,
                    tp_axis=axis,
                    expert_id=expert,
                    expert_count=experts,
                )
        for proj, n, k, axis in (
            ("w1", mid, d, 0),
            ("w3", mid, d, 0),
            ("w2", d, mid, 1),
        ):
            linear(f"{ffn}.shared_experts.{proj}", n, k, scope=scope, tp_axis=axis)
        if draft:
            if layer_id == config.num_nextn_predict_layers - 1:
                r = config.dspark_markov_rank
                for name in ("embed", "head"):
                    linear(
                        f"{prefix}.markov_head.{name}",
                        config.vocab_size,
                        r,
                        quantized=False,
                        scope=scope,
                        tp_axis=0,
                    )
                linear(
                    prefix + ".confidence_head.proj",
                    1,
                    d + r,
                    quantized=False,
                    scope=scope,
                )
                add(prefix + ".norm.weight", (d,), scope=scope)
            if layer_id == 0:
                linear(
                    prefix + ".main_proj",
                    d,
                    d * len(config.dspark_target_layer_ids),
                    scope=scope,
                )
                add(prefix + ".main_norm.weight", (d,), scope=scope)
    engram_dim = (
        (config.engram_max_ngram_size - 1)
        * config.engram_n_heads
        * config.engram_head_dim
    )
    for layer, rows in zip(config.engram_layer_ids, config.engram_num_embeddings):
        prefix = f"layers.{layer}.engram"
        add(
            prefix + ".embed.weight",
            (rows, config.engram_head_dim),
            "F8_E4M3",
            scope="engram_table",
            block=(1, 32),
        )
        add(
            prefix + ".embed.scale",
            (rows, config.engram_head_dim // 32),
            "F8_E8M0",
            scope="engram_table",
            block=(1, 32),
        )
        linear(prefix + ".wkv", (hc + 1) * d, engram_dim)
        for name in ("k_weight", "q_weight"):
            add(prefix + "." + name, (hc, d))
    vision = config._multimodal_config.vision_config
    vd, vi = vision.hidden_size, vision.intermediate_size
    for name in ("image_start", "image_end", "image_newline"):
        add(name, (d,), scope="vision")
    for name, n, k in (
        ("aligner.w1", d, vd * vision.downsample_ratio**2),
        ("aligner.w2", d, d),
        ("vision.patch_embed.proj", vd, 3 * vision.patch_size**2),
    ):
        linear(name, n, k, quantized=False, scope="vision")
        add(name + ".bias", (n,), scope="vision")
    add("vision.norm.weight", (vd,), scope="vision")
    for i in range(vision.num_hidden_layers):
        prefix = f"vision.blocks.{i}"
        for name in ("norm1", "norm2"):
            add(f"{prefix}.{name}.weight", (vd,), scope="vision")
        for name, n, k in (
            ("attn.wqkv", 3 * vd, vd),
            ("attn.wo", vd, vd),
            ("mlp.w1", 2 * vi, vd),
            ("mlp.w2", vd, vi),
        ):
            linear(f"{prefix}.{name}", n, k, quantized=False, scope="vision")
            if name.startswith("attn."):
                add(f"{prefix}.{name}.bias", (n,), scope="vision")
    return specs


def build_weight_manifest(
    schema, *, tp_rank=0, tp_size=1, ep_rank=0, ep_size=1, scopes=("backbone",)
) -> tuple[LocalWeight, ...]:
    """Assign local source slices and explicit exclusions before touching payloads."""
    if not 0 <= tp_rank < tp_size or not 0 <= ep_rank < ep_size:
        raise ValueError("Invalid TP/EP rank or size")
    if set(scopes) - {"backbone", "vision", "draft"}:
        raise ValueError("Unknown checkpoint scope")
    result = []
    for spec in schema.values():
        if spec.scope == "engram_table":
            action = "host" if "backbone" in scopes else "skip"
            result.append(LocalWeight(spec, None, action, "Engram mmap provider"))
            continue
        if spec.scope not in scopes:
            result.append(
                LocalWeight(spec, None, "skip", f"{spec.scope} explicitly excluded")
            )
            continue
        axis = spec.tp_axis
        if spec.expert_id is not None and ep_size > 1:
            if spec.expert_count % ep_size:
                raise ValueError("Experts must divide evenly over EP ranks")
            if spec.expert_id // (spec.expert_count // ep_size) != ep_rank:
                result.append(
                    LocalWeight(spec, None, "skip", "Expert owned by another EP rank")
                )
                continue
            axis = (
                None  # EP owns whole experts; do not also split their intermediate dim.
            )
        if tp_size == 1:
            axis = None
        start = length = 0
        if axis is not None:
            if spec.shape[axis] % tp_size:
                raise ValueError(f"TP partition crosses source blocks: {spec.name}")
            length = spec.shape[axis] // tp_size
            start = tp_rank * length
            if spec.block and spec.dtype != "F8_E8M0":
                packing = 2 if spec.dtype == "I8" and axis == 1 else 1
                if (length * packing) % spec.block[axis]:
                    raise ValueError(f"TP partition crosses source blocks: {spec.name}")
        is_scale = spec.name.endswith(".scale")
        target = (
            spec.name.removesuffix(".scale") + ".weight_scale"
            if is_scale
            else spec.name
        )
        action = "dequant_scale" if spec.dequantize and is_scale else "load"
        result.append(
            LocalWeight(
                spec,
                None if action == "dequant_scale" else target,
                action,
                "Native source layout",
                axis,
                start,
                length,
            )
        )
    return tuple(result)


class CheckpointReader:
    """Lazy CPU safetensors mappings; only a requested rank's slice is copied.

    Keep this reader alive while using its Engram tables. No GPU device context
    is used when mapping the tables, and generic parameter loading excludes them.
    """

    def __init__(self, directory, schema):
        self.directory = Path(directory)
        index = json.loads(
            (self.directory / "model.safetensors.index.json").read_text()
        )
        self.weight_map = index["weight_map"]
        self._stack = ExitStack()
        self._handles = {}
        missing = schema.keys() - self.weight_map.keys()
        unexpected = self.weight_map.keys() - schema.keys()
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint schema mismatch: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}"
            )
        seen = set()
        for shard in sorted(set(self.weight_map.values())):
            with (self.directory / shard).open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
            for name, data in header.items():
                if name == "__metadata__":
                    continue
                if name in seen or self.weight_map.get(name) != shard:
                    raise ValueError(f"Index/shard disagreement: {name}")
                spec = schema[name]
                if tuple(data["shape"]) != spec.shape or data["dtype"] != spec.dtype:
                    raise ValueError(f"Source shape/dtype mismatch for {name}: {data}")
                seen.add(name)
        if seen != set(schema):
            raise ValueError("Checkpoint shards are incomplete")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._stack.close()
        self._handles.clear()

    def _handle(self, name):
        shard = self.weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = self._stack.enter_context(
                safe_open(self.directory / shard, framework="pt", device="cpu")
            )
        return self._handles[shard]

    def read(self, entry: LocalWeight):
        if entry.action not in ("load", "dequant_scale"):
            raise ValueError(
                f"{entry.source.name} is owned by {entry.action}, not the GPU parameter loader"
            )
        tensor = self._handle(entry.source.name).get_slice(entry.source.name)
        indices = [slice(None)] * len(entry.source.shape)
        if entry.axis is not None:
            indices[entry.axis] = slice(entry.start, entry.start + entry.length)
        tensor = tensor[tuple(indices)]
        return (
            tensor.view(torch.float4_e2m1fn_x2)
            if entry.source.dtype == "I8"
            else tensor
        )

    def engram_tables(self, config):
        # Reuse the table implementation from PR #2185, not a second lookup path.
        tables = {}
        for layer, rows in zip(config.engram_layer_ids, config.engram_num_embeddings):
            name = f"layers.{layer}.engram.embed"
            tensors = [
                self._handle(name + suffix).get_tensor(name + suffix)
                for suffix in (".weight", ".scale")
            ]
            tables[layer] = HostEmbeddingTable(
                tensors[0],
                num_rows=rows,
                head_dim=config.engram_head_dim,
                scale=tensors[1],
            )
        return tables
