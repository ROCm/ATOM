# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next's geometry (HF / ATOM ``qwen4_exp``), checked against a checkpoint's ``config.json``, and the
geometry of the layer pieces built from it.

The pieces (``GDNGeometry``, ``AttnGeometry``, ``HCGeometry``, ``MoEGeometry``, ``IndexerGeometry``) are what the
stages take as arguments: stages never read the model's constants. They are frozen (hashable): stages put them in
their build keys.

Every per-layer size of the model is fixed: ``geometry`` raises unless the text config matches ``G``; the model-level
counts (layers, experts, layer types, PLE layers) are read from it. Layer kinds (``Geometry.layer_kind``): ``delta``
(Gated DeltaNet + MoE under two hyper-connections), ``delta_ple`` (the same after the PLE block; ``ple_layer_ids`` is
1-based), ``qsa`` (query-sparse attention + MoE) and ``mtp`` (the MTP drafter's layer, index ``layers``: a QSA layer
with its own weights and caches whose RoPE positions run ``ROPE_OFFSET["mtp"]`` ahead of its cache rows: ATOM's draft
writes token ``t + 1``'s key at row ``t``).
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch


@dataclass(frozen=True)
class GDNGeometry:
    """A gated delta net mixer (Qwen3.5 / Qwen3-Next style).

    q, k: ``k_heads`` heads of ``k_dim``; v (and z): ``v_heads`` heads of ``v_dim``; each
    k-head serves ``v_heads // k_heads`` consecutive v-heads. The causal conv has
    ``conv_kernel`` taps over the concatenated pre-conv qkv rows (``conv_dim``). The
    gated output norm uses ``eps``. The recurrent state [v_heads, k_dim, v_dim] (V
    contiguous) is *stored* in ``state_dtype`` and computed in fp32.
    """

    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv_kernel: int
    eps: float
    state_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.v_heads % self.k_heads:
            raise ValueError(f"{self}: v_heads must be a multiple of k_heads")

    @property
    def rep(self) -> int:
        """v-heads per k-head."""
        return self.v_heads // self.k_heads

    @property
    def key_width(self) -> int:
        return self.k_heads * self.k_dim

    @property
    def value_width(self) -> int:
        return self.v_heads * self.v_dim

    @property
    def conv_dim(self) -> int:
        """Pre-conv qkv channels: q | k | v."""
        return 2 * self.key_width + self.value_width


@dataclass(frozen=True)
class AttnGeometry:
    """Gated multi-head attention with GQA and partial RoPE (Qwen3.5 full attention).

    ``q_heads`` query heads over ``kv_heads`` KV heads of ``head_dim``; RoPE rotates the
    first ``rotary_dim`` dims (base ``rope_theta``, HF's half-split rotation); q and k
    are RMS-normed per head (zero-centred ``1 + w``, ``eps``) before RoPE. The fused
    projection is q_proj (per head [q, gate]) | k_proj | v_proj.
    """

    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    eps: float

    def __post_init__(self):
        if (
            self.q_heads % self.kv_heads
            or self.rotary_dim % 2
            or self.rotary_dim > self.head_dim
        ):
            raise ValueError(f"unsupported attention geometry {self}")

    @property
    def rep(self) -> int:
        """Query heads per KV head."""
        return self.q_heads // self.kv_heads

    @property
    def q_width(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass(frozen=True)
class HCGeometry:
    """Hyper-connections: ``streams`` residual streams of ``hidden`` (the flat row is
    stream-major, ``streams * hidden``); each sub-layer's HC mixes them through a low-rank
    path of ``lowrank`` columns and normalizes each stream with ``eps``."""

    hidden: int
    streams: int
    lowrank: int
    eps: float

    @property
    def width(self) -> int:
        """The flat stream row: streams * hidden."""
        return self.streams * self.hidden


@dataclass(frozen=True)
class MoEGeometry:
    """A routed MoE over rows of ``hidden``: ``experts`` routed SwiGLU experts of
    ``inter`` (softmax router, the top ``top_k`` renormalized) plus ``shared`` always-on
    experts of ``shared_inter`` with a sigmoid gate."""

    hidden: int
    experts: int
    top_k: int
    inter: int
    shared: int = 1
    shared_inter: int = 0

    def __post_init__(self):
        if not 0 < self.top_k <= self.experts:
            raise ValueError(f"unsupported MoE geometry {self}")


@dataclass(frozen=True)
class IndexerGeometry:
    """A block indexer for sparse attention: ``heads`` index query heads of ``head_dim``
    score one key per block of ``compress`` tokens; a query attends the tokens of its
    ``top_blocks`` best blocks plus the incomplete tail. Queries and block keys are
    RMS-normed per head (zero-centred, ``eps``) and rotated on their first
    ``rotary_dim`` dims (base ``rope_theta``, half-split rotation)."""

    heads: int
    head_dim: int
    compress: int
    top_blocks: int
    rotary_dim: int
    rope_theta: float
    eps: float

    def __post_init__(self):
        if (
            self.rotary_dim % 2
            or self.rotary_dim > self.head_dim
            or self.top_blocks < 1
        ):
            raise ValueError(f"unsupported indexer geometry {self}")


# ReplaySSM record capacities the GDN programs are built for (``blocks.gdn``: the replay loop is unrolled); the
# deployment contract refuses a larger ``ATOM_REPLAYSSM_CACHE_LEN`` (``config``)
REPLAY_MAX = 64

QSA_TYPES = ("full_attention", "qwen_sparse_attention")
QSA_KINDS = ("qsa", "mtp")  # kinds whose mixer is QSA (one recipe, one kernel)
ROPE_OFFSET = {"qsa": 0, "mtp": 1}  # RoPE position - cache row, per QSA kind


@dataclass(frozen=True)
class Geometry:
    hidden: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    eps: float = 1e-6
    # Gated DeltaNet (linear attention)
    gdn_k_heads: int = 16
    gdn_v_heads: int = 48
    gdn_k_dim: int = 128
    gdn_v_dim: int = 128
    conv_kernel: int = 4
    # QSA: gated GQA with partial RoPE and the block indexer
    q_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 10_000_000.0
    index_heads: int = 4
    index_dim: int = 128
    index_budget: int = 2048  # tokens: the top index_budget / index_compress blocks
    index_compress: int = 4
    # MoE
    top_k: int = 10
    moe_inter: int = 640
    shared_inter: int = 640
    ple_dim: int = 2560
    # model-level counts
    layers: int = 48
    experts: int = 512
    layer_types: tuple = field(
        default=tuple(
            "full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)
        )
    )
    ple_layers: tuple = (1,)  # 0-based

    @property
    def hc_width(self) -> int:
        return self.hc_count * self.hidden  # 10240: the four residual streams

    @property
    def gdn_value_width(self) -> int:
        return self.gdn_v_heads * self.gdn_v_dim  # 6144

    @property
    def conv_dim(self) -> int:
        # 10240: q | k | v through the conv
        return 2 * self.gdn_k_heads * self.gdn_k_dim + self.gdn_value_width

    @property
    def q_width(self) -> int:
        return self.q_heads * self.head_dim  # 6144

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim  # 512

    @property
    def hc(self) -> HCGeometry:
        return HCGeometry(self.hidden, self.hc_count, self.hc_lowrank, self.eps)

    @property
    def gdn(self) -> GDNGeometry:
        """The recurrent state is stored fp32 (the checkpoint's ``mamba_ssm_dtype``, ATOM's state pool)."""
        return GDNGeometry(
            self.gdn_k_heads,
            self.gdn_v_heads,
            self.gdn_k_dim,
            self.gdn_v_dim,
            self.conv_kernel,
            self.eps,
            torch.float32,
        )

    @property
    def attn(self) -> AttnGeometry:
        return AttnGeometry(
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            self.eps,
        )

    @property
    def indexer(self) -> IndexerGeometry:
        blocks = self.index_budget // self.index_compress  # 512 selected blocks at most
        return IndexerGeometry(
            self.index_heads,
            self.index_dim,
            self.index_compress,
            blocks,
            self.rotary_dim,
            self.rope_theta,
            self.eps,
        )

    @property
    def moe(self) -> MoEGeometry:
        return MoEGeometry(
            self.hidden, self.experts, self.top_k, self.moe_inter, 1, self.shared_inter
        )

    def layer_kind(self, layer: int) -> str:
        """The kind of decoder layer ``layer``; index ``layers`` is the MTP drafter's (``mtp``)."""
        if layer == self.layers:
            return "mtp"
        if not 0 <= layer < self.layers:
            raise IndexError(f"layer {layer} of {self.layers}")
        if self.layer_types[layer] in QSA_TYPES:
            if layer in self.ple_layers:
                raise ValueError("PLE on a QSA layer is not supported")
            return "qsa"
        return "delta_ple" if layer in self.ple_layers else "delta"


G = Geometry()

# Per-layer sizes the kernels are written for: config.json key -> Geometry field.
_FIXED = {
    "hidden_size": "hidden",
    "hc_count": "hc_count",
    "hc_lowrank": "hc_lowrank",
    "rms_norm_eps": "eps",
    "linear_num_key_heads": "gdn_k_heads",
    "linear_num_value_heads": "gdn_v_heads",
    "linear_key_head_dim": "gdn_k_dim",
    "linear_value_head_dim": "gdn_v_dim",
    "linear_conv_kernel_dim": "conv_kernel",
    "num_attention_heads": "q_heads",
    "num_key_value_heads": "kv_heads",
    "head_dim": "head_dim",
    "indexer_n_heads": "index_heads",
    "indexer_head_dim": "index_dim",
    "indexer_budget": "index_budget",
    "indexer_compress_ratio": "index_compress",
    "num_experts_per_tok": "top_k",
    "moe_intermediate_size": "moe_inter",
    "shared_expert_intermediate_size": "shared_inter",
    "ple_embed_dim": "ple_dim",
}
# Fixed by the model's code rather than a size: checked too.
_REQUIRED = {
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "indexer_kv_heads": 1,
    "output_gate_type": "sigmoid",
    "hidden_act": "silu",
    "mamba_ssm_dtype": "float32",
}


def geometry(config: dict) -> Geometry:
    """The ``Geometry`` of an HF ``config.json`` dict (or its ``text_config``); ``ValueError`` on any per-layer size
    other than :data:`G`'s."""
    text = config.get("text_config", config)
    bad = {
        key: (text[key], getattr(G, name))
        for key, name in _FIXED.items()
        if text.get(key) != getattr(G, name)
    }
    bad.update(
        {
            key: (text.get(key), want)
            for key, want in _REQUIRED.items()
            if text.get(key) != want
        }
    )
    rope = text.get("rope_parameters", {})
    rotary = int(
        text["head_dim"]
        * rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0))
    )
    if rotary != G.rotary_dim:
        bad["rotary_dim"] = (rotary, G.rotary_dim)
    if float(rope.get("rope_theta", 0)) != G.rope_theta:
        bad["rope_theta"] = (rope.get("rope_theta"), G.rope_theta)
    if bad:
        raise ValueError(
            f"unsupported Qwen3.8-Flash-Next geometry (got, supported): {bad}"
        )
    geo = replace(
        G,
        layers=int(text["num_hidden_layers"]),
        experts=int(text["num_experts"]),
        layer_types=tuple(text["layer_types"]),
        ple_layers=tuple(i - 1 for i in text.get("ple_layer_ids", ())),
    )
    if len(geo.layer_types) != geo.layers:
        raise ValueError(f"{len(geo.layer_types)} layer_types for {geo.layers} layers")
    for i in range(geo.layers):
        geo.layer_kind(i)  # raises on PLE-on-QSA
    return geo


def load_geometry(checkpoint) -> Geometry:
    """The served geometry, read from ``config.json`` in the local ``checkpoint`` directory."""
    return geometry(json.loads((Path(checkpoint) / "config.json").read_text()))
