"""Decode geometry of Qwen3.8-Flash-Next (model type ``qwen4_exp``) at TP1/EP1.

The kernels are specialized at build time on these sizes. :func:`load` reads
them from a checkpoint's ``config.json`` and checks that every per-layer size
matches the one the kernels were written for. Only the model-level counts
(layers, experts, vocab) may differ: the 4-layer CI fixture uses 16 experts.
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

GRID = 256  # one CTA per CU on MI355X
THREADS = 512  # 8 waves per CTA


@dataclass(frozen=True)
class Qwen4Config:
    hidden: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    eps: float = 1e-6

    # Gated DeltaNet (linear attention)
    delta_k_heads: int = 16
    delta_v_heads: int = 48
    delta_k_dim: int = 128
    delta_v_dim: int = 128
    delta_conv: int = 4

    # Full attention with the sparse indexer (QSA)
    q_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64  # partial_rotary_factor 0.25 of head_dim
    rope_theta: float = 1e7
    index_heads: int = 4
    index_dim: int = 128
    index_budget: int = 2048
    index_compress: int = 4

    # MoE
    experts: int = 512
    top_k: int = 10
    moe_inter: int = 640
    shared_inter: int = 640

    # PLE
    ple_dim: int = 2560
    ple_conv: int = 4

    layers: int = 48
    vocab: int = 248320
    layer_types: tuple = field(default=())
    ple_layers: tuple = (1,)  # 0-based; config.json's ple_layer_ids count from 1

    @property
    def hc_width(self):
        return self.hc_count * self.hidden

    @property
    def delta_key(self):
        return self.delta_k_heads * self.delta_k_dim

    @property
    def delta_value(self):
        return self.delta_v_heads * self.delta_v_dim

    @property
    def delta_qkv(self):
        """Channels through the causal conv: q, k and v."""
        return 2 * self.delta_key + self.delta_value

    @property
    def delta_gqa(self):
        return self.delta_v_heads // self.delta_k_heads

    @property
    def qsa_proj(self):
        """Fused QSA projection width: q, gate, k, v, index q, index k."""
        q = self.q_heads * self.head_dim
        kv = self.kv_heads * self.head_dim
        return 2 * q + 2 * kv + self.index_heads * self.index_dim + self.index_dim

    def kind(self, layer):
        """Kernel variant serving ``layer``: ``delta``, ``delta_ple`` or ``qsa``."""
        if self.layer_types[layer] in QSA_TYPES:
            assert layer not in self.ple_layers, "PLE on a QSA layer is not supported"
            return "qsa"
        return "delta_ple" if layer in self.ple_layers else "delta"


# Configs name the QSA layers "full_attention" or "qwen_sparse_attention"
# (transformers' config normalizes the first to the second).
QSA_TYPES = ("full_attention", "qwen_sparse_attention")

DEFAULT = Qwen4Config(
    layer_types=tuple(
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)
    )
)

# Per-layer sizes the kernels are written for; a checkpoint must match them.
_FIXED = {
    "hidden": "hidden_size",
    "hc_count": "hc_count",
    "hc_lowrank": "hc_lowrank",
    "eps": "rms_norm_eps",
    "delta_k_heads": "linear_num_key_heads",
    "delta_v_heads": "linear_num_value_heads",
    "delta_k_dim": "linear_key_head_dim",
    "delta_v_dim": "linear_value_head_dim",
    "delta_conv": "linear_conv_kernel_dim",
    "q_heads": "num_attention_heads",
    "kv_heads": "num_key_value_heads",
    "head_dim": "head_dim",
    "index_heads": "indexer_n_heads",
    "index_dim": "indexer_head_dim",
    "index_budget": "indexer_budget",
    "index_compress": "indexer_compress_ratio",
    "top_k": "num_experts_per_tok",
    "moe_inter": "moe_intermediate_size",
    "shared_inter": "shared_expert_intermediate_size",
    "ple_dim": "ple_embed_dim",
    "ple_conv": "ple_conv_kernel_size",
}


def load(checkpoint):
    """Geometry of the checkpoint directory ``checkpoint`` (its ``config.json``)."""
    return from_dict(json.loads((Path(checkpoint) / "config.json").read_text()))


def from_dict(text):
    """Geometry from a ``config.json`` dict (or a config's ``to_dict()``)."""
    text = text.get("text_config", text)
    for name, key in _FIXED.items():
        if key in text and text[key] != getattr(DEFAULT, name):
            raise ValueError(
                f"{key} = {text[key]}, kernels need {getattr(DEFAULT, name)}"
            )
    rotary = int(text["head_dim"] * text.get("partial_rotary_factor", 0.25))
    if rotary != DEFAULT.rotary_dim:
        raise ValueError(f"rotary dims {rotary}, kernels need {DEFAULT.rotary_dim}")
    return replace(
        DEFAULT,
        experts=text["num_experts"],
        layers=text["num_hidden_layers"],
        vocab=text["vocab_size"],
        layer_types=tuple(text["layer_types"]),
        ple_layers=tuple(i - 1 for i in text.get("ple_layer_ids", ())),
        rope_theta=float(
            text.get("rope_parameters", {}).get("rope_theta", DEFAULT.rope_theta)
        ),
    )
