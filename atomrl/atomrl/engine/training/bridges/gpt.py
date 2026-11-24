# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Standard Megatron-Core GPTModel <-> Llama/Qwen-style HF weights.

The default layout every family starts from: GQA attention with a fused QKV and
optional q/k layernorm, a SwiGLU MLP, and for MoE a router, routed experts and
an optional shared expert (Qwen-MoE naming, ``mlp.shared_expert.*``). Validated
on Qwen3 dense and Qwen3-MoE. Families whose HF names or tensors differ get
their own rules (see ``dsv3``); a model that is close but not identical shows up
as "match no rule" warnings rather than silently missing tensors -- e.g. a QKV
bias (Qwen2) or Mixtral's ``block_sparse_moe.experts.{e}.w1/w2/w3``.

Megatron GQA ``linear_qkv`` layout is interleaved per KV group:
``[g0: q0..q_{r-1}, k0, v0, g1: ...]`` with ``r = num_heads / num_kv_groups``.
``linear_fc1`` is the fused SwiGLU ``[gate; up]`` (gate first).

The TransformerEngine spec fuses the input / pre-MLP RMSNorms into the following
linear (``linear_qkv.layer_norm_weight`` / ``linear_fc1.layer_norm_weight``); the
local spec and every MoE layer keep them standalone (``input_layernorm`` /
``pre_mlp_layernorm``). Export accepts both spellings; loading picks one with
``te``.

Routed experts come in three Megatron layouts: grouped GEMM
(``linear_fc{1,2}.weight{E}``), sequential (``local_experts.{E}.linear_fc{1,2}``)
and the legacy fused GroupedMLP (``weight1`` / ``weight2`` holding all of a
rank's experts).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from atomrl.engine.training.bridges.core import (
    TOP_LEVEL_RULES,
    HfTensor,
    Rule,
    Site,
    WeightBridge,
    gate_up,
    pp_layer_range,
    rename,
    routed_expert_rules,
    split,
    stacked_expert_rules,
)

__all__ = [
    "GPT",
    "GPTDims",
    "GPTMoEDims",
    "build_moe_dims",
    "hf_expert_fc1",
    "hf_expert_fc2",
    "hf_to_megatron",
    "megatron_to_hf",
    "non_expert_hf_to_megatron",
]


@dataclass
class GPTDims:
    num_layers: int = 36
    hidden: int = 4096
    num_heads: int = 32
    num_kv_groups: int = 8
    head_dim: int = 128
    ffn: int = 12288
    vocab: int = 151936
    num_experts: int = 0
    moe_ffn: int = 0
    shared_expert_ffn: int = 0
    shared_expert_gate: bool = False


@dataclass
class GPTMoEDims(GPTDims):
    """Dense ``GPTDims`` + MoE-specific sizes.

    ``ffn`` is unused for pure-routed Qwen3-MoE (no dense MLP); ``moe_ffn`` is the
    per-expert intermediate size (HF ``moe_intermediate_size``). ``shared_ffn`` > 0
    enables a shared expert (HF ``shared_expert_intermediate_size``); Qwen3-30B-A3B
    has none, so it defaults to 0.
    """

    num_experts: int = 128
    moe_ffn: int = 768
    shared_ffn: int = 0


def build_moe_dims(hf_cfg: dict) -> GPTMoEDims:
    """Construct ``GPTMoEDims`` from a HF MoE ``config.json`` dict (Qwen3-MoE keys)."""
    head_dim = hf_cfg.get(
        "head_dim", hf_cfg["hidden_size"] // hf_cfg["num_attention_heads"]
    )
    num_experts = (
        hf_cfg.get("num_experts")
        or hf_cfg.get("n_routed_experts")
        or hf_cfg.get("num_local_experts")
    )
    moe_ffn = hf_cfg.get("moe_intermediate_size") or hf_cfg.get("intermediate_size")
    shared_ffn = int(hf_cfg.get("shared_expert_intermediate_size", 0) or 0)
    return GPTMoEDims(
        num_layers=hf_cfg["num_hidden_layers"],
        hidden=hf_cfg["hidden_size"],
        num_heads=hf_cfg["num_attention_heads"],
        num_kv_groups=hf_cfg["num_key_value_heads"],
        head_dim=head_dim,
        ffn=hf_cfg.get("intermediate_size", 0),
        vocab=hf_cfg["vocab_size"],
        num_experts=int(num_experts),
        moe_ffn=int(moe_ffn),
        shared_ffn=shared_ffn,
    )


def _hf_qkv_to_megatron(q, k, v, d: GPTDims) -> torch.Tensor:
    r = d.num_heads // d.num_kv_groups
    q = q.reshape(d.num_kv_groups, r, d.head_dim, d.hidden)
    k = k.reshape(d.num_kv_groups, 1, d.head_dim, d.hidden)
    v = v.reshape(d.num_kv_groups, 1, d.head_dim, d.hidden)
    qkv = torch.cat([q, k, v], dim=1)  # (groups, r+2, head_dim, hidden)
    return qkv.reshape(-1, d.hidden).contiguous()


def _megatron_qkv_to_hf(qkv, d: GPTDims):
    r = d.num_heads // d.num_kv_groups
    qkv = qkv.reshape(d.num_kv_groups, r + 2, d.head_dim, d.hidden)
    q = qkv[:, :r].reshape(d.num_heads * d.head_dim, d.hidden).contiguous()
    k = qkv[:, r : r + 1].reshape(d.num_kv_groups * d.head_dim, d.hidden).contiguous()
    v = (
        qkv[:, r + 1 : r + 2]
        .reshape(d.num_kv_groups * d.head_dim, d.hidden)
        .contiguous()
    )
    return q, k, v


# ------------------------------------------------------------------ export

_LAYER = "decoder.layers.{L}."
_HF = "model.layers.{L}."


def _legacy_fc1(t: torch.Tensor, site: Site) -> Iterator[HfTensor]:
    """GroupedMLP ``weight1`` [hidden, n_local * 2 * moe_ffn] -> per-expert gate, up."""
    f = site.dims.moe_ffn
    for i in range(t.shape[1] // (2 * f)):
        w = t[:, i * 2 * f : (i + 1) * 2 * f].t()
        p = f"model.layers.{site.layer}.mlp.experts.{site.expert_offset + i}."
        yield p + "gate_proj.weight", w[:f].contiguous()
        yield p + "up_proj.weight", w[f:].contiguous()


def _legacy_fc2(t: torch.Tensor, site: Site) -> Iterator[HfTensor]:
    """GroupedMLP ``weight2`` [n_local * moe_ffn, hidden] -> per-expert down."""
    f = site.dims.moe_ffn
    for i in range(t.shape[0] // f):
        p = f"model.layers.{site.layer}.mlp.experts.{site.expert_offset + i}."
        yield p + "down_proj.weight", t[i * f : (i + 1) * f].t().contiguous()


GPT = WeightBridge(
    "gpt",
    [
        *TOP_LEVEL_RULES,
        # attention
        rename(
            _LAYER + "self_attention.linear_qkv.layer_norm_weight",
            _HF + "input_layernorm.weight",
        ),
        rename(_LAYER + "input_layernorm.weight", _HF + "input_layernorm.weight"),
        split(
            _LAYER + "self_attention.linear_qkv.weight",
            [_HF + f"self_attn.{p}_proj.weight" for p in "qkv"],
            _megatron_qkv_to_hf,
        ),
        rename(
            _LAYER + "self_attention.q_layernorm.weight",
            _HF + "self_attn.q_norm.weight",
        ),
        rename(
            _LAYER + "self_attention.k_layernorm.weight",
            _HF + "self_attn.k_norm.weight",
        ),
        rename(
            _LAYER + "self_attention.linear_proj.weight",
            _HF + "self_attn.o_proj.weight",
        ),
        rename(
            _LAYER + "mlp.linear_fc1.layer_norm_weight",
            _HF + "post_attention_layernorm.weight",
        ),
        rename(
            _LAYER + "pre_mlp_layernorm.weight", _HF + "post_attention_layernorm.weight"
        ),
        # dense MLP
        gate_up(_LAYER + "mlp.linear_fc1.weight", _HF + "mlp."),
        rename(_LAYER + "mlp.linear_fc2.weight", _HF + "mlp.down_proj.weight"),
        # MoE
        rename(_LAYER + "mlp.router.weight", _HF + "mlp.gate.weight"),
        *routed_expert_rules(),
        *stacked_expert_rules(),
        Rule(_LAYER + "mlp.experts.weight1", _legacy_fc1),
        Rule(_LAYER + "mlp.experts.weight2", _legacy_fc2),
        gate_up(
            _LAYER + "mlp.shared_experts.linear_fc1.weight", _HF + "mlp.shared_expert."
        ),
        rename(
            _LAYER + "mlp.shared_experts.linear_fc2.weight",
            _HF + "mlp.shared_expert.down_proj.weight",
        ),
        rename(
            _LAYER + "mlp.shared_experts.gate_weight",
            _HF + "mlp.shared_expert_gate.weight",
        ),
    ],
)


def megatron_to_hf(
    named_params: Iterable[tuple[str, torch.Tensor]],
    d: GPTDims,
    *,
    layer_offset: int = 0,
    expert_offset: int = 0,
) -> Iterator[HfTensor]:
    """Stream ``(hf_name, tensor)`` from Megatron named params, one tensor at a time.

    Args:
        named_params: Megatron ``(name, tensor)`` pairs, possibly a lazy gather.
        d: Model dims (the QKV split and the legacy GroupedMLP need them).
        layer_offset: Global index of the first layer when names are
            stage-local (the legacy engine under PP); 0 for global names.
        expert_offset: Global index of the first expert when names are
            EP-rank-local; 0 for global names.
    """
    return GPT.megatron_to_hf(
        named_params, d, layer_offset=layer_offset, expert_offset=expert_offset
    )


# ------------------------------------------------------------------ load


def hf_to_megatron(
    hf: dict[str, torch.Tensor],
    d: GPTDims,
    te: bool = False,
    ep_rank: int = 0,
    ep_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    layers_per_pp_rank: list[int] | None = None,
    use_grouped_mlp: bool = True,
) -> dict[str, torch.Tensor]:
    """Return a Megatron GPTModel state_dict (bf16 tensors on CPU).

    For MoE models (``d.num_experts > 0``), only the local experts for the
    given ``ep_rank`` are included in the returned state dict.

    With PP > 1, only the layers and embedding/output owned by this PP rank
    are included.  Megatron's decoder layers are numbered locally (0-based
    within the stage), while HF layers use the global index.

    ``te=True`` targets the TransformerEngine layer spec (fused layernorm
    names); ``te=False`` the local spec. This only affects dense layernorm keys.

    ``use_grouped_mlp``: when True, fuse experts into GroupedMLP weight1/weight2;
    when False, use per-expert SequentialMLP local_experts.{i}.linear_fc1/fc2.
    """
    in_ln = (
        "self_attention.linear_qkv.layer_norm_weight"
        if te
        else "input_layernorm.weight"
    )
    mlp_ln = "mlp.linear_fc1.layer_norm_weight" if te else "pre_mlp_layernorm.weight"
    m: dict[str, torch.Tensor] = {}
    is_first_pp = pp_rank == 0
    is_last_pp = pp_rank == pp_size - 1

    if is_first_pp:
        m["embedding.word_embeddings.weight"] = hf["model.embed_tokens.weight"]
    if is_last_pp:
        m["decoder.final_layernorm.weight"] = hf["model.norm.weight"]
        m["output_layer.weight"] = hf["lm_head.weight"]

    is_moe = d.num_experts > 0
    if is_moe:
        num_local = d.num_experts // ep_size
        expert_offset = ep_rank * num_local

    layer_offset, num_local_layers = pp_layer_range(
        d,
        pp_rank,
        pp_size,
        layers_per_pp_rank,
    )

    for local_i in range(num_local_layers):
        global_i = layer_offset + local_i
        hp = f"model.layers.{global_i}."
        mp = f"decoder.layers.{local_i}."

        # -- attention (shared between dense and MoE) --
        m[mp + in_ln] = hf[hp + "input_layernorm.weight"]
        m[mp + "self_attention.linear_qkv.weight"] = _hf_qkv_to_megatron(
            hf[hp + "self_attn.q_proj.weight"],
            hf[hp + "self_attn.k_proj.weight"],
            hf[hp + "self_attn.v_proj.weight"],
            d,
        )
        if hp + "self_attn.q_norm.weight" in hf:
            m[mp + "self_attention.q_layernorm.weight"] = hf[
                hp + "self_attn.q_norm.weight"
            ]
            m[mp + "self_attention.k_layernorm.weight"] = hf[
                hp + "self_attn.k_norm.weight"
            ]
        m[mp + "self_attention.linear_proj.weight"] = hf[hp + "self_attn.o_proj.weight"]
        m[mp + mlp_ln] = hf[hp + "post_attention_layernorm.weight"]

        if is_moe:
            # -- MoE: router --
            m[mp + "mlp.router.weight"] = hf[hp + "mlp.gate.weight"]

            # -- MoE: experts --
            if use_grouped_mlp:
                # GroupedMLP: fuse all local experts into weight1/weight2
                gate_ups = []
                downs = []
                for e in range(expert_offset, expert_offset + num_local):
                    gate = hf[hp + f"mlp.experts.{e}.gate_proj.weight"]
                    up = hf[hp + f"mlp.experts.{e}.up_proj.weight"]
                    down = hf[hp + f"mlp.experts.{e}.down_proj.weight"]
                    gate_ups.append(torch.cat([gate, up], dim=0).t())
                    downs.append(down.t())
                m[mp + "mlp.experts.weight1"] = torch.cat(gate_ups, dim=1).contiguous()
                m[mp + "mlp.experts.weight2"] = torch.cat(downs, dim=0).contiguous()
            else:
                # SequentialMLP: per-expert linear_fc1/linear_fc2
                for local_i_e, e in enumerate(
                    range(expert_offset, expert_offset + num_local)
                ):
                    gate = hf[hp + f"mlp.experts.{e}.gate_proj.weight"]
                    up = hf[hp + f"mlp.experts.{e}.up_proj.weight"]
                    down = hf[hp + f"mlp.experts.{e}.down_proj.weight"]
                    ep_prefix = mp + f"mlp.experts.local_experts.{local_i_e}."
                    m[ep_prefix + "linear_fc1.weight"] = torch.cat(
                        [gate, up], dim=0
                    ).contiguous()
                    m[ep_prefix + "linear_fc2.weight"] = down

            # -- MoE: shared expert (optional) --
            if d.shared_expert_ffn > 0:
                m[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                    [
                        hf[hp + "mlp.shared_expert.gate_proj.weight"],
                        hf[hp + "mlp.shared_expert.up_proj.weight"],
                    ],
                    dim=0,
                ).contiguous()
                m[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[
                    hp + "mlp.shared_expert.down_proj.weight"
                ]
                if d.shared_expert_gate:
                    m[mp + "mlp.shared_experts.gate_weight"] = hf[
                        hp + "mlp.shared_expert_gate.weight"
                    ]
        else:
            # -- Dense MLP --
            m[mp + "mlp.linear_fc1.weight"] = torch.cat(
                [hf[hp + "mlp.gate_proj.weight"], hf[hp + "mlp.up_proj.weight"]], dim=0
            ).contiguous()
            m[mp + "mlp.linear_fc2.weight"] = hf[hp + "mlp.down_proj.weight"]
    return m


def non_expert_hf_to_megatron(hf: dict, d: GPTMoEDims) -> dict[str, torch.Tensor]:
    """Global-named Megatron tensors for every NON-expert MoE param (TE spec).

    Attention/embedding/output identical to the dense TE layout; the MoE pre-MLP
    RMSNorm is emitted as the standalone ``pre_mlp_layernorm.weight`` and the
    router as ``mlp.router.weight``.
    """
    m: dict[str, torch.Tensor] = {}
    m["embedding.word_embeddings.weight"] = hf["model.embed_tokens.weight"]
    m["decoder.final_layernorm.weight"] = hf["model.norm.weight"]
    m["output_layer.weight"] = hf["lm_head.weight"]
    for i in range(d.num_layers):
        hp = f"model.layers.{i}."
        mp = f"decoder.layers.{i}."
        m[mp + "self_attention.linear_qkv.layer_norm_weight"] = hf[
            hp + "input_layernorm.weight"
        ]
        m[mp + "self_attention.linear_qkv.weight"] = _hf_qkv_to_megatron(
            hf[hp + "self_attn.q_proj.weight"],
            hf[hp + "self_attn.k_proj.weight"],
            hf[hp + "self_attn.v_proj.weight"],
            d,
        )
        m[mp + "self_attention.q_layernorm.weight"] = hf[hp + "self_attn.q_norm.weight"]
        m[mp + "self_attention.k_layernorm.weight"] = hf[hp + "self_attn.k_norm.weight"]
        m[mp + "self_attention.linear_proj.weight"] = hf[hp + "self_attn.o_proj.weight"]
        # MoE: standalone pre-mlp RMSNorm (NOT fused into a linear).
        m[mp + "pre_mlp_layernorm.weight"] = hf[hp + "post_attention_layernorm.weight"]
        m[mp + "mlp.router.weight"] = hf[hp + "mlp.gate.weight"]
        # optional shared expert (fused gate;up + down + gate score)
        if d.shared_ffn > 0 and (hp + "mlp.shared_expert.gate_proj.weight") in hf:
            m[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                [
                    hf[hp + "mlp.shared_expert.gate_proj.weight"],
                    hf[hp + "mlp.shared_expert.up_proj.weight"],
                ],
                dim=0,
            ).contiguous()
            m[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[
                hp + "mlp.shared_expert.down_proj.weight"
            ]
            if (hp + "mlp.shared_expert_gate.weight") in hf:
                m[mp + "mlp.shared_experts.gate_weight"] = hf[
                    hp + "mlp.shared_expert_gate.weight"
                ]
    return m


def hf_expert_fc1(hf: dict, d: Any, layer: int, global_e: int) -> torch.Tensor:
    """Fused SwiGLU ``[gate; up]`` for one global expert -> [2*moe_ffn, hidden]."""
    hp = f"model.layers.{layer}.mlp.experts.{global_e}."
    return torch.cat(
        [hf[hp + "gate_proj.weight"], hf[hp + "up_proj.weight"]], dim=0
    ).contiguous()


def hf_expert_fc2(hf: dict, d: Any, layer: int, global_e: int) -> torch.Tensor:
    """Down projection for one global expert -> [hidden, moe_ffn]."""
    hp = f"model.layers.{layer}.mlp.experts.{global_e}."
    return hf[hp + "down_proj.weight"].contiguous()
