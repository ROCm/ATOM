# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/models/ds_v4_1_flash/weight_converter.py``, the
# packing route of ``hf_loader.py`` (``_logical`` / ``_pack`` and the Engram tables), ``rope.py`` and the parameter
# slots of ``e2e_graph.py`` (``LAYER_SLOTS`` / ``GLOBAL_SLOTS``).
"""A TP8 rank's packed DeepSeek-V4.1-Flash decode ("D") parameters from the HF checkpoint, on the CPU: the tensors
TileRT's ``dsv41_e2e_prepare_rank`` takes, by the names of its parameter slots.

- ``pack_scope`` / ``LAYER_GROUPS`` / ``MTP_GROUPS`` / ``GLOBAL_GROUPS``: which packer (``pack``) takes which logical
  tensors (``ckpt``) of a scope (``global``, ``layers.<i>``, ``mtp.<m>``) into which group.
- ``pack_rank_scopes``: a scope's ``{group: {name: tensor}}`` for some ranks, as TileRT's HF loader leaves it (its
  ``K[rank][scope]``): bf16 gammas kept, all routed experts, Engram's projection packed apart from its tables
  (``engram_table``), the draft layers' MoE in the rx layout.
- ``scope_params`` / ``global_params`` / ``rank_params``: those by parameter slot name -- ``<scope>.<group>.<name>``
  for the ``LAYER_SLOTS`` of every layer and draft layer, and the ``GLOBAL_SLOTS`` (``embed``, ``head_w``,
  ``final_gamma``, the RoPE tables, the Engram hash constants, draft layer 0's ``main_proj`` and the last draft
  layer's markov head).

A rank's parameters come to about 66 GB (TileRT's run, RoPE tables of 1M positions), 25.3 GB of them its two Engram
table shards; ``rank_params`` yields them scope by scope and leaves the tables to ``engram_table`` unless asked.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

from . import pack as P
from .ckpt import (
    NUM_PES,
    CkptIndex,
    ModelArgs,
    engram_row_range,
    get_index,
    load_global_all,
    load_layer_all,
    load_mtp_all,
)

Packer = Callable[..., dict[str, torch.Tensor]]


@dataclass(frozen=True)
class Group:
    packer: str
    inputs: tuple[str, ...]
    # all; kv_source, index_source, engram (layers); mtp0, mtp_last (draft layers)
    when: str = "all"


_HC = tuple(f"hc_{h}_{k}" for h in ("attn", "ffn") for k in ("fn", "base", "scale"))
_EXPERTS = ("w1", "w1_s", "w3", "w3_s", "w2", "w2_s", "expert_ids")
_SHARED = ("sw1", "sw1_s", "sw3", "sw3_s", "sw2", "sw2_s")
LAYER_GROUPS: dict[str, Group] = {
    "mhc": Group("pack_mhc", (*_HC, "attn_norm", "ffn_norm")),
    "engram": Group(
        "pack_engram",
        ("e_table", "e_table_s", "e_row_lo", "e_wkv", "e_wkv_s", "e_qk"),
        "engram",
    ),
    "q_proj": Group("pack_q_proj", ("wq_a", "wq_a_s", "q_norm", "wq_b", "wq_b_s")),
    "window_kv": Group("pack_window_kv", ("wkv", "wkv_s", "kv_norm")),
    "compressor": Group("pack_compressor", ("c_wkv", "c_wgate", "c_norm"), "kv_source"),
    "indexer": Group(
        "pack_indexer",
        ("i_wq_b", "i_wq_b_s", "i_wproj", "i_wk", "i_k_norm"),
        "index_source",
    ),
    "attn_projo": Group(
        "pack_attn_projo", ("attn_sink", "wo_a", "wo_a_s", "wo_b", "wo_b_s")
    ),
    "moe": Group("pack_moe", ("gate_w", "gate_b", *_EXPERTS, *_SHARED)),
}
MTP_GROUPS: dict[str, Group] = {
    **{k: g for k, g in LAYER_GROUPS.items() if g.when == "all"},
    "main_proj": Group(
        "pack_main_proj", ("main_proj", "main_proj_s", "main_norm"), "mtp0"
    ),
    "dspark_head": Group(
        "pack_dspark_head", ("mtp_norm", "markov_embed", "markov_head"), "mtp_last"
    ),
}
GLOBAL_GROUPS: dict[str, Group] = {
    "head": Group("pack_head", ("head",)),
    "final_norm": Group("pack_final_norm", ("norm",)),
}
PACKERS: dict[str, Packer] = {
    "pack_mhc": P.pack_mhc,
    "pack_final_norm": P.pack_final_norm,
    "pack_engram": P.pack_engram,
    "pack_q_proj": P.pack_q_proj,
    "pack_window_kv": P.pack_window_kv,
    "pack_compressor": P.pack_compressor,
    "pack_indexer": P.pack_indexer,
    "pack_attn_projo": P.pack_attn_projo,
    "pack_moe": P.pack_moe,
    "pack_head": P.pack_head,
    "pack_main_proj": P.pack_main_proj,
    "pack_dspark_head": P.pack_dspark_head,
}

# the parameter slots of TileRT's e2e graph: per scope (each of the 40 layers, then the 3 draft layers) these
# (group, name) pairs, an absent one empty; the indexer's under the packer's names (IX_PACKED)
LAYER_SLOTS = (
    *(
        ("mhc", f"{h}_{k}")
        for h in ("attn", "ffn")
        for k in ("fn", "scale", "base", "gamma")
    ),
    ("q_proj", "qa_w"),
    ("q_proj", "qa_s"),
    ("q_proj", "q_norm"),
    ("q_proj", "qb_w"),
    ("q_proj", "qb_s"),
    ("window_kv", "kv_w"),
    ("window_kv", "kv_s"),
    ("window_kv", "kv_norm"),
    ("compressor", "c_w"),
    ("compressor", "c_norm"),
    ("indexer", "i_wq_b"),
    ("indexer", "i_wq_b_s"),
    ("indexer", "i_wproj"),
    ("indexer", "i_wk"),
    ("indexer", "i_k_norm"),
    ("attn_projo", "attn_sink"),
    ("attn_projo", "wo_a"),
    ("attn_projo", "wo_a_s"),
    ("attn_projo", "wo_b"),
    ("attn_projo", "wo_b_s"),
    *(("moe", k) for k in ("gate_w", "gate_b", "ug", "ug_s", "dn", "dn_s")),
    *(("moe", k) for k in ("sug", "sug_s", "sdn", "sdn_s")),
    *(("engram", k) for k in ("table", "table_s", "row_range", "wkv", "wkv_s", "qk")),
)
GLOBAL_SLOTS = (
    "embed",
    "head_w",
    "final_gamma",
    "rope0",
    "ropec",
    "token_map",
    "eng_mult",
    "eng_primes",
    "eng_offsets",
    "main_w",
    "main_w_s",
    "main_norm",
    "mtp_norm",
    "markov_embed",
    "markov_head",
)
IX_PACKED = {
    "i_wq_b": "wq_b",
    "i_wq_b_s": "wq_b_s",
    "i_wproj": "weights_proj",
    "i_wk": "wk",
    "i_k_norm": "k_norm",
}
_FP8 = (torch.float8_e4m3fn, torch.float8_e8m0fnu)


# ---------------------------------------------------------------------------------------------------------------
# weight_converter.py


def _call_packer(fn: Packer, tensors: dict, extra: dict) -> dict[str, torch.Tensor]:
    sig = inspect.signature(fn)
    has_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    kw = {k: v for k, v in tensors.items() if has_kw or k in sig.parameters}
    kw.update({k: v for k, v in extra.items() if k in sig.parameters})
    out = fn(**kw)
    if not isinstance(out, dict):
        raise TypeError(f"packer {fn.__qualname__} returned {type(out)}, want dict")
    return out


def _applies(g: Group, scope: str, args: ModelArgs) -> bool:
    if g.when == "all":
        return True
    if scope.startswith("layers."):
        layer = int(scope.split(".")[1])
        return {
            "kv_source": layer in args.kv_source_layers,
            "index_source": layer in args.index_source_layers,
            "engram": layer in args.engram_layer_ids,
        }.get(g.when, False)
    if scope.startswith("mtp."):
        m = int(scope.split(".")[1])
        return {"mtp0": m == 0, "mtp_last": m == args.n_mtp_layers - 1}.get(
            g.when, False
        )
    return False


def groups_of(scope: str) -> dict[str, Group]:
    if scope == "global":
        return GLOBAL_GROUPS
    return LAYER_GROUPS if scope.startswith("layers.") else MTP_GROUPS


def pack_scope(
    logical: dict[str, torch.Tensor],
    groups: dict[str, Group],
    scope: str,
    rank: int,
    args: ModelArgs,
    options: dict | None = None,
) -> dict[str, torch.Tensor]:
    """``{"<group>.<name>": tensor}``: each group of ``groups`` that applies to ``scope`` packed from ``logical``
    (a logical tensor no group takes is an error)."""
    out: dict[str, torch.Tensor] = {}
    used: set[str] = set()
    extra = {
        "rank": rank,
        "mype": rank,
        "scope": scope,
        "args": args,
        **(options or {}),
    }
    if scope.startswith("layers."):
        extra["layer"] = int(scope.split(".")[1])
    for gname, g in groups.items():
        if not _applies(g, scope, args):
            continue
        tensors = {k: logical[k] for k in g.inputs if k in logical}
        used |= set(tensors)
        for k, v in _call_packer(PACKERS[g.packer], tensors, extra).items():
            out[f"{gname}.{k}"] = v
    left = set(logical) - used
    if left:
        raise KeyError(f"{scope}: logical tensors without a group: {sorted(left)}")
    return out


# ---------------------------------------------------------------------------------------------------------------
# hf_loader.py, on the CPU

_GAMMA_SRC = {
    "attn_norm": "attn_norm.weight",
    "ffn_norm": "ffn_norm.weight",
    "norm": "norm.weight",
}
_ENGRAM_PROJ = ("e_row_lo", "e_wkv", "e_wkv_s", "e_qk")


def _logical(
    idx: CkptIndex, scope: str, ranks: list[int], consts: dict | None, args: ModelArgs
) -> list[dict]:
    kind, _, i = scope.partition(".")
    if kind == "global":
        per = load_global_all(idx, ranks, with_embed=False)
    elif kind == "layers":
        per = load_layer_all(
            idx, int(i), ranks, engram_consts=consts, args=args, engram_tables=False
        )
    else:
        per = load_mtp_all(idx, int(i), ranks, args=args)
    for d in per:  # the checkpoint's own bf16 gammas
        for k, name in _GAMMA_SRC.items():
            src = name if scope == "global" else f"{scope}.{name}"
            if k in d and src in idx and idx.info(src).dtype == torch.bfloat16:
                d[k] = idx.read(src)
    return per


def _pack(scope: str, rank: int, g: dict, args: ModelArgs) -> dict[str, dict]:
    groups = {k: v for k, v in groups_of(scope).items() if k != "engram"}
    g = dict(g)
    eng = None
    if "e_row_lo" in g:
        eng = {k: g.pop(k) for k in _ENGRAM_PROJ}
    flat = pack_scope(g, groups, scope, rank, args, options={"chunk": 1024})
    if eng is not None:
        for k, v in P.pack_engram_proj(
            eng["e_wkv"], eng["e_wkv_s"], eng["e_qk"]
        ).items():
            flat[f"engram.{k}"] = v
    out: dict[str, dict] = {}
    for k, v in flat.items():
        grp, name = k.split(".", 1)
        out.setdefault(grp, {})[name] = v
    moe = out.get("moe")
    if scope.startswith("mtp.") and moe and moe["gate_b"].shape[0] == 128:
        out["moe"] = P.rx_from_main(moe)
    return out


def pack_rank_scopes(
    ckpt: str | CkptIndex,
    scope: str,
    ranks: list[int] | None = None,
    consts: dict | None = None,
    args: ModelArgs | None = None,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    """``scope``'s packed ``{group: {name: tensor}}`` for each rank of ``ranks`` (default all 8). An Engram layer's
    ``engram`` group holds ``wkv`` / ``wkv_s`` / ``qk``, not the tables (``engram_table``), and needs ``consts``
    (``engram_hash_consts``'s or ``compute_engram_consts``'s: the row ranges)."""
    idx = get_index(ckpt)
    args = args or ModelArgs()
    rs = list(range(NUM_PES)) if ranks is None else list(ranks)
    per = _logical(idx, scope, rs, consts, args)
    return [_pack(scope, r, per[i], args) for i, r in enumerate(rs)]


def engram_table(
    ckpt: str | CkptIndex,
    layer: int,
    rank: int,
    consts: dict,
    rows: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """``layers.<layer>.engram``'s ``table`` (uint8 [n, 256]), ``table_s`` (uint8 [n, 8]) and ``row_range`` (int64
    ``[lo, hi]``) of rank ``rank``: the checkpoint's rows ``[lo, hi)`` of its hash columns (``hashcol``). ``rows``:
    only the shard's rows ``[a, b)`` (``0 <= a < b <= hi - lo``) of table / table_s, to stream the 12 GB shard.
    """
    idx = get_index(ckpt)
    lo, hi = engram_row_range(consts, layer, rank)
    a, b = (0, hi - lo) if rows is None else rows
    assert 0 <= a < b <= hi - lo, (a, b, lo, hi)
    out = {}
    for key, name in (("table", "embed.weight"), ("table_s", "embed.scale")):
        full = f"layers.{layer}.engram.{name}"
        ti = idx.info(full)
        t = torch.empty((b - a, ti.row_bytes), dtype=torch.uint8)
        idx.read_rows(full, lo + a, lo + b, out=t.view(ti.dtype))
        out[key] = t
    out["row_range"] = torch.tensor([lo, hi], dtype=torch.int64)
    return out


# ---------------------------------------------------------------------------------------------------------------
# rope.py


def freqs_f32(
    dim: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    freqs = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 0.001)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_table(
    seqlen: int,
    dim: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """float32 [seqlen, dim]: cos | sin of each position's angles (computed in float64)."""
    freqs = freqs_f32(dim, original_seq_len, base, factor, beta_fast, beta_slow)
    ang = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs).double()
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).float().contiguous()


def rope_table_r0(
    seqlen: int | None = None, args: ModelArgs | None = None
) -> torch.Tensor:
    """``rope0``: the uncompressed layers' table (no YaRN)."""
    args = args or ModelArgs()
    return rope_table(
        seqlen or args.max_seq_len,
        args.rope_head_dim,
        0,
        args.rope_theta,
        args.rope_factor,
        args.beta_fast,
        args.beta_slow,
    )


def rope_table_rc(
    seqlen: int | None = None, args: ModelArgs | None = None
) -> torch.Tensor:
    """``ropec``: the compressed layers' table (YaRN)."""
    args = args or ModelArgs()
    return rope_table(
        seqlen or args.max_seq_len,
        args.rope_head_dim,
        args.original_seq_len,
        args.compress_rope_theta,
        args.rope_factor,
        args.beta_fast,
        args.beta_slow,
    )


# ---------------------------------------------------------------------------------------------------------------
# the parameter slots (e2e_graph.py)


def _slot_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype in _FP8 else t


def scope_params(
    scope: str, groups: dict[str, dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """``{"<scope>.<group>.<name>": tensor}`` of the ``LAYER_SLOTS`` ``groups`` (a ``pack_rank_scopes`` entry, plus an
    ``engram_table`` for the tables) fills; an absent slot is left out (TileRT passes it empty).
    """
    out = {}
    for grp, name in LAYER_SLOTS:
        gd = groups.get(grp, {})
        t = gd.get(name)
        if t is None and grp == "indexer":
            t = gd.get(IX_PACKED.get(name, ""))
        if t is not None:
            out[f"{scope}.{grp}.{name}"] = _slot_tensor(t)
    return out


def global_params(
    embed: torch.Tensor,
    glob: dict[str, dict[str, torch.Tensor]],
    mtp0: dict[str, dict[str, torch.Tensor]],
    mtp_last: dict[str, dict[str, torch.Tensor]],
    eng: dict,
    rope0: torch.Tensor,
    ropec: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The ``GLOBAL_SLOTS``: ``embed`` (the checkpoint's), the ``global`` scope's head / final norm, the RoPE tables,
    the Engram hash constants (``pack.engram_hash_params``), ``mtp.0``'s main_proj and the last draft layer's
    markov head."""
    g = (
        embed,
        glob["head"]["head_w"],
        glob["final_norm"]["gamma"],
        rope0,
        ropec,
        eng["token_map"],
        eng["mult"],
        eng["primes"],
        eng["offsets"],
        mtp0["main_proj"]["w"],
        mtp0["main_proj"]["w_s"],
        mtp0["main_proj"]["main_norm"],
        mtp_last["dspark_head"]["mtp_norm"],
        mtp_last["dspark_head"]["markov_embed"],
        mtp_last["dspark_head"]["markov_head"],
    )
    return {k: _slot_tensor(t) for k, t in zip(GLOBAL_SLOTS, g)}


def scopes(args: ModelArgs | None = None) -> list[str]:
    """The packed scopes in the slots' order: the 40 layers, then the draft layers."""
    args = args or ModelArgs()
    return [f"layers.{i}" for i in range(args.n_layers)] + [
        f"mtp.{m}" for m in range(args.n_mtp_layers)
    ]


def rank_params(
    ckpt: str | CkptIndex,
    rank: int,
    consts: dict,
    *,
    max_seq_len: int | None = None,
    args: ModelArgs | None = None,
    engram_tables: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Rank ``rank``'s parameters as ``(slot name, tensor)``: the ``GLOBAL_SLOTS`` first, then each scope's
    ``LAYER_SLOTS`` (one scope in memory at a time). ``consts``: ``compute_engram_consts``'s (the token map is a
    global slot). ``max_seq_len``: the RoPE tables' rows (default ``args.max_seq_len``). ``engram_tables``: also the
    two 12 GB table shards (else leave them to ``engram_table``)."""
    idx = get_index(ckpt)
    args = args or ModelArgs()
    seqlen = max_seq_len or args.max_seq_len
    glob = pack_rank_scopes(idx, "global", [rank], consts, args)[0]
    mtp0 = pack_rank_scopes(idx, "mtp.0", [rank], consts, args)[0]
    last = f"mtp.{args.n_mtp_layers - 1}"
    mtp_last = pack_rank_scopes(idx, last, [rank], consts, args)[0]
    yield from global_params(
        idx.read("embed.weight"),
        glob,
        mtp0,
        mtp_last,
        P.engram_hash_params(consts),
        rope_table_r0(seqlen, args),
        rope_table_rc(seqlen, args),
    ).items()
    del glob
    for sc in scopes(args):
        if sc == "mtp.0":
            groups = mtp0
        elif sc == last:
            groups = mtp_last
        else:
            groups = pack_rank_scopes(idx, sc, [rank], consts, args)[0]
        if engram_tables and "engram" in groups:
            groups["engram"].update(
                engram_table(idx, int(sc.split(".")[1]), rank, consts)
            )
        yield from scope_params(sc, groups).items()
        del groups
