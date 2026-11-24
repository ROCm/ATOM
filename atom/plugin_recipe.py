# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plugin recipes: declarative module/seam routing and engine/model options.

Supports both ``atom.plugin_recipe/1`` (seam tables) and ``atom.plugin_recipe/2``
(hierarchical ``nn.Module`` subtree takeover + declarative ``engine``,
``model_options``, and ``dspark_options`` blocks so deployments need no
proliferation of ``ATOM_*`` environment variables).

Graph modes: ``any`` (default) serves every pass; ``eager`` serves uncaptured
passes; ``capture`` with ``buckets`` serves captured passes whose padded rows
match one of the buckets. ``python -m atom.plugin_recipe PATH`` validates and
dumps a recipe or recipe directory on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from atom import plugin_host

RECIPE_SCHEMA = "atom.plugin_recipe/1"
RECIPE_SCHEMA_V2 = "atom.plugin_recipe/2"
SCHEMAS = (RECIPE_SCHEMA, RECIPE_SCHEMA_V2)
MODES = ("any", "eager", "capture")
ON_DECLINE = ("native", "error")
SHADOW = ("error", "shadow", "nested")
MONO_MODES = ("auto", "off")
PREFILL_ADMIT_ORDERS = ("fifo", "short_new", "srpt")
SEG_FUSE_ITEMS = (1, 7)
QPATH_FUSE_ITEMS = ("kv", "q")
_SLOT_RE = re.compile(r"^[A-Za-z0-9_*]+(\.[A-Za-z0-9_*]+)*$")


class RecipeRefused(ValueError):
    """A recipe that can never be valid here; ``reason`` is its short code."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def _strict_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise RecipeRefused("recipe-schema", f"{where} must be a bool")
    return value


def _strict_int(value: Any, where: str, *, min_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeRefused("recipe-schema", f"{where} must be an int")
    if min_value is not None and value < min_value:
        raise RecipeRefused("recipe-schema", f"{where} must be >= {min_value}")
    return value


def _strict_number(value: Any, where: str, *, min_value: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RecipeRefused("recipe-schema", f"{where} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise RecipeRefused("recipe-schema", f"{where} must be finite")
    if min_value is not None and value < min_value:
        raise RecipeRefused("recipe-schema", f"{where} must be >= {min_value}")
    return value


def _one_of(value: Any, where: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise RecipeRefused("recipe-schema", f"{where} must be one of {choices}")
    return value


def _opt_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RecipeRefused("recipe-schema", f"{where} must be a non-empty string")
    return value.strip()


def _row_counts(value: Any, where: str) -> str | tuple[int, ...] | None:
    """A file of row counts (a path) or the row counts themselves."""
    if value is None:
        return None
    if isinstance(value, str):
        return _opt_str(value, where)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in value
    ):
        raise RecipeRefused(
            "recipe-schema", f"{where} must be a path or a list of positive ints"
        )
    return tuple(value)


def _items(value: Any, where: str, known: tuple) -> tuple:
    """A list of item names or numbers out of ``known``, in ``known``'s order."""
    if not isinstance(value, (list, tuple)) or any(
        isinstance(x, bool) or x not in known for x in value
    ):
        raise RecipeRefused(
            "recipe-schema", f"{where} must be a list of items out of {known}"
        )
    return tuple(x for x in known if x in value)


_WIDTHS_RE = re.compile(r"^\d+(-\d+)?([ ,]+\d+(-\d+)?)*$")


def _widths(value: Any, where: str, *, words: tuple[str, ...]) -> str | None:
    """Request widths: one of ``words``, or a list such as "1-32,48,64"."""
    text = _opt_str(value, where)
    if text is None or text.lower() in words or _WIDTHS_RE.match(text):
        return text
    raise RecipeRefused(
        "recipe-schema", f"{where} must be one of {words} or a list like '1-32,48'"
    )


def _freeze_placement(
    value: Any, where: str
) -> str | tuple[tuple[str, tuple[int, ...]], ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, tuple) and all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], tuple)
        for item in value
    ):
        return value
    if not isinstance(value, Mapping):
        raise RecipeRefused(
            "recipe-schema",
            f"{where} must be a JSON path string, a layer->perm mapping, or null",
        )
    items: list[tuple[str, tuple[int, ...]]] = []
    for layer_key, perm in sorted(value.items(), key=lambda kv: str(kv[0])):
        if not isinstance(perm, (list, tuple)) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in perm
        ):
            raise RecipeRefused(
                "recipe-schema", f"{where}[{layer_key!r}] must be an int permutation"
            )
        items.append((str(layer_key), tuple(int(x) for x in perm)))
    return tuple(items)


@dataclass(frozen=True)
class EngineOptions:
    """Scheduler and DP-routing options configured from a recipe (PR #3644)."""

    prefill_short_first_rows: int = 0
    prefill_short_first_long_rows: int = 4096
    prefill_short_first_max_skips: int = 32
    prefill_cost_ceil: bool = False
    prefill_decode_interval_busy: int = 0
    prefill_decode_interval_busy_seqs: int = 48
    # Load-dependent decode interval (0 = off): see `PrefillDelayer`.
    prefill_decode_interval_slope: float = 0.0
    prefill_decode_interval_slope_from: int = 0
    prefill_long_cap_busy: int = 0
    prefill_long_cap_busy_seqs: int = 48
    dp_session_prefix_owner_tokens: int = 0
    dp_session_prefix_owner_max_extra_reqs: int = 8
    # The order a rank admits the prefill work it holds: see
    # `Scheduler._order_waiting_prefills`.
    prefill_admit_order: str = "fifo"
    prefill_admit_max_age_ms: int = 3000
    # No transparent huge pages for the server and every process it starts
    # (`atom.utils.disable_thp`).
    thp_disable: bool = False
    # Pin the pages of a copied Engram shard against page migration
    # (`atom.model_ops.engram.host_memory._pin_pages`).
    engram_pin_pages: bool = False
    # Start-up: no MoRI EP op handles at model build
    # (`atom.model_ops.fused_moe.deferred_mori_op`); dummy prefill forwards
    # at these row counts and the draft's context projection at these decode
    # widths before serving (`ModelRunner._maybe_warmup`).
    mori_defer_handles: bool = False
    prefill_warmup_ladder: str | tuple[int, ...] | None = None
    decode_warmup_widths: str | None = None
    decode_warmup_tails: str | None = None

    @classmethod
    def from_mapping(cls, raw: Any) -> EngineOptions:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeRefused("recipe-schema", "engine must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RecipeRefused("recipe-schema", f"unknown engine option(s): {unknown}")
        defaults = cls()
        options = cls(
            prefill_short_first_rows=_strict_int(
                raw.get("prefill_short_first_rows", defaults.prefill_short_first_rows),
                "engine.prefill_short_first_rows",
                min_value=0,
            ),
            prefill_short_first_long_rows=_strict_int(
                raw.get(
                    "prefill_short_first_long_rows",
                    defaults.prefill_short_first_long_rows,
                ),
                "engine.prefill_short_first_long_rows",
                min_value=1,
            ),
            prefill_short_first_max_skips=_strict_int(
                raw.get(
                    "prefill_short_first_max_skips",
                    defaults.prefill_short_first_max_skips,
                ),
                "engine.prefill_short_first_max_skips",
                min_value=0,
            ),
            prefill_cost_ceil=_strict_bool(
                raw.get("prefill_cost_ceil", defaults.prefill_cost_ceil),
                "engine.prefill_cost_ceil",
            ),
            prefill_decode_interval_busy=_strict_int(
                raw.get(
                    "prefill_decode_interval_busy",
                    defaults.prefill_decode_interval_busy,
                ),
                "engine.prefill_decode_interval_busy",
                min_value=0,
            ),
            prefill_decode_interval_busy_seqs=_strict_int(
                raw.get(
                    "prefill_decode_interval_busy_seqs",
                    defaults.prefill_decode_interval_busy_seqs,
                ),
                "engine.prefill_decode_interval_busy_seqs",
                min_value=1,
            ),
            prefill_decode_interval_slope=_strict_number(
                raw.get(
                    "prefill_decode_interval_slope",
                    defaults.prefill_decode_interval_slope,
                ),
                "engine.prefill_decode_interval_slope",
                min_value=0,
            ),
            prefill_decode_interval_slope_from=_strict_int(
                raw.get(
                    "prefill_decode_interval_slope_from",
                    defaults.prefill_decode_interval_slope_from,
                ),
                "engine.prefill_decode_interval_slope_from",
                min_value=0,
            ),
            prefill_long_cap_busy=_strict_int(
                raw.get("prefill_long_cap_busy", defaults.prefill_long_cap_busy),
                "engine.prefill_long_cap_busy",
                min_value=0,
            ),
            prefill_long_cap_busy_seqs=_strict_int(
                raw.get(
                    "prefill_long_cap_busy_seqs", defaults.prefill_long_cap_busy_seqs
                ),
                "engine.prefill_long_cap_busy_seqs",
                min_value=1,
            ),
            dp_session_prefix_owner_tokens=_strict_int(
                raw.get(
                    "dp_session_prefix_owner_tokens",
                    defaults.dp_session_prefix_owner_tokens,
                ),
                "engine.dp_session_prefix_owner_tokens",
                min_value=0,
            ),
            dp_session_prefix_owner_max_extra_reqs=_strict_int(
                raw.get(
                    "dp_session_prefix_owner_max_extra_reqs",
                    defaults.dp_session_prefix_owner_max_extra_reqs,
                ),
                "engine.dp_session_prefix_owner_max_extra_reqs",
                min_value=0,
            ),
            prefill_admit_order=_one_of(
                raw.get("prefill_admit_order", defaults.prefill_admit_order),
                "engine.prefill_admit_order",
                PREFILL_ADMIT_ORDERS,
            ),
            prefill_admit_max_age_ms=_strict_int(
                raw.get("prefill_admit_max_age_ms", defaults.prefill_admit_max_age_ms),
                "engine.prefill_admit_max_age_ms",
                min_value=0,
            ),
            thp_disable=_strict_bool(
                raw.get("thp_disable", defaults.thp_disable), "engine.thp_disable"
            ),
            engram_pin_pages=_strict_bool(
                raw.get("engram_pin_pages", defaults.engram_pin_pages),
                "engine.engram_pin_pages",
            ),
            mori_defer_handles=_strict_bool(
                raw.get("mori_defer_handles", defaults.mori_defer_handles),
                "engine.mori_defer_handles",
            ),
            prefill_warmup_ladder=_row_counts(
                raw.get("prefill_warmup_ladder", defaults.prefill_warmup_ladder),
                "engine.prefill_warmup_ladder",
            ),
            decode_warmup_widths=_widths(
                raw.get("decode_warmup_widths", defaults.decode_warmup_widths),
                "engine.decode_warmup_widths",
                words=("capture",),
            ),
            decode_warmup_tails=_widths(
                raw.get("decode_warmup_tails", defaults.decode_warmup_tails),
                "engine.decode_warmup_tails",
                words=("capture",),
            ),
        )
        if options.prefill_decode_interval_slope > 0 and (
            options.prefill_decode_interval_busy < 1
        ):
            raise RecipeRefused(
                "recipe-schema",
                "engine.prefill_decode_interval_slope needs "
                "engine.prefill_decode_interval_busy, its ceiling",
            )
        if options.prefill_short_first_rows and options.prefill_admit_order != "fifo":
            raise RecipeRefused(
                "recipe-schema",
                "engine.prefill_short_first_rows and engine.prefill_admit_order "
                f"{options.prefill_admit_order!r} both reorder prefill admission: "
                "set one",
            )
        return options

    def dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelOptions:
    """Target-model options configured from a recipe (PRs #3529, #3645)."""

    target_model_cls: str | None = None
    draft_model_cls: str | None = None
    ffn_mhc_fuse_quant: bool = False
    shared_act_quant: bool = False
    fused_attn: bool = False
    fused_attn_min_rows: int = 0
    maxima_grid: int = 0
    expert_placement: str | tuple[tuple[str, tuple[int, ...]], ...] | None = None
    block_transition: bool = False
    mono_mode: str = "auto"
    # Launch fusions of the non-MoE decode segment, by item number
    # (`atom.models.deepseek_v41_plugin.seg_fuse`): 1, the FFN mHC writes the
    # shared expert's MXFP8 input pair; 7, the shared expert's SiLU-mul and
    # w2's input quant as AITER's one launch.
    seg_fuse: tuple[int, ...] = ()
    # A decode step's MXFP8 attention also writes the inverse RoPE group
    # quant of its output, wo_a's input pair (AITER's
    # `mla_sparse_prefill_v41_inv_rope_quant`).
    attn_out_quant_fuse: bool = False
    # Launch fusions of the decode query path, by item name
    # (`atom.models.deepseek_v41_plugin.qpath_fuse`): "kv", the q/kv norm
    # also rotates, quantizes and stores the KV row; "q", wq_b also writes
    # the rotated MXFP8 query (the `v41.attn.query_rope_quant` seam).
    qpath_fuse: tuple[str, ...] = ()
    # The RoPE / window launch runs the kernel `qpath_fuse` records its
    # rotation forms from (`atom.model_ops.deepseek_v41.rope_window_reference`).
    rope_window_reference: bool = False
    # wq_b runs ahead of the index scorer on every step (the order of the
    # steps `qpath_fuse` plans).
    wq_b_before_index: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> ModelOptions:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeRefused("recipe-schema", "model_options must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RecipeRefused(
                "recipe-schema", f"unknown model_options field(s): {unknown}"
            )
        defaults = cls()
        mono_mode = raw.get("mono_mode", defaults.mono_mode)
        if mono_mode not in MONO_MODES:
            raise RecipeRefused(
                "recipe-schema",
                f"model_options.mono_mode must be one of {MONO_MODES}",
            )
        options = cls(
            target_model_cls=_opt_str(
                raw.get("target_model_cls", defaults.target_model_cls),
                "model_options.target_model_cls",
            ),
            draft_model_cls=_opt_str(
                raw.get("draft_model_cls", defaults.draft_model_cls),
                "model_options.draft_model_cls",
            ),
            ffn_mhc_fuse_quant=_strict_bool(
                raw.get("ffn_mhc_fuse_quant", defaults.ffn_mhc_fuse_quant),
                "model_options.ffn_mhc_fuse_quant",
            ),
            shared_act_quant=_strict_bool(
                raw.get("shared_act_quant", defaults.shared_act_quant),
                "model_options.shared_act_quant",
            ),
            fused_attn=_strict_bool(
                raw.get("fused_attn", defaults.fused_attn),
                "model_options.fused_attn",
            ),
            fused_attn_min_rows=_strict_int(
                raw.get("fused_attn_min_rows", defaults.fused_attn_min_rows),
                "model_options.fused_attn_min_rows",
                min_value=0,
            ),
            maxima_grid=_strict_int(
                raw.get("maxima_grid", defaults.maxima_grid),
                "model_options.maxima_grid",
                min_value=0,
            ),
            expert_placement=_freeze_placement(
                raw.get("expert_placement", defaults.expert_placement),
                "model_options.expert_placement",
            ),
            block_transition=_strict_bool(
                raw.get("block_transition", defaults.block_transition),
                "model_options.block_transition",
            ),
            mono_mode=mono_mode,
            seg_fuse=_items(
                raw.get("seg_fuse", defaults.seg_fuse),
                "model_options.seg_fuse",
                SEG_FUSE_ITEMS,
            ),
            attn_out_quant_fuse=_strict_bool(
                raw.get("attn_out_quant_fuse", defaults.attn_out_quant_fuse),
                "model_options.attn_out_quant_fuse",
            ),
            qpath_fuse=_items(
                raw.get("qpath_fuse", defaults.qpath_fuse),
                "model_options.qpath_fuse",
                QPATH_FUSE_ITEMS,
            ),
            rope_window_reference=_strict_bool(
                raw.get("rope_window_reference", defaults.rope_window_reference),
                "model_options.rope_window_reference",
            ),
            wq_b_before_index=_strict_bool(
                raw.get("wq_b_before_index", defaults.wq_b_before_index),
                "model_options.wq_b_before_index",
            ),
        )
        for item, other in ((1, "ffn_mhc_fuse_quant"), (7, "shared_act_quant")):
            if item in options.seg_fuse and getattr(options, other):
                raise RecipeRefused(
                    "recipe-schema",
                    f"model_options.seg_fuse item {item} and model_options.{other} "
                    "replace the same launch: set one",
                )
        return options

    def placement_spec(self) -> str | dict[str, list[int]] | None:
        if self.expert_placement is None or isinstance(self.expert_placement, str):
            return self.expert_placement
        return {k: list(v) for k, v in self.expert_placement}

    def dump(self) -> dict[str, Any]:
        data = asdict(self)
        data["expert_placement"] = self.placement_spec()
        return data


@dataclass(frozen=True)
class DSparkOptions:
    """DSpark speculative-decode options configured from a recipe (PRs #3639, #3645)."""

    fused_prologue: bool = False
    two_buffer_attn: bool = False
    lean_head: bool = False
    verify_argmax_once: bool = False
    # The draft head's vocab split over the DP group (pure DP attention):
    # `atom.model_ops.dspark_dp_head`. `_max_rows`: block rows above which the
    # replicated head serves.
    dp_sharded_head: bool = False
    dp_sharded_head_max_rows: int = 1024
    # Per-step sampling work nothing reads (`atom.model_ops.sampler`): the
    # greedy select of a batch with no greedy row; under forced acceptance,
    # the scoring of the target's own draw.
    sampler_skip_greedy_pick: bool = False
    verify_skip_unused_target_draw: bool = False
    # The independent-noise draw as the race and then the greedy rows' argmax
    # and select, three calls, in place of the one fused call.
    sampler_split_independent_noise: bool = False
    # The draft's context projection reads the taps' bf16 concatenation and
    # quantizes it itself, in place of the one launch that quantizes the taps
    # where they lie.
    context_taps_concat: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> DSparkOptions:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeRefused("recipe-schema", "dspark_options must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RecipeRefused(
                "recipe-schema", f"unknown dspark_options field(s): {unknown}"
            )
        defaults = cls()
        return cls(
            fused_prologue=_strict_bool(
                raw.get("fused_prologue", defaults.fused_prologue),
                "dspark_options.fused_prologue",
            ),
            two_buffer_attn=_strict_bool(
                raw.get("two_buffer_attn", defaults.two_buffer_attn),
                "dspark_options.two_buffer_attn",
            ),
            lean_head=_strict_bool(
                raw.get("lean_head", defaults.lean_head),
                "dspark_options.lean_head",
            ),
            verify_argmax_once=_strict_bool(
                raw.get("verify_argmax_once", defaults.verify_argmax_once),
                "dspark_options.verify_argmax_once",
            ),
            dp_sharded_head=_strict_bool(
                raw.get("dp_sharded_head", defaults.dp_sharded_head),
                "dspark_options.dp_sharded_head",
            ),
            dp_sharded_head_max_rows=_strict_int(
                raw.get("dp_sharded_head_max_rows", defaults.dp_sharded_head_max_rows),
                "dspark_options.dp_sharded_head_max_rows",
                min_value=1,
            ),
            sampler_skip_greedy_pick=_strict_bool(
                raw.get("sampler_skip_greedy_pick", defaults.sampler_skip_greedy_pick),
                "dspark_options.sampler_skip_greedy_pick",
            ),
            verify_skip_unused_target_draw=_strict_bool(
                raw.get(
                    "verify_skip_unused_target_draw",
                    defaults.verify_skip_unused_target_draw,
                ),
                "dspark_options.verify_skip_unused_target_draw",
            ),
            sampler_split_independent_noise=_strict_bool(
                raw.get(
                    "sampler_split_independent_noise",
                    defaults.sampler_split_independent_noise,
                ),
                "dspark_options.sampler_split_independent_noise",
            ),
            context_taps_concat=_strict_bool(
                raw.get("context_taps_concat", defaults.context_taps_concat),
                "dspark_options.context_taps_concat",
            ),
        )

    def dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Plan:
    """What the host runs on: module/seam tables and options, frozen before build.

    ``tables`` keys are ``"any"``, ``"eager"`` or a capture bucket (int) for
    seam routing; ``modules`` uses the same keys for ``nn.Module`` slot paths.
    A table of ``None`` (the ``--plugin-enable`` shorthand) means "whatever the
    enabled plugins register, in ``providers`` order".
    """

    plan_id: str
    strategy: str
    model_arch: str | None
    tables: Mapping[Any, Mapping[str, tuple[str, ...]] | None]
    providers: tuple[str, ...]
    aiter: tuple[str, ...]
    arms: Mapping[str, tuple[str, ...]]
    shadowed: tuple[tuple[str, str, str], ...] = ()
    strict_overlap: bool = False
    on_decline: str = "native"
    expect: Mapping[str, Any] = MappingProxyType({})
    requires: Mapping[str, Any] = MappingProxyType({})
    recipe_digests: tuple[str, ...] = ()
    modules: Mapping[Any, Mapping[str, tuple[str, ...]] | None] = MappingProxyType({})
    engine: EngineOptions = EngineOptions()
    model_options: ModelOptions = ModelOptions()
    dspark_options: DSparkOptions = DSparkOptions()
    schema_version: int = 1
    # Process environment a v2 recipe sets for the server it configures
    # (`apply_env`): native ATOM / AITER switches that are not recipe options.
    env: Mapping[str, str] = MappingProxyType({})

    @property
    def has_options(self) -> bool:
        return (
            self.engine != EngineOptions()
            or self.model_options != ModelOptions()
            or self.dspark_options != DSparkOptions()
            or bool(self.env)
        )

    @property
    def empty(self) -> bool:
        return not self.providers and not self.aiter and not self.has_options

    def __reduce__(self):
        state = {f.name: getattr(self, f.name) for f in fields(self)}
        state["tables"] = {
            k: None if t is None else dict(t) for k, t in self.tables.items()
        }
        state["modules"] = {
            k: None if t is None else dict(t) for k, t in self.modules.items()
        }
        for name in ("arms", "expect", "requires", "env"):
            state[name] = dict(state[name])
        return _rebuild_plan, (state,)

    def dump(self) -> dict[str, Any]:
        out = {
            "plan_id": self.plan_id,
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "model_arch": self.model_arch,
            "tables": {
                str(key): (
                    None if table is None else {k: list(v) for k, v in table.items()}
                )
                for key, table in self.tables.items()
            },
            "modules": {
                str(key): (
                    None if table is None else {k: list(v) for k, v in table.items()}
                )
                for key, table in self.modules.items()
            },
            "providers": list(self.providers),
            "aiter": list(self.aiter),
            "arms": {k: list(v) for k, v in self.arms.items()},
            "shadowed": [list(x) for x in self.shadowed],
            "strict_overlap": self.strict_overlap,
            "on_decline": self.on_decline,
            "expect": dict(self.expect),
            "requires": dict(self.requires),
            "recipe_digests": list(self.recipe_digests),
            "engine": self.engine.dump(),
            "model_options": self.model_options.dump(),
            "dspark_options": self.dspark_options.dump(),
            "env": dict(self.env),
        }
        return out


def _rebuild_plan(state: dict[str, Any]) -> Plan:
    state["tables"] = MappingProxyType(
        {
            k: None if t is None else MappingProxyType(t)
            for k, t in state["tables"].items()
        }
    )
    state["modules"] = MappingProxyType(
        {
            k: None if t is None else MappingProxyType(t)
            for k, t in state.get("modules", {}).items()
        }
    )
    for name in ("arms", "expect", "requires", "env"):
        state[name] = MappingProxyType(state.get(name, {}))
    return Plan(**state)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _names(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(v, str) and v for v in value
    ):
        raise RecipeRefused("recipe-schema", f"{where} must be a list of names")
    return tuple(value)


def _api_ok(requires: str, api: int = plugin_host.HOOK_API) -> bool:
    return plugin_host.api_satisfies(requires, api)


def shorthand(enable: Iterable[str] | str | None) -> Plan | None:
    """``--plugin-enable a,b``: every name to both halves, owners by this order."""
    if isinstance(enable, str):
        enable = enable.split(",")
    names = tuple(dict.fromkeys(n.strip() for n in enable or () if n.strip()))
    if not names:
        return None
    return Plan(
        plan_id=_digest(["enable", names, plugin_host.HOOK_API]),
        strategy="",
        model_arch=None,
        tables=MappingProxyType({"any": None}),
        modules=MappingProxyType({"any": None}),
        providers=names,
        aiter=names,
        arms=MappingProxyType({"native": (), "plugin": names, "identity": names}),
    )


def _check_slot_pattern(slot: str) -> None:
    if not isinstance(slot, str) or not slot or not _SLOT_RE.match(slot):
        raise RecipeRefused("recipe-schema", f"invalid module slot pattern {slot!r}")


ENV_PREFIXES = (
    "ATOM_",
    "AITER_",
    "FLYDSL_",
    "PYTORCH_",
    "HIP_",
    "HSA_",
    "GPU_",
    "NCCL_",
    "RCCL_",
    "TORCH_",
)
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _parse_env(raw: Any, is_v2: bool) -> dict[str, str]:
    if raw is None:
        return {}
    if not is_v2:
        raise RecipeRefused(
            "recipe-schema", "env requires schema 'atom.plugin_recipe/2'"
        )
    if not isinstance(raw, Mapping):
        raise RecipeRefused("recipe-schema", "env must map NAME -> value")
    out = {}
    for name, value in raw.items():
        if not (
            isinstance(name, str)
            and _ENV_RE.match(name)
            and name.startswith(ENV_PREFIXES)
        ):
            raise RecipeRefused(
                "recipe-schema", f"env name {name!r}: one of {ENV_PREFIXES}, upper case"
            )
        if name.startswith("ATOM_PLUGIN"):
            raise RecipeRefused("recipe-schema", f"env {name}: the host's own switch")
        if isinstance(value, bool):
            value = "1" if value else "0"
        if not isinstance(value, (str, int, float)):
            raise RecipeRefused("recipe-schema", f"env {name}: a string or a number")
        out[name] = str(value)
    return out


def _parse(document: Mapping[str, Any], *, model_arch: str | None) -> dict[str, Any]:
    """One recipe document (v1 or v2), checked against this ATOM's contracts."""
    if not isinstance(document, Mapping):
        raise RecipeRefused("recipe-schema", "recipe must be a JSON object")
    schema = document.get("schema")
    if schema not in SCHEMAS:
        raise RecipeRefused(
            "recipe-schema", f"expected schema in {SCHEMAS!r}, got {schema!r}"
        )
    is_v2 = schema == RECIPE_SCHEMA_V2
    arch = (
        document.get("arch")
        if is_v2 and "arch" in document
        else document.get("model_arch")
    )
    if not isinstance(arch, str) or not arch:
        raise RecipeRefused("recipe-schema", "model_arch (or arch) is required")
    if model_arch is not None and arch != model_arch:
        raise RecipeRefused("recipe-model-mismatch", f"{arch} != {model_arch}")
    strategy = document.get("strategy", "")
    if not isinstance(strategy, str):
        raise RecipeRefused("recipe-schema", "strategy must be a string")
    graph = document.get("graph", {"mode": "any"})
    mode = graph.get("mode", "any") if isinstance(graph, Mapping) else None
    if mode not in MODES:
        raise RecipeRefused("recipe-schema", f"graph.mode must be one of {MODES}")
    buckets: tuple[int, ...] = ()
    if mode == "capture":
        raw = graph.get("buckets")
        if (
            not isinstance(raw, (list, tuple))
            or not raw
            or not all(
                isinstance(b, int) and not isinstance(b, bool) and b > 0 for b in raw
            )
        ):
            raise RecipeRefused("recipe-schema", "capture recipes need int buckets")
        buckets = tuple(sorted(set(raw)))

    requires = document.get("requires", {})
    if not isinstance(requires, Mapping):
        raise RecipeRefused("recipe-schema", "requires must be an object")
    api = str(requires.get("atom_hook_api", f">={plugin_host.HOOK_API}"))
    if not _api_ok(api):
        raise RecipeRefused(
            "recipe-pin-mismatch", f"atom_hook_api {api} vs {plugin_host.HOOK_API}"
        )
    for seam, contract in dict(requires.get("seams", {})).items():
        spec = plugin_host.SEAMS.get(seam)
        if spec is None:
            raise RecipeRefused("recipe-unknown-seam", f"requires.seams: {seam}")
        if contract != spec.contract:
            raise RecipeRefused(
                "recipe-contract-mismatch",
                f"{seam}: recipe {contract} != ATOM {spec.contract}",
            )

    engine_opts = EngineOptions.from_mapping(document.get("engine"))
    model_opts = ModelOptions.from_mapping(document.get("model_options"))
    dspark_opts = DSparkOptions.from_mapping(document.get("dspark_options"))
    env = _parse_env(document.get("env"), is_v2)

    raw_plugins = document.get("plugins")
    has_explicit_plugins = raw_plugins is not None
    if raw_plugins is None:
        raw_plugins = []
    if not isinstance(raw_plugins, (list, tuple)):
        raise RecipeRefused("recipe-schema", "plugins must be a list")
    plugins: dict[str, Mapping[str, Any]] = {}
    for raw in raw_plugins:
        name = raw.get("name") if isinstance(raw, Mapping) else None
        if not isinstance(name, str) or not name or name in plugins:
            raise RecipeRefused("recipe-schema", "plugins need unique names")
        kind = raw.get("kind", "layer")
        if kind not in ("layer", "aiter"):
            raise RecipeRefused("recipe-schema", f"{name}: kind {kind!r}")
        if "serves" in raw or not is_v2:
            _names(raw.get("serves", []), f"{name}.serves")
        plugins[name] = raw

    extra_aiter = (
        _names(document.get("aiter", ()), "aiter") if "aiter" in document else ()
    )
    for name in extra_aiter:
        plugins.setdefault(name, {"name": name, "kind": "aiter", "serves": []})

    plan = document.get("plan", {})
    if not isinstance(plan, Mapping):
        raise RecipeRefused("recipe-schema", "plan must be an object")
    seams = plan.get("seams", document.get("seams", {}))
    if not isinstance(seams, Mapping):
        raise RecipeRefused("recipe-schema", "plan.seams must map seam -> owners")
    raw_modules = document.get("modules", plan.get("modules", {}))
    if not isinstance(raw_modules, Mapping):
        raise RecipeRefused("recipe-schema", "modules must map slot -> owners")
    if not is_v2 and raw_modules:
        raise RecipeRefused(
            "recipe-schema", "modules requires schema 'atom.plugin_recipe/2'"
        )

    defaults = document.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise RecipeRefused("recipe-schema", "defaults must be an object")
    allow_partition = bool(defaults.get("allow_partition", is_v2))
    shadow_policy = defaults.get("shadow_policy", "nested" if is_v2 else "error")
    on_decline = defaults.get("on_decline", "native")
    if shadow_policy not in SHADOW or on_decline not in ON_DECLINE:
        raise RecipeRefused("recipe-schema", "shadow_policy / on_decline")

    table: dict[str, tuple[str, ...]] = {}
    module_table: dict[str, tuple[str, ...]] = {}

    for seam, owners in seams.items():
        if seam not in plugin_host.SEAMS:
            raise RecipeRefused("recipe-unknown-seam", seam)
        owners = _names(owners, f"plan.seams[{seam}]")
        if (
            owners
            and plugin_host.SEAMS[seam].form == "window"
            and (
                len(owners) != 1
                or seams.get("v41.block.ffn")
                or raw_modules.get("layers.*.ffn")
            )
        ):
            raise RecipeRefused(
                "recipe-window-conflict",
                "transition needs one owner and native FFN",
            )
        if len(set(owners)) != len(owners):
            raise RecipeRefused("recipe-schema", f"{seam}: an owner twice")
        for owner in owners:
            ref = plugins.get(owner)
            if ref is None and is_v2 and not has_explicit_plugins:
                ref = plugins[owner] = {
                    "name": owner,
                    "kind": "layer",
                    "serves": [seam],
                }
            if ref is None:
                raise RecipeRefused("recipe-unknown-provider", f"{seam}: {owner}")
            if ref.get("kind", "layer") != "layer":
                raise RecipeRefused("recipe-conflict", f"{owner} is an AITER provider")
            if "serves" in ref and seam not in ref.get("serves", []):
                raise RecipeRefused(
                    "recipe-plugin-drift", f"{owner} does not declare {seam}"
                )
        if len(owners) > 1 and not allow_partition:
            raise RecipeRefused(
                "recipe-conflict",
                f"{seam} has {list(owners)}: set defaults.allow_partition "
                "(disjoint cells) or keep one",
            )
        if owners:
            table[seam] = owners
            alias = plugin_host.SEAM_TO_MODULE_SLOT.get(seam)
            if alias is not None:
                module_table.setdefault(alias, owners)

    for slot, owners in raw_modules.items():
        _check_slot_pattern(slot)
        owners = _names(owners, f"modules[{slot}]")
        if len(set(owners)) != len(owners):
            raise RecipeRefused("recipe-schema", f"modules[{slot}]: an owner twice")
        if len(owners) > 1 and not allow_partition:
            raise RecipeRefused(
                "recipe-conflict",
                f"modules[{slot}] has {list(owners)}: set defaults.allow_partition "
                "or keep one",
            )
        for owner in owners:
            ref = plugins.get(owner)
            if ref is None and not has_explicit_plugins:
                ref = plugins[owner] = {"name": owner, "kind": "layer"}
            if ref is None:
                raise RecipeRefused(
                    "recipe-unknown-provider", f"modules[{slot}]: {owner}"
                )
            if ref.get("kind", "layer") != "layer":
                raise RecipeRefused("recipe-conflict", f"{owner} is an AITER provider")
            serves = ref.get("serves")
            seam_alias = plugin_host.MODULE_SLOT_TO_SEAM.get(slot)
            if (
                serves is not None
                and slot not in serves
                and (seam_alias is None or seam_alias not in serves)
            ):
                raise RecipeRefused(
                    "recipe-plugin-drift", f"{owner} does not declare {slot}"
                )
        if owners:
            module_table[slot] = owners
            seam_alias = plugin_host.MODULE_SLOT_TO_SEAM.get(slot)
            if seam_alias is not None and seam_alias not in table:
                table[seam_alias] = owners

    shadowed: list[tuple[str, str, str]] = []
    for child, owners in table.items():
        parent = plugin_host.SEAMS[child].parent
        if parent in table:
            for owner in owners:
                if owner in table[parent]:
                    continue
                if shadow_policy == "error":
                    raise RecipeRefused(
                        "recipe-shadow",
                        f"{parent} owner {table[parent][0]} shadows {child} owner {owner}",
                    )
                shadowed.append((child, owner, table[parent][0]))

    arms = document.get("arms", {})
    if not isinstance(arms, Mapping):
        raise RecipeRefused("recipe-schema", "arms must map name -> providers")
    for arm, members in arms.items():
        unknown = [m for m in _names(members, f"arms[{arm}]") if m not in plugins]
        if unknown:
            raise RecipeRefused("recipe-unknown-provider", f"arms[{arm}]: {unknown}")
    runtime = document.get("runtime", {})
    expect = runtime.get("expect", {}) if isinstance(runtime, Mapping) else None
    if not isinstance(expect, Mapping):
        raise RecipeRefused("recipe-schema", "runtime.expect must be an object")
    return {
        "schema_version": 2 if is_v2 else 1,
        "digest": _digest(document),
        "model_arch": arch,
        "strategy": strategy,
        "mode": mode,
        "buckets": buckets,
        "plugins": plugins,
        "table": table,
        "module_table": module_table,
        "shadowed": shadowed,
        "arms": {k: tuple(v) for k, v in arms.items()},
        "strict_overlap": bool(defaults.get("strict_overlap", False)),
        "on_decline": on_decline,
        "expect": dict(expect),
        "requires": dict(requires),
        "engine": engine_opts,
        "model_options": model_opts,
        "dspark_options": dspark_opts,
        "env": env,
    }


def resolve(
    documents: Iterable[Mapping[str, Any]], *, model_arch: str | None = None
) -> Plan:
    """One strategy's recipes (one per graph mode / bucket set) -> Plan."""
    parsed = [_parse(d, model_arch=model_arch) for d in documents]
    if not parsed:
        raise RecipeRefused("recipe-schema", "no recipe")
    first = parsed[0]
    for other in parsed[1:]:
        for field_name in (
            "model_arch",
            "strategy",
            "on_decline",
            "strict_overlap",
            "engine",
            "model_options",
            "dspark_options",
            "env",
        ):
            if other[field_name] != first[field_name]:
                raise RecipeRefused("recipe-set-conflict", f"{field_name} differs")
        if other["expect"] != first["expect"]:
            raise RecipeRefused("recipe-set-conflict", "runtime.expect differs")
    tables: dict[Any, Mapping[str, tuple[str, ...]]] = {}
    module_tables: dict[Any, Mapping[str, tuple[str, ...]]] = {}
    for recipe in parsed:
        keys = recipe["buckets"] or (recipe["mode"],)
        for key in keys:
            clash = key in tables or (
                key in ("any", "eager") and {"any", "eager"} & set(tables)
            )
            if clash:
                raise RecipeRefused("recipe-set-conflict", f"two recipes serve {key!r}")
            tables[key] = MappingProxyType(dict(recipe["table"]))
            module_tables[key] = MappingProxyType(dict(recipe["module_table"]))

    providers: dict[str, None] = {}
    aiter: dict[str, None] = {}
    arms: dict[str, tuple[str, ...]] = {}
    for recipe in parsed:
        for name, ref in recipe["plugins"].items():
            if ref.get("kind", "layer") == "aiter":
                aiter[name] = None
        for table_owners in recipe["table"].values():
            providers.update(dict.fromkeys(table_owners))
        for mod_owners in recipe["module_table"].values():
            providers.update(dict.fromkeys(mod_owners))
        for arm, members in recipe["arms"].items():
            if arm in arms and arms[arm] != members:
                raise RecipeRefused("recipe-set-conflict", f"arm {arm} differs")
            arms[arm] = members
    everyone = (*providers, *aiter)
    arms.setdefault("native", ())
    arms.setdefault("plugin", everyone)
    arms.setdefault("identity", everyone)
    digests = tuple(sorted(r["digest"] for r in parsed))
    contracts = {
        seam: plugin_host.SEAMS[seam].contract
        for table in tables.values()
        for seam in table
        if seam in plugin_host.SEAMS
    }
    shadowed = tuple(s for r in parsed for s in r["shadowed"])
    schema_version = max(r["schema_version"] for r in parsed)
    id_payload: list[Any] = [
        digests,
        plugin_host.HOOK_API,
        contracts,
        sorted(arms),
        list(providers),
    ]
    if schema_version >= 2:
        id_payload.append(
            {
                "modules": {
                    str(k): {s: list(o) for s, o in v.items()}
                    for k, v in sorted(module_tables.items(), key=lambda kv: str(kv[0]))
                },
                "engine": first["engine"].dump(),
                "model_options": first["model_options"].dump(),
                "dspark_options": first["dspark_options"].dump(),
                "env": dict(sorted(first["env"].items())),
            }
        )
    plan_id = _digest(id_payload)
    return Plan(
        plan_id=plan_id,
        strategy=first["strategy"],
        model_arch=first["model_arch"],
        tables=MappingProxyType(tables),
        providers=tuple(providers),
        aiter=tuple(aiter),
        arms=MappingProxyType(arms),
        shadowed=shadowed,
        strict_overlap=first["strict_overlap"],
        on_decline=first["on_decline"],
        expect=MappingProxyType(first["expect"]),
        requires=MappingProxyType(first["requires"]),
        recipe_digests=digests,
        modules=MappingProxyType(module_tables),
        engine=first["engine"],
        model_options=first["model_options"],
        dspark_options=first["dspark_options"],
        env=MappingProxyType(dict(first["env"])),
        schema_version=schema_version,
    )


def _read(path: str | Path) -> tuple[str, ...]:
    """A recipe file, or a directory holding one strategy's recipes, as
    canonical JSON text."""
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise RecipeRefused("recipe-schema", f"no *.json under {path}")
    documents = []
    for file in files:
        try:
            documents.append(_canonical(json.loads(file.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as error:
            raise RecipeRefused("recipe-schema", f"{file}: {error}") from error
    return tuple(documents)


def load(path: str | Path, *, model_arch: str | None = None) -> Plan:
    """A recipe file, or a directory holding one strategy's recipes."""
    return resolve([json.loads(d) for d in _read(path)], model_arch=model_arch)


def _plan_of(documents: tuple[str, ...], enable: tuple[str, ...]) -> Plan | None:
    """Recipes plus ``--plugin-enable``, which may only narrow them."""
    names = shorthand(enable)
    if not documents:
        return names
    plan = resolve([json.loads(d) for d in documents])
    if names is None:
        return plan
    extra = [n for n in names.providers if n not in plan.arms["plugin"]]
    if extra:
        raise RecipeRefused("recipe-unknown-provider", f"--plugin-enable adds {extra}")
    arms = dict(plan.arms)
    arms["plugin"] = arms["identity"] = names.providers
    return replace(
        plan,
        arms=MappingProxyType(arms),
        plan_id=_digest([plan.plan_id, "enable", names.providers]),
    )


@dataclass(frozen=True)
class RecipeSource:
    """What ``Config`` carries: the recipes as canonical JSON text and the
    ``--plugin-enable`` names, plain data a rank's config copies and pickles;
    every process resolves it to the same ``Plan`` (``plan_id`` checks it)."""

    documents: tuple[str, ...] = ()
    enable: tuple[str, ...] = ()
    plan_id: str = ""
    empty: bool = True

    def resolve(self) -> Plan | None:
        plan = _plan_of(self.documents, self.enable)
        if plan is not None and plan.plan_id != self.plan_id:
            raise RecipeRefused(
                "recipe-set-conflict", "the recipes resolved to another plan here"
            )
        return plan


def from_cli(
    recipe: str | None, enable: str | None, arm: str = "plugin"
) -> RecipeSource | None:
    """``--plugin-recipe`` / ``--plugin-enable`` / ``--plugin-arm``, resolved
    once to refuse a bad recipe before any worker starts; None: no plugin."""
    documents = _read(recipe) if recipe else ()
    names = shorthand(enable)
    names = () if names is None else names.providers
    plan = _plan_of(documents, names)
    if plan is None:
        return None
    if arm not in plan.arms:
        raise RecipeRefused("recipe-unknown-arm", f"{arm!r} not in {sorted(plan.arms)}")
    apply_env(plan)
    apply_host_options(plan)
    return RecipeSource(documents, names, plan.plan_id, plan.empty)


def apply_host_options(plan: Plan | None) -> None:
    """The engine options that act on the server's main process, before any
    worker starts: ``engine.thp_disable`` (inherited by every process the
    server starts). Nothing under the kill switch, as `engine_options_for`."""
    from atom.utils import disable_thp, envs

    if plan is None or not envs.ATOM_PLUGIN_HOST:
        return
    disable_thp("server", enabled=plan.engine.thp_disable)


def apply_env(plan: Plan | None, environ=None) -> dict[str, str]:
    """Set a recipe's ``env`` in this process before any worker starts (the
    workers inherit it). A variable already set to another value refuses the
    recipe: the deployment has one source of truth. Returns what was set."""
    environ = os.environ if environ is None else environ
    if plan is None or not plan.env:
        return {}
    clash = {
        k: (environ[k], v)
        for k, v in plan.env.items()
        if k in environ and environ[k] != v
    }
    if clash:
        raise RecipeRefused(
            "recipe-env-conflict",
            f"set in the environment and differently in the recipe: {clash}",
        )
    applied = {k: v for k, v in plan.env.items() if k not in environ}
    environ.update(applied)
    return applied


def _document_text(document: Any, where: str) -> str:
    """One recipe document (JSON text, bytes or a decoded object) as JSON text."""
    if isinstance(document, Mapping):
        return _canonical(dict(document))
    if isinstance(document, (bytes, bytearray)):
        document = document.decode("utf-8")
    if isinstance(document, str):
        try:
            decoded = json.loads(document)
        except ValueError as error:
            raise RecipeRefused("recipe-schema", f"{where}: {error}") from error
        if not isinstance(decoded, Mapping):
            raise RecipeRefused("recipe-schema", f"{where}: a recipe is an object")
        return document
    raise RecipeRefused("recipe-schema", f"{where}: {type(document).__name__}")


def _enable_names(enable: Any) -> tuple[str, ...]:
    if enable is None:
        return ()
    if isinstance(enable, str):
        enable = enable.split(",")
    return tuple(
        dict.fromkeys(n.strip() for n in _names(enable, "enable") if n.strip())
    )


def from_payload(payload: Any) -> RecipeSource | None:
    """The ``RecipeSource`` a hot-plug payload describes."""
    if payload is None:
        return None
    if isinstance(payload, RecipeSource):
        return payload
    plan_id = None
    if isinstance(payload, Mapping):
        if "schema" in payload:
            documents, enable = (payload,), ()
        else:
            documents = payload.get("documents", payload.get("recipe", ()))
            enable = payload.get("enable", ())
            plan_id = payload.get("plan_id")
    elif isinstance(payload, (str, bytes)):
        documents, enable = (payload,), ()
    else:
        try:
            documents = tuple(payload)
        except TypeError as error:
            raise RecipeRefused(
                "recipe-schema", f"payload documents: {type(payload).__name__}"
            ) from error
    if isinstance(documents, (str, bytes, Mapping)):
        documents = (documents,)
    texts = tuple(
        _document_text(document, f"documents[{index}]")
        for index, document in enumerate(documents)
    )
    names = _enable_names(enable)
    plan = _plan_of(texts, names)
    if plan is None:
        return RecipeSource(texts, names, "", True)
    if plan_id is not None and plan_id != plan.plan_id:
        raise RecipeRefused(
            "recipe-set-conflict",
            f"payload plan_id {plan_id!r} does not resolve to the same plan here",
        )
    return RecipeSource(texts, names, plan.plan_id, plan.empty)


def engine_options_for(config: Any) -> EngineOptions:
    """Engine-level scheduling/DP options resolved from ``config``'s recipe."""
    explicit = getattr(config, "plugin_engine_options", None)
    if isinstance(explicit, EngineOptions):
        return explicit
    if isinstance(explicit, Mapping):
        return EngineOptions.from_mapping(explicit)
    from atom.utils import envs

    # As the model options: any arm (native included) runs one engine; only
    # the kill switch drops the recipe.
    if not envs.ATOM_PLUGIN_HOST:
        return EngineOptions()
    source = getattr(config, "plugin_source", None)
    if isinstance(source, Plan):
        return source.engine
    if not isinstance(source, RecipeSource):
        return EngineOptions()
    plan = source.resolve()
    return EngineOptions() if plan is None else plan.engine


def check_runtime(plan: Plan, config: Any) -> None:
    """``runtime.expect``: the server must be the one the recipe was made for."""
    has_transition = any(
        isinstance(key, int) and table and table.get("v41.block.transition")
        for key, table in plan.tables.items()
    )
    if has_transition:
        compile_cfg = getattr(config, "compilation_config", None)
        mode = str(getattr(compile_cfg, "cudagraph_mode", "")).rsplit(".", 1)[-1]
        transition_opted_in = (
            plan.model_options.block_transition
            or os.getenv("ATOM_V41_TRANSITION_AB") == "1"
        )
        if (
            not transition_opted_in
            or mode != "FULL"
            or getattr(compile_cfg, "level", None) != 0
            or getattr(config, "tensor_parallel_size", None) != 4
            or getattr(
                getattr(config, "parallel_config", None), "data_parallel_size", 1
            )
            != 1
            or any(
                getattr(config, n, 1) != 1
                for n in (
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                    "pipeline_parallel_size",
                )
            )
            or any(
                getattr(config, n, False)
                for n in (
                    "enable_dp_attention",
                    "enable_expert_parallel",
                    "enable_tbo",
                    "enable_decoder_swa_bounded_replay",
                    "enforce_eager",
                )
            )
        ):
            raise RecipeRefused(
                "recipe-window-topology",
                "capture transition needs opted-in FULL level-0 TP4 without DPA/EP/TBO/replay",
            )
    actual = {
        "level": lambda c: getattr(
            getattr(c, "compilation_config", None), "level", None
        ),
        "cudagraph_mode": lambda c: str(
            getattr(getattr(c, "compilation_config", None), "cudagraph_mode", None)
        ).rsplit(".", 1)[-1],
        "tensor_parallel_size": lambda c: getattr(c, "tensor_parallel_size", None),
        "model_arch": lambda c: (
            getattr(getattr(c, "hf_config", None), "architectures", None) or [None]
        )[0],
        "mono": lambda c: _mono_enabled(plan),
    }
    for name, expected in plan.expect.items():
        read = actual.get(name)
        if read is None:
            raise RecipeRefused("recipe-schema", f"runtime.expect has no {name!r}")
        value = read(config)
        if value != expected:
            raise RecipeRefused(
                "recipe-runtime-mismatch",
                f"{name}: expected {expected!r}, got {value!r}",
            )


def _mono_enabled(plan: Plan | None = None) -> bool:
    if plan is not None and plan.model_options.mono_mode == "off":
        return False
    from atom.utils import envs

    return bool(envs.ATOM_MONO_ENABLE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m atom.plugin_recipe")
    parser.add_argument("path", help="recipe file or a strategy's recipe directory")
    parser.add_argument(
        "--installed",
        action="append",
        default=[],
        help="name=seam_or_slot,...: what an installed plugin registers (drift check)",
    )
    args = parser.parse_args(argv)
    try:
        plan = load(args.path)
        for item in args.installed:
            name, _, targets = item.partition("=")
            served = set(filter(None, targets.split(",")))
            for table in plan.tables.values():
                for seam, owners in (table or {}).items():
                    slot = plugin_host.SEAM_TO_MODULE_SLOT.get(seam)
                    if name in owners and seam not in served and slot not in served:
                        raise RecipeRefused(
                            "recipe-plugin-drift", f"installed {name} lacks {seam}"
                        )
            for table in plan.modules.values():
                for slot, owners in (table or {}).items():
                    seam = plugin_host.MODULE_SLOT_TO_SEAM.get(slot)
                    if name in owners and slot not in served and seam not in served:
                        raise RecipeRefused(
                            "recipe-plugin-drift", f"installed {name} lacks {slot}"
                        )
    except RecipeRefused as refused:
        print(f"REFUSED {refused}")
        return 2
    print(json.dumps(plan.dump(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
