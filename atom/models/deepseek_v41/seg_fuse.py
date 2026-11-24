# SPDX-License-Identifier: MIT
"""`ATOM_DSV41_SEG_FUSE`: opt-in launch fusion around V4.1's shared expert.

A comma list of item numbers; unset or empty leaves every path as it was (this
file then only parses the variable: no op is registered, no AITER module is
loaded). Each item replaces two back-to-back launches of a decode or prefill
step by one whose output is bit for bit the pair's:

  1  the FFN seam's Mega-MHC also writes the MXFP8 pair of its normed output
     (the instantiation the attention seam already runs) and the shared
     expert's `gate_up_proj` reads that pair: its own `quantize_fp8` launch
     goes away.
  7  the shared expert's `silu_and_mul` and the `quantize_fp8` in front of
     `w2` become AITER's `silu_mul_mxfp8` (gfx950).
 10  the shared expert's `w2` GEMM and the router's top-k selection become one
     launch (AITER `gemm_a8w8_blockscale_topk`: the GEMM's tuned tile in the
     launch's first workgroups, the selection in the trailing ones; the two
     read nothing of each other). The router GEMM is issued before the shared
     expert's activation instead of after `w2`, and the selection is handed to
     the routed experts' call. Needs item 7 on the layer, the target router
     (384 experts, top 6) and TP 1; a row count whose `w2` row is not such a
     tile, or an image step, runs the two launches as before.

The items apply only where the shared expert runs on the layer's own stream
(`MoE.single_stream_moe_forward`: no dual stream, no comm-fused dispatch) with
native MXFP8 `gate_up_proj` / `w2`; anywhere else an item is ignored, and so is
one whose AITER entry this AITER build does not have (items 7 and 10). The
boot lines below say what each layer took, which is not what was asked for
where an item's conditions fail.

Item 9 of the same numbering (the `routed + shared` add folded into the next
attention seam) is accepted and ignored: the routed combine adds the shared
output itself (`FusedMoE.combine_accepts_shared`).
"""

import logging
import os

logger = logging.getLogger("atom")

KNOWN_ITEMS = frozenset({1, 7, 10})
# Accepted for existing launch lines, not implemented here (see the docstring).
SUPERSEDED_ITEMS = frozenset({9})


def parse_items(raw: str | None) -> frozenset[int]:
    """The implemented item numbers of an `ATOM_DSV41_SEG_FUSE` value. Anything
    else in the list is logged and ignored: a server start must not fail on it."""
    items = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit() and int(part) in KNOWN_ITEMS:
            items.add(int(part))
        elif part.isdigit() and int(part) in SUPERSEDED_ITEMS:
            logger.info(
                "ATOM_DSV41_SEG_FUSE: item %s is not needed here (the routed "
                "combine adds the shared expert's output); ignored",
                part,
            )
        else:
            logger.warning(
                "ATOM_DSV41_SEG_FUSE: item %r is not implemented (known: %s); ignored",
                part,
                sorted(KNOWN_ITEMS),
            )
    return frozenset(items)


ITEMS = parse_items(os.environ.get("ATOM_DSV41_SEG_FUSE"))


def enabled(item: int) -> bool:
    return item in ITEMS


def _spans(ids) -> str:
    """Sorted layer ids as `1-12, 14-18`."""
    runs: list[list[int]] = []
    for i in sorted(ids):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _by_reason(outcomes: dict[int, str]) -> str:
    """`{layer: reason}` as `0, 13: reason; 39: reason`."""
    reasons: dict[str, list[int]] = {}
    for layer, reason in outcomes.items():
        reasons.setdefault(reason, []).append(layer)
    return "; ".join(f"{_spans(ids)}: {why}" for why, ids in reasons.items())


# -- items 1 and 7 ------------------------------------------------------------

_noted: set[tuple[bool, bool]] = set()


def note_layer(layer: str, ffn_pair: bool, act_quant: bool) -> None:
    """With an item on: one boot line per process and outcome, at the first
    layer that has it -- what a layer's FFN took."""
    if not ITEMS or (ffn_pair, act_quant) in _noted:
        return
    _noted.add((ffn_pair, act_quant))
    logger.info(
        "dsv41 seg fuse (ATOM_DSV41_SEG_FUSE=%s): ffn input pair from the norm=%s, "
        "act+quant in one launch=%s (first at %s)",
        ",".join(str(item) for item in sorted(ITEMS)),
        ffn_pair,
        act_quant,
        layer,
    )


_act_quant_missing_noted = False


def act_quant_op(width: int):
    """Item 7's op (`v41_silu_mul_mxfp8`) for a shared expert of this per-rank
    width, or None where it cannot run: another arch, a width that is not a
    multiple of 32, or an AITER without `silu_mul_mxfp8` (logged once)."""
    global _act_quant_missing_noted
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950" or width <= 0 or width % 32:
        return None
    try:
        from atom.model_ops.deepseek_v41.seg_fuse_ops import v41_silu_mul_mxfp8
    except ImportError as error:
        if not _act_quant_missing_noted:
            _act_quant_missing_noted = True
            logger.warning(
                "ATOM_DSV41_SEG_FUSE item 7 declined: this AITER has no "
                "silu_mul_mxfp8 (%s)",
                error,
            )
        return None
    return v41_silu_mul_mxfp8


# -- item 10: `w2` and the top-k selection in one launch ----------------------

# Per model ("target" / "draft"): layer id -> None where the layer takes item
# 10, else why not; one boot line a model once its last layer is built, and
# one line per (model, row count) the first time a pass runs it.
_tail: dict[str, dict[int, str | None]] = {"target": {}, "draft": {}}
_tail_rows_logged: set[tuple[str, int, bool]] = set()
_tail_summarized: set[str] = set()


def _tail_reason(module) -> str | None:
    """Why this MoE layer cannot take item 10, or None. Load-time facts only:
    the weights are not loaded yet, so what depends on them (a preshuffled
    `w2`, its tuned row at a row count) is decided per call."""
    experts = getattr(module.experts, "global_num_experts", None)
    top_k = getattr(module.experts, "top_k", None)
    if (experts, top_k) != (384, 6):
        return (
            f"router {experts} experts / top {top_k}: only the target router "
            "(384 / 6) is validated"
        )
    if not module.seg_fuse_act_quant:
        return "item 7 (act + quant in one launch) is not on this layer"
    if module.tp_size != 1:
        return f"TP {module.tp_size}"
    w2 = module.shared_experts.w2
    if w2.native_a8_group_rows != 32 or w2.bias is not None or w2.is_output_padded:
        return "w2 is not a plain native MXFP8 32x32 GEMM"
    try:
        from aiter.ops.gemm_topk_tail import gemm_a8w8_blockscale_topk  # noqa: F401
    except ImportError:
        return "this AITER has no gemm_a8w8_blockscale_topk"
    return None


def tail_layer(module, layer_id: int, layers: int | None) -> bool:
    """Item 10 at model build: whether this layer takes it. Logs one line per
    model when its last layer (``layers`` - 1, where known) has been seen, and
    at the first layer otherwise: the layers that took it and, per reason, the
    ones that did not."""
    experts = getattr(module.experts, "global_num_experts", None)
    group = "target" if experts == 384 else "draft"
    reason = _tail_reason(module)
    seen = _tail[group]
    seen[layer_id] = reason
    # The draft's layers carry ids behind the target's, so its count never
    # reaches ``layers``: its line is printed at its first layer (every draft
    # layer declines for the same reason).
    if len(seen) == layers or (group == "draft" and len(seen) == 1):
        _tail_summary(group)
    return reason is None


def _tail_summary(group: str) -> None:
    """The model's boot line, once: at its last layer where the layer count is
    known, else at the first pass that runs item 10."""
    if group in _tail_summarized:
        return
    _tail_summarized.add(group)
    seen = _tail[group]
    took = [i for i, why in seen.items() if why is None]
    declined = {i: why for i, why in seen.items() if why is not None}
    logger.info(
        "dsv41 seg fuse item 10 (%s): w2 + top-k in one launch TOOK layers [%s] "
        "(%d of %d; the tile is the w2 row's, logged per row count at the first "
        "pass); declined [%s]",
        group,
        _spans(took),
        len(took),
        len(seen),
        _by_reason(declined) if declined else "none",
    )


def note_tail_rows(experts: int, rows: int, host: str | None) -> None:
    """Item 10, once per (model, row count, outcome): the tile that hosts the
    selection at this row count, or that the two launches ran."""
    group = "target" if experts == 384 else "draft"
    _tail_summary(group)
    key = (group, rows, host is not None)
    if key in _tail_rows_logged:
        return
    _tail_rows_logged.add(key)
    if host is not None:
        logger.info(
            "dsv41 seg fuse item 10 (%s): rows=%d ONE launch, host tile %s",
            group,
            rows,
            host,
        )
    else:
        logger.info(
            "dsv41 seg fuse item 10 (%s): rows=%d two launches as before (AITER "
            "has no host tile for this w2 row)",
            group,
            rows,
        )
