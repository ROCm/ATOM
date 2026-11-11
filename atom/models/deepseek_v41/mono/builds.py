# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What a V4.1 mono width was built as, fused or unfused.

One record a width (``WidthBuilds``, keyed by the step's row count), holding
the launchers and -- the reason it is a record and not a row count -- the K2
build key they came from. A step's shape is not a function of its rows alone,
so ``runner`` reads it off the build in hand rather than deriving it again.

A width is normally built *fused*: K2 carries the indexer, so a selecting
layer's scoring, its top-k and the pool rows the attention reads all happen
inside the one launch (``kernels.index_score``). At some widths that build
needs a byte of scratch a lane -- at TP 4 a 32- or 48-row K2 on the MFMA FP4
index layout comes out at 256 VGPRs instead of 255 -- and a persistent kernel
whose waves need scratch cannot be dispatched whole, so ``WidthBuilds`` refuses
the width and its steps take the original model's path.

The unfused variant is what ``WidthBuilds``' ``lean`` seam rebuilds such a
width as: K2 without its index stage (``index_bound_max`` 0, the shape the
DSpark draft already builds), with ATOM's own indexer run between K1 and K2
and its selection published into the buffers K2 would otherwise have filled
itself (``runner.V41MonoDecodeRunner._publish_selection``). The same
configuration, on kernels byte-identical to these, is what the out-of-tree
AITER provider drives, so this is a build choice, not a new kernel.

``strict`` is the other half of the same story: a width refused even after the
lean rebuild is still a silent fallback to the original model's path, and
``ATOM_DSV41_MONO_STRICT`` makes the runner say so rather than let a
measurement report it as mono.
"""

from dataclasses import dataclass, replace

from atom.utils import envs


@dataclass(frozen=True)
class RowsBuild:
    """The kernels of an S-row step: K1's build a layer and their launchers,
    and K2 (or, with ``check``, K2a and K2b apart).

    ``post_key`` is the K2 build key the launchers came from: the one place a
    step's ``tok_par`` and its K2 scratch layout are recorded, so the forward
    and the debug paths ask the build what it is instead of recomputing it
    from the row count and hoping the two answers agree. ``unfused``: K2 was
    built without its indexer and the runner scores between K1 and K2."""

    keys: list
    launchers: dict
    layer_post: object = None
    attn_post: object = None
    moe: object = None
    post_key: object = None
    unfused: bool = False

    @property
    def tok_par(self) -> bool:
        """Whether this build's kernels run token parallel. Read it off the
        build, never off the row count: the unfused variant of a width that
        would run token parallel does not."""
        return bool(self.post_key is not None and self.post_key.tok_par)


def unfused_post_key(key):
    """``key`` with K2's index stage out.

    ``index_bound_max`` 0 builds the K2 that reads a selection instead of
    making one; ``tok_par`` off because the indexer is what a token-parallel
    step splits (each rank selects its own rows and writes every rank's copy,
    ``index_score._st_out``) -- without it in the kernel every rank scores
    every row on the host side and keeps its own copy.
    """
    return replace(key, index_bound_max=0, tok_par=False)


def lean_width(built: RowsBuild, spilled: list, rebuild):
    """``built`` rebuilt unfused when one of its K2 launchers needs scratch,
    else None (``WidthBuilds``: nothing leaner, so the width is refused).

    ``rebuild()``: the same width's unfused build. Only a fused build has a
    leaner form, and only a K2 spill is what the unfused form fixes -- a K1
    that spills on its own is rebuilt by nothing here, nor is a K2b, which the
    unfused key does not change (``LayerPostBuild.halves``), nor a build whose
    K2 has no index stage to drop (one already unfused, or the DSpark draft's).
    """
    key = built.post_key
    if built.unfused or key is None or key.index_bound_max == 0:
        return None
    k2 = [f for f in (built.layer_post, built.attn_post) if f is not None]
    if not any(f in spilled for f in k2):
        return None
    return rebuild()


def strict() -> bool:
    """``ATOM_DSV41_MONO_STRICT``: refuse to fall back silently."""
    return bool(envs.ATOM_DSV41_MONO_STRICT)


def refuse(rows: int) -> None:
    """Raise for a step width mono will not serve, under ``strict``.

    Off (the default) this does nothing and the step takes the original
    model's path, as every unserved width does. On, it is an error: the only
    reason to ask for a width is a measurement, and a measurement that quietly
    reports the original path is worse than no measurement.
    """
    if not strict():
        return
    raise RuntimeError(
        f"V4.1 mono: width {rows} refused (the warning above names why); its "
        "steps would take the original path (ATOM_DSV41_MONO_STRICT=1)"
    )
