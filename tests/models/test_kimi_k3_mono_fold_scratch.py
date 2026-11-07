# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The width guard (``WidthBuilds``: no kernel with scratch) on the K2 builds a
TP8 Kimi-K3 server makes with ``ATOM_MONO_MLA_POST=1 ATOM_MONO_MLA_PREP=3``.

K2 sits at its 256-VGPR limit with 100+ SGPRs spilled to VGPR lanes. Some fold
builds (post and / or prep) then come out of the backend with a private segment
that no instruction touches: the SGPR allocator leaves a spill slot whose spills
and reloads all went away (rematerialized), SILowerSGPRSpills hands lanes only
to referenced slots, so the orphan (16 B) reaches frame finalization, which adds
an emergency scavenging slot (4 B): 20 B a lane. The post builds at 12 / 16
rows spill one VGPR outright (8 B). The guard refuses any of them, which used
to cost the whole width its mono path; ``runner._lean_width`` rebuilds those
under another LLVM scheduler (``k2.FOLD_SCHEDULERS``: a different schedule
changes which values the allocator spills), and gives a layer its default K2
build only when none of those fits.

Compile-only: every K2 key of a width (the model's 93 layers, folds on the 24
MLA layers) through ``WidthBuilds`` with the runner's ``_lean_width``, as the
server builds it. The width must be served, every served launcher free of
scratch, the default builds untouched; it prints which layers keep their
folds, and under which scheduler. Slow (a minute or more a width), so opt-in:
``ATOM_MONO_FOLD_SCRATCH_WIDTHS=6,12`` (the server's 1- and 2-request widths at
query length 6; any of 1 .. 16), with a GPU or ``ARCH=gfx950``.
"""

import os
from dataclasses import replace

import pytest

WIDTHS = [
    int(w) for w in os.environ.get("ATOM_MONO_FOLD_SCRATCH_WIDTHS", "").split(",") if w
]
LAYERS = 93
MLA = [*range(3, LAYERS - 1, 4), LAYERS - 1]


def test_folds_stop_at_16_rows():
    """Wider steps (18, 24, ... rows at query length 6) build no fold: their
    K2s are the default builds, so the folds cannot cost them anything."""
    post = pytest.importorskip("atom.models.kimi_k3.mono.kernels.stages.mla_post_plan")
    prep = pytest.importorskip("atom.models.kimi_k3.mono.kernels.stages.mla_prep")
    assert post.MAX_ROWS == prep.MAX_ROWS == 16


def server_k2_keys(rows: int) -> dict:
    """``K3MonoDecodeRunner._k2_keys`` at ``rows`` with every MLA layer's post
    and sub-mode 3 prep (the build-key fields that shape the code; the model's
    eps values only change constants)."""
    from atom.models.kimi_k3.mono.config import layer_block
    from atom.models.kimi_k3.mono.kernels import layer as k2
    from atom.models.kimi_k3.mono.kernels.stages import mla_prep as k2pr

    keys = {}
    for layer in range(1, LAYERS):
        blocks, write_idx = layer_block(layer)
        kw = {"mla_post": layer in MLA and rows <= 16}
        prep_blocks = layer_block(layer + 1)[0]
        if layer + 1 in MLA and rows <= 16 and k2pr.red_fits(rows, prep_blocks + 1):
            kw.update(mla_prep=3, prep_nblocks=prep_blocks)
        keys[layer] = k2.K2Build(tokens=rows, nblocks=blocks + (write_idx >= 0), **kw)
    return keys


@pytest.mark.skipif(not WIDTHS, reason="opt-in: ATOM_MONO_FOLD_SCRATCH_WIDTHS")
@pytest.mark.parametrize("rows", WIDTHS)
def test_the_servers_fold_builds_pass_the_scratch_guard(rows, caplog):
    pytest.importorskip("flydsl", reason="K2 builds need FlyDSL")
    pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")
    if not os.environ.get("ARCH"):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("compile-only needs a GPU or ARCH (e.g. gfx950)")
    from atom.models.kimi_k3.mono import runner
    from atom.models.kimi_k3.mono.kernels import layer as k2
    from atom.mono.runtime.compile import compile_kernels, kernel_names, scratch_bytes
    from atom.mono.runtime.widths import WidthBuilds

    keys = server_k2_keys(rows)

    def build(width):
        return runner.WidthBuild(
            k1_keys={},
            k1_launchers={},
            k2_keys=keys,
            k2_launchers={key: k2.build(key) for key in set(keys.values())},
        )

    widths = WidthBuilds(
        build, runner.WidthBuild.kernels, None, "Kimi-K3 mono", lean=runner._lean_width
    )
    caplog.set_level("INFO", logger="atom")
    assert widths.prepare(rows), caplog.text
    served = widths[rows]
    for key, launcher in served.k2_launchers.items():
        assert scratch_bytes(launcher) == 0, kernel_names(launcher)
    # a layer keeps its fold build (under LLVM's default scheduler or one of
    # ``k2.FOLD_SCHEDULERS``) or falls back to its default build
    for layer, key in served.k2_keys.items():
        rescheduled = {replace(keys[layer], sched=s) for s in k2.FOLD_SCHEDULERS}
        assert key in {keys[layer], k2.unfolded(keys[layer]), *rescheduled}, layer
    folds = sorted(
        layer for layer, key in served.k2_keys.items() if key.mla_post or key.mla_prep
    )
    wanted = sorted(
        layer for layer, key in keys.items() if key.mla_post or key.mla_prep
    )
    scheds = {layer: key.sched for layer, key in served.k2_keys.items() if key.sched}
    print(f"\n{rows} rows: folds kept on K2 of layers {folds} (of {wanted})")
    print(f"  rescheduled: {scheds}")
    for key in sorted(set(keys.values()), key=str):
        launcher = k2.build(key)
        compile_kernels([(launcher, k2.abi_for(key))])
        print(f"  {kernel_names(launcher)[0]}: scratch {scratch_bytes(launcher)} B")
