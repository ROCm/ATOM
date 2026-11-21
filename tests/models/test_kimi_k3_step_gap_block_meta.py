# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The block metadata mirror's host side (``step_gap.block_meta``): that the
methods it mirrors are still the ones it was written against, what it refuses,
how it is installed. (The values are compared on the GPU by
``tools/kimi_k3_step_gap_block_meta_check.py``.)"""

import types
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("aiter", reason="the drafter imports aiter")
pytest.importorskip("triton", reason="the mirror's kernel is Triton")

from atom.models.kimi_k3.step_gap import block_meta as bm
from atom.models.kimi_k3.step_gap import install as gap_install
from atom.spec_decode.dspark_proposer import DSparkProposer
from atom.utils import envs


def _drafter(cls=DSparkProposer, with_draft=True, mla=True, dcp=8, pool=None):
    drafter = cls.__new__(cls)
    drafter.speculative_config = NS(use_dspark_with_draft=lambda: with_draft)
    drafter.model = NS(kv_pool_spec=pool, block_attention_is_mla=mla)
    drafter.dcp_world_size, drafter.dcp_rank = dcp, 0
    return drafter


def test_the_mirrored_methods_are_the_ones_recorded():
    """An edit to any of the three turns the mirror off (``lean_refusal``):
    bring ``block_meta`` up to date with it, then record the new digest."""
    for name, function in (
        ("_build_paged_block_metadata", DSparkProposer._build_paged_block_metadata),
        ("_block_page_idx", DSparkProposer._block_page_idx),
        ("_paged_pad_ctx_len", DSparkProposer._paged_pad_ctx_len.fget),
    ):
        assert bm.source_digest(function) in bm.MIRRORED[name], name


def test_the_digest_ignores_layout_and_decorators_only():
    def one(a, b):
        return a + b

    def two(a, b):
        return a - b

    assert bm.source_digest(one) != bm.source_digest(two)
    assert len(bm.source_digest(one)) == 16


def test_the_served_drafter_is_covered():
    assert bm.lean_refusal(_drafter()) is None


@pytest.mark.parametrize(
    ("drafter", "why"),
    [
        (lambda: NS(), "not a DSparkProposer"),
        (lambda: _drafter(with_draft=False), "inside the target"),
        (lambda: _drafter(mla=False), "MLA"),
        (lambda: _drafter(dcp=1), "DCP"),
    ],
)
def test_what_the_mirror_does_not_cover(drafter, why):
    assert why in bm.lean_refusal(drafter())


def test_a_drafter_whose_method_changed_keeps_its_own():
    class Edited(DSparkProposer):
        @staticmethod
        def _block_page_idx(block_positions, tokens_per_entry, block_tables):
            return None, None

    drafter = _drafter(Edited)
    assert "_block_page_idx is not the method mirrored" in bm.lean_refusal(drafter)
    assert not bm.install(drafter)
    assert "_build_paged_block_metadata" not in drafter.__dict__


def test_install_binds_the_instance_only():
    drafter, other = _drafter(), _drafter()
    assert bm.install(drafter)
    bound = drafter._build_paged_block_metadata
    assert isinstance(bound, types.MethodType) and bound.__self__ is drafter
    assert bound.__func__ is bm.build_paged_block_metadata
    assert other._build_paged_block_metadata.__func__ is (
        DSparkProposer._build_paged_block_metadata
    )


@pytest.mark.parametrize(("value", "on"), [(None, False), ("0", False), ("1", True)])
def test_the_switch_is_off_by_default(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv("ATOM_KIMI_MONO_STEP_LEAN", raising=False)
    else:
        monkeypatch.setenv("ATOM_KIMI_MONO_STEP_LEAN", value)
    assert envs.ATOM_KIMI_MONO_STEP_LEAN is on
    drafter = _drafter()
    gap_install.install(None, None, drafter)
    assert ("_build_paged_block_metadata" in drafter.__dict__) is on
    gap_install.install(None, None, None)  # no drafter: nothing to bind


def _recording(monkeypatch):
    """The proposer's own method replaced by a recorder (after any digest was
    taken): what a drafter runs when the mirror is not installed."""
    calls = []
    monkeypatch.setattr(
        DSparkProposer,
        "_build_paged_block_metadata",
        lambda self, *args, **kwargs: calls.append((self, args, kwargs)),
    )
    return calls


def test_a_digest_mismatch_keeps_production_serving(monkeypatch):
    """The mirrored source changed upstream: the installer goes on without the
    mirror, raises nothing, and the drafter's next build is the proposer's."""
    monkeypatch.setenv("ATOM_KIMI_MONO_STEP_LEAN", "1")
    monkeypatch.setitem(bm.MIRRORED, "_build_paged_block_metadata", ("0" * 16,))
    drafter = _drafter()
    assert "_build_paged_block_metadata is not the method mirrored" in bm.lean_refusal(
        drafter
    )
    gap_install.install(None, None, drafter)  # must not raise
    assert "_build_paged_block_metadata" not in drafter.__dict__
    calls = _recording(monkeypatch)
    drafter._build_paged_block_metadata(
        "fc", "md", "anchors", scheduled_bs=1, max_seqlen_k=9
    )
    assert calls == [
        (drafter, ("fc", "md", "anchors"), {"scheduled_bs": 1, "max_seqlen_k": 9})
    ]


def test_a_source_that_cannot_be_read_keeps_production_serving(monkeypatch):
    """No source to digest (a deployment without the .py files): the same."""
    monkeypatch.setenv("ATOM_KIMI_MONO_STEP_LEAN", "1")

    def no_source(function):
        raise OSError("could not get source code")

    monkeypatch.setattr(bm.inspect, "getsource", no_source)
    drafter = _drafter()
    assert "cannot be compared" in bm.lean_refusal(drafter)
    assert bm.install(drafter) is False
    gap_install.install(None, None, drafter)  # must not raise
    assert "_build_paged_block_metadata" not in drafter.__dict__
    calls = _recording(monkeypatch)
    drafter._build_paged_block_metadata(
        "fc", "md", "anchors", scheduled_bs=2, max_seqlen_k=3
    )
    assert len(calls) == 1 and calls[0][0] is drafter


def test_install_never_raises(monkeypatch):
    def broken(drafter):
        raise RuntimeError("anything")

    monkeypatch.setattr(bm, "lean_refusal", broken)
    drafter = _drafter()
    assert bm.install(drafter) is False
    assert "_build_paged_block_metadata" not in drafter.__dict__


def test_the_runtime_images_proposer_is_accepted_too(monkeypatch):
    """Its three methods digest to the second entries; its sharded branch is
    taken on DCP alone, with no ``_draft_is_dcp_sharded`` to ask."""
    real = bm.source_digest
    image = {
        DSparkProposer._build_paged_block_metadata: bm.MIRRORED[
            "_build_paged_block_metadata"
        ][1],
        DSparkProposer._paged_pad_ctx_len.fget: bm.MIRRORED["_paged_pad_ctx_len"][1],
    }
    monkeypatch.setattr(
        bm, "source_digest", lambda function: image.get(function) or real(function)
    )

    class Image(DSparkProposer):
        _draft_is_dcp_sharded = None

    assert bm.lean_refusal(_drafter(Image)) is None
    assert "DCP" in bm.lean_refusal(_drafter(Image, dcp=1))
    assert bm.install(_drafter(Image)) is True
    for digests in bm.MIRRORED.values():
        assert isinstance(digests, tuple) and all(len(d) == 16 for d in digests)


def test_a_draft_with_a_pool_of_its_own_is_not_sharded():
    """Where the proposer tells such a draft apart (``_draft_is_dcp_sharded``;
    the runtime image's does not: its method shards on DCP alone)."""
    if not hasattr(DSparkProposer, "_draft_is_dcp_sharded"):
        pytest.skip("this proposer shards on DCP alone")
    assert "DCP" in bm.lean_refusal(_drafter(pool=NS()))
