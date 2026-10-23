# SPDX-License-Identifier: MIT
"""ATOM_SPEC_LOSSLESS_SAMPLING and ATOM_ENABLE_RELAXED_MTP refuse each other at
drafter construction, before anything else and for every drafter flavor, so
whether the pair errors cannot depend on PP, PCP, or the draft model."""

import pytest

pytest.importorskip("aiter", reason="the drafter imports AITER's parallel state")

from atom.spec_decode.drafter import Drafter


class _Bare(Drafter):
    def _resolve_mtp_k(self):
        return 1

    def propose(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize("lossless, relaxed", [(1, 1), (1, 0), (0, 1), (0, 0)])
def test_relaxed_mtp_and_lossless_sampling_refuse_each_other_first(
    monkeypatch, lossless, relaxed
):
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", str(lossless))
    monkeypatch.setenv("ATOM_ENABLE_RELAXED_MTP", str(relaxed))
    drafter = _Bare.__new__(_Bare)
    if lossless and relaxed:
        with pytest.raises(ValueError, match="ATOM_ENABLE_RELAXED_MTP"):
            Drafter.__init__(drafter, None, None, None)
    else:
        # Past the check, construction fails on the missing config instead.
        with pytest.raises(AttributeError):
            Drafter.__init__(drafter, None, None, None)
