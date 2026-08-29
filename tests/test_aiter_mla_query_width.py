from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="requires AITER to import aiter_mla")

from atom.model_ops.attentions.aiter_mla import _decode_query_width


def test_batch_query_width_overrides_runner_drafter_width():
    runner = SimpleNamespace(drafter=SimpleNamespace(mtp_k=3))
    producer_target_only_batch = SimpleNamespace(num_spec_query_tokens=1)

    assert _decode_query_width(producer_target_only_batch, runner) == 1


def test_runner_width_is_only_a_legacy_fallback():
    runner = SimpleNamespace(drafter=SimpleNamespace(mtp_k=3))

    assert _decode_query_width(SimpleNamespace(), runner) == 4


def test_decode_query_width_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        _decode_query_width(
            SimpleNamespace(num_spec_query_tokens=0), SimpleNamespace()
        )
