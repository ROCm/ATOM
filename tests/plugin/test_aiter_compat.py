import pytest

pytest.importorskip("aiter.ops.inverse_rope_group_quant")

from atom.model_ops import aiter_compat


def _call_inverse_rope_adapter(scale_layout="row"):
    inputs = [object() for _ in range(4)]
    result = aiter_compat.inverse_rope_group_quant(
        *inputs,
        num_groups=4,
        quant_group_size=128,
        scale_layout=scale_layout,
        x_fp8=object(),
        x_scale=object(),
    )
    return inputs, result


def test_inverse_rope_adapter_uses_new_scale_layout_keyword(monkeypatch):
    calls = []
    expected = (object(), object())

    def fake_inverse_rope(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        aiter_compat,
        "_inverse_rope_scale_keyword",
        lambda: "scale_layout",
    )
    monkeypatch.setattr(
        aiter_compat,
        "_aiter_inverse_rope_group_quant",
        fake_inverse_rope,
    )

    inputs, result = _call_inverse_rope_adapter()

    assert result is expected
    assert calls[0][0] == tuple(inputs)
    assert calls[0][1]["scale_layout"] == "row"
    assert "scale_shuffle" not in calls[0][1]


def test_inverse_rope_adapter_maps_row_layout_to_old_keyword(monkeypatch):
    calls = []
    expected = (object(), object())

    def fake_inverse_rope(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        aiter_compat,
        "_inverse_rope_scale_keyword",
        lambda: "scale_shuffle",
    )
    monkeypatch.setattr(
        aiter_compat,
        "_aiter_inverse_rope_group_quant",
        fake_inverse_rope,
    )

    inputs, result = _call_inverse_rope_adapter()

    assert result is expected
    assert calls[0][0] == tuple(inputs)
    assert calls[0][1]["scale_shuffle"] is False
    assert "scale_layout" not in calls[0][1]

    with pytest.raises(NotImplementedError, match="only maps the row"):
        aiter_compat.inverse_rope_group_quant(
            *inputs,
            num_groups=4,
            scale_layout="mfma_tile",
        )
    assert len(calls) == 1


def test_inverse_rope_adapter_rejects_unknown_aiter_signature(monkeypatch):
    def inverse_rope_without_layout_keyword():
        pass

    monkeypatch.setattr(
        aiter_compat,
        "_aiter_inverse_rope_group_quant",
        inverse_rope_without_layout_keyword,
    )
    aiter_compat._inverse_rope_scale_keyword.cache_clear()

    with pytest.raises(RuntimeError, match="Unsupported aiter"):
        aiter_compat._inverse_rope_scale_keyword()
