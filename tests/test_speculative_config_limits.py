from types import SimpleNamespace

from atom.config import _resolve_sequential_drafter_max_spec


def _spec(method="mtp", **draft_attrs):
    return SimpleNamespace(
        method=method,
        draft_model_hf_config=SimpleNamespace(**draft_attrs),
    )


def test_plain_mtp_defaults_to_mtp8_horizon():
    assert _resolve_sequential_drafter_max_spec(_spec()) == 8


def test_eagle3_keeps_historic_default_horizon():
    assert _resolve_sequential_drafter_max_spec(_spec(method="eagle3")) == 4


def test_checkpoint_declared_horizon_takes_precedence():
    assert (
        _resolve_sequential_drafter_max_spec(_spec(max_speculative_tokens=12)) == 12
    )
