# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""``_check_fp8_usable``: refuse an FP8 request the run cannot honour.

Both failure modes it covers are silent-or-misleading without it. A
``build_config`` family (DSv3/DSv4) never receives fp8_kwargs, so the run would
train in BF16 while reporting FP8; and an unsupported recipe only surfaces at
the first forward, minutes in, as a TE traceback.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron", reason="Megatron backend is optional in CPU CI")


@pytest.fixture(autouse=True)
def _ignore_te_blockwise_pin(monkeypatch):
    """These tests are about the recipe/config preflight, not the TE version gate.

    On any TE older than the gate requires it would fire first and mask what each
    test is actually checking.
    """
    monkeypatch.setenv("ATOMRL_SKIP_TE_BLOCKWISE_CHECK", "1")


from atomrl.engine.training.megatron_native_engine import MegatronNativeEngine
from megatron.core.transformer.transformer_config import TransformerConfig

check = MegatronNativeEngine._check_fp8_usable
KW = {"fp8": "e4m3", "fp8_recipe": "tensorwise"}


def _cfg(**over):
    base = {
        "num_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 8,
        "ffn_hidden_size": 256,
    }
    return TransformerConfig(**base, bf16=True, **over)


def test_a_config_without_fp8_is_refused_as_a_silent_bf16_downgrade():
    # What build_config produces: fp8 never applied.
    with pytest.raises(RuntimeError, match="would train in BF16"):
        check(KW, _cfg())


def test_the_refusal_names_the_config_type_so_the_family_is_identifiable():
    class MLATransformerConfig(SimpleNamespace):
        pass

    with pytest.raises(RuntimeError, match="MLATransformerConfig"):
        check(KW, MLATransformerConfig(fp8=None))


def test_the_refusal_does_not_blame_the_users_format():
    # Before the guard this raised "E4M3 and HYBRID are the only supported FP8
    # formats" -- at a user who had in fact passed e4m3.
    with pytest.raises(RuntimeError) as ei:
        check(KW, _cfg())
    assert "only supported FP8 formats" not in str(ei.value)


def test_a_supported_recipe_on_an_fp8_config_passes():
    check(KW, _cfg(fp8="e4m3", fp8_recipe="tensorwise"))


def test_delayed_is_supported_too():
    check(
        {"fp8": "e4m3", "fp8_recipe": "delayed"}, _cfg(fp8="e4m3", fp8_recipe="delayed")
    )


@pytest.fixture
def te_rejects_the_recipe(monkeypatch):
    """Make TE refuse whatever recipe it is handed.

    These two tests used to hardcode ``blockwise`` as the unsupported recipe,
    which was true on TE 2.15 and is false from 2.20 -- blockwise working is the
    point of this feature. What they are really about is the plumbing: when TE
    says no, we refuse up front with a message naming the recipe. So drive that
    from a stub rather than from whatever TE happens to support today.
    """
    q = pytest.importorskip(
        "transformer_engine.pytorch.quantization", exc_type=ImportError
    )

    def _reject(_recipe):
        raise RuntimeError("stub: this build cannot run that recipe")

    monkeypatch.setattr(q, "check_recipe_support", _reject, raising=False)


@pytest.mark.parametrize("recipe", ["blockwise", "mxfp8"])
def test_a_recipe_this_build_cannot_run_is_refused_up_front(
    recipe, te_rejects_the_recipe
):
    kw = {"fp8": "e4m3", "fp8_recipe": recipe}
    with pytest.raises(RuntimeError, match="not usable on this build"):
        check(kw, _cfg(fp8="e4m3", fp8_recipe=recipe))


def test_the_recipe_refusal_quotes_the_recipe_that_was_asked_for(te_rejects_the_recipe):
    kw = {"fp8": "e4m3", "fp8_recipe": "blockwise"}
    with pytest.raises(RuntimeError, match="'blockwise'"):
        check(kw, _cfg(fp8="e4m3", fp8_recipe="blockwise"))


def test_a_recipe_this_build_can_run_is_allowed_through(monkeypatch):
    """The complement: when TE accepts the recipe, the preflight must not refuse it.

    Stubbed in the same direction as te_rejects_the_recipe, so this passes on both
    the container's TE 2.15 (no blockwise) and a 2.20+ prefix (blockwise) -- the
    suite should not depend on which TE happens to be on PYTHONPATH.
    """
    q = pytest.importorskip(
        "transformer_engine.pytorch.quantization", exc_type=ImportError
    )

    monkeypatch.setattr(q, "check_recipe_support", lambda _r: None, raising=False)
    check(
        {"fp8": "e4m3", "fp8_recipe": "blockwise"},
        _cfg(fp8="e4m3", fp8_recipe="blockwise"),
    )


def test_a_value_error_from_megatron_is_wrapped_not_leaked():
    # Without TE the check returns before asking Megatron for the recipe.
    pytest.importorskip("transformer_engine.pytorch.quantization", exc_type=ImportError)
    # get_fp8_recipe raises ValueError, which an `except RuntimeError` would miss.
    with pytest.raises(RuntimeError, match="not usable on this build"):
        check(
            {"fp8": "e4m3", "fp8_recipe": "tensorwise"},
            SimpleNamespace(fp8="nonsense", fp8_recipe="tensorwise"),
        )
