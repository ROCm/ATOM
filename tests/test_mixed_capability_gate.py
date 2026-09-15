# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""`--enable-mixed-prefill-decode` must refuse unsupported configs at LAUNCH.

The refusals used to be seven `assert`s and two `raise`s spread over five
files, all of them fired only once a mixed batch had been built. Two problems
followed. A user learned the combination was unsupported some way into serving
rather than at startup; and `python -O` strips asserts, so under it the
combinations did not refuse at all -- they ran with whatever wrong numerics the
assert was standing in front of.

The table lives in `Config._validate_mixed_prefill_decode`. The runtime sites
stay as backstops, but every one of them is now a `raise`.
"""

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mixed_refusals():
    """Every assert/raise in atom/ whose message is about mixed batching."""
    out = []
    for p in sorted((_ROOT / "atom").rglob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "mixed" not in src.lower():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assert, ast.Raise)):
                t = ast.unparse(n).lower()
                if "mixed" in t and (
                    "enable-mixed" in t or "is_mixed" in t or "mixed prefill" in t
                ):
                    rel = p.relative_to(_ROOT)
                    out.append((isinstance(n, ast.Assert), f"{rel}:{n.lineno}"))
    return out


def test_no_mixed_refusal_is_an_assert():
    """An `assert` here is a refusal that disappears under `python -O`."""
    asserts = [where for is_assert, where in _mixed_refusals() if is_assert]
    assert not asserts, (
        "mixed-batch refusals that vanish under `python -O`, leaving the "
        f"unsupported path to run with wrong numerics: {asserts}"
    )


def test_there_are_refusals_at_all():
    """Guards the test above from passing because it found nothing."""
    found = _mixed_refusals()
    assert len(found) >= 5, (
        f"only {len(found)} mixed refusal(s) found; the scan probably stopped "
        "matching (it keys off the message wording), which would make the "
        "assert-free check above vacuous"
    )


def test_the_config_table_exists_and_is_called():
    cfg = (_ROOT / "atom/config.py").read_text(encoding="utf-8")
    tree = ast.parse(cfg)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_validate_mixed_prefill_decode" in names

    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "__post_init__":
            body = ast.unparse(n)
            if "_validate_mixed_prefill_decode()" in body:
                return
    pytest.fail(
        "_validate_mixed_prefill_decode is defined but never called from a "
        "__post_init__; a table nobody runs refuses nothing"
    )


def test_the_table_returns_early_when_the_flag_is_off():
    """Cost and blast radius are both zero for everyone not using the flag."""
    cfg = (_ROOT / "atom/config.py").read_text(encoding="utf-8")
    tree = ast.parse(cfg)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == (
            "_validate_mixed_prefill_decode"
        ):
            first = ast.unparse(n.body[1] if len(n.body) > 1 else n.body[0])
            assert "enable_mixed_prefill_decode" in first and "return" in first, (
                "the table must short-circuit when the flag is off; it runs in "
                f"every Config.__post_init__. Got: {first[:90]}"
            )
            return
    pytest.fail("no _validate_mixed_prefill_decode")


# ── behavioural: the checks above are source-level and let a real bug through ──
#
# `_validate_mixed_prefill_decode` shipped reading `self.hf_text_config`, which
# is a ModelRunner attribute and not a Config one. Every source-level test
# passed; every launch with the flag on died with AttributeError inside the
# validator. These call it.


class _FakeHF:
    """Minimal HF text config. MLA is 'has a latent rank' (see selector)."""

    def __init__(self, *, kv_lora_rank=512, index_topk=None):
        self.kv_lora_rank = kv_lora_rank
        self.num_attention_heads = 16
        self.num_hidden_layers = 2
        if index_topk is not None:
            self.index_topk = index_topk


class _FakeConfig:
    """Just enough surface for the validator, bound to the real function."""

    _validate_mixed_prefill_decode = None  # bound below

    def __init__(self, *, enabled=True, hf=None, spec=None):
        self.enable_mixed_prefill_decode = enabled
        self.hf_config = hf if hf is not None else _FakeHF()
        self.speculative_config = spec


def _bind():
    from atom.config import Config

    _FakeConfig._validate_mixed_prefill_decode = Config._validate_mixed_prefill_decode


def test_validator_runs_against_a_real_config_surface():
    """The regression: it read an attribute Config does not have."""
    _bind()
    # Dense MLA, no spec -> supported, must not raise (and must not
    # AttributeError on the way to deciding that).
    _FakeConfig()._validate_mixed_prefill_decode()


def test_validator_refuses_speculative_decode():
    _bind()
    cfg = _FakeConfig(spec=object())
    with pytest.raises(ValueError, match="speculative decoding"):
        cfg._validate_mixed_prefill_decode()


def test_validator_refuses_sparse_mla():
    _bind()
    cfg = _FakeConfig(hf=_FakeHF(index_topk=2048))
    with pytest.raises(ValueError, match="sparse MLA"):
        cfg._validate_mixed_prefill_decode()


# Every member of the taxonomy, named. The first version of this table tested
# only MHA, passed, and shipped a validator that refused DeepSeek-V4 -- the one
# model the feature exists for. A family list that does not enumerate itself
# lets exactly that through.
_FAMILY_CASES = [
    # (model_type, kv_lora_rank, supported?)
    ("deepseek_v4", 512, True),  # the feature's reason to exist
    ("deepseek_v3", 512, True),  # dense MLA, the other prepare_mixed
    ("kimi_linear", 512, False),  # hybrid; no prepare_mixed
    ("qwen3_next", None, False),  # GDN
    ("llama", None, False),  # MHA
]


@pytest.mark.parametrize("model_type,rank,supported", _FAMILY_CASES)
def test_every_attention_family_is_classified_deliberately(model_type, rank, supported):
    _bind()
    hf = _FakeHF(kv_lora_rank=rank)
    hf.model_type = model_type
    cfg = _FakeConfig(hf=hf)
    if supported:
        cfg._validate_mixed_prefill_decode()  # must not raise
    else:
        with pytest.raises(ValueError, match="attention family"):
            cfg._validate_mixed_prefill_decode()


def test_v4_is_not_refused_as_non_mla():
    """The regression, pinned on its own.

    `Family.is_mla` is `self in (MLA, KIMI_MLA)` and V4 is deliberately absent
    from it -- so `not is_mla` reads as "refuse V4". The validator must key off
    a whitelist of families that implement `prepare_mixed`, not off `is_mla`.
    """
    _bind()
    from atom.utils.selector import Family

    assert not Family.V4.is_mla, (
        "premise changed: if V4 now reports is_mla, re-read the validator -- "
        "this test is guarding against an inference that would become valid"
    )
    hf = _FakeHF(kv_lora_rank=512)
    hf.model_type = "deepseek_v4"
    _FakeConfig(hf=hf)._validate_mixed_prefill_decode()


def test_validator_is_inert_when_the_flag_is_off():
    """Runs in every Config.__post_init__; must cost and refuse nothing."""
    _bind()
    cfg = _FakeConfig(enabled=False, hf=_FakeHF(index_topk=2048), spec=object())
    cfg._validate_mixed_prefill_decode()  # would refuse twice over if it ran


def test_all_reasons_are_reported_not_just_the_first():
    """A user fixing one conflict should not discover the next on relaunch."""
    _bind()
    cfg = _FakeConfig(hf=_FakeHF(index_topk=2048), spec=object())
    with pytest.raises(ValueError) as e:
        cfg._validate_mixed_prefill_decode()
    assert "speculative decoding" in str(e.value)
    assert "sparse MLA" in str(e.value)
