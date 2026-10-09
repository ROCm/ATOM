# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`Config.true_vocab_size`: the padding tail of a padded vocabulary.

Qwen3-8B carries 151936 embedding rows for 151665 real tokens. Those rows are
neither zero nor -inf on that checkpoint -- they are copies of an existing
embedding -- so the sampler reaches them and can return an id the tokenizer
cannot decode, while the trainer masks exactly those positions on its side.
"""

from dataclasses import fields

import pytest
import torch

from atom.config import Config

# Importing RLHFModelRunner pulls in aiter, which resolves the chip
# architecture through rocminfo at import time. A skip, not a try/except: the
# only condition that may skip this module is "no ROCm device", and anything
# else going wrong with the import is a failure worth seeing.
if not torch.cuda.is_available():
    pytest.skip(
        "importing RLHFModelRunner resolves the chip architecture through aiter",
        allow_module_level=True,
    )

from atom.model_engine.model_runner import ModelRunner
from atom.rollout.model_runner_ext import RLHFModelRunner


def _config(**overrides):
    values = {"true_vocab_size": 0}
    values.update(overrides)
    return type("_Cfg", (), values)()


def _checked(true_vocab_size, vocab_size=151936):
    runner = object.__new__(RLHFModelRunner)
    runner._true_vocab_size = true_vocab_size
    runner._check_true_vocab_size(_config(hf_config=_config(vocab_size=vocab_size)))


# ── what the number is allowed to be ──────────────────────────────────────


def test_true_vocab_size_survives_the_engine_kwarg_filter():
    """`LLMEngine.__init__` keeps only kwargs naming a `Config` field.

    That filter is what lets a caller pass `true_vocab_size` to an ATOM build
    that does not know it. It also means a rename here turns the mask off in
    silence rather than raising, so pin the name.
    """
    engine_kwargs = {"true_vocab_size": 151665, "not_a_config_field": 1}
    config_fields = {f.name for f in fields(Config)}
    assert {k: v for k, v in engine_kwargs.items() if k in config_fields} == {
        "true_vocab_size": 151665
    }


def test_a_value_above_the_checkpoint_is_refused():
    """It counts real tokens, so it cannot exceed the rows there are.

    One vocabulary's count against another's checkpoint masks nothing, which
    is the exact failure this path exists to prevent -- so it raises instead.
    """
    with pytest.raises(ValueError, match="exceeds the checkpoint"):
        _checked(200000)


def test_a_negative_value_is_refused():
    """The other way a wrong number masks nothing instead of saying so."""
    with pytest.raises(ValueError, match="expected >= 0"):
        _checked(-1)


def test_a_value_inside_the_checkpoint_is_accepted():
    _checked(151665)


def test_the_default_masks_nothing_and_checks_nothing():
    """0 is every model whose embedding matrix matches its tokenizer."""
    _checked(0, vocab_size=0)


# ── what it does to the logits ────────────────────────────────────────────


def _postprocess(monkeypatch, true_vocab_size, logits):
    seen = {}

    def _capture(self, batch, passed_logits, *args, **kwargs):
        seen["logits"] = passed_logits
        return "sentinel"

    monkeypatch.setattr(ModelRunner, "postprocess", _capture)
    runner = object.__new__(RLHFModelRunner)
    runner._true_vocab_size = true_vocab_size
    result = runner.postprocess(None, logits, None, None, None, True, None)
    assert result == "sentinel"
    return seen["logits"]


def test_the_padding_tail_is_the_only_thing_masked(monkeypatch):
    logits = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    original = logits.clone()

    masked = _postprocess(monkeypatch, 3, logits)

    assert torch.equal(masked[:, :3], original[:, :3])
    assert torch.isneginf(masked[:, 3:]).all()


def test_greedy_would_have_sampled_a_padding_id(monkeypatch):
    """The counters cannot show this: it is about which id comes back."""
    logits = torch.tensor([[1.0, 2.0, 9.0]])
    assert int(logits.argmax(-1)) == 2  # a padding row, unmasked

    masked = _postprocess(monkeypatch, 2, logits.clone())

    assert int(masked.argmax(-1)) == 1


def test_zero_leaves_the_logits_untouched(monkeypatch):
    logits = torch.tensor([[1.0, 2.0, 9.0]])

    masked = _postprocess(monkeypatch, 0, logits.clone())

    assert torch.equal(masked, logits)


def test_a_vocabulary_that_is_not_padded_is_untouched(monkeypatch):
    """`true_vocab_size == logits.shape[-1]`: nothing to mask, and no slice
    that would silently drop the last real token."""
    logits = torch.tensor([[1.0, 2.0, 9.0]])

    masked = _postprocess(monkeypatch, 3, logits.clone())

    assert torch.equal(masked, logits)
