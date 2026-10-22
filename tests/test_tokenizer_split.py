# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random
import sys

import pytest

from atom.model_engine.tokenizer_split import (
    _WS_LIMIT,
    install_fast_whitespace_split,
    longest_class_run,
)


def _kimi_split(s, max_consecutive_slice_len):
    # Verbatim from Kimi-K3 tokenization_kimi.TikTokenTokenizer.
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
        is_now_space = s[i].isspace()

        if current_slice_is_space ^ is_now_space:
            current_slice_len = 1
            current_slice_is_space = is_now_space
        else:
            current_slice_len += 1
            if current_slice_len > max_consecutive_slice_len:
                yield s[slice_start:i]
                slice_start = i
                current_slice_len = 1
    yield s[slice_start:]


def _make_tokenizer():
    class KimiLike:
        _split_whitespaces_or_nonwhitespaces = staticmethod(_kimi_split)

    return KimiLike()


def _brute_longest(s):
    best = run = 0
    for i, ch in enumerate(s):
        run = run + 1 if i and ch.isspace() == s[i - 1].isspace() else 1
        best = max(best, run)
    return best


_ALPHABET = "ab 你\t\n\u3000\u00a0\u2028\x1c\u200b\ud800" + chr(0x1F600)


def _random_text(rng, n):
    out = []
    while len(out) < n:
        out.extend(rng.choice(_ALPHABET) * rng.choice((1, 1, 2, 5, 40)))
    return "".join(out[:n])


def test_every_space_code_point_is_below_the_table_limit():
    assert all(not chr(c).isspace() for c in range(_WS_LIMIT, sys.maxunicode + 1))


@pytest.mark.parametrize("seed", range(20))
def test_longest_class_run_matches_brute_force(seed):
    rng = random.Random(seed)
    s = _random_text(rng, rng.randint(0, 400))
    assert longest_class_run(s) == _brute_longest(s)


def test_longest_class_run_edges():
    assert longest_class_run("") == 0
    assert longest_class_run("x") == 1
    assert longest_class_run("   ") == 3
    assert longest_class_run("ab  c") == 2
    assert longest_class_run("\ud800\udfff") == 2


@pytest.mark.parametrize("limit", [1, 2, 3, 7, 50])
@pytest.mark.parametrize("seed", range(10))
def test_fast_split_yields_the_original_pieces(limit, seed):
    tok = _make_tokenizer()
    assert install_fast_whitespace_split(tok)
    rng = random.Random(seed)
    for n in (0, 1, 10, 300):
        s = _random_text(rng, n)
        fast = list(tok._split_whitespaces_or_nonwhitespaces(s, limit))
        assert fast == list(_kimi_split(s, limit))


def test_fast_split_takes_the_loop_only_for_an_over_long_run():
    tok = _make_tokenizer()
    install_fast_whitespace_split(tok)
    split = type(tok)._split_whitespaces_or_nonwhitespaces
    assert list(split("ab cd", 2)) == ["ab cd"]
    assert list(split("abc d", 2)) == ["ab", "c d"]
    assert list(split("", 25_000)) == [""]


def test_install_is_idempotent_and_skips_other_tokenizers():
    tok = _make_tokenizer()
    assert install_fast_whitespace_split(tok)
    fast = type(tok).__dict__["_split_whitespaces_or_nonwhitespaces"]
    assert install_fast_whitespace_split(tok)
    assert type(tok).__dict__["_split_whitespaces_or_nonwhitespaces"] is fast
    assert not install_fast_whitespace_split(object())
