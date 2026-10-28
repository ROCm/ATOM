# SPDX-License-Identifier: MIT
"""Compile-cache source hashing handles CPython's frozen module filenames."""

import pytest

from atom.utils.backends import _hash_traced_files


@pytest.mark.parametrize(
    "synthetic",
    [
        "<string>",
        "<frozen os>",
        "<frozen genericpath>",
        "<frozen posixpath>",
        "<frozen importlib._bootstrap>",
    ],
)
def test_synthetic_sources_preserve_real_source_invalidation(tmp_path, synthetic):
    source = tmp_path / "forward.py"
    source.write_text("def forward(x): return x + 1\n")
    files = [synthetic, str(source)]
    before = _hash_traced_files(files)
    assert _hash_traced_files(files) == before
    assert _hash_traced_files([str(source)]) != before
    source.write_text("def forward(x): return x + 2\n")
    assert _hash_traced_files(files) != before


def test_missing_real_source_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        _hash_traced_files([str(tmp_path / "missing.py")])
