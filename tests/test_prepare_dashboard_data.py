"""Tests for benchmark dashboard data validation and normalization."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".github" / "scripts"))

from prepare_dashboard_data import entry_count, normalize, parse_data


def dashboard_data(suffix=b""):
    payload = b'{"entries":{"Benchmark":[{"commit":{"id":"abc"},"benches":[]}]}}'
    return b"window.BENCHMARK_DATA = " + payload + suffix


def test_normalize_removes_all_terminal_semicolons_and_preserves_crlf():
    original = dashboard_data(b";;\r\n")

    normalized, changed = normalize(original)

    assert changed
    assert normalized == dashboard_data(b"\r\n")
    assert entry_count(parse_data(normalized)[0]) == 1


def test_normalize_leaves_valid_data_unchanged():
    original = dashboard_data()

    normalized, changed = normalize(original)

    assert not changed
    assert normalized == original


def test_parse_data_rejects_javascript_nonstandard_constants():
    text = b'window.BENCHMARK_DATA = {"entries":{"Benchmark":[' b'{"value":NaN}]}}'

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        parse_data(text)


def test_parse_data_allows_empty_entries_for_first_write():
    text = b'window.BENCHMARK_DATA = {"entries":{}}'

    data, changed = parse_data(text)

    assert not changed
    assert entry_count(data) == 0


def test_parse_data_reports_wrong_types():
    with pytest.raises(TypeError, match="entries"):
        parse_data(b'window.BENCHMARK_DATA = {"entries": []}')
