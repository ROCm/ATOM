# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for proxy utility functions (CPU-only, no network)."""

from __future__ import annotations

import pytest

import atom.kv_transfer.disaggregation.proxy as proxy_mod
from atom.kv_transfer.disaggregation.proxy import (
    _append_whole_dict_unique,
    _extract_ip_port,
    example_round_robin_dp_loader,
)

# ---------------------------------------------------------------------------
# _extract_ip_port
# ---------------------------------------------------------------------------


class TestExtractIpPort:
    def test_valid_url(self):
        ip, port = _extract_ip_port("http://10.0.0.5:8000/v1/completions")
        assert ip == "10.0.0.5"
        assert port == "8000"

    def test_https_url(self):
        ip, port = _extract_ip_port("https://192.168.1.1:443/api")
        assert ip == "192.168.1.1"
        assert port == "443"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Cannot extract"):
            _extract_ip_port("not-a-url")

    def test_ipv6_not_matched(self):
        with pytest.raises(ValueError):
            _extract_ip_port("http://[::1]:8000/v1")

    def test_no_port_raises(self):
        with pytest.raises(ValueError):
            _extract_ip_port("http://10.0.0.1/v1")


# ---------------------------------------------------------------------------
# example_round_robin_dp_loader
# ---------------------------------------------------------------------------


class TestRoundRobin:
    def test_even_distribution(self):
        results = [example_round_robin_dp_loader(i, dp_size=3) for i in range(6)]
        assert results == [0, 1, 2, 0, 1, 2]

    def test_single_dp(self):
        assert example_round_robin_dp_loader(5, dp_size=1) == 0

    def test_large_request_number(self):
        assert example_round_robin_dp_loader(1000, dp_size=7) == 1000 % 7


# ---------------------------------------------------------------------------
# _append_whole_dict_unique
# ---------------------------------------------------------------------------


class TestAppendWholeDictUnique:
    @pytest.fixture(autouse=True)
    def _reset_transfer_type(self):
        """Reset global TRANSFER_TYPE before each test."""
        proxy_mod.TRANSFER_TYPE = None
        yield
        proxy_mod.TRANSFER_TYPE = None

    def test_append_new(self):
        target = []
        result = _append_whole_dict_unique(
            target, {"host": "10.0.0.1", "port": 8000, "transfer_mode": "read"}
        )
        assert result is True
        assert len(target) == 1

    def test_dedup_ignores_index(self):
        target = []
        d1 = {"host": "10.0.0.1", "port": 8000, "index": 1, "transfer_mode": "read"}
        d2 = {"host": "10.0.0.1", "port": 8000, "index": 2, "transfer_mode": "read"}
        _append_whole_dict_unique(target, d1)
        result = _append_whole_dict_unique(target, d2)
        assert result is False
        assert len(target) == 1

    def test_different_dicts_both_appended(self):
        target = []
        _append_whole_dict_unique(target, {"host": "10.0.0.1", "transfer_mode": "read"})
        _append_whole_dict_unique(target, {"host": "10.0.0.2", "transfer_mode": "read"})
        assert len(target) == 2

    def test_transfer_mode_mismatch_raises(self):
        target = []
        _append_whole_dict_unique(target, {"host": "10.0.0.1", "transfer_mode": "read"})
        with pytest.raises(ValueError, match="mismatched"):
            _append_whole_dict_unique(
                target, {"host": "10.0.0.2", "transfer_mode": "write"}
            )

    def test_sets_global_transfer_type(self):
        target = []
        _append_whole_dict_unique(target, {"host": "10.0.0.1", "transfer_mode": "read"})
        assert proxy_mod.TRANSFER_TYPE == "read"
