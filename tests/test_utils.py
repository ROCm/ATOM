# SPDX-License-Identifier: MIT
# Tests for atom/utils/__init__.py — pure functions only (no GPU / ZMQ required)

from atom.utils import (
    is_valid_ipv6_address,
    split_host_port,
    join_host_port,
    get_tcp_uri,
    get_device_indices,
    _is_torch_equal_or_newer,
    split_zmq_path,
    make_zmq_path,
)

import pytest

# ── IPv6 validation ────────────────────────────────────────────────────────


class TestIsValidIpv6Address:
    def test_valid_ipv6(self):
        assert is_valid_ipv6_address("::1") is True

    def test_valid_ipv6_full(self):
        assert is_valid_ipv6_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334") is True

    def test_invalid_ipv6(self):
        assert is_valid_ipv6_address("192.168.1.1") is False

    def test_invalid_string(self):
        assert is_valid_ipv6_address("not-an-ip") is False

    def test_empty_string(self):
        assert is_valid_ipv6_address("") is False


# ── split_host_port ────────────────────────────────────────────────────────


class TestSplitHostPort:
    def test_ipv4(self):
        host, port = split_host_port("192.168.1.1:8080")
        assert host == "192.168.1.1"
        assert port == 8080

    def test_ipv6_bracketed(self):
        host, port = split_host_port("[::1]:8080")
        assert host == "::1"
        assert port == 8080

    def test_localhost(self):
        host, port = split_host_port("localhost:3000")
        assert host == "localhost"
        assert port == 3000


# ── join_host_port ─────────────────────────────────────────────────────────


class TestJoinHostPort:
    def test_ipv4(self):
        assert join_host_port("192.168.1.1", 8080) == "192.168.1.1:8080"

    def test_ipv6(self):
        assert join_host_port("::1", 8080) == "[::1]:8080"

    def test_hostname(self):
        assert join_host_port("localhost", 3000) == "localhost:3000"


# ── get_tcp_uri ────────────────────────────────────────────────────────────


class TestGetTcpUri:
    def test_ipv4(self):
        assert get_tcp_uri("192.168.1.1", 8080) == "tcp://192.168.1.1:8080"

    def test_ipv6(self):
        assert get_tcp_uri("::1", 8080) == "tcp://[::1]:8080"


# ── split_zmq_path ────────────────────────────────────────────────────────


class TestSplitZmqPath:
    def test_tcp_path(self):
        scheme, host, port = split_zmq_path("tcp://192.168.1.1:5555")
        assert scheme == "tcp"
        assert host == "192.168.1.1"
        assert port == "5555"

    def test_ipc_path(self):
        scheme, host, port = split_zmq_path("ipc:///tmp/test.sock")
        assert scheme == "ipc"

    def test_inproc_path(self):
        scheme, host, port = split_zmq_path("inproc://my-endpoint")
        assert scheme == "inproc"
        assert port == ""

    def test_no_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid zmq path"):
            split_zmq_path("no-scheme-here")

    def test_tcp_missing_host_raises(self):
        with pytest.raises(ValueError, match="Invalid zmq path"):
            split_zmq_path("tcp://")

    def test_non_tcp_with_port_raises(self):
        with pytest.raises(ValueError, match="Invalid zmq path"):
            split_zmq_path("ipc://host:1234")


# ── make_zmq_path ─────────────────────────────────────────────────────────


class TestMakeZmqPath:
    def test_without_port(self):
        assert make_zmq_path("ipc", "/tmp/test.sock") == "ipc:///tmp/test.sock"

    def test_with_port_ipv4(self):
        assert make_zmq_path("tcp", "192.168.1.1", 5555) == "tcp://192.168.1.1:5555"

    def test_with_port_ipv6(self):
        assert make_zmq_path("tcp", "::1", 5555) == "tcp://[::1]:5555"

    def test_inproc(self):
        assert make_zmq_path("inproc", "my-endpoint") == "inproc://my-endpoint"


# ── get_device_indices ─────────────────────────────────────────────────────


class TestGetDeviceIndices:
    def test_rank0_world2(self):
        result = get_device_indices("HIP_VISIBLE_DEVICES", 0, 2)
        assert result == "0,1"

    def test_rank1_world2(self):
        result = get_device_indices("HIP_VISIBLE_DEVICES", 1, 2)
        assert result == "2,3"

    def test_rank0_world1(self):
        result = get_device_indices("HIP_VISIBLE_DEVICES", 0, 1)
        assert result == "0"

    def test_rank2_world4(self):
        result = get_device_indices("HIP_VISIBLE_DEVICES", 2, 4)
        assert result == "8,9,10,11"


# ── _is_torch_equal_or_newer ──────────────────────────────────────────────


class TestIsTorchEqualOrNewer:
    def test_newer(self):
        assert _is_torch_equal_or_newer("2.5.0", "2.4.0") is True

    def test_equal(self):
        assert _is_torch_equal_or_newer("2.4.0", "2.4.0") is True

    def test_older(self):
        assert _is_torch_equal_or_newer("2.3.0", "2.4.0") is False

    def test_dev_version(self):
        assert _is_torch_equal_or_newer("2.5.0.dev20240101", "2.4.0") is True

    def test_rc_version(self):
        assert _is_torch_equal_or_newer("2.4.0rc1", "2.4.0") is False
