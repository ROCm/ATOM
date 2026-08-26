# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node DP's mutual-exclusion matrix (M-CFG).

One case per row: the combination is rejected, and the message says what to do
about it. A validator that only reports "invalid" moves the work to whoever
reads the traceback, which for a hang-shaped failure is the expensive path.

`_validate_multinode` reads attributes and nothing else, so it is exercised
against a namespace rather than a real Config -- building one loads an HF
config off disk, which has nothing to do with what is under test.
"""

from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

try:
    from atom.config import Config, ParallelConfig
except ImportError as _e:  # transformers/aiter absent on a bare runner
    skip_if_dependency_missing(_e, "requires atom.config import env")

# Whatever get_open_port() returned when ParallelConfig was defined. Leaving the
# field at this value is what "the user never passed --data-parallel-base-port"
# looks like from inside __post_init__.
BASE_PORT_DEFAULT = next(
    f.default for f in ParallelConfig.__dataclass_fields__.values()
    if f.name == "data_parallel_base_port"
)


def _pc(**over):
    """Node 1 of a two-node, eight-engines-per-node run."""
    kw = {
        "data_parallel_size": 16,
        "data_parallel_size_local": 8,
        "data_parallel_rank": 8,
        "data_parallel_base_port": 40000,
        "data_parallel_master_ip": "10.0.0.1",
        "decode_context_parallel_size": 1,
    }
    kw.update(over)
    pc = SimpleNamespace(**kw)
    pc.is_multinode_dp = (
        kw["data_parallel_size_local"] < kw["data_parallel_size"]
        or kw["data_parallel_rank"] > 0
    )
    return pc


def _cfg(*, pc=None, **over):
    """A config that passes every multi-node check unless a field is overridden."""
    kw = {
        "parallel_config": pc if pc is not None else _pc(),
        "enable_dp_attention": True,
        "enable_expert_parallel": True,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "moe_backend": "standard",
        "enable_rapidserve": False,
        "plugin_config": None,
        "enable_tbo": False,
    }
    kw.update(over)
    return SimpleNamespace(**kw)


def _reject(cfg) -> str:
    with pytest.raises(ValueError) as exc:
        Config._validate_multinode(cfg)
    return str(exc.value)


class TestAllowed:
    def test_the_target_configuration_passes(self):
        Config._validate_multinode(_cfg())

    def test_single_node_skips_every_check(self):
        # Same violations, one node: none of them are multi-node's business.
        pc = _pc(data_parallel_size=8, data_parallel_size_local=8, data_parallel_rank=0)
        Config._validate_multinode(
            _cfg(
                pc=pc,
                enable_dp_attention=False,
                enable_expert_parallel=False,
                moe_backend="mega",
                enable_rapidserve=True,
            )
        )


class TestRejected:
    """One row of the matrix each. Every message must carry a fix."""

    @pytest.mark.parametrize(
        "override, expected",
        [
            ({"enable_dp_attention": False}, "--enable-dp-attention"),
            ({"enable_expert_parallel": False}, "--enable-expert-parallel"),
            ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
            ({"prefill_context_parallel_size": 2}, "prefill_context_parallel_size"),
            ({"moe_backend": "mega"}, "moe-backend"),
            ({"enable_rapidserve": True}, "rapidserve"),
            ({"plugin_config": object()}, "plugin"),
        ],
    )
    def test_row_is_rejected_with_a_fix(self, override, expected):
        message = _reject(_cfg(**override))
        assert expected in message
        assert "Why:" in message and "Fix:" in message

    def test_decode_context_parallel_is_rejected(self):
        message = _reject(_cfg(pc=_pc(decode_context_parallel_size=2)))
        assert "decode_context_parallel_size" in message
        assert "Fix:" in message

    def test_message_names_the_topology(self):
        message = _reject(_cfg(enable_dp_attention=False))
        assert "2 nodes" in message
        assert "data_parallel_size=16" in message


class TestRendezvous:
    def test_unset_base_port_is_rejected(self):
        # The default is per-process, so two nodes never agree on it and the
        # symptom is a hang rather than a connection error.
        message = _reject(_cfg(pc=_pc(data_parallel_base_port=BASE_PORT_DEFAULT)))
        assert "data-parallel-base-port" in message
        assert "silent hang" in message

    @pytest.mark.parametrize("ip", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_warns_but_is_allowed(self, ip, caplog):
        # Gate 1 runs both logical nodes on one box, where loopback is correct.
        # Rejecting it would block our own single-machine verification path.
        with caplog.at_level("WARNING"):
            Config._validate_multinode(_cfg(pc=_pc(data_parallel_master_ip=ip)))
        assert "loopback" in caplog.text or "this machine" in caplog.text

    def test_routable_ip_is_silent(self, caplog):
        with caplog.at_level("WARNING"):
            Config._validate_multinode(_cfg(pc=_pc(data_parallel_master_ip="10.0.0.1")))
        assert caplog.text == ""


class TestTboWarning:
    def test_tbo_warns_and_continues(self, caplog):
        with caplog.at_level("WARNING"):
            Config._validate_multinode(_cfg(enable_tbo=True))
        assert "TBO" in caplog.text
