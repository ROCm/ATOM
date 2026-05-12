# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/arg_utils.py CLI parsing

import argparse

import pytest

from atom.model_engine.arg_utils import EngineArgs


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], True),
        (["--enable-prefix-caching"], True),
        (["--enable_prefix_caching"], True),
        (["--no-enable-prefix-caching"], False),
        (["--no-enable_prefix_caching"], False),
    ],
)
def test_enable_prefix_caching_cli(argv, expected):
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    ns = parser.parse_args(argv)
    assert ns.enable_prefix_caching is expected
