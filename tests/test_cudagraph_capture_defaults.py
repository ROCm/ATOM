# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

from atom.config import Config
from atom.model_engine.arg_utils import EngineArgs, parse_size_list
from atom.utils.arg_parser import FlexibleArgumentParser


def test_default_capture_sizes_preserve_old_buckets_and_add_step32():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(parser.parse_args([]))
    assert parse_size_list(args.cudagraph_capture_sizes) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        16,
        32,
        48,
        64,
        128,
        160,
        192,
        224,
        256,
        512,
    ]


def test_explicit_capture_sizes_are_not_expanded():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(
        parser.parse_args(["--cudagraph-capture-sizes", "[64,128,256]"])
    )
    declared = parse_size_list(args.cudagraph_capture_sizes)
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=declared,
        )
    )
    sizes = Config.capture_sizes.fget(cfg)
    assert sizes == [64, 128, 256]
    sizes.reverse()
    assert declared == [64, 128, 256]
