# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4.1's spaced DSML tags, using the shared V4 recovery and streaming engine."""

from typing import ClassVar

from .deepseekv4_tool_parser import DsmlParser, _compile_patterns


class DsmlV41Parser(DsmlParser):
    NAME: ClassVar[str] = "dsml_v41"
    DSML_PREFIX: ClassVar[str] = "｜DSML｜ "
    SECTION: ClassVar[str] = "calls"
    PARAM_RE, INVOKE_RE = _compile_patterns(DSML_PREFIX, SECTION)
    START_MARKERS: ClassVar[tuple[str, ...]] = (
        "<｜DSML｜ calls>",
        "<｜DSML｜ invoke",
        "<invoke name=",
        "<calls>",
    )
    DETECT_MARKERS: ClassVar[tuple[str, ...]] = (
        "<｜DSML｜ calls>",
        "<｜DSML｜ invoke",
        "<｜DSML｜ parameter",
    )
    CALL_OPENERS: ClassVar[tuple[str, ...]] = ("<｜DSML｜ calls>", "<calls>")
    CALL_CLOSERS: ClassVar[tuple[str, ...]] = ("</｜DSML｜ calls>", "</calls>")
    CALL_SELF_CLOSERS: ClassVar[tuple[str, ...]] = (
        "</｜DSML｜ invoke>",
        "</invoke>",
    )
    CALL_CONTINUES: ClassVar[tuple[str, ...]] = (
        "<｜DSML｜ parameter",
        "<parameter",
        "</｜DSML｜ invoke>",
        "</invoke>",
    )
