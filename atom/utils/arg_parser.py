# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A drop-in ArgumentParser that accepts both spellings of every long flag.

Historically some ATOM flags were registered kebab-case (``--tensor-parallel-size``)
and some snake_case (``--kv_cache_dtype``), so users had to remember which was
which. Rather than spelling out both forms on every ``add_argument`` call, this
parser adds the missing counterpart automatically.
"""

import argparse

__all__ = ["FlexibleArgumentParser"]


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser accepting both ``--kebab-case`` and ``--snake_case`` flags.

    For each long option registered in one spelling, the dash/underscore
    counterpart is auto-registered as an alias for the same ``dest``. Only the
    flag *name* (after the leading ``--``) is transformed, so option *values* are
    never touched — this is safe for JSON-valued flags such as
    ``--online_quant_config '{"use_index_cache": true}'``. Short flags (``-tp``)
    and positionals are left untouched, and ``dest`` still derives from the first
    (original) option string, so existing callers are unaffected.
    """

    def add_argument(self, *names, **kwargs):
        expanded = list(names)
        for opt in names:
            if not opt.startswith("--"):
                continue
            stem = opt[2:]
            for alt in (
                "--" + stem.replace("_", "-"),
                "--" + stem.replace("-", "_"),
            ):
                # Skip the option itself, in-call duplicates, and any collision
                # with an already-registered flag (a real argument wins over an
                # auto-generated alias — never raise "conflicting option string").
                if alt not in expanded and alt not in self._option_string_actions:
                    expanded.append(alt)
        return super().add_argument(*expanded, **kwargs)
