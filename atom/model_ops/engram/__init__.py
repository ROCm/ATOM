# SPDX-License-Identifier: MIT
"""Engram: n-gram memory read out of two ~98 GB host tables.

The modules beside this one -- `mapping`, `tables`, `host` -- are the host
half: the n-gram hash layout, the memory-mapped tables and their page-locking,
and the prefetch/staging runtime that drives them. They import numpy and torch
and nothing else, so a CPU-only machine can import and test all of it.

`device/` is the other half, and everything in it reaches Triton. The split is
the directory rather than a convention: a CPU pre-checks runner has no Triton,
and an import that raises during collection takes the whole pytest session down
rather than one file. `test_import_boundary.py` is the executable form of that
rule.

Nothing is re-exported here on purpose -- a re-export would pull `device/` into
`import atom.model_ops.engram` and undo the split.
"""
