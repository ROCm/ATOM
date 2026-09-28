# SPDX-License-Identifier: MIT
"""`atom.model_ops.engram`'s host half must import with no Triton installed.

The CPU pre-checks runner has neither Triton nor aiter, and an import that
raises during collection does not skip one file -- pytest reports `Interrupted`
and the job runs zero tests. That has happened twice on this package, so the
rule that keeps the host modules free of device imports is executed here rather
than left to review.

It imports the modules rather than reading their source: a grep would pass a
module whose Triton import arrives through a third one. And it imports them in
a CHILD interpreter rather than this one -- re-importing a module in-process
leaves a second copy of its classes alive, which is the thing
`tests/conftest.py`'s duplicate guard exists to stop.
"""

import subprocess
import sys
import textwrap

HOST_MODULES = (
    "atom.model_ops.engram.mapping",
    "atom.model_ops.engram.tables",
    "atom.model_ops.engram.host",
    "atom.model_ops.deepseek_v41.draft_block",
)

# Answering "not here" is what a missing package looks like: `find_spec` gives
# None and `import` raises on its own, from the import machinery. A finder that
# raises instead would turn a module's clean find_spec-and-skip into an error.
_PROBE = textwrap.dedent("""
    import importlib, sys
    from importlib.machinery import PathFinder

    BLOCKED = ("triton", "aiter")
    found = PathFinder.find_spec

    def find_spec(name, path=None, target=None):
        return None if name.split(".")[0] in BLOCKED else found(name, path, target)

    PathFinder.find_spec = staticmethod(find_spec)

    # Without this the run passes on any box that simply has Triton installed.
    for name in BLOCKED:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        raise SystemExit(f"probe is not armed: {name} still imported")

    for name in sys.argv[1:]:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            raise SystemExit(f"{name} reaches a device package: {error}") from None
    """)


def test_host_modules_import_without_triton():
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, *HOST_MODULES],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr.strip() or done.stdout.strip()
