# SPDX-License-Identifier: MIT
# Tests for the offline profiling example's trace-directory resolution.

import pytest

from atom.examples.profile_offline import (
    DEFAULT_PROFILER_DIR,
    resolve_profiler_dir,
)

_DEFAULT_NOTICE = (
    "Warning: --torch-profiler-dir not specified and "
    f"ATOM_TORCH_PROFILER_DIR unset, using default: {DEFAULT_PROFILER_DIR}"
)


@pytest.fixture(autouse=True)
def _clear_profiler_env(monkeypatch):
    monkeypatch.delenv("ATOM_TORCH_PROFILER_DIR", raising=False)


@pytest.mark.parametrize(
    "cli_value, env_value, expected_dir, expected_notice",
    [
        # The example always profiles: with neither source set it keeps its
        # own default instead of leaving the directory at None.
        (None, None, DEFAULT_PROFILER_DIR, _DEFAULT_NOTICE),
        # The environment now reaches the example, as it does the server.
        (
            None,
            "/tmp/env-traces",
            "/tmp/env-traces",
            (
                "Warning: --torch-profiler-dir not specified, using "
                "ATOM_TORCH_PROFILER_DIR: /tmp/env-traces"
            ),
        ),
        # An explicit flag still outranks the environment.
        ("/tmp/cli-traces", None, "/tmp/cli-traces", None),
        ("/tmp/cli-traces", "/tmp/env-traces", "/tmp/cli-traces", None),
        # Empty reads as unset on both sides: the variable (the rule shared
        # with the engine) and, in this example, the flag.
        (None, "", DEFAULT_PROFILER_DIR, _DEFAULT_NOTICE),
        (
            "",
            "/tmp/env-traces",
            "/tmp/env-traces",
            (
                "Warning: --torch-profiler-dir not specified, using "
                "ATOM_TORCH_PROFILER_DIR: /tmp/env-traces"
            ),
        ),
        ("", None, DEFAULT_PROFILER_DIR, _DEFAULT_NOTICE),
    ],
)
def test_resolve_profiler_dir(
    monkeypatch, cli_value, env_value, expected_dir, expected_notice
):
    if env_value is not None:
        monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", env_value)

    directory, notice = resolve_profiler_dir(cli_value)

    assert directory == expected_dir
    assert notice == expected_notice
