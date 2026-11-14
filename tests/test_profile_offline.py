# SPDX-License-Identifier: MIT
# Tests for the offline profiling example's trace-directory resolution.

import pytest

from atom.examples import profile_offline


class _EngineArgsReached(Exception):
    """Stops main() once the resolved directory has been handed over."""


@pytest.fixture(autouse=True)
def _clear_profiler_env(monkeypatch):
    monkeypatch.delenv("ATOM_TORCH_PROFILER_DIR", raising=False)


@pytest.mark.parametrize(
    "cli_value, env_value, expected",
    [
        # The example always profiles: with neither source set it keeps its
        # own default instead of leaving the directory at None.
        (None, None, profile_offline.DEFAULT_PROFILER_DIR),
        # The environment reaches the example, as it does the server.
        (None, "/tmp/env-traces", "/tmp/env-traces"),
        # An explicit flag still outranks the environment.
        ("/tmp/cli-traces", None, "/tmp/cli-traces"),
        ("/tmp/cli-traces", "/tmp/env-traces", "/tmp/cli-traces"),
        # Empty reads as unset on both sides: the variable (the rule shared
        # with the engine) and, in this example, the flag.
        (None, "", profile_offline.DEFAULT_PROFILER_DIR),
        ("", "/tmp/env-traces", "/tmp/env-traces"),
        ("", None, profile_offline.DEFAULT_PROFILER_DIR),
    ],
)
def test_main_resolves_profiler_dir(
    monkeypatch, capsys, cli_value, env_value, expected
):
    if env_value is not None:
        monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", env_value)

    seen = {}

    class _RecordingEngineArgs:
        @staticmethod
        def from_cli_args(args):
            seen["dir"] = args.torch_profiler_dir
            raise _EngineArgsReached

    monkeypatch.setattr(profile_offline, "EngineArgs", _RecordingEngineArgs)
    argv = ["profile_offline.py"]
    if cli_value is not None:
        argv += ["--torch-profiler-dir", cli_value]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(_EngineArgsReached):
        profile_offline.main()

    assert seen["dir"] == expected
    assert f"Writing profiler traces to {expected}" in capsys.readouterr().out
