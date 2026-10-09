# SPDX-License-Identifier: MIT

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STATE_PREFIX = "ATOM_LOGGER_STATE="
_LOGGER_PROBE = textwrap.dedent(f"""
    import json
    import logging

    class RecordingHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    root_handler = RecordingHandler()
    logging.getLogger().addHandler(root_handler)

    from atom.utils import getLogger

    atom_logger = getLogger()
    root_handler.messages.clear()
    atom_logger.warning("propagation probe")
    print(
        {_STATE_PREFIX!r}
        + json.dumps(
            {{
                "level": atom_logger.level,
                "handler_levels": [
                    handler.level for handler in atom_logger.handlers
                ],
                "propagate": atom_logger.propagate,
                "root_messages": root_handler.messages,
            }}
        )
    )
    """)


def _fresh_logger_state(level_name: str | None) -> dict:
    env = os.environ.copy()
    python_path = [str(_REPO_ROOT)]
    if inherited_path := env.get("PYTHONPATH"):
        python_path.append(inherited_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["ATOM_LOG_MORE"] = "0"
    if level_name is None:
        env.pop("ATOM_LOG_LEVEL", None)
    else:
        env["ATOM_LOG_LEVEL"] = level_name

    result = subprocess.run(
        [sys.executable, "-c", _LOGGER_PROBE],
        cwd=_REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_STATE_PREFIX):
            return json.loads(line.removeprefix(_STATE_PREFIX))
    raise AssertionError(
        "logger probe did not report state\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_logger_defaults_to_info_without_propagation():
    state = _fresh_logger_state(None)

    assert state == {
        "level": logging.INFO,
        "handler_levels": [logging.INFO],
        "propagate": False,
        "root_messages": [],
    }


@pytest.mark.parametrize(
    ("level_name", "expected_level"),
    [
        ("DEBUG", logging.DEBUG),
        ("WARN", logging.WARNING),
        ("  warning  ", logging.WARNING),
    ],
)
def test_atom_log_level_sets_logger_and_handler(level_name, expected_level):
    state = _fresh_logger_state(level_name)

    assert state["level"] == expected_level
    assert state["handler_levels"] == [expected_level]


def test_invalid_atom_log_level_falls_back_to_info():
    state = _fresh_logger_state("not-a-logging-level")

    assert state["level"] == logging.INFO
    assert state["handler_levels"] == [logging.INFO]
