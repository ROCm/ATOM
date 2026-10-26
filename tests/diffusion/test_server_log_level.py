# SPDX-License-Identifier: MIT
"""--log-level has to reach the atom loggers directly: their records stop at
ATOM's own handler and never reach the root handler basicConfig() installs."""

import logging
import os

import pytest

from atom.diffusion.entrypoints.diffusion_server import (
    apply_atom_log_level,
    build_parser,
)
from atom.utils import getLogger, set_atom_log_level


@pytest.fixture
def atom_logger(monkeypatch):
    # setenv first so monkeypatch remembers the original value and also undoes
    # the export apply_atom_log_level() writes to os.environ itself.
    monkeypatch.setenv("ATOM_LOG_LEVEL", "INFO")
    monkeypatch.delenv("ATOM_LOG_LEVEL")
    atom = getLogger()
    restore = logging.getLevelName(atom.level)
    yield atom
    set_atom_log_level(restore)


def test_log_level_flag_is_unset_by_default():
    assert build_parser().parse_args(["--model", "/ckpt"]).log_level is None


def test_explicit_log_level_reaches_this_process_and_the_workers(atom_logger):
    assert apply_atom_log_level("debug") == "DEBUG"
    assert atom_logger.level == logging.DEBUG
    # What the spawned workers inherit and read when they import atom.utils.
    assert os.environ["ATOM_LOG_LEVEL"] == "debug"


def test_without_the_flag_atom_log_level_stays_in_charge(atom_logger, monkeypatch):
    monkeypatch.setenv("ATOM_LOG_LEVEL", "warning")

    assert apply_atom_log_level(None) == "WARNING"
    assert atom_logger.level == logging.WARNING
    assert os.environ["ATOM_LOG_LEVEL"] == "warning"


def test_without_the_flag_or_the_variable_the_level_is_info(atom_logger):
    assert apply_atom_log_level(None) == "INFO"
    assert atom_logger.level == logging.INFO
    assert "ATOM_LOG_LEVEL" not in os.environ
