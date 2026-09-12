# SPDX-License-Identifier: MIT
"""Pinned model math, isolated per differential test."""

import os

import pytest

from .reference import load_reference


@pytest.fixture
def reference():
    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for pinned model methods")
    with load_reference(directory) as module:
        yield module
