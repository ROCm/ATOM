# SPDX-License-Identifier: MIT
"""Mono validation settings for clients with no engine/GPU dependencies."""

import os

# Match the registered client-only defaults in atom/utils/envs.py. That file
# must also run standalone during isolated builds; importing the registry via
# atom.utils would pull engine dependencies into clients. Tests check parity.


def configured_profile() -> str:
    return os.getenv("ATOM_V4_MONO_PROFILE", "")


def configured_policy() -> str:
    return os.getenv("ATOM_V4_MONO_POLICY", "auto")
