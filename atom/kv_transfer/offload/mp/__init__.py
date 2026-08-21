# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache multiprocess offload support for ATOM native models.

``connector`` is the stable factory-facing shell. Model subpackages such as
``glm52`` declare :class:`MPModelConnectorPlugin` subclasses that the registry
discovers automatically.
"""
