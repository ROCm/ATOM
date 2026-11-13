# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.8 mono layer programs: whole decoder layers composed from FlyDSL stages (``blocks``) by
``recipe.compose_layer`` with the per-scenario schedules (``schedules``), placed by ``program`` and emitted as one
resident kernel (``emit`` / ``kernel``); ``device`` holds their gfx950 primitives, ``weights`` the weight layouts the
stages read.
"""
