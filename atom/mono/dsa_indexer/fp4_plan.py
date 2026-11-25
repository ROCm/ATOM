# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""E2M1 magnitude rounding on Python values and traced values alike."""

from atom.mono.plan.arith import sel


def e2m1_magnitude(magnitude, cast=float):
    """Nearest-even magnitude code, saturated to the largest E2M1 value."""
    code = 0
    for edge, value, inclusive in (
        (0.25, 1, False),
        (0.75, 2, True),
        (1.25, 3, False),
        (1.75, 4, True),
        (2.5, 5, False),
        (3.5, 6, True),
        (5.0, 7, False),
    ):
        past = magnitude >= cast(edge) if inclusive else magnitude > cast(edge)
        code = sel(past, value, code)
    return code
