# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Schedule knob dicts (a model's ``schedules`` entries): plain Python, no torch."""

import copy


def merge_knobs(base: dict, over: dict) -> dict:
    """``over`` deep-merged onto a copy of ``base``: a dict value merges into the dict it replaces, any other value
    (None included) replaces it."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = (
            merge_knobs(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else copy.deepcopy(v)
        )
    return out
