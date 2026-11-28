# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 extra state written by ROCm TransformerEngine reads back through Megatron."""

from __future__ import annotations

import io
import pickle

import pytest

recipe = pytest.importorskip("transformer_engine.common.recipe")
safe_globals = pytest.importorskip("megatron.core.safe_globals")

from atomrl.engine.training import megatron_native_engine as eng


@pytest.mark.parametrize("make", ["Float8CurrentScaling", "DelayedScaling"])
def test_rocm_te_recipe_extra_state_loads(make, monkeypatch):
    monkeypatch.setattr(
        safe_globals.SafeUnpickler,
        "_SAFE_CLASSES",
        safe_globals.SafeUnpickler._SAFE_CLASSES,
    )
    blob = pickle.dumps(getattr(recipe, make)())
    if hasattr(recipe, "_FormatHelperFP8"):
        # ROCm TE: refused until the engine adds its helper classes.
        with pytest.raises(pickle.UnpicklingError, match="_FormatHelperFP8"):
            safe_globals.SafeUnpickler(io.BytesIO(blob)).load()
    eng._allow_rocm_te_recipes()
    assert type(safe_globals.SafeUnpickler(io.BytesIO(blob)).load()).__name__ == make
