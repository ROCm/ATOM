# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from atom.plugin.sglang import prepare_model_for_sglang
from atom.sampling_params import SamplingDefaults, SamplingParams

__all__ = [
    "LLMEngine",
    "SamplingDefaults",
    "SamplingParams",
    "prepare_model_for_sglang",
]

# `LLMEngine` is resolved on first attribute access rather than at import time.
# Importing it here pulls the whole engine -- zmq sockets, the model runner,
# AITER kernels -- into *any* `import atom.<anything>`, because Python imports
# a package before its submodule. `import atom.config` paid for a GPU inference
# engine to read a dataclass, which is why the test suite hand-stubbed
# `atom.config` instead of importing it.
#
# The two names above stay eager: both cost only dataclasses, logging and
# typing.


def __getattr__(name: str):
    if name == "LLMEngine":
        from atom.model_engine.llm_engine import LLMEngine

        return LLMEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
