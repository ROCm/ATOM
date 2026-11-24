# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SwiGLU-OAI of a split-layout gate_up + MXFP8 quant of the result, I = 3072 output columns, FlyDSL
(gfx950). The torch contract is ``reference.py``."""


def __getattr__(name):
    if name in __all__:
        from . import host

        return getattr(host, name)
    raise AttributeError(name)


__all__ = ["swiglu_oai_mxfp8", "warmup_swiglu_oai_mxfp8"]
