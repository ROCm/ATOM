# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM wrapper over aiter's ``FusionResources`` (windows/control/epochs).

aiter owns allocation and the quiesce/reset protocol; ATOM binds it to the SP
group and the graph lifecycle and adds the token->local-M mapping and
consensus. Nothing here runs per replay.
"""

import logging
from typing import Callable, Optional

from . import consensus
from .stages import aiter_gf

logger = logging.getLogger("atom.gemm_fusion")


class FusionResourcesWrapper:
    def __init__(self, sp_group, world: int, inner=None, reduce_min: Optional[Callable] = None):
        self.group = sp_group
        self.world = world
        gf = aiter_gf()
        caps = gf.detect_capabilities(sp_group) if inner is None else frozenset()
        self.inner = inner if inner is not None else gf.FusionResources(sp_group, caps=caps)
        self.caps = getattr(self.inner, "caps", caps)
        if reduce_min is None and world > 1:
            reduce_min = consensus.torch_min_reducer(
                getattr(sp_group, "cpu_group", None) or None, device=_device()
            )
        self._reduce_min = reduce_min

    def local_m(self, tokens: int) -> int:
        return tokens // self.world

    def match_ctx(self, m: int, graph_mode: str = "eager"):
        gf = aiter_gf()
        arch, cu = _arch_cu()
        return gf.MatchCtx(
            arch=arch,
            cu=cu,
            world=self.world,
            m=m,
            available=frozenset(self.caps),
            graph_mode=graph_mode,
        )

    def agree(self, site: str, tokens: int, plan, n: int, errored: bool = False) -> str:
        vid = getattr(getattr(plan, "info", None), "vid", None) or getattr(plan, "vid", "native")
        return consensus.agree(self._reduce_min, site, tokens, vid, n, errored)

    def ensure(self, reqs):
        self.inner.ensure(reqs)

    @property
    def bound(self):
        return self.inner.bound

    def quiesce_and_reset(self):
        self.inner.quiesce_and_reset()

    def release(self):
        self.inner.release()


def _device():
    import torch

    return torch.device("cuda", torch.cuda.current_device())


def _arch_cu():
    import torch
    from aiter.jit.utils.chip_info import get_gfx_runtime

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return get_gfx_runtime(), props.multi_processor_count
