# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM wrapper over aiter's ``FusionResources`` (aiter/gemm_fusion/resources.py).

aiter owns window allocation and the quiesce/reset protocol; ATOM binds it to
the SP group and the graph lifecycle and adds token->local-M mapping,
consensus, and the register/finalize ordering policy below. Nothing here runs
per replay.

Ordering policy (aiter: ``ensure()`` after ``finalize()`` raises; every rank
must ensure identical requests in identical order):
  * ``ensure`` happens while resolving buckets (``before_graph_capture`` or
    the first call), then ``finalize()`` runs once, collectively, before any
    plan executes (explicitly before capture, otherwise lazily on first run).
  * A bucket resolved after finalize whose plan needs NEW windows falls back to
    native on every rank (reason ``after_finalize``); it never raises.
"""

import logging
from typing import Callable, Optional

from . import consensus
from .stages import aiter_gf

logger = logging.getLogger("atom.gemm_fusion")


def plan_owner(plan) -> str:
    """Window-ownership key: aiter keys windows per plan instance build key."""
    return getattr(plan, "build_key", None) or getattr(plan, "vid", "plan")


class FusionResourcesWrapper:
    def __init__(
        self,
        sp_group=None,
        world: int = 1,
        *,
        rank: Optional[int] = None,
        pg=None,
        inner=None,
        reduce_min: Optional[Callable] = None,
        device: Optional[int] = None,
        arch: Optional[str] = None,
        cu: Optional[int] = None,
    ):
        self.group = sp_group
        self.world = world
        gf = aiter_gf()
        if rank is None:
            rank = sp_group.rank_in_group
        if pg is None and sp_group is not None:
            # object broadcast / barrier go over the CPU group when present
            pg = getattr(sp_group, "cpu_group", None) or getattr(
                sp_group, "device_group", None
            )
        self.rank, self.pg = rank, pg
        if inner is None:
            import torch

            device = torch.cuda.current_device() if device is None else device
            caps = gf.detect_capabilities(sp_group)
            inner = gf.FusionResources(
                rank=rank, world=world, group=pg, device=device, caps=caps
            )
        self.inner = inner
        self.caps = frozenset(getattr(inner, "caps", ()))
        self._arch = arch or gf.detect_arch()
        self._cu = cu if cu is not None else gf.detect_cu()
        if reduce_min is None and world > 1:
            reduce_min = consensus.torch_min_reducer(pg, device=_device())
        self._reduce_min = reduce_min

    # -- matching / consensus ----------------------------------------------
    def local_m(self, tokens: int) -> int:
        return tokens // self.world

    def match_ctx(self, m: int, graph_mode: str = "eager"):
        return aiter_gf().MatchCtx(
            arch=self._arch,
            cu=self._cu,
            world=self.world,
            m=m,
            available=self.caps,
            graph_mode=graph_mode,
        )

    def agree(self, site: str, tokens: int, plan, n: int, errored: bool = False) -> str:
        vid = getattr(plan, "vid", None) or "native"
        return consensus.agree(self._reduce_min, site, tokens, vid, n, errored)

    # -- windows / lifecycle -----------------------------------------------
    @property
    def finalized(self) -> bool:
        return bool(getattr(self.inner, "finalized", False))

    def can_ensure(self, plan) -> bool:
        return (not self.finalized) or plan_owner(plan) in self.inner.windows

    def ensure(self, plan) -> None:
        self.inner.ensure(plan_owner(plan), plan.resources())

    def finalize(self) -> None:
        """Collective; idempotent."""
        if not self.finalized:
            self.inner.finalize()

    @property
    def bound(self):
        """Object handed to ``Plan.run(res=...)``: aiter's FusionResources."""
        return self.inner

    def quiesce_and_reset(self) -> None:
        self.inner.quiesce_and_reset()

    def release(self) -> None:
        self.inner.release()


def _device():
    import torch

    return torch.device("cuda", torch.cuda.current_device())
