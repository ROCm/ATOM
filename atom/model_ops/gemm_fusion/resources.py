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

from . import consensus
from .stages import aiter_gf

logger = logging.getLogger("atom.gemm_fusion")


def plan_owner(plan) -> str:
    """Window-ownership key. aiter plans call ``res.ensure(self.owner, ...)`` and look
    their windows up by ``self.owner + "/rx"`` etc., and (since aiter f8a8ebf7) expose the
    same string as ``build_key`` for exactly this purpose, so ATOM must use that key.
    ``vid`` alone is shared by every plan of a variant and would collide."""
    return (
        getattr(plan, "build_key", None)
        or getattr(plan, "owner", None)
        or getattr(plan, "vid", "plan")
    )


def fusion_resources_cls(gf):
    """aiter's ``FusionResources`` class: the package export when present, else its
    defining submodule (``aiter.gemm_fusion.resources``). Never a silent stub."""
    cls = getattr(gf, "FusionResources", None)
    if cls is not None:
        return cls
    import importlib

    name = getattr(gf, "__name__", "aiter.gemm_fusion")
    return importlib.import_module(name + ".resources").FusionResources


class FusionResourcesWrapper:
    def __init__(self, sp_group, world: int, profile: str):
        gf = aiter_gf()
        self.group = sp_group
        self.world = world
        self.rank = sp_group.rank_in_group
        # object broadcast / barrier go over the CPU group when present
        self.pg = getattr(sp_group, "cpu_group", None) or getattr(
            sp_group, "device_group", None
        )
        import torch

        device = torch.cuda.current_device()
        caps = gf.detect_capabilities(sp_group)
        # The window budget (and the VMM heap derived from it) comes from the profile's
        # window_budget. The profile is passed EXPLICITLY: aiter reads no environment variable to
        # choose one, so omitting it leaves the budget unknown and the first (small, 4 MiB)
        # request sizes every window (E2 new_on02: a later 40 MiB large-M plan is rejected).
        self.inner = fusion_resources_cls(gf)(
            rank=self.rank,
            world=world,
            group=self.pg,
            device=device,
            caps=caps,
            profile=profile,
        )
        self.caps = frozenset(getattr(self.inner, "caps", ()))
        self._arch = gf.detect_arch()
        self._cu = gf.detect_cu()
        self._reduce_min = (
            consensus.torch_min_reducer(self.pg, device=_device())
            if world > 1
            else None
        )

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def cu(self) -> int:
        return self._cu

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
