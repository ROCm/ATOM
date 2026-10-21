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
import time

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
        self.cpu_pg = getattr(sp_group, "cpu_group", None)
        self.pg = self.cpu_pg or getattr(sp_group, "device_group", None)
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

    def release(self) -> None:
        self.inner.release()

    # -- device error word, step counter, reset (aiter R12 host API) --------------------------
    # EVERY call into aiter's error-word / STEP / reset API is in this block (names from the R12
    # design card, D2.6). Re-pointing ATOM to the names aiter actually ships is a change to this
    # block only; ``hooks`` and ``site`` call these methods and nothing else.
    def step_tensor(self):
        """int64 [1] device view of the rank-uniform STEP counter (ATOM is its only writer)."""
        return self.inner.step_tensor()

    def error_readback_async(self, stream=None) -> None:
        """Enqueue a 4-byte D2H copy of the local error word on ``stream`` (default: the current
        stream), ordered after every launch already enqueued on it. Never syncs, never inside capture.
        """
        if stream is None:
            import torch

            stream = torch.cuda.current_stream()
        self.inner.error_readback_async(stream)

    def error_readback_ready(self):
        """The value of the readback enqueued last, or None while its copy has not completed."""
        return self.inner.error_readback_result()

    def error_word(self, timeout_s: float = 30.0) -> int:
        """The error word of the readback enqueued last, waiting for its copy if it is not complete
        yet. The wait is normally instant (the copy was enqueued two or more steps ago); it is
        bounded so a wedged device cannot hang a step silently."""
        deadline = time.monotonic() + timeout_s
        while True:
            v = self.error_readback_ready()
            if v is not None:
                return int(v)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"gemm_fusion: the device error readback did not complete in {timeout_s}s"
                )
            time.sleep(0.0005)

    def reset_after_error(self) -> None:
        """Collective reset after some rank's error word was set (every rank calls it)."""
        self.inner.reset()

    def clean_epoch(self) -> None:
        """Collective epoch reset at a clean point (after prepare, around graph capture).

        Maps to the same aiter ``reset()`` today. Kept apart from ``reset_after_error`` because the
        design card calls ``reset()`` legal only after an observed error; whether it is legal at a
        clean point is an open question for aiter, and the answer changes only this method.
        """
        self.inner.reset()

    def reduce_max(self, values):
        """Elementwise MAX all-reduce of small ints over the SP group; trivial on one rank.

        Uses a CPU tensor on the group's gloo CPU group (never a device tensor), so the verdict costs
        no device sync. Falls back to a device tensor on the device group only when the SP group
        has no CPU group."""
        values = [int(v) for v in values]
        if self.world <= 1 or self.pg is None:
            return values
        import torch
        import torch.distributed as dist

        if self.cpu_pg is not None:
            t = torch.tensor(values, dtype=torch.int64)
        else:
            t = torch.tensor(values, dtype=torch.int64, device=_device())
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.pg)
        return t.tolist()


def _device():
    import torch

    return torch.device("cuda", torch.cuda.current_device())
