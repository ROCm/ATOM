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


# The low call_id bits a LARGE plan's handshake tag carries, per aiter source. Route A
# (r19, LARGE_ABI generation 6) repartitions the r13 13-bit m_real / 19-bit call_id word to
# 14-bit m_real / 18-bit call_id, so it needs narrower tag ids than r13 / r18.
_LARGE_TAG_BITS_FALLBACK = {
    "r11": 19,
    "r12": 19,
    "r13": 19,
    "r14": 19,
    "r18": 19,
    "r19": 18,
}
_LARGE_SRC = "sp_qkv.large.v1"


def large_tag_bits(src: str) -> int:
    """Low ``call_id`` bits of a large plan's handshake tag, from aiter's authoritative
    ``specs.large_tag_bits`` when the installed build exposes it (route A), else the
    fallback table above. Unknown sources keep aiter's own default (19)."""
    try:
        from aiter.ops.gemm_fusion.sp_qkv.specs import large_tag_bits as _aiter_width

        return int(_aiter_width(src))
    except Exception:  # noqa: BLE001 - absent on older aiter / CPU-only tests
        return _LARGE_TAG_BITS_FALLBACK.get(src, 19)


def large_plan_tag_bits(plan) -> int | None:
    """The low ``call_id`` bits of a LARGE plan's handshake tag (19 for the r13 / r18
    generations, 18 for route A's r19), or None when the plan's tag carries the whole
    ``call_id`` (every SMALL plan, and any ABI generation this wrapper does not know).

    The guard is the plan *kind* AND ABI generation, never ``src`` alone: ``r19`` also names
    a SMALL source (``SMALL_ABI`` 3) whose tags carry the full id, while the route-A r19 is
    the LARGE source at generation 6. The ``vid`` gate separates them; ``vid`` is absent
    only on test doubles, which are then classified by ABI as before. A future large source
    of the same tag width must be added to ``_LARGE_TAG_BITS_FALLBACK`` (or aiter should
    expose the width on the plan instead of ATOM names)."""
    vid = getattr(plan, "vid", None)
    if vid is not None and vid != _LARGE_SRC:
        return None
    abi = getattr(plan, "abi", None)
    # LARGE_ABI: 4 = r13 / r18 (LG_COMM_ABI_VERSION 5, 19 call bits),
    #           5 = r14 (LG_COMM_ABI_VERSION 6, same 19-bit tag word),
    #           6 = r19 route A (LG_COMM_ABI_VERSION 7, 18 call bits).
    if abi not in (4, 5, 6):
        return None
    src = getattr(plan, "src", None)
    if src is None:
        return None
    return large_tag_bits(src)


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

    # -- device error word, step counter, reset, call ids (aiter R12 host API, aiter 9b081852) ---
    # EVERY call into aiter's error-word / STEP / reset / call-id API is in this block. ``hooks`` and
    # ``site`` call these methods and nothing else, so a rename on aiter's side is a change here only.
    def step_tensor(self):
        """int64 [1] device view of the rank-uniform STEP counter (ATOM is its only writer)."""
        return self.inner.step_tensor()

    def error_readback_async(self, step_idx: int, stream=None) -> None:
        """Enqueue one 4-byte async D2H copy of the local error word into the pinned ring slot of
        ``step_idx`` (``step_idx % 32``), in stream order on ``stream`` (default: the current stream),
        i.e. after every launch already enqueued on it. Never syncs; aiter refuses it inside graph
        capture."""
        self.inner.readback_enqueue(step_idx, stream)

    def error_word(self, step_idx: int) -> int:
        """The error word captured at ``step_idx`` (0 = clean). Non-blocking first; when the copy is
        not complete yet the call blocks (``wait=True``). Every rank takes this same uniform path
        because the caller evaluates a snapshot a fixed number of steps after enqueueing it (the copy
        is complete by then), so the wait is normally instant."""
        v = self.inner.readback_consume(step_idx, False)
        if v is None:
            v = self.inner.readback_consume(step_idx, True)
        return int(v)

    def error_description(self) -> str:
        """A human-readable decode of the device error page, for the log only. A synchronous D2H
        read: call it after a sync, never on the hot path. Never raises (a diagnostics read must not
        block the recovery it belongs to)."""
        try:
            info = self.inner.error_info()
            if info is None:
                return "no error page"
            err, step, _diag = info
            return f"{err.describe()} step={step}"
        except Exception as e:  # noqa: BLE001
            return f"error page unreadable: {e!r}"

    def reset_after_error(self) -> None:
        """Collective reset after some rank's error word was set (every rank calls it). aiter drains,
        barriers, zeroes and verifies with an all-gather of every rank's error word, up to its
        ``reset_max_rounds``, and raises its ``ResetError`` when the word cannot be cleared.
        """
        self.inner.reset()

    @property
    def launch_gate(self) -> bool:
        """True for the whole of aiter's collective reset: ``Plan.run`` raises while it is set, and a
        captured-graph replay (no Python) would not notice, so ATOM reads it before every forward and
        graph replay. A plain bool read (False on an aiter without the gate)."""
        return bool(getattr(self.inner, "launch_gate", False))

    def reset_error_words(self, exc):
        """``{rank: error word}`` when ``exc`` is aiter's ``ResetError`` (a reset that could not reach
        an all-zero error state: the communicator cannot be trusted), else None."""
        cls = getattr(aiter_gf(), "ResetError", None)
        if cls is not None and isinstance(exc, cls):
            return dict(exc.words)
        return None

    def clean_epoch(self) -> None:
        """Collective epoch reset at a clean point (after prepare, around graph capture). Uses
        aiter's ``quiesce_and_reset`` (kept as an alias of ``reset``), the call these points have
        always made, since the design calls ``reset()`` itself the response to an observed error.
        """
        self.inner.quiesce_and_reset()

    @staticmethod
    def _callid_module():
        """aiter's call-id module, or None (older build: its plans ignore ``call_id``)."""
        try:
            import importlib

            return importlib.import_module(
                getattr(aiter_gf(), "__name__", "aiter.gemm_fusion") + ".callid"
            )
        except ImportError:
            return None

    def check_call_ids(
        self, keys, *, key_fn=None, tag_bits=None, tag_vids=None, per_tokens=False
    ):
        """aiter's static call-id uniqueness check over ``(site, tokens, vid, n_fused)`` keys.
        Returns its report (``.ok``, ``.refused_keys``), or None when this aiter has no call-id
        module (older build: its plans ignore ``call_id``). Rank-uniform: same keys, same install.

        R16 pad-M: ``per_tokens`` (only launches of ONE token count can meet in a handshake),
        ``tag_bits`` / ``tag_vids`` (a variant whose tag carries only the low bits of ``call_id``)
        and ``key_fn`` (the id function the launches really use) are passed through when given.
        """
        callid = self._callid_module()
        if callid is None:
            logger.warning(
                "gemm_fusion: the installed aiter has no callid module; the call_id uniqueness "
                "check is skipped"
            )
            return None
        kw = {}
        if key_fn is not None:
            kw["key_fn"] = key_fn
        if tag_bits is not None:
            kw["tag_bits"] = tag_bits
            kw["tag_vids"] = tag_vids
        if per_tokens:
            kw["per_tokens"] = True
        return callid.check_call_ids(keys, **kw)

    def tag_unique_ids(self, keys, tag_bits=None):
        """``{key: call_id}`` whose LOW ``tag_bits`` bits are distinct and non-zero within every
        token count (aiter's ``tag_unique_ids``, for plans whose tag carries only those bits), or
        None when this aiter has no such helper. ``tag_bits=None`` keeps aiter's default (19, the
        r13 / r18 large tag); route A's r19 large plans repartition the tag to 18 call bits and
        must pass it. A pure function of the key set."""
        callid = self._callid_module()
        fn = None if callid is None else getattr(callid, "tag_unique_ids", None)
        if fn is None:
            return None
        if tag_bits is None:
            return fn(keys)
        try:
            return fn(keys, tag_bits=tag_bits)
        except TypeError:
            # An aiter older than the route-A tag-width argument can only assign 19-bit ids;
            # returning them for a narrower tag would be wrong, so report "no ids" and let the
            # caller refuse the buckets (native) instead of launching with unknown low bits.
            return None

    @staticmethod
    def plan_tag_bits(plan):
        """Delegates to the single module-level helper (``large_plan_tag_bits``); kept as a
        wrapper method because the site/hook call sites historically asked the resources object.
        """
        return large_plan_tag_bits(plan)

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
