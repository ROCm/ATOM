# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``GemmFusionSite``: one GEMM followed by a declared epilogue chain.

Model-agnostic apart from the QKV split sizes. ``resolve`` asks aiter which
stage prefix a profile covers for a token bucket (cached, consensus-checked);
``run`` executes the fused prefix through the aiter plan and the remainder with
today's ATOM code. Every call yields a ``Decision`` record.

Binding: the per-call tensors ATOM receives (x_scale, positions, slot_mapping)
are not address-stable, so aiter's pointer bindings are made once, outside graph
capture, against site-owned staging buffers; each call copies the small inputs
into them (device-side, capturable).
"""

import logging
import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from . import consensus, hooks
from .resources import plan_owner
from .stages import aiter_gf

logger = logging.getLogger("atom.gemm_fusion")


class GemmFusionContractError(RuntimeError):
    """A profile row covers the spec but its contract fails (all ranks)."""


class GemmFusionBindError(RuntimeError):
    """An invariant the fused path relies on was violated (KV page mismatch, a
    re-bound KV pool, a consumer on the wrong stream). Never downgraded to a
    one-rank fallback."""


@dataclass(frozen=True)
class Resolution:
    plan: object  # aiter Plan or None
    n_fused: int  # stages covered (0 = all native)
    vid: str = "native"
    reason: str = ""
    # R16 pad-M: the T of the PREPARED bucket this call runs in when it differs from the call's own
    # (0 = the call's T is itself a prepared bucket). The plan, binding and outputs are the bucket's;
    # the kernel is only handed the real rows.
    bucket_tokens: int = 0


@dataclass
class Decision:
    site: str
    tokens: int
    vid: str
    n_fused: int
    path: str  # "native" | "fused"
    reason: str = ""
    bind_mode: str = ""  # "staged" | "raw" (1-stage plans take raw pointers)
    m_real: int = 0  # local rows the kernel was handed (T / world)
    bucket_m: int = 0  # local rows of the prepared bucket whose plan ran (0: native)


def _plan_vid(plan) -> str:
    if plan is None:
        return "native"
    return getattr(plan, "vid", None) or type(plan).__name__


def _rows2d(t):
    """[max_pos, 1, 1, rot/2] (ATOM rotary) or [max_pos, rot/2] -> [max_pos, rot/2]."""
    return t if t.dim() == 2 else t.reshape(t.shape[0], -1)


_KV_TORCH_DTYPE = {"fp8": "float8_e4m3fn", "bf16": "bfloat16"}


def _cache_layout_reason(impl, state, *, page, with_index):
    """Pure check of the bound caches against what the fused prologue writes (None = they match).

    The signature (``shuffle_page{page}_{kv_dtype}_ptok`` and the index layout) is built from
    configuration; aiter writes through raw pointers, so a tensor of another dtype or SHUFFLE geometry
    would be written at the wrong addresses with no error. Checked once at bind time (outside any
    step) and, through ``agree_prepared``, native on EVERY rank on a mismatch:
      K       [N, kv_heads, head_dim//x, page, x]   V  [N, kv_heads, page//x, head_dim, x]
              dtype of the signature, ``x`` = 16 bytes per element (the innermost, contiguous axis; one
              page row ``x`` elements apart)
      scales  fp8 only, float32 ``[N, kv_heads, page]``, innermost contiguous; None for bf16
      index   (indexer layers) FP8, contiguous ``[N, 128, 128]`` = flat ``[slot][128]``
    """
    kv_dtype = getattr(impl, "kv_cache_dtype", "bf16")
    name = _KV_TORCH_DTYPE.get(kv_dtype)
    if name is None:
        return "kv_layout"
    want = getattr(torch, name)
    x = 16 // (1 if kv_dtype == "fp8" else 2)
    nkv, hd = impl.num_kv_heads, impl.head_dim
    k = state.get("k_cache")
    v = state.get("v_cache_shuffle", state.get("v_cache"))
    ks = state.get("k_scale_view", state.get("k_scale"))
    vs = state.get("v_scale_view", state.get("v_scale"))
    if k is None or v is None or k.dim() != 5 or v.dim() != 5:
        return "kv_layout"
    pg = k.shape[3] if page is None else page
    if (
        k.dtype != want
        or v.dtype != want
        or (k.shape[1], k.shape[2] * x, k.shape[3], k.shape[4]) != (nkv, hd, pg, x)
        or (v.shape[0], v.shape[1], v.shape[2] * x, v.shape[3], v.shape[4])
        != (k.shape[0], nkv, pg, hd, x)
        or k.stride(4) != 1
        or k.stride(3) != x
        or v.stride(4) != 1
        or v.stride(3) != x
    ):
        return "kv_layout"
    if kv_dtype == "fp8":
        for s in (ks, vs):
            if (
                s is None
                or s.dtype != torch.float32
                or s.dim() != 3
                or tuple(s.shape) != (k.shape[0], nkv, pg)
                or s.stride(2) != 1
            ):
                return "kv_layout"
    elif ks is not None or vs is not None:
        return "kv_layout"
    if with_index:
        ic = getattr(impl, "index_cache", None)
        if (
            ic is None
            or getattr(impl, "index_cache_dtype", None) != "fp8"
            or getattr(impl, "index_head_dim", None) != 128
            or ic.dtype != torch.float8_e4m3fn
            or ic.dim() != 3
            or tuple(ic.shape[1:]) != (128, 128)
            or not ic.is_contiguous()
        ):
            return "index_layout"
    return None


class GemmFusionSite:
    def __init__(
        self,
        name,
        linear,
        stages,
        native_fns,
        *,
        attn,
        q_size,
        kv_size,
        index_q_size=0,
        index_k_size=0,
        sparse=False,
        skip_index=False,
    ):
        gf = aiter_gf()
        self.name = name
        self.linear = linear
        self.stages = tuple(stages)
        self.native = native_fns
        self.attn = attn
        self.q_size, self.kv_size = q_size, kv_size
        self.index_q_size, self.index_k_size = index_q_size, index_k_size
        self.sparse, self.skip_index = sparse, skip_index
        gemm = gf.GemmSpec(
            quant="fp8_ptpc",
            n=linear.output_size,
            k=linear.input_size,
        )
        self.spec = gf.FusionSpec(gemm, self.stages)
        self._by_tokens: dict = {}
        self._wcache: dict = {}
        self._staging: dict = {}
        # bucket -> why THIS rank refused to prepare it (bound caches do not match the signature)
        self._prepare_refusals: dict = {}
        # R16 pad-M, ready phase: a T that was not prepared resolves, as a PURE function of
        # (T, profile), to the prepared bucket that serves it (or native). _padded_raw: before the
        # call-id verdict; _padded: final.
        self._padded: dict = {}
        self._padded_raw: dict = {}
        # True only on the first registered site (hooks.register_site): it advances the device STEP
        # counter once per forward. Set before registration; registration decides it.
        self.ticks_step = False
        hooks.register_site(self)

    # -- layout helpers --------------------------------------------------
    def split_qkv(self, qkv):
        """Views of q/k/v as today's model forward produces them."""
        if self.sparse:
            q, k, v, _, _ = qkv.split(
                [
                    self.q_size,
                    self.kv_size,
                    self.kv_size,
                    self.index_q_size,
                    self.index_k_size,
                ],
                dim=-1,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def declared_page(self):
        """KV page size encoded in the site's HeadNormRopeCache.kv_cache string
        (``shuffle_page{N}_{dtype}_ptok``); aiter reads the page only from there."""
        for st in self.stages:
            kv = getattr(st, "kv_cache", None)
            if kv:
                m = re.search(r"_page(\d+)_", kv)
                return int(m.group(1)) if m else None
        return None

    def bind_caches(self, kv_cache_data) -> None:
        key = f"layer_{self.attn.impl.layer_num}"
        if key not in kv_cache_data:
            raise KeyError(f"gemm_fusion site {self.name}: no KV cache entry {key}")
        # The page in the signature must be the page of the cache aiter will write.
        # K shuffle layout: [blocks, kv_heads, head_dim//x, page, x] -> page = shape[3].
        # Sparse layers re-view the pool as page-16 (any multiple of 16 is fine);
        # dense layers use the pool's own page.
        declared = self.declared_page()
        k = getattr(kv_cache_data[key], "k_cache", None)
        if declared is not None and k is not None and k.dim() == 5:
            pool_page = int(k.shape[3])
            ok = (pool_page % declared == 0) if self.sparse else (pool_page == declared)
            if not ok:
                raise GemmFusionBindError(
                    f"{self.name}: signature declares KV page {declared} but the bound cache "
                    f"has page {pool_page} ({'sparse' if self.sparse else 'dense'} layer)"
                )

    # -- resolution ------------------------------------------------------
    def _static_prologue_ok(self) -> bool:
        """Static (config/env) eligibility of the fused norm/rope/cache stage,
        folded into the cross-rank agreement so it can never differ per call."""
        impl = getattr(self.attn, "impl", None)
        fn = getattr(impl, "gemm_fusion_prologue_supported", None)
        # Fail safe: an impl that cannot say it supports a fused prologue (no impl,
        # or no method) is not eligible -> native on every rank (via the agreement).
        return False if fn is None else bool(fn())

    def _static_weight_reason(self):
        """Why the QKV weight cannot be handed to an aiter plan as-is (None = it can).

        aiter's plans take ``w`` as a 16x16-PRESHUFFLED fp8 ``[N, K]`` matrix and ``w_scale`` with at
        least N rows (``variants.py`` / ``variants_large.py`` contracts; the spec declares
        ``w_layout="shuffle_16x16"`` but only the weight itself knows what it holds). ATOM stores it
        so unless ``ATOM_USE_TRITON_GEMM=1`` (unshuffled) or a loader path skipped the shuffle; a
        plan fed an unshuffled weight would compute wrong values with no error, so it is checked
        here, once, at resolve time (after ``process_weights_after_loading``), and agreed across ranks.
          weight_layout : ``weight.is_shuffled`` is not True
          weight_shape  : K of the weight is not ``input_size``, or its rows are fewer than N. A
                          weight whose N was padded for the preshuffle GEMM (``is_output_padded``)
                          has MORE rows than N and is fine: the first N rows of a 16-row-block
                          shuffle are a contiguous prefix and aiter reads exactly N rows. An
                          unpadded weight must have exactly N rows.
        """
        lin = self.linear
        w = getattr(lin, "weight", None)
        if w is None or getattr(w, "is_shuffled", False) is not True:
            return "weight_layout"
        n, k = lin.output_size, lin.input_size
        shape = tuple(getattr(w, "shape", ()))
        if len(shape) != 2 or shape[1] != k:
            return "weight_shape"
        if getattr(lin, "is_output_padded", False) is True:
            if shape[0] < n:
                return "weight_shape"
        elif shape[0] != n:
            return "weight_shape"
        ws_shape = tuple(getattr(getattr(lin, "weight_scale", None), "shape", ()))
        if not ws_shape or ws_shape[0] < n:
            return "weight_shape"
        return None

    def resolve(self, tokens: int, graph_mode: str = "eager") -> Resolution:
        """Resolve a bucket. Cache key is ``(tokens, graph_mode)``.

        Only the collective ``preparing`` phase runs match/consensus/ensure. Before
        it every call is native (``not_prepared``); after it a token count that was not
        prepared is resolved WITHOUT a collective by aiter's pure ``bucket_for_m`` (R16 pad-M):
        the prepared bucket whose plan serves that many real rows runs with the real rows only
        (``pad_hit``), else it is native (``late_bucket`` / ``uncovered`` ...). Both are decided
        identically on every rank: a function of (T, profile) and the buckets agreed at prepare.
        """
        key = (tokens, graph_mode)
        r = self._by_tokens.get(key)
        if r is not None:
            if isinstance(r, Exception):
                raise r
            return r
        res = hooks.resources()
        phase = hooks.phase()
        if res is None:
            return Resolution(None, 0, "native", "profile_off")
        if phase == "pre":
            return Resolution(None, 0, "native", "not_prepared")  # not cached
        if phase == "ready":
            r = self._padded.get(key)
            if r is None:
                r = self._padded_raw_resolution(tokens, graph_mode, res)
                if r.bucket_tokens and r.plan is not None:
                    # a padded key is a different (token count, variant) than any prepared one: its
                    # call_id is checked among the launches of THIS token count (pure, no collective)
                    ids = hooks.token_call_ids(tokens, graph_mode)
                    if (self.name, tokens, r.vid, r.n_fused) in ids.refused:
                        r = Resolution(None, 0, r.vid, "call_id_refused")
                self._padded[key] = r
            return r
        gf = aiter_gf()
        contract_err = getattr(gf, "ContractError", ())
        errored, plan, n, msg = False, None, 0, ""
        try:
            # profile passed EXPLICITLY (config.gemm_fusion.profile via hooks), never left to
            # aiter's AITER_GEMM_FUSION_PROFILE env fallback (absent in spawned workers)
            plan, n = gf.match_prefix(
                self.spec,
                res.match_ctx(res.local_m(tokens), graph_mode),
                hooks.profile(),
            )
        except contract_err as e:  # covered but contract fails
            errored, msg = True, str(e)
        static_reason = (
            None  # why a covered row is still not fusable (static, per layer)
        )
        if plan is not None:
            # Both are properties of the layer, not of the call, so every rank computes the same
            # answer; they are folded into the agreement below (plan=None => native on all ranks).
            static_reason = (
                self._static_weight_reason()
            )  # applies to 1- and 2-stage plans: both read w
            if static_reason is None and n >= 2 and not self._static_prologue_ok():
                static_reason = "prologue_unsupported"
            if static_reason is not None:
                plan, n = None, 0
        outcome = res.agree(self.name, tokens, plan, n, errored=errored)
        if outcome == "error":
            r = GemmFusionContractError(msg or "contract error on all ranks")
            self._by_tokens[key] = r
            raise r
        if outcome == "disagree" or plan is None:
            if outcome == "disagree":
                reason = "rank_disagree"
            else:
                # a profile row covered the spec but this layer cannot be fused (static
                # eligibility, identical on every rank): say so, do not report it as a
                # profile miss
                reason = static_reason or "uncovered"
            r = Resolution(None, 0, "native", reason)
        elif not res.can_ensure(plan):
            r = Resolution(
                None, 0, "native", "after_finalize"
            )  # uniform: same order on all ranks
        else:
            # ONE plan object per (site, bucket), shared by the eager and capture keys.
            # match_prefix builds a NEW plan each call and a small-M plan keeps its
            # pointer bindings in a per-INSTANCE dict, so a plan prepared for "eager"
            # would be a different, unbound object when the capture/replay path asks
            # for "capture" ("new binding required during capture"). The profile row
            # does not depend on graph_mode, so the first match is the one to keep.
            other = self._by_tokens.get(
                (tokens, "capture" if graph_mode == "eager" else "eager")
            )
            if (
                other is not None
                and not isinstance(other, Exception)
                and other.plan is not None
                and other.n_fused == n
                and _plan_vid(other.plan) == _plan_vid(plan)
                and plan_owner(other.plan) == plan_owner(plan)
            ):
                plan = other.plan
            res.ensure(plan)
            hooks.register_plan(plan)
            r = Resolution(plan, n, _plan_vid(plan), "profile_hit")
        self._by_tokens[key] = r
        return r

    @classmethod
    def _pad_capable(cls, plan, n_fused) -> bool:
        """Whether ``plan`` can launch fewer real rows than the bucket it was built for: aiter marks
        those plans ``pad_m``; a small plan must also take x_scale / positions / slot_mapping per
        launch (abi >= 3), the large one takes none of them."""
        if not getattr(plan, "pad_m", False):
            return False
        return cls.per_launch(plan) if n_fused >= 2 else True

    def _padded_raw_resolution(self, tokens, graph_mode, res):
        key = (tokens, graph_mode)
        got = self._padded_raw.get(key)
        if got is None:
            got = self._padded_raw[key] = self._pad_match(tokens, graph_mode, res)
        return got

    def _pad_match(self, tokens, graph_mode, res):
        """The resolution of a token count that was NOT prepared (ready phase), before the call-id
        verdict. Pure: aiter's ``bucket_for_m`` depends only on (spec, arch, cu, world, m_real) and the
        immutable profile, and whether that bucket is usable is what every rank agreed at prepare.
        """
        late = Resolution(None, 0, "native", "late_bucket")
        bucket_for_m = getattr(aiter_gf(), "bucket_for_m", None)
        world = hooks.world_or_1()
        if bucket_for_m is None or tokens % world:
            return late  # an aiter without pad-M, or T that is not a whole number of local rows
        m_real = res.local_m(tokens)
        try:
            bucket_m = bucket_for_m(
                self.spec, res.match_ctx(m_real, graph_mode), hooks.profile()
            )
        # deterministic in (spec, m, profile): the same on every rank
        except Exception:
            logger.exception("%s: bucket_for_m(m=%d) failed; native", self.name, m_real)
            return Resolution(None, 0, "native", "pad_error")
        if bucket_m is None:
            return Resolution(None, 0, "native", "uncovered")
        if bucket_m < m_real:
            return Resolution(None, 0, "native", "pad_error")
        bucket_t = int(bucket_m) * world
        rb = self._by_tokens.get((bucket_t, graph_mode))
        if rb is None or isinstance(rb, Exception):
            return late  # the bucket itself was never prepared
        if rb.plan is None:
            return Resolution(None, 0, "native", rb.reason)  # the bucket is native too
        if bucket_t == tokens:
            return rb
        if not self._pad_capable(rb.plan, rb.n_fused):
            return Resolution(None, 0, rb.vid, "pad_unsupported")
        return Resolution(rb.plan, rb.n_fused, rb.vid, "pad_hit", bucket_t)

    def raw_resolution(self, tokens, graph_mode):
        """The resolution of ``(tokens, mode)`` BEFORE its call-id verdict, or None (not ready, an
        error entry). Independent of the order sites resolved in: it is what the per-token-count
        call-id check (``hooks.token_call_ids``) is built from."""
        r = self._by_tokens.get((tokens, graph_mode))
        if r is not None:
            return None if isinstance(r, Exception) else r
        res = hooks.resources()
        if res is None or hooks.phase() != "ready":
            return None
        return self._padded_raw_resolution(tokens, graph_mode, res)

    @staticmethod
    def _runtime_mode() -> str:
        return (
            "capture"
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            else "eager"
        )

    def _prepared_ok(self, tokens: int, r: Resolution, prepared: bool) -> bool:
        """This rank's local verdict that bucket ``tokens`` is safe to launch for resolution
        ``r``. Every rank-local invariant the per-call path would otherwise re-check (and raise
        on) is folded in here, at prepare time, and then AGREED across ranks by
        ``agree_prepared``: a failure downgrades the bucket to native on ALL ranks instead
        of raising on one rank in the middle of a step.
          * prepared    : this rank bound the plan (copies into its own staging buffers)
          * plan instance : the plan prepared is the plan this resolution will launch
                            (an unshared instance would need a new binding at run/capture)
        """
        if not prepared:
            return False
        if r.n_fused < 2:
            return (
                True  # 1-stage large-M: raw per-launch pointers, nothing bound per site
            )
        st = self._staging.get(tokens)
        if st is None or not st.prepared:
            return False
        prepared_plan = st.plan
        if prepared_plan is not None and prepared_plan is not r.plan:
            logger.error(
                "%s: bucket %d resolves to a plan instance other than the one prepared",
                self.name,
                tokens,
            )
            return False
        return True

    def agree_prepared(self, tokens: int, prepared: bool) -> None:
        """Collective: every rank must have prepared (or not) the same buckets.
        A mismatch downgrades the bucket to native on ALL ranks. This is the ONLY place the
        "can this bucket launch" invariants are decided; the per-call path trusts it."""
        res = hooks.resources()
        for mode in ("eager", "capture"):
            r = self._by_tokens.get((tokens, mode))
            if r is None or isinstance(r, Exception) or r.plan is None:
                continue
            ok = self._prepared_ok(tokens, r, prepared)
            outcome = res.agree(self.name + "#prepared", tokens, None, int(bool(ok)))
            if outcome != "agree" or not ok:
                if outcome != "agree":
                    why = "prepare_disagree"
                else:
                    why = self._prepare_refusals.get(tokens, "unprepared")
                self._by_tokens[(tokens, mode)] = Resolution(None, 0, "native", why)

    def refuse_call_ids(self, refused) -> int:
        """Downgrade every bucket whose ``(site, tokens, vid, n_fused)`` key is in ``refused`` (a
        call_id collision or a zero id, from ``hooks._refuse_colliding_call_ids``) to native. The keys
        and the check are rank-uniform, so every rank downgrades the same buckets. Returns how many.
        """
        n = 0
        for (tokens, mode), r in list(self._by_tokens.items()):
            if getattr(r, "plan", None) is None:
                continue
            if (self.name, tokens, r.vid, r.n_fused) in refused:
                self._by_tokens[(tokens, mode)] = Resolution(
                    None, 0, r.vid, "call_id_refused"
                )
                n += 1
        return n

    # -- KV pointer guard (bind time, collective) --------------------------------
    def kv_ptrs(self, kv_cache_data):
        """Addresses of this layer's KV caches and scales: what a binding is made on."""
        layer = kv_cache_data.get(f"layer_{self.attn.impl.layer_num}")
        if layer is None:
            return None
        return tuple(
            self._ptr(getattr(layer, n, None))
            for n in ("k_cache", "v_cache", "k_scale", "v_scale")
        )

    def kv_moved(self, kv_cache_data) -> bool:
        """True when a bucket was already prepared against KV caches whose addresses differ
        from ``kv_cache_data``'s (a re-allocated pool). The plan keeps raw pointers from
        prepare, so launching it would write the OLD cache. Evaluated by ``hooks.bind_kv_cache``
        and agreed across ranks there; it is never evaluated per call."""
        now = self.kv_ptrs(kv_cache_data)
        return any(
            st.prepared and st.kv_ptrs not in (None, now)
            for st in self._staging.values()
        )

    # -- persistent staging (aiter binds raw pointers once per tensor set) ------
    def _stage(self, tokens, device, dtype):
        """Per-bucket buffers whose addresses never move: aiter's plan binds ``outs`` (and, for an
        r17/r18 plan, x_scale/positions/slot_mapping) by pointer outside capture, and the per-call
        tensors ATOM receives are not address-stable, so an r17/r18 call copies its small per-call
        inputs into these (device-side, capturable). An r19 plan takes those inputs per launch: the
        ``xs/pos/slot`` arrays then only give ``prepare`` conforming tensors to validate.
        """
        st = self._staging.get(tokens)
        if st is None:
            impl = self.attn.impl
            m = tokens // max(1, hooks.world_or_1())
            outs = {
                "q": torch.empty(
                    (tokens, impl.num_heads, impl.head_dim), dtype=dtype, device=device
                )
            }
            if self.sparse and not self.skip_index:  # indexer: index_q, NO k/v
                outs["index_q"] = torch.empty(
                    (
                        tokens,
                        impl.index_q_size // impl.index_head_dim,
                        impl.index_head_dim,
                    ),
                    dtype=dtype,
                    device=device,
                )
            else:  # dense / reuse
                kv = (tokens, impl.num_kv_heads, impl.head_dim)
                outs["k"] = torch.empty(kv, dtype=dtype, device=device)
                outs["v"] = torch.empty(kv, dtype=dtype, device=device)
            st = SimpleNamespace(
                outs=outs,
                xs=torch.empty((m, 1), dtype=torch.float32, device=device),
                # T = padded tokens long, never shorter: the kernel reads all T entries. Initialised to
                # the pad-row values (position 0, slot -1) so no launch can ever see garbage.
                pos=torch.zeros((tokens,), dtype=torch.int64, device=device),
                slot=torch.full((tokens,), -1, dtype=torch.int64, device=device),
                prepared=False,
                plan=None,
                kv_ptrs=None,
                m=m,
            )
            self._staging[tokens] = st
        return st

    def _f32(self, key, t):
        c = self._wcache.get(key)
        if c is None or c[0] is not t:
            c = (t, t.float().contiguous())
            self._wcache[key] = c
        return c[1]

    def build_meta(self, st, state, ctx=None) -> dict:
        """``Plan.run(meta=...)``: the authoritative aiter sp_qkv.small.v1 key contract.

        ``ctx`` given (an r19 per-launch call): positions / slot_mapping are the caller's own
        tensors. Without it (prepare, and r17/r18 plans) they are the site's T-long staging arrays.
        """
        impl = self.attn.impl
        rot = impl.rotary_emb
        meta = {
            "positions": st.pos if ctx is None else ctx.positions,
            "slot_mapping": st.slot if ctx is None else state["slot_mapping"],
            "k_cache": state["k_cache"],
            # the prologue writes the SHUFFLE view of V (== what rope_cache writes)
            "v_cache": state.get("v_cache_shuffle", state["v_cache"]),
            "k_scale": state.get("k_scale_view", state.get("k_scale")),
            "v_scale": state.get("v_scale_view", state.get("v_scale")),
            "cos_cache": _rows2d(rot.cos_cache),
            "sin_cache": _rows2d(rot.sin_cache),
            "q_norm_w": self._f32("qw", impl.q_norm.weight),
            "k_norm_w": self._f32("kw", impl.k_norm.weight),
        }
        if self.sparse and not self.skip_index:
            meta["index_cache"] = impl.index_cache
            meta["index_q_norm_w"] = self._f32("iqw", impl.index_q_norm.weight)
            meta["index_k_norm_w"] = self._f32("ikw", impl.index_k_norm.weight)
        return meta

    @staticmethod
    def _ptr(t):
        return 0 if t is None else t.data_ptr()

    @staticmethod
    def decide_path(
        *,
        resolution,
        is_dummy_run,
        x_dtype_is_fp8,
        x_scale_ok,
        x_k_ok=True,
        meta_len_ok=True,
        tokens_ok=True,
    ):
        """The per-call fused/native decision as a PURE function of rank-uniform inputs
        (resolution agreed at resolve, the broadcast batch's dummy flag, and the input's
        dtype/shape facts, including whether x's K equals the layer's ``input_size``). No rank,
        no env, no device state: the same inputs give the same ``(path, reason)`` on every rank.
        -> (path, reason).

        Padded rows (``real_tokens < padded_tokens``) are NOT a reason to go native: both plan
        kinds run them. The small-M plan reads positions/slot_mapping for all T = 4*M rows. An
        r17/r18 plan gets T-length arrays the site stages (pad rows are position 0, slot -1; see
        ``_stage_inputs``); an r19 plan (``plan.abi >= 3``) takes the caller's arrays per launch and
        its kernel pads the rows past ``t_valid`` the same way.

        ``meta_len_ok``: ``positions`` and ``slot_mapping`` hold exactly ``real_tokens`` entries
        each. The r19 kernel reads ``t_valid = min(numel(positions), numel(slot_mapping), T)`` rows as
        real ones (frozen into a captured graph), and r17/r18 staging slices both to ``real_tokens``:
        a ``slot_mapping`` shorter than the real tokens would silently turn real rows into pad rows
        (position 0, cache write skipped) and a longer one is metadata the site did not stage. Both are
        native decisions instead of a silent shrink or a rank-local raise.

        ``tokens_ok`` (r19 only): the per-launch ``positions`` / ``slot_mapping`` / ``x_scale``
        satisfy the plan's own input contract. The plan RAISES on a violation, and a raise on one
        rank is a fatal host exception, so a violation must be a native decision on every rank.
        """
        if resolution.plan is None:
            return "native", resolution.reason
        if is_dummy_run:
            return "native", "dummy_run"
        if not x_dtype_is_fp8:
            return "native", "unquantized_input"
        if not x_scale_ok:
            return "native", "xscale_layout"
        if not x_k_ok:
            return (
                "native",
                "x_k_mismatch",
            )  # the plan's K is the spec's (input_size); never feed it another
        if not meta_len_ok:
            return "native", "meta_len"
        if not tokens_ok:
            return "native", "meta_layout"
        return "fused", resolution.reason

    @staticmethod
    def input_facts(x, x_scale):
        """The two rank-uniform input facts ``decide_path`` needs, read from the tensors:
        (x is fp8 e4m3 with a scale, x_scale is a [rows, 1] float32 column)."""
        fp8 = x_scale is not None and x.dtype == torch.float8_e4m3fn
        ok = (
            x_scale is not None
            and x_scale.dim() == 2
            and tuple(x_scale.shape) == (x.shape[0], 1)
            and x_scale.dtype == torch.float32
        )
        return fp8, ok

    @staticmethod
    def per_launch_ok(x_scale, positions, slot_mapping):
        """Whether the arrays an r19 plan reads per launch meet its contract (aiter ``inputs``):
        ``x_scale`` contiguous; ``positions`` and ``slot_mapping`` int64, contiguous, non-empty. A
        rank-uniform fact (SPMD: the same tensors on every rank), read from metadata only.
        """
        return x_scale.is_contiguous() and all(
            t.dtype == torch.int64 and t.is_contiguous() and t.numel() >= 1
            for t in (positions, slot_mapping)
        )

    @staticmethod
    def meta_len_ok(positions, slot_mapping, real_tokens) -> bool:
        """``numel(positions) == numel(slot_mapping) == real_tokens``: the lengths every consumer of
        the per-call token arrays (t_valid, ``_stage_inputs``) assumes. Metadata only, rank-uniform.
        """
        return int(positions.numel()) == int(slot_mapping.numel()) == int(real_tokens)

    def cache_layout_reason(self, state):
        """Why the caches in ``state`` are not what this site's signature describes (None = they
        are): ``"kv_layout"`` (K/V/scales) or ``"index_layout"`` (indexer keys). Checked at bind time
        on the bound tensors, because the signature is built from configuration before the KV pool
        exists and a mismatch would be written through with no error. See ``_cache_layout_reason``.
        """
        return _cache_layout_reason(
            self.attn.impl,
            state,
            page=self.declared_page(),
            with_index=bool(self.sparse and not self.skip_index),
        )

    @staticmethod
    def per_launch(plan) -> bool:
        """True for a plan whose small-M kernel takes x_scale / positions / slot_mapping per launch
        (R12, ``abi >= 3``); r17/r18 plans bind them by pointer and need the site's staging.
        """
        return getattr(plan, "abi", 2) >= 3

    # -- binding -----------------------------------------------------------------
    def _bind_plan(self, r, tokens, st, state, device, kv_cache_data):
        """Create aiter's pointer binding for a 2-stage plan outside capture and mark the
        bucket prepared. ``st.xs/pos/slot`` are the buffers the binding is made on."""
        x = torch.zeros(
            (st.m, self.linear.input_size), dtype=torch.float8_e4m3fn, device=device
        )
        res = hooks.resources()
        res.finalize()
        prep = getattr(r.plan, "prepare", None)
        if prep is not None:
            prep(
                x=x,
                x_scale=st.xs,
                w_scale=self.linear.weight_scale,
                outs=st.outs,
                meta=self.build_meta(st, state),
                res=res.bound,
            )
        st.prepared = True
        st.plan = r.plan  # the exact plan instance whose binding was just created
        st.kv_ptrs = (
            self.kv_ptrs(kv_cache_data) if kv_cache_data else None
        )  # for kv_moved()
        return True

    @staticmethod
    def _prepare_one_stage(r):
        """1-stage prefix (large M): the plan takes raw per-launch pointers, no staging; just
        attach + init control after finalize, before capture."""
        hooks.resources().finalize()
        prep = getattr(r.plan, "prepare", None)
        if prep is not None:
            prep(res=hooks.resources().bound)
        return True

    def prepare_bucket(self, tokens, device, kv_cache_data, out_dtype):
        """Bind the plan for one bucket OUTSIDE graph capture (padded == real).

        Called after ``bind_kv_cache`` and the bucket's resolution, before capture."""
        r = self.resolve(tokens, "eager")
        if r.plan is None:
            return False
        if r.n_fused < 2:
            return self._prepare_one_stage(r)
        st = self._stage(tokens, device, out_dtype)
        if st.prepared:
            return True
        state = self.attn.impl.gemm_fusion_prologue_state(
            _ShimFwdCtx(kv_cache_data, st.slot, self.attn.impl)
        )
        if state is None:
            return False
        why = self.cache_layout_reason(state)
        if why is not None:
            # False here is AGREED across ranks by agree_prepared: the bucket is native on all of them
            if not self._prepare_refusals:
                logger.warning(
                    "%s: the bound caches do not match the fused signature (%s); native on every "
                    "rank",
                    self.name,
                    why,
                )
            self._prepare_refusals[tokens] = why
            return False
        return self._bind_plan(r, tokens, st, state, device, kv_cache_data)

    # -- execution -------------------------------------------------------
    def _prepare(self, ctx, x, x_scale):
        """-> (Resolution, reason, state, staging).

        PER-CALL DECISION. ``decide_path`` is a pure function of rank-uniform inputs, so every
        rank takes the same path for the same call:
          resolution   : agreed at resolve (collective consensus), cached, and downgraded
                         uniformly by ``agree_prepared`` (prepared / plan instance are folded
                         into THAT agreement, at prepare time).
          dummy_run    : ``batch.is_dummy_run`` comes from ONE pickled batch that
                         ``AsyncIOProcManager.call_func`` broadcasts to every worker of the
                         engine core (rpc_broadcast_mq.enqueue); SP ranks share that core.
          x dtype / x_scale shape / K : follow from the model (SPMD) and equal local rows.
        Padded rows do not change the path: ``real_tokens`` = positions.shape[-1] (global,
        positions stay whole under SP) and ``padded_tokens`` = local rows * world only decide how
        ``_stage_inputs`` fills the T-length token arrays.
        There is NO rank-local raise in this method: the "unprepared bucket", "prologue state
        unavailable" and "other plan instance" conditions are unreachable after
        ``agree_prepared`` (they downgrade the bucket to native on every rank, or fail
        ``hooks.bind_kv_cache`` on every rank), so nothing here can make one rank leave a step
        while its peers wait in a fused kernel.

        What this does NOT make safe: a non-uniform decision caused by a bug or a fault (a
        one-rank exception before launch). A divergence that blocks mid-step (one rank in the
        native RCCL all_to_all, the others in the fused CCO kernel) never reaches the step-end
        device error check in ``hooks.check_step`` and is FAIL-STOP at the process-group timeout;
        the fused kernels' own tags report a divergence that still completes.
        """
        r = self.resolve(ctx.padded_tokens, self._runtime_mode())
        if r.plan is not None and self._tag_refused(r, ctx.padded_tokens):
            r = Resolution(None, 0, r.vid, "call_id_refused")
        reason = r.reason
        if r.plan is None:
            return r, reason, None, None
        fp8, scale_ok = self.input_facts(x, x_scale)
        state = None
        tokens_ok = True
        meta_len_ok = True
        if r.n_fused >= 2:
            # eligibility was agreed at resolve; never None here
            state = ctx.prologue_state()
            meta_len_ok = self.meta_len_ok(
                ctx.positions, state["slot_mapping"], ctx.real_tokens
            )
            if meta_len_ok and self.per_launch(r.plan) and fp8 and scale_ok:
                tokens_ok = self.per_launch_ok(
                    x_scale, ctx.positions, state["slot_mapping"]
                )
        path, why = self.decide_path(
            resolution=r,
            is_dummy_run=ctx.is_dummy_run,
            x_dtype_is_fp8=fp8,
            x_scale_ok=scale_ok,
            x_k_ok=x.shape[-1] == self.linear.input_size,  # SPMD: same on every rank
            meta_len_ok=meta_len_ok,
            tokens_ok=tokens_ok,
        )
        if path == "native":
            return Resolution(None, 0, "native", why), why, None, None
        if r.n_fused < 2:
            return r, reason, None, None
        # the PREPARED bucket's staging (agree_prepared): the call's own T unless it was padded
        st = self._staging[r.bucket_tokens or ctx.padded_tokens]
        return r, reason, state, st

    def _stage_inputs(self, x_scale, ctx, state, st):
        """Copy the per-call inputs into the buffers the plan is bound to (device-side,
        capturable).

        ``st.pos`` / ``st.slot`` are T = padded_tokens long: the small-M kernel reads
        positions[t] / slot_mapping[t] for EVERY t < T (all ranks' rows, after the exchange), while
        ATOM's metadata covers only the ``real`` rows. Rows [real, T) are the SP divisibility pad,
        which is the GLOBAL tail (so only the last rank(s) hold pad rows in x, but every rank's
        kernel reads all T entries): they get position 0 and slot -1, the values ATOM already uses
        for decode batch-padding rows, and the kernel skips cache writes for slot < 0. The tail is
        rewritten on every call because the same bucket can be reached with a different ``real``.
        """
        n = ctx.real_tokens
        st.xs.copy_(x_scale)
        st.pos[:n].copy_(ctx.positions[:n])
        st.slot[:n].copy_(state["slot_mapping"][:n])
        if n < ctx.padded_tokens:
            st.pos[n:].zero_()
            st.slot[n:].fill_(-1)

    def _execute(self, r, x, x_scale, ctx, state, st, out_dtype):
        """Run the fused prefix; return its ``outs`` dict."""
        res = hooks.resources()
        res.finalize()  # collective + idempotent; before any plan executes
        if r.n_fused >= 2:
            # the plan keeps its pointer bindings per INSTANCE; that the plan launched here is the
            # one prepared is verified at prepare time (`_prepared_ok` via `agree_prepared`),
            # not per call, so no rank can raise here while its peers launch.
            if self.per_launch(r.plan):
                # r19: x_scale / positions / slot_mapping are per-launch kernel arguments (not part
                # of the binding) and t_valid is read from their lengths: no staging copies.
                outs, meta, xs = st.outs, self.build_meta(st, state, ctx), x_scale
            else:
                self._stage_inputs(x_scale, ctx, state, st)  # r17/r18
                outs, meta, xs = st.outs, self.build_meta(st, state), st.xs
        else:
            outs, meta, xs = {}, {}, x_scale  # large M: raw per-launch pointers
        r.plan.run(
            x=x,
            x_scale=xs,
            w=self.linear.weight,
            w_scale=self.linear.weight_scale,
            outs=outs,
            meta=meta,
            res=res.bound,
            stream=torch.cuda.current_stream(),
            # the launch's identity in the kernels' handshake tags (R12 plans require it; older plans
            # ignore it). Distinct non-zero keys were checked statically (prepare_buckets for the
            # prepared buckets, per token count for padded ones), and a refused key never reaches
            # here (its bucket is native).
            call_id=self._call_id(r, ctx.padded_tokens),
        )
        if r.n_fused < 2:
            # Zero-copy view of the plan's receive window; valid until this rank
            # launches the next large epoch (of ANY plan under a shared window), i.e. past
            # the native remainder that consumes it (same stream, in order). Reuse layers get
            # the [T,2560] row-stride-2304 view: the IQ|IK columns alias the next
            # row and are never read (skip branch), and rope_cache's
            # qkv.contiguous() copies the view with those semantics intact.
            plan = r.plan
            # R16: the view is over the REAL rows (T = 4 * m_real), passed explicitly because a graph
            # replay runs no Python and the plan's own "last launch" would be stale
            rows = (
                (ctx.padded_tokens // hooks.world_or_1(),)
                if getattr(plan, "pad_m", False)
                else ()
            )
            if getattr(plan, "mode", None) == "reuse" and hasattr(plan, "reuse_view"):
                view = plan.reuse_view(res.bound, *rows)
            else:
                view = plan.recv_view(res.bound, *rows)
            outs = {"packed": view}
        return outs

    def _tag_refused(self, r, tokens) -> bool:
        """A plan whose tag carries only the low bits of ``call_id`` whose id could not be assigned
        (rank-uniform, per token count): its call is native."""
        bits = getattr(hooks.resources(), "plan_tag_bits", None)
        if bits is None or not bits(r.plan):
            return False
        ids = hooks.token_call_ids(tokens, self._runtime_mode())
        return (self.name, tokens, r.vid, r.n_fused) in ids.refused

    def _call_id(self, r, tokens) -> int:
        """The launch's identity in the handshake tag. Small plans: the 31-bit ``decision_key``. A plan
        whose tag carries only the LOW bits of call_id (the large R12-ABI sources r13 / r18): the id
        aiter's ``tag_unique_ids`` assigned for this token count, whose low bits are distinct and
        non-zero by construction (a ``decision_key`` collides on them)."""
        bits = getattr(hooks.resources(), "plan_tag_bits", None)
        if bits is not None and bits(r.plan):
            ids = hooks.token_call_ids(tokens, self._runtime_mode())
            got = ids.ids.get((self.name, tokens, r.vid, r.n_fused))
            if got is not None:
                return got
        return consensus.decision_key(self.name, tokens, r.vid, r.n_fused)

    def _bind_label(self, r, path, st):
        if path == "fused" and r.n_fused < 2:
            return "raw"
        if path == "fused" and self.per_launch(r.plan):
            return "per_launch"
        return "staged"

    def _decision(self, ctx, r, reason, path, st=None):
        world = hooks.world_or_1()
        m_real = ctx.padded_tokens // world
        # the coverage record of pad-M: the rows the kernel got vs the rows of the bucket whose plan ran
        bucket_m = (
            0 if r.plan is None else (r.bucket_tokens or ctx.padded_tokens) // world
        )
        d = Decision(
            self.name,
            ctx.padded_tokens,
            r.vid,
            r.n_fused,
            path,
            reason,
            self._bind_label(r, path, st),
            m_real,
            bucket_m,
        )
        return d

    def _remainder_after_exchange(self, r, outs, ctx, consumer):
        """Run the native remainder on a large-M recv_view. The view is only valid on the
        launching stream until this rank's next large launch, so the launch and every
        consumer must be on ``torch.cuda.current_stream()``."""
        s0 = torch.cuda.current_stream()
        out = consumer(outs["packed"])
        if torch.cuda.current_stream() != s0:
            raise GemmFusionBindError("recv_view consumer ran on a different stream")
        return out

    def run(self, x, x_scale, ctx):
        """Full QKV+attention. -> ((output, scale), Decision)."""
        if self.ticks_step:
            # once per forward per rank, on every path (fused or native), before anything can return
            hooks.tick()
        r, reason, state, st = self._prepare(ctx, x, x_scale)
        if r.plan is None:
            out = self.native.full(self, x, x_scale, ctx)
            return out, self._decision(ctx, r, reason, "native")
        outs = self._execute(r, x, x_scale, ctx, state, st, _out_dtype())
        if r.n_fused >= 2:
            out = self.native.after_prologue(self, outs, state, ctx)
        else:
            out = self._remainder_after_exchange(
                r,
                outs,
                ctx,
                lambda packed: self.native.after_exchange(self, packed, ctx),
            )
        return out, self._decision(ctx, r, reason, "fused", st)


class _ShimFwdCtx:
    """Just enough forward context for ``gemm_fusion_prologue_state`` before
    capture: the bound KV caches and a slot-mapping holder."""

    def __init__(self, kv_cache_data, slot_mapping, impl):
        self.kv_cache_data = kv_cache_data
        self.attn_metadata = SimpleNamespace(slot_mapping=slot_mapping)


def _out_dtype():
    from atom.config import get_current_atom_config

    return get_current_atom_config().torch_dtype
