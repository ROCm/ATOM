# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The single-launch step's runner: per-width builds, the bound pointer tables, one launch a step.

What ``ModelRunner`` asks of it on a served step (``ModelRunner._single_step``):
- ``run_step(running_bs, forward_context)``: the step metadata (``write_step_meta`` / ``write_draft_step_meta``) and the
  one launch, on the current stream (graph-capturable: every pointer it binds stays put);
- ``verify_outputs(bs)``: the rejection sampler's ``(output_token_ids [bs, k + 1] int32, -1 past the emitted tokens;
  verdict [3, bs] int32: accepted drafts, rejected drafts, anchor row)``;
- ``draft_ids(bs)``: the next step's draft tokens, as ``Drafter.propose`` returns them.

Construction (collective over the TP group, ``LazyRunner``): the kernel modules (``kernels/``: the whole step is
``kernels.layer4l_step_fly``), the scratch (``buffers.scratch_bytes``, once at the widest step),
the peer receive buffers (``SymmetricPeerBuffer``), the routed-expert planes (``planes.PLANES``). The weight tables
and the step metadata (``buffers``: WEIGHT / STEP) are not bound yet, so construction still refuses at its end and
``LazyRunner`` keeps the production path."""

import importlib
import logging
import os
import sys
import time

import numpy as np
import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.models.deepseek_v41.mono.gfx1250_single import buffers, planes
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.peer_memory import SymmetricPeerBuffer
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _base_model(model):
    for m in model.modules():
        if hasattr(m, "embed") and hasattr(m, "head") and hasattr(m, "norm") and hasattr(m, "layers"):
            return m
    raise MonoUnsupported("no V4.1 base model")


def model_dir():
    """The checkpoint directory (the engine's ``--model``), for the bring-up checks."""
    from atom.config import get_current_atom_config

    return get_current_atom_config().model

KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
NL, NDR = 40, 3
ROWS_PER_REQ = 6  # a DSpark verify request's target rows (5 drafts + 1)
TP = 4
# the receive regions: 4 TP regions x 4 sources x 40960 16-byte words, the sampling and Markov exchange after them
# (step_tail_ct3 SOFF / SOFF2), 2 MB-rounded, doubled for the step epoch
# the recipe's draft temperature scale (dspark_prob_sample.DRAFT_TSCALE): q = softmax(z * invT / TSCALE)
DRAFT_TSCALE = float(os.environ.get("ATOM_DSPARK_PROB_DRAFT_TSCALE", "1.0"))
RECV_BYTES = 2 * (
    (16 * 40960 * 16 + 4 * 4 * 48 * 256 * 16 + (2 << 20) - 1) // (2 << 20) * (2 << 20)
)


def import_kernels() -> dict:
    """The kernel modules (this package's kernels/: the whole step is layer4l_step_fly)."""
    if KERNELS not in sys.path:
        sys.path.insert(0, KERNELS)
    names = (
        "hx",
        "attn_stages",
        "ffn_stages",
        "moe_stages",
        "ctx_stages",
        "head_fly",
        "sample4r_fly",
        "sample4p_fly",
        "markov_prob_fly",
        "step_tail_ct3",
        "layer4l_step_fly",
    )
    return {n: importlib.import_module(n) for n in names}


class _StepTimer:
    """Bring-up (ATOM_MONO_SINGLE_TIMING=1): where a served step's time goes. Events on the compute stream at the step's
    entry (``a``), just before the kernel (``k``) and after it (``z``); host wall times at the same points. Logged every
    ``every`` steps (GPU spans resolved once the events complete, no sync on the hot path)."""

    def __init__(self, rank, every=64):
        import time

        self.time, self.rank, self.every = time, rank, every
        self.pend, self.acc, self.n, self.last_z, self.last_wall = [], {}, 0, None, None
        self.pool = []   # pinned host copies of the kernel's stamps (st [256][32] int64, 100 MHz)

    # the kernel's stamp slots (max over CTAs): 0 layer 0 start, 12 layer 1 start, 25 target end / head, 19 sampling,
    # 20 verify done (then the draft context / metadata), 26 draft layers start, 22 draft head done, 24 Markov done
    SPANS = (("k_layer0", 0, 12), ("k_target_per_layer", 12, 25), ("k_head", 25, 19), ("k_sample", 19, 20),
             ("k_verify", 20, 21), ("k_ctx_meta", 21, 26), ("k_draft3_dhead", 26, 22), ("k_markov", 22, 24),
             ("k_all", 0, 24))

    def copy_st(self, rec, st):
        buf = self.pool.pop() if self.pool else torch.empty(256 * 32, dtype=torch.int64, pin_memory=True)
        buf.copy_(st.view(torch.int64)[: 256 * 32], non_blocking=True)
        rec["st"] = buf

    def mark(self, name, rec):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        rec[name] = ev
        rec["w_" + name] = self.time.perf_counter()

    def step(self, rec, rows):
        rec["prev_z"], rec["prev_wall"], rec["rows"] = self.last_z, self.last_wall, rows
        self.last_z, self.last_wall = rec["z"], rec["w_a"]
        self.pend.append(rec)
        while self.pend and self.pend[0]["z"].query():
            r = self.pend.pop(0)
            d = dict(gpu_prep=r["a"].elapsed_time(r["k"]), kernel=r["k"].elapsed_time(r["z"]),
                     host_prep=1e3 * (r["w_k"] - r["w_a"]), host_launch=1e3 * (r["w_z"] - r["w_k"]))
            prev = r["w_a"]
            for nm, tt in r.get("hm", []):
                d["h_" + nm] = 1e3 * (tt - prev)
                prev = tt
            if "pre" in r:
                d["h_runstep_pre"] = 1e3 * r["pre"][0]
                d["h_engram_stage"] = 1e3 * r["pre"][1]
            if r["prev_z"] is not None:
                d["gpu_gap"] = r["prev_z"].elapsed_time(r["a"])
                d["host_period"] = 1e3 * (r["w_a"] - r["prev_wall"])
            if "st" in r:
                m = r["st"].view(256, 32).max(0).values.double()
                for nm, a_, b_ in self.SPANS:
                    us = float(m[b_] - m[a_]) / 100.0
                    d[nm] = us / (39 if nm == "k_target_per_layer" else 1) / 1e3   # ms (per layer: ms too)
                self.pool.append(r["st"])
            for k_, v in d.items():
                self.acc[k_] = self.acc.get(k_, 0.0) + v
            self.acc["rows"] = self.acc.get("rows", 0) + r["rows"]
            self.n += 1
            if self.n % self.every == 0:
                logger.warning("V4.1 single-launch timing (rank %d, %d steps, ms/step): %s", self.rank, self.every,
                               {k_: round(v / self.every, 3) for k_, v in self.acc.items()})
                self.acc = {}


class SingleStepRunner:
    def __init__(self, model, drafter, atom_config) -> None:
        if get_tensor_model_parallel_world_size() != TP:
            raise MonoUnsupported(
                f"TP {get_tensor_model_parallel_world_size()} (the step kernel is TP{TP})"
            )
        self.rank = get_tensor_model_parallel_rank()
        self.device = torch.device("cuda", torch.cuda.current_device())
        cache = get_forward_context().attn_metadata.cache
        window = cache.geometry.window(0, cache.num_pages)
        try:
            self.K = import_kernels()
        except (
            Exception
        ) as err:  # a kernel source that does not import: the deployment cannot run it
            raise MonoUnsupported(
                f"step kernel import: {type(err).__name__}: {err}"
            ) from err
        # the kernel's ring slots (window + block width) and pool row width (bf16 elements) are compile-time constants
        AS = self.K["attn_stages"]
        if int(window.ring_slots) != AS.KG_RING:
            raise MonoUnsupported(f"window ring {window.ring_slots} slots (the step kernel's: {AS.KG_RING})")
        miss = buffers.check_catalog(self.K)
        if miss:
            raise MonoUnsupported(f"pointer catalog out of date: {miss[:4]}")
        self.scratch = {
            n: torch.zeros(b, dtype=torch.uint8, device=self.device)
            for n, b in buffers.scratch_bytes(self.K, NL + NDR).items()
        }
        # fine-grained, not uncached: the step's few-word peer pushes (Markov, sampling) waited ~80-100 us each for an
        # uncached store to become visible (CT12 11-08: the fixture's fine-grained receive buffers did not)
        self.peer = SymmetricPeerBuffer(
            RECV_BYTES, rank=self.rank, npes=TP, group=get_tp_group().cpu_group, fine_grained=True
        )
        self.recv = [int(a) for a in self.peer.addresses.tolist()]
        # the routed experts a layer, as the loader packed them (``planes.capture``): target "layers.{i}.ffn", the
        # draft's "mtp.{j}.ffn"
        want = [f"layers.{i}.ffn" for i in range(NL)] + [
            f"mtp.{j}.ffn" for j in range(NDR)
        ]
        missing = [p for p in want if not any(k.endswith(p) for k in planes.PLANES)]
        if missing:
            raise MonoUnsupported(
                f"no expert plane for {missing[:3]} (planes.capture not run at load)"
            )
        self.planes = {
            p: next(v for k, v in planes.PLANES.items() if k.endswith(p)) for p in want
        }
        logger.info(
            "V4.1 single-launch step: %.1f MB scratch, %.1f MB receive, %d planes (rank %d)",
            sum(t.numel() for t in self.scratch.values()) / 2**20,
            RECV_BYTES / 2**20,
            len(self.planes),
            self.rank,
        )
        check = os.environ.get("ATOM_MONO_SINGLE_CHECK_PLANES")
        if (
            check
        ):  # bring-up: a layer's packed planes against the investigation packer's files
            self._check_planes(check, model, drafter)
        if os.environ.get("ATOM_MONO_SINGLE_CHECK_ROWS"):  # bring-up: layer 3's table row against the checkpoint
            self._check_rows(model, int(os.environ["ATOM_MONO_SINGLE_CHECK_ROWS"]))
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import StepTables

        self.tables = StepTables(
            self.K, model, drafter, self.scratch, self.recv, self.rank, get_tp_group().device_group
        )
        logger.info(
            "V4.1 single-launch step tables: fp %d, tp %d, ctx %d pointers, %d layer rows (rank %d)",
            self.tables.fptab.numel(),
            self.tables.tp.numel(),
            self.tables.ctxtab.numel(),
            len(self.tables.rows),
            self.rank,
        )
        self.model, self.drafter = model, drafter
        self.base = _base_model(model)
        self.builds = {}
        # ATOM_MONO_SINGLE_GRAPH (default 1): the step's device preparation and first launch as one graph a width
        # (``launch``): the ~20 small preparation operations cost ~0.3 ms of host time and GPU gaps a step otherwise
        self.graph_prep = os.environ.get("ATOM_MONO_SINGLE_GRAPH", "1") == "1"
        self._graphs, self._seen = {}, {}
        self._fast = {}  # rows -> the width's launchers (``_fast_launchers``)
        # ATOM_MONO_SINGLE_SPLIT (default 1): the step as two launches (``prepare``); the verdict's copies and the output D2H wait
        # for the first only (``verify_outputs`` / ``verify_ready``)
        self.split = os.environ.get("ATOM_MONO_SINGLE_SPLIT", "1") == "1"
        self._ev_verify = torch.cuda.Event()
        self.verify_ready = torch.cuda.Event()
        self._verify_stream = torch.cuda.Stream()
        self._timer = _StepTimer(self.rank, int(os.environ.get("ATOM_MONO_SINGLE_TIMING_EVERY", "64"))) if os.environ.get("ATOM_MONO_SINGLE_TIMING") == "1" else None
        self._ptrlog_on = os.environ.get("ATOM_MONO_SINGLE_PTRLOG") == "1"
        self._eng_rows = None
        self._noq = os.environ.get("ATOM_MONO_SINGLE_NOQ") == "1"   # (diagnostic) every draft verified one-hot
        self.shadow = bool(os.environ.get("ATOM_MONO_SINGLE_SHADOW"))
        # bring-up: the launch's target layers (< NL: no step tail / draft; its xh is the normed collapse after layer
        # shadow_layers - 1 with layer shadow_layers' attn_norm, compared to the production state there)
        self.shadow_layers = int(os.environ.get("ATOM_MONO_SINGLE_SHADOW_NL", str(NL)))
        # serving (the launch's sampled ids, verdict and drafts replace the production step's): opt-in while the
        # probability-ratio verify is not in the kernel
        self.serve = os.environ.get("ATOM_MONO_SINGLE_SERVE") == "1"
        if not (self.shadow or self.serve):
            raise MonoUnsupported("single-launch step: ATOM_MONO_SINGLE_SHADOW=1 (compare) or ATOM_MONO_SINGLE_SERVE=1")
        if self.serve and self.shadow_layers != NL:
            raise MonoUnsupported("single-launch step: serving runs all the layers (ATOM_MONO_SINGLE_SHADOW_NL unset)")
        # every width built now (each a compile or a disk-cache load, then a TP barrier: ``prepare``), not on its first
        # step: a width first met mid-serve stalled every request for its JIT (~18 s cold)
        if os.environ.get("ATOM_MONO_SINGLE_PRECOMPILE", "1") == "1":
            import time

            t0 = time.perf_counter()
            for rows in range(ROWS_PER_REQ, buffers.T_MAX + 1, ROWS_PER_REQ):
                self.prepare(rows)
            logger.info("V4.1 single-launch step: %d widths built in %.1f s (rank %d)", len(self.builds),
                        time.perf_counter() - t0, self.rank)

    def _check_planes(self, directory, model, drafter) -> None:
        """``{directory}/L3_r{rank}.bin`` (routed, 384 experts) and ``L3_sh_r{rank}.bin`` (shared, FP8) against
        layer 3's planes packed here; logged, not raised (a bring-up check)."""
        routed = self.planes["layers.3.ffn"].cpu().numpy()
        ref = np.fromfile(
            os.path.join(directory, f"L3_r{self.rank}.bin"), dtype=np.uint8
        )
        mods = [
            m
            for n, m in model.named_modules()
            if n.endswith("layers.3.ffn.shared_experts")
        ]
        shared = planes.pack_shared(mods[0]).cpu().numpy() if mods else None
        ref_sh = np.fromfile(
            os.path.join(directory, f"L3_sh_r{self.rank}.bin"), dtype=np.uint8
        )
        logger.warning(
            "V4.1 single-launch step plane check (rank %d): routed %s (%d B vs %d B), shared %s",
            self.rank,
            routed.size == ref.size and bool((routed.reshape(-1) == ref).all()),
            routed.size,
            ref.size,
            (
                None
                if shared is None
                else (
                    shared.size == ref_sh.size
                    and bool((shared.reshape(-1) == ref_sh).all())
                )
            ),
        )

    def _check_rows(self, model, li) -> None:
        """Layer li's layer-table row (``layer_weights.layer_row``) against the checkpoint read here: the FP8 rank shards
        and scales byte for byte, the f32 / bf16 vectors exactly, the hc fn's hi + lo to its f32; logged, not raised."""
        import json

        from safetensors import safe_open

        from atom.models.deepseek_v41.mono.gfx1250_single import layer_weights

        blocks = [m for n, m in model.named_modules() if n.endswith(f"layers.{li}")]
        nxt = [m for n, m in model.named_modules() if n.endswith(f"layers.{li + 1}")]
        row = layer_weights.layer_row(blocks[0], nxt[0])
        root = model_dir()
        wm = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]

        def ck(name):
            with safe_open(os.path.join(root, wm[name]), "pt") as f:
                return f.get_tensor(name)

        r, P = self.rank, f"layers.{li}."
        res = {}

        def same(k, got, want):
            got = got.cpu()
            want = want.cpu()
            res[k] = bool(got.shape == want.shape and torch.equal(got.view(torch.uint8), want.contiguous().view(torch.uint8)))

        def slab(t, rows=None, cols=None):
            if rows is not None:
                n = t.shape[0] // 4
                t = t[r * n:(r + 1) * n]
            if cols is not None:
                n = t.shape[1] // 4
                t = t[:, r * n:(r + 1) * n]
            return t

        wq = torch.cat([ck(P + "attn.wq_a.weight"), ck(P + "attn.wkv.weight")])
        wqs = torch.cat([ck(P + "attn.wq_a.scale"), ck(P + "attn.wkv.scale")]) if (P + "attn.wq_a.scale") in wm else None
        same("wqkv", row["wqkv"], wq)
        if wqs is not None:
            same("wqkv_s", row["wqkv_s"], wqs)
        same("wqb", row["wqb"], slab(ck(P + "attn.wq_b.weight"), rows=1))
        same("woa", row["woa"], slab(ck(P + "attn.wo_a.weight"), rows=1))
        same("wob", row["wob"], slab(ck(P + "attn.wo_b.weight"), cols=1))
        for k, n in (("wqb_s", "attn.wq_b.scale"), ("woa_s", "attn.wo_a.scale")):
            if P + n in wm:
                same(k, row[k], slab(ck(P + n), rows=1))
        if P + "attn.wo_b.scale" in wm:
            same("wob_s", row["wob_s"], slab(ck(P + "attn.wo_b.scale"), cols=1))
        q = ck(P + "attn.q_norm.weight").float()
        same("qnw", row["qnw"][: q.numel()], q)
        res["qnw_pad_zero"] = bool((row["qnw"][q.numel():] == 0).all())
        same("kvnw", row["kvnw"], ck(P + "attn.kv_norm.weight").float())
        same("sink", row["sink"], slab(ck(P + "attn.attn_sink").view(-1, 1), rows=1).view(-1).float())
        same("hsc", row["hsc"], ck(P + "hc_ffn_scale").float())
        same("hbs", row["hbs"], ck(P + "hc_ffn_base").float())
        same("nw", row["nw"], ck(P + "ffn_norm.weight").to(torch.bfloat16))
        same("gate", row["gate"], ck(P + "ffn.gate.weight").to(torch.bfloat16))
        same("gbias", row["gbias"], ck(P + "ffn.gate.bias").float())
        fn = ck(P + "hc_ffn_fn").float().reshape(-1)
        hl = (row["fnh"].float() + row["fnl"].float()).cpu()[: fn.numel()]
        res["fn_hi_lo_maxrel"] = float(((hl - fn).abs() / fn.abs().clamp_min(1e-30)).max())
        Q = f"layers.{li + 1}."
        same("ahsc", row["ahsc"], ck(Q + "hc_attn_scale").float())
        same("ahbs", row["ahbs"], ck(Q + "hc_attn_base").float())
        same("anw", row["anw"], ck(Q + "attn_norm.weight").to(torch.bfloat16))
        logger.warning("V4.1 single-launch step row check (layer %d, rank %d): %s", li, self.rank, json.dumps(res))

    def prepare(self, rows: int) -> bool:
        if rows % 6 or not 0 < rows <= buffers.T_MAX:
            return False
        if rows not in self.builds:
            M = self.K["layer4l_step_fly"]
            nl = self.shadow_layers
            kw = dict(NL=nl, ndr=NDR if nl == NL else 0, trd=rows // 6 * 5)
            if self.split:   # two launches: target..verification, then the draft (the host reads the verdict between)
                launch = tuple(M.build(rows, self.rank, part=pt, **kw)[0] for pt in ("target", "draft"))
            else:
                launch = (M.build(rows, self.rank, **kw)[0],)
            # compile (JIT or disk cache) before the first launch of this width, then wait for every rank: the step
            # kernel's cross-rank polls give up after 2 s, and ranks compiling a width seconds apart made that first
            # launch time out and compute on stale / zero peer data (KV / index / compressor state stays wrong after)
            from atom.mono.runtime.compile import compile_only

            with compile_only():
                for ln in launch:
                    ln(*self._launch_args(), stream=torch.cuda.current_stream())
            torch.distributed.barrier(group=get_tp_group().cpu_group)
            self.builds[rows] = launch
        return True

    def _fast_launchers(self, rows):
        """The width's launchers. ATOM_MONO_SINGLE_FASTLAUNCH (default 1): each @flyc.jit launcher's first call goes
        through flyc.compile (which launches it) and the CompiledFunction it returns is called after that -- no
        per-call signature binding / cache key / globals check (~0.15 ms of host time a launch)."""
        if os.environ.get("ATOM_MONO_SINGLE_FASTLAUNCH", "1") != "1":
            return [lambda *a, ln=ln: ln(*a[:-1], stream=a[-1]) for ln in self.builds[rows]]
        fast = self._fast.get(rows)
        if fast is None:
            import flydsl.compiler as flyc

            def first(i):
                def call(*a):
                    fn = flyc.compile(self.builds[rows][i], *a)
                    if fn is None:
                        raise MonoUnsupported(f"single-launch step: no compiled launcher for {rows} rows")
                    self._fast[rows][i] = fn
                return call

            fast = self._fast[rows] = [first(i) for i in range(len(self.builds[rows]))]
        return fast

    def _launch_args(self) -> tuple:
        T_ = self.tables
        return (*self.recv, self.scratch["cnt"].data_ptr(), self.scratch["st"].data_ptr(),
                float(self.base.layers[0].attn.softmax_scale), T_.fptab.data_ptr(), T_.tp.data_ptr())

    def _check_timeouts(self) -> None:
        """The kernel's poll-timeout counter (hx.note_timeout: scratch cyc int32 [0] count, [1] last site), read a
        launch late without a sync; any timeout means that step computed on stale peer data."""
        prev = getattr(self, "_to_pend", None)
        if prev is not None and prev[1].query():
            n, site = prev[0].tolist()
            if n != getattr(self, "_to_seen", 0):
                logger.error("V4.1 single-launch step: %d cross-rank poll timeouts so far (rank %d, last site %d)",
                             n, self.rank, site)
                self._to_seen = n
            prev = None
        if prev is None:
            buf = getattr(self, "_to_buf", None)
            if buf is None:
                buf = self._to_buf = torch.empty(2, dtype=torch.int32, pin_memory=True)
            buf.copy_(self.scratch["cyc"].view(torch.int32)[:2], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            self._to_pend = (buf, ev)

    def _step_inputs(self, input_ids, forward_context):
        """The STEP buffers (``tables.StepTables.step``) of this step, on the current stream."""
        engs, _ = self._host_inputs(input_ids, forward_context)
        self._device_inputs(input_ids, forward_context, engs)

    def precheck(self, batch, forward_context) -> bool:
        """Can the kernel serve this step's request shapes (no device work, nothing staged)? The same limits
        ``_sampling`` / ``_host_inputs`` refuse with MonoUnsupported, checked before the step commits to the kernel (an
        exception after the Engram rows are staged left the worker without a fallback: CT11 11-08, a 4.4k-token prompt
        with 4096 outputs past the index bound). Identical on every rank (the batch and its spans are)."""
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import BCOLS, ILW, T as T_MAX

        why = None
        if batch is not None and (np.asarray(batch.top_ks[: batch.total_seqs_num]) != -1).any():
            why = "top_k"
        md = getattr(forward_context, "attn_metadata", None)
        if why is None and self.tables.idx and md is not None:
            step, geom = md.step, md.cache.geometry
            end = max((sp.end for sp in step.requests), default=1)
            bt = step.block_tables
            if any(end // r > ILW for r in (1, 2)):
                why = f"index bound {end} > {ILW}"
            elif -(-end // geom.block_size) > BCOLS or bt.shape[0] > 8:
                why = f"block table {bt.shape[0]} x {-(-end // geom.block_size)}"
            elif self.tables.comp:
                for r in (1, 2):
                    plan = step.plans[r]
                    if plan.num_compress > T_MAX or plan.num_write > T_MAX:
                        why = f"plan {r} rows"
        if why is None:
            self._precheck_why = None
        else:
            if why.split()[0] != getattr(self, "_precheck_why", None):   # once a kind (the bound values change a step)
                self._precheck_why = why.split()[0]
                logger.warning("V4.1 single-launch step: not served (%s), the production path runs (rank %d)", why, self.rank)
            return False
        return True

    def _host_inputs(self, input_ids, forward_context):
        """The step's host-side checks and the rare host-to-device refills (geometry), before the device preparation;
        returns (the Engram layers' embedding rows, the key of every address / shape ``_device_inputs`` captures)."""
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import BCOLS, ILW, T as T_MAX

        md = forward_context.attn_metadata
        cache, step = md.cache, md.step
        rows = input_ids.numel()
        if self._ptrlog_on:
            self._ptrlog(input_ids, md)
        S = self.tables.step
        geom = cache.geometry
        self.tables.set_geometry(cache, self._layer_ids())
        # the window geometry key, recomputed only when the cache's geometry / page count changes (geom.window() every
        # step was most of this function's host time)
        wk = (id(geom), cache.num_pages, cache.num_slots)
        if wk != getattr(self, "_wkey", None):
            window = geom.window(0, cache.num_pages)
            self._wkey, self._gk = wk, (geom.window_size, window.slot_rows, window.ring_slots, cache.num_slots)
        gk = self._gk
        self._hm("hi_geom")
        if gk != getattr(self, "_dgeom_key", None):   # (static for a cache: an H2D copy only when it changes)
            self._dgeom_key = gk
            S["dgeom"][:4].copy_(torch.tensor(gk, dtype=torch.int32))
        rope = self.drafter.model.rope
        if getattr(self, "_ctx_rope", None) is None:   # f32 copies of the draft's RoPE rows, once: a gather a step
            self._ctx_rope = (rope.cos_cache.view(-1, 32).float(), rope.sin_cache.view(-1, 32).float())
        key = [rows, input_ids.data_ptr(), gk, geom.block_size, self.tables._geom_key]
        if self.tables.idx:
            # the step's furthest position, from the host spans (no device read): a row at position p sees (p + 1) // r
            # compressed rows, p < span.end
            end = max((sp.end for sp in step.requests), default=1)
            for r in (1, 2):
                if end // r > ILW:
                    raise MonoUnsupported(f"index bound {end // r} > {ILW}")
            bt = step.block_tables
            if -(-end // geom.block_size) > BCOLS or bt.shape[0] > 8:
                raise MonoUnsupported(f"block table: {bt.shape[0]} requests x {-(-end // geom.block_size)} columns > 8 x {BCOLS}")
            key += [t.data_ptr() for t in (step.positions, step.slots, step.batch_ids, step.visible[1], step.visible[2])]
            key += [bt.data_ptr(), tuple(bt.shape), bt.stride(0)]
            if self.tables.comp:
                for r in (1, 2):
                    plan = step.plans[r]
                    for src, nl in ((plan.compress_plan_gpu, plan.num_compress), (plan.write_plan_gpu, plan.num_write)):
                        if nl > T_MAX:
                            raise MonoUnsupported(f"plan {r}: {nl} rows > {T_MAX}")
                        key += [src.data_ptr(), tuple(src.shape)]
        self._hm("hi_key")
        # the Engram layers' embedding rows (staged on Engram's side stream: ``get`` makes this stream wait for them)
        emb = getattr(md, "engram_embeddings", None)
        engs = {}
        for k, w in self.tables.engram.items():
            e = (dict.get(emb, k) if self._eng_rows is not None else emb.get(k)) if emb is not None else None
            if e is None or e.shape[-2] < rows or e.shape[-1] != w["engemb"].shape[1]:
                raise MonoUnsupported(f"engram layer {k}: embedding rows {None if e is None else tuple(e.shape)}")
            engs[k] = e
            key += [e.data_ptr(), tuple(e.shape)]
        er = self._eng_rows
        if er is not None:   # the staged Engram inputs the graph captures (fixed addresses: EngramStep)
            st = er.staging.step
            key += [er.width, id(er.staging)] + [None if t_ is None else t_.data_ptr() for t_ in (
                st.input_ids, st.live, st.dead, st.batch_ids, st.cu_seqlens, st.positions, st.cursor, st.history_index,
                st.candidates)]
        self._hm("host_inputs")
        return engs, tuple(key)

    def _device_inputs(self, input_ids, forward_context, engs):
        """Every device operation of the step's preparation: from fixed addresses only (captured in the width's
        graph, ``launch``), sizes fixed by the key of ``_host_inputs``."""
        from atom.models.deepseek_v41.mono.step_meta import write_step_meta
        from atom.models.deepseek_v41.mono.gfx1250_single.tables import BCOLS, T as T_MAX

        md = forward_context.attn_metadata
        cache, step = md.cache, md.step
        rows = input_ids.numel()
        req = rows // 6
        S = self.tables.step
        geom = cache.geometry
        window = geom.window(0, cache.num_pages)
        S["sparam"].copy_(self._sp_pin, non_blocking=True)
        write_step_meta(step, geom, window, S["kvw"][:rows], S["kmeta"][: rows * 4].view(rows, 4), {}, (0, rows))
        S["rpos"][:rows].copy_(step.positions[:rows])
        S["tids"][:rows].copy_(input_ids.view(-1))
        S["dids"][: req * 5].copy_(input_ids.view(req, 6)[:, 1:].reshape(-1))
        # each request's state slot (the draft context's rows): not its batch position once requests have finished
        S["dgeom"][8 : 8 + req].copy_(step.slots[:req])
        S["ctx_kvw"][:rows].copy_(S["kvw"][:rows])
        pos = step.positions[:rows]
        torch.index_select(self._ctx_rope[0], 0, pos, out=S["ctx_rcos"][: rows * 32].view(rows, 32))
        torch.index_select(self._ctx_rope[1], 0, pos, out=S["ctx_rsin"][: rows * 32].view(rows, 32))
        if self.tables.idx:
            S["ibat"][:rows].copy_(step.batch_ids[:rows])
            for r in (1, 2):
                S[f"ibound{r}"][:rows].copy_(step.visible[r][:rows])
            # the requests' page tables: a fixed BCOLS columns (past the step's reach: never read)
            bt = step.block_tables
            n = bt.shape[0]
            cc = min(bt.shape[1], BCOLS)
            S["btab"][: n * BCOLS].view(n, BCOLS)[:, :cc].copy_(bt[:, :cc])
            if self.tables.comp:   # the compress / ring-write plans, -1 padded (pos < 0: skipped); the plan's rows past
                # its live count are sentinels (its fixed-capacity layout) or absent (a tight one)
                for r in (1, 2):
                    plan = step.plans[r]
                    for name, src in (("cplan", plan.compress_plan_gpu), ("wplan", plan.write_plan_gpu)):
                        dst = S[f"{name}{r}"]
                        dst.fill_(-1)
                        m = min(src.shape[0], T_MAX)
                        if m:
                            dst[: m * 4].copy_(src[:m].reshape(-1))
                S["slots"][:n].copy_(step.slots[:n])
        for k, w in self.tables.engram.items():
            e = engs[k]
            w["engemb"][:rows].copy_(e.reshape(-1, e.shape[-1])[:rows])
        self.scratch["cnt"].zero_()
        self.scratch["ss"].zero_()
        # Markov's per-(position, request) global argmax is an atomic max: it must restart from INT64_MIN every step
        # (as the fixture does), else the drafts keep earlier steps' winners
        self.scratch["mbest"].view(torch.int64).fill_(-(1 << 63))
        self._hm("device_inputs")

    def _ptrlog(self, input_ids, md) -> None:
        """Bring-up (ATOM_MONO_SINGLE_PTRLOG=1): the addresses of ATOM's per-step inputs the launch's preparation reads
        (graph capture of the preparation needs them fixed), the first 6 steps of each width."""
        n = self.__dict__.setdefault("_ptrlog_n", {})
        rows = input_ids.numel()
        n[rows] = n.get(rows, 0) + 1
        if n[rows] > 6:
            return
        st = md.step
        p = dict(ids=input_ids.data_ptr(), pos=st.positions.data_ptr(), slots=st.slots.data_ptr(),
                 bid=st.batch_ids.data_ptr(), bt=st.block_tables.data_ptr(), v1=st.visible[1].data_ptr(),
                 v2=st.visible[2].data_ptr())
        for r in (1, 2):
            pl = st.plans[r]
            p[f"cp{r}"] = pl.compress_plan_gpu.data_ptr()
            p[f"wp{r}"] = pl.write_plan_gpu.data_ptr()
        emb = getattr(md, "engram_embeddings", None)
        for k in self.tables.engram:
            e = dict.get(emb, k) if isinstance(emb, dict) else None   # (no consume: a raw lookup)
            p[f"eng{k}"] = e.data_ptr() if e is not None else 0
        logger.warning("PTRLOG rank %d rows %d step %d: %s", self.rank, rows, n[rows], {k: hex(v) for k, v in p.items()})

    def _hm(self, name) -> None:
        """Bring-up timing (ATOM_MONO_SINGLE_TIMING): a host time mark inside the step's preparation."""
        r = getattr(self, "_t_rec", None)
        if r is not None:
            import time

            r.setdefault("hm", []).append((name, time.perf_counter()))

    def _sampling(self, batch, req) -> None:
        """Each target row's sampling from its request (``ScheduledBatch`` temperatures / top_ps / top_ks), as the
        recipe's prob drafting + block verification (``dspark_prob_sample``, PROB_DRAFT / BLOCK_VERIFY): greedy
        (temperature 0) -> argmax drafts and the argmax match; else the drafts are sampled from q (1 / temperature
        times ``ATOM_DSPARK_PROB_DRAFT_TSCALE``) and verified against p (the top-p set of logits / temperature) by the
        probability ratio. q is the kernel's own (``qlog``) when the request's drafts came from the last kernel launch
        (``hasq``, its row there ``qsrc``), else the one-hot rule. Noise seeds mix a step counter and the batch row.
        Top-k is not in the kernel: refused."""
        if batch is None:
            raise MonoUnsupported("single-launch step: no batch for the sampling parameters")
        temps = np.asarray(batch.temperatures[:req], dtype=np.float32)
        top_ps = np.asarray(batch.top_ps[:req], dtype=np.float32)
        top_ks = np.asarray(batch.top_ks[:req])
        if (top_ks != -1).any():
            raise MonoUnsupported(f"single-launch step: top_k {top_ks.tolist()}")
        rids = list(batch.req_ids[:req])
        self._step_no = getattr(self, "_step_no", 0) + 1
        n = req * ROWS_PER_REQ
        # the batch's static columns (inverse temperatures, top-p, greedy) cached by request set: in steady decode only
        # the seeds and the draft-source columns change a step (rebuilding all with numpy was ~0.06 ms of host time
        # after the verdict)
        skey = (req, tuple(rids), temps.tobytes(), top_ps.tobytes())
        sc = getattr(self, "_sp_static", None)
        if sc is None or sc[0] != skey:
            greedy = temps == 0
            sp = np.zeros((buffers.T_MAX, 8), dtype=np.float32)
            sp[:, 0], sp[:, 1], sp[:, 4] = 1.0, 1.0, 1.0
            sp.view(np.int32)[:, 3] = 1  # rows past the batch: greedy
            inv_t = np.where(greedy, 1.0, 1.0 / np.maximum(temps, 1e-6)).astype(np.float32)
            for c, v in ((0, inv_t), (1, np.where(greedy, 1.0, top_ps)), (4, inv_t / DRAFT_TSCALE)):
                sp[:n, c] = np.repeat(v, ROWS_PER_REQ)  # a request's 6 verify rows share its sampling
            sp.view(np.int32)[:n, 3] = np.repeat(greedy.astype(np.int32), ROWS_PER_REQ)
            rr = np.repeat(np.arange(req, dtype=np.int64), ROWS_PER_REQ)
            sc = self._sp_static = (skey, sp, rr * 0x85EBCA77, rr * 0x27D4EB2F + 0x165667B1)
        sp = sc[1]
        ints = sp.view(np.int32)
        prev = getattr(self, "_qrow", {})
        ints[:n, 5] = np.repeat(np.array([r in prev for r in rids], dtype=np.int32), ROWS_PER_REQ)
        if self._noq:   # (diagnostic, ATOM_MONO_SINGLE_NOQ) as if no draft's q were available: one-hot verification
            ints[:n, 5] = 0
        ints[:n, 6] = np.repeat(np.array([prev.get(r, 0) for r in rids], dtype=np.int32), ROWS_PER_REQ)
        ints[:n, 2] = ((self._step_no * 0x9E3779B1 + sc[2]) & 0x7FFFFFFF).astype(np.int32)
        ints[:n, 7] = ((self._step_no * 0xC2B2AE35 + sc[3]) & 0x7FFFFFFF).astype(np.int32)
        # staged in a fixed pinned buffer; ``_device_inputs`` copies it (non-blocking: a pageable copy waited for the
        # stream to drain -- the previous step's draft launch -- stalling this step's host preparation). Rewritten a
        # step later only after that step's first launch, which follows the copy, has completed (``verify_ready``).
        if getattr(self, "_sp_pin", None) is None:
            self._sp_pin = torch.empty(sp.shape, dtype=torch.float32, pin_memory=True)
        self._sp_pin.numpy()[:] = sp
        # this launch's drafts (and their q in qlog / qids) by request: committed once the launch is issued
        self._qrow_next = {r: i for i, r in enumerate(rids)}

    def forget_drafts(self, req_ids) -> None:
        """These requests' next drafts are not the kernel's (the production drafter ran): no q for their verification."""
        prev = getattr(self, "_qrow", None)
        if prev:
            for r in req_ids:
                prev.pop(r, None)

    def _layer_ids(self):
        ids = getattr(self, "_layer_ids_c", None)
        if ids is None:
            ids = [blk.attn.spec.layer_id for blk in self.tables.target_blocks]
            ids = self._layer_ids_c = ids + [blk.attn.spec.layer_id for blk in self.tables.draft_blocks]
        return ids

    def launch(self, input_ids, forward_context, batch=None) -> None:
        rows = input_ids.numel()
        if not self.prepare(rows):
            raise MonoUnsupported(f"single-launch step: {rows} rows")
        self._sampling(batch, rows // ROWS_PER_REQ)
        self._hm("sampling")
        engs, key = self._host_inputs(input_ids, forward_context)
        if os.environ.get("ATOM_MONO_SINGLE_ADDR_MAP") and not getattr(self, "_addr_mapped", False):
            self._addr_mapped = True
            self._write_addr_map(forward_context)
        if getattr(self, "_t_rec", None) is not None:
            self._timer.mark("k", self._t_rec)
        launches = self._fast_launchers(rows)
        args = (*self._launch_args(), torch.cuda.current_stream())

        er = self._eng_rows

        def first():   # the step's device preparation (the Engram stage first) and its first launch
            if er is not None:
                er.stage()
                # join Engram's side stream BEFORE the launch: the step kernel occupies every WGP and spins on its peers,
                # so a side-stream collective (the embedding all-gather) queued behind it on one GPU while its peers' step
                # kernels wait for this GPU's is a cross-GPU circular wait; the launch also reads the embedding rows
                er.join()
            self._device_inputs(input_ids, forward_context, engs)
            launches[0](*args[:-1], torch.cuda.current_stream())

        g = self._graphs.get(rows)
        if g is not None and g[0] == key:
            g[1].replay()
        elif self.graph_prep and self._seen.get(rows) == key:
            # the same addresses / shapes as the width's previous (eager) step: capture its preparation and first
            # launch, then replay -- a key change (a reallocation, a geometry or plan-capacity change) recaptures
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, capture_error_mode="thread_local"):
                first()
            self._graphs[rows] = (key, gr)
            logger.info("V4.1 single-launch step: captured the preparation graph of %d rows (rank %d)", rows, self.rank)
            gr.replay()
        else:
            self._seen[rows] = key
            first()
        if len(launches) > 1:   # (split) the verdict is ready here; the draft launch follows on the same stream
            self._ev_verify.record()
            launches[1](*args)
        self._check_timeouts()
        self._qrow = self._qrow_next

    def _write_addr_map(self, forward_context) -> None:
        """Bring-up (ATOM_MONO_SINGLE_ADDR_MAP=<dir>): every device range the launch can reach, [start, end) by name, so a
        memory fault's address names its buffer; one JSON a rank."""
        import json

        T_, out = self.tables, []

        def add(name, t):
            if isinstance(t, torch.Tensor) and t.is_cuda:
                span = (max((sz - 1) * st for sz, st in zip(t.shape, t.stride())) + 1 if t.numel() else 0) * t.element_size()
                out.append((t.data_ptr(), t.data_ptr() + span, name))

        def walk(name, o):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(f"{name}.{k}", v)
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    walk(f"{name}[{i}]", v)
            else:
                add(name, o)

        walk("scratch", self.scratch)
        walk("step", T_.step)
        for k in ("isel", "irow", "ilog", "ibmax", "icand", "idx_w", "comp_w", "keep", "src", "ctx_src", "fptab", "ctxtab", "tp",
                  "dummy"):
            walk(k, getattr(T_, k, None))
        for i, a in enumerate(self.recv):
            out.append((a, a + RECV_BYTES, f"recv[{i}]"))
        cache = forward_context.attn_metadata.cache
        add("pool", cache.pool)
        for o, u in getattr(cache, "index_units", {}).items():
            add(f"index_units[{o}].values", u.values)
        for o in getattr(cache, "compress_indices", {}):
            for n, t in zip(("kv", "score"), cache.compress_state(o)):
                add(f"compress_{n}[{o}]", t)
        # not the launch's, but what else runs beside it: Engram's UVA (page-locked host) tables its side stream reads,
        # and the step metadata ATOM fills
        rows_ = getattr(forward_context.attn_metadata, "engram_embeddings", None)
        host = getattr(getattr(rows_, "staging", None), "host", None)
        for lid, table in (getattr(getattr(host, "prefetcher", None), "_tables", None) or {}).items():
            try:
                shard = table.registered_shard()
            except Exception:  # noqa: BLE001 -- not registered (no UVA)
                continue
            for n, t in zip(("weight", "scales"), shard[:2]):
                if isinstance(t, torch.Tensor):
                    span = t.numel() * t.element_size()
                    out.append((t.data_ptr(), t.data_ptr() + span, f"engram_uva[{lid}].{n}"))
        walk("md.step", vars(forward_context.attn_metadata.step) if hasattr(forward_context.attn_metadata.step, "__dict__")
             else {})
        out.sort()
        path = os.path.join(os.environ["ATOM_MONO_SINGLE_ADDR_MAP"], f"addr-map-rank{self.rank}.json")
        with open(path, "w") as f:
            json.dump([(hex(a), hex(b), n) for a, b, n in out], f, indent=0)
        logger.info("single-launch address map: %d ranges -> %s", len(out), path)

    def _draft_window_rows(self, forward_context, rows):
        """(pool view [N][512] bf16, per draft layer the pool rows of this step's ``rows`` target rows): the draft
        context KV the kernel's ctx stage writes and ``DSpark.write_context_kv`` (swa_write) writes after the forward."""
        cache = forward_context.attn_metadata.cache
        pool = cache.pool.view(torch.bfloat16).view(-1, 512)
        kvw = self.tables.step["kvw"][:rows].long()
        lids = self._layer_ids()[NL : NL + NDR]
        out = []
        for lid in lids:
            start = int(cache.geometry.window(lid, cache.num_pages).ring_start)
            out.append(torch.where(kvw >= 0, start + kvw, torch.full_like(kvw, -1)))
        return pool, out

    def shadow_ctx_after_launch(self, forward_context, rows) -> None:
        """Keep the draft context rows the kernel just wrote (compared with the production's next step)."""
        torch.cuda.synchronize()
        pool, idx = self._draft_window_rows(forward_context, rows)
        self._ctx_kernel = [(i, pool[i.clamp_min(0)].clone(), i >= 0) for i in idx]
        self._ctx_pos = forward_context.attn_metadata.step.positions[:rows].tolist()

    def arm_native_draft_capture(self) -> None:
        """Wrap the production drafter's ``draft_hidden`` once: keep its anchors and its normed block hidden (the
        head's input) of the last call, for ``shadow_draft_hidden``."""
        model = self.drafter.model
        if getattr(model, "_single_capture", False):
            return
        orig = model.draft_hidden

        def wrapped(anchor_ids, anchors, *a, **k):
            out = orig(anchor_ids, anchors, *a, **k)
            self._native_draft = (anchor_ids.detach().clone(), anchors.detach().clone(), out[0].detach().float().clone())
            return out

        model.draft_hidden = wrapped
        model._single_capture = True

    def shadow_draft_after_launch(self, rows) -> None:
        """Keep the kernel's draft block of this launch: its anchors (token, position) and normed hidden (dh)."""
        torch.cuda.synchronize()
        req = rows // ROWS_PER_REQ
        anc = self.scratch["anc"].view(torch.int32)[:req].long().clone()
        pos = self.tables.step["rpos_d"][: req * 5].view(req, 5)[:, 0].clone() - 1
        dh = self.scratch["dh"].view(torch.bfloat16).view(-1, 5120)[: req * 5].float().clone()
        self._kernel_draft = (anc, pos, dh)

    def shadow_draft_hidden(self) -> None:
        """The kernel's draft hidden of the step before against the production's (same request: same anchor token and
        position), relL2 / cosine per block row k."""
        kd, nd = getattr(self, "_kernel_draft", None), getattr(self, "_native_draft", None)
        self._native_draft = None
        if kd is None or nd is None or self.rank != 0:
            return
        anc, pos, dh = kd
        n_ids, n_pos, n_h = nd
        n_ids, n_pos = n_ids.view(-1).long(), n_pos.view(-1).long()
        W = 5
        stats = [[] for _ in range(W)]
        matched = 0
        for q in range(anc.numel()):
            hit = torch.nonzero((n_ids == anc[q]) & (n_pos == pos[q])).flatten()
            if hit.numel() != 1:
                continue
            j = int(hit[0])
            matched += 1
            for k in range(W):
                g, r = dh[q * W + k], n_h[j * W + k]
                stats[k].append(float(torch.nn.functional.cosine_similarity(g, r, dim=0)))
        if matched:
            logger.warning("V4.1 single-launch shadow draft hidden (%d requests matched of %d): cos by block row %s",
                           matched, anc.numel(), [round(sum(v) / len(v), 4) for v in stats])
        else:
            logger.warning("V4.1 single-launch shadow draft hidden: no request matched (kernel anchors %s pos %s; "
                           "production %s pos %s)", anc.tolist()[:4], pos.tolist()[:4], n_ids.tolist()[:4], n_pos.tolist()[:4])

    def shadow_ctx_compare(self, forward_context) -> None:
        """The production's draft context rows (written after its forward: compute_draft_kv) at the rows the kernel
        wrote in the step before, against the kernel's values; relL2 per draft layer."""
        prev = getattr(self, "_ctx_kernel", None)
        if not prev or self.rank != 0:
            return
        torch.cuda.synchronize()
        pool = forward_context.attn_metadata.cache.pool.view(torch.bfloat16).view(-1, 512)
        rel = []

        def r(a, b):
            return round(float((a - b).norm() / b.norm().clamp_min(1e-30)), 4)

        for i, got, ok in prev:
            if not ok.any():
                rel.append(None)
                continue
            ref = pool[i.clamp_min(0)].float()[ok]
            g = got.float()[ok]
            cos = torch.nn.functional.cosine_similarity(g, ref, dim=1)
            # the 448 latent columns and the 64 RoPE ones apart; and the kernel's row against the reference's
            # neighbours (a position / ring offset error shows as a better match one row over)
            rel.append(dict(all=r(g, ref), nope=r(g[:, :448], ref[:, :448]), rope=r(g[:, 448:], ref[:, 448:]),
                            cos_min=round(float(cos.min()), 4), rms=(round(float(g.pow(2).mean().sqrt()), 4),
                                                                   round(float(ref.pow(2).mean().sqrt()), 4))))
        logger.warning("V4.1 single-launch shadow draft ctx KV (rows of the step before): per draft layer %s", rel)
        # a row-placement error: each kernel row's best-matching production row within +-8 rows of where it was written
        i, got, ok = prev[0]
        best = {}
        for k in torch.nonzero(ok).flatten().tolist():
            g = got[k].float()
            cs = []
            for d in range(-8, 9):
                ref = pool[int(i[k]) + d].float()
                cs.append(float(torch.nn.functional.cosine_similarity(g, ref, dim=0)))
            d = max(range(17), key=lambda j: cs[j]) - 8
            best[d] = best.get(d, 0) + 1
        cos0 = [round(float(torch.nn.functional.cosine_similarity(got[k].float(), pool[int(i[k])].float(), dim=0)), 3)
                if bool(ok[k]) else None for k in range(len(i))]
        logger.warning("V4.1 single-launch shadow draft ctx KV row cos (draft layer 1, by row: request-row k = row %% 6, "
                       "position %s): %s", getattr(self, "_ctx_pos", None), cos0)
        cache = forward_context.attn_metadata.cache
        geo = [(int(w.ring_start), int(w.slot_rows), int(w.ring_slots)) for w in
               (cache.geometry.window(lid, cache.num_pages) for lid in self._layer_ids()[NL - 1 : NL + NDR])]
        logger.warning("V4.1 single-launch shadow draft ctx KV: best-matching row offset (draft layer 1) %s; "
                       "windows (ring_start, slot_rows, ring_slots) of the last target layer and the draft layers %s",
                       dict(sorted(best.items())), geo)

    def shadow_taps(self, rows) -> None:
        """The kernel's draft-context taps (aux8 / auxs: the inputs of target layers 37..39, MXFP8 with 32-wide E8M0
        groups) against the production drafter's captured taps (``Drafter._aux_buffers``, bf16) of this forward."""
        bufs = getattr(self.drafter, "_aux_buffers", None)
        if not bufs or self.rank != 0:
            return
        torch.cuda.synchronize()
        H = 5120
        a8 = self.scratch["aux8"].view(torch.float8_e4m3fn)[: rows * 3 * H].view(rows, 3 * H).float()
        code = self.scratch["auxs"].view(torch.uint8)[: rows * 3 * H // 32].view(rows, 3 * H // 32).float()
        got = (a8.view(rows, -1, 32) * torch.exp2(code - 127)[..., None]).view(rows, 3, H)
        rel = []
        for j, b in enumerate(bufs[:3]):
            ref = b[:rows].float()
            rel.append(round(float((got[:, j] - ref).norm() / ref.norm().clamp_min(1e-30)), 4))
        logger.warning("V4.1 single-launch shadow taps (%d rows): relL2 per tap (layers 37 / 38 / 39 inputs) %s", rows, rel)

    def shadow_drafts(self, input_ids, forward_context) -> None:
        """The kernel's drafts of the step before (``mids``, by request slot) against the production drafter's, which
        this step's verify rows carry (rows q·6 + 1..5), for the requests whose anchor token agrees (row q·6); per
        draft position, logged. Then this step's drafts are kept for the next call."""
        torch.cuda.synchronize()
        step = forward_context.attn_metadata.step
        req = input_ids.numel() // ROWS_PER_REQ
        ids = input_ids.view(req, ROWS_PER_REQ).cpu()
        slots = step.slots[:req].tolist()
        prev = getattr(self, "_prev_drafts", {})
        same = [0] * 5
        n = 0
        for q, sl in enumerate(slots):
            if sl in prev and prev[sl][0] == int(ids[q, 0]):
                n += 1
                for k in range(5):
                    same[k] += int(prev[sl][1][k] == int(ids[q, 1 + k]))
        if n and self.rank == 0:
            logger.warning(
                "V4.1 single-launch shadow drafts (%d requests with the same anchor): kernel = production by position %s",
                n, [round(c / n, 3) for c in same],
            )
        out = self.scratch["outtok"].view(torch.int32)[: 8 * ROWS_PER_REQ].view(8, ROWS_PER_REQ)[:req].cpu()
        acc = self.scratch["verdict"].view(torch.int32)[:8][:req].cpu()
        mids = self.scratch["mids"].view(torch.int32)[: 8 * 5].view(8, 5)[:req].cpu()
        self._prev_drafts = {sl: (int(out[q, int(acc[q])]), mids[q].tolist()) for q, sl in enumerate(slots)}

    def shadow_selection(self, rows, forward_context) -> None:
        """The kernel's top-512 of each selecting layer it ran (``tables`` isel: index columns; irow: pool rows) against
        the production scorer's ``step.selected[layer]``; logged as the mean / min per-row overlap of the two sets."""
        T_ = self.tables
        if not T_.idx:
            return
        torch.cuda.synchronize()
        step = forward_context.attn_metadata.step
        for Ly in sorted(T_.idx_w):
            if Ly >= self.shadow_layers:
                continue
            nat = step.selected.get(T_.specs[Ly].layer_id)
            if nat is None:
                continue
            nat = nat.reshape(-1, nat.shape[-1])[:rows].long().cpu()
            sel = T_.isel[Ly].view(-1, 512)[:rows].long().cpu()
            row = T_.irow[Ly].view(-1, 512)[:rows].long().cpu()
            out = {}
            for name, got in (("isel", sel), ("irow", row)):
                ov = []
                for q in range(rows):
                    a = set(nat[q][nat[q] >= 0].tolist())
                    b = set(got[q][sel[q] >= 0].tolist())
                    ov.append(len(a & b) / max(len(a), len(b), 1))
                out[name] = (sum(ov) / len(ov), min(ov))
            # the pool rows the attention reads: page(btab[request][col // irpp]) * cprow + imain + col % irpp
            L = self.K["layer4l_step_fly"].LKEYS
            irpp, imain, cprow = (T_.rows[Ly][L.index(k)] for k in ("irpp", "imain", "cprow"))
            bt = step.block_tables.long().cpu()
            req = step.batch_ids[:rows].long().cpu().clamp_min(0)
            col = sel.clamp_min(0)
            want = bt[req[:, None], col // irpp] * cprow + imain + col % irpp
            ok = (want == row) | (sel < 0)
            logger.info("single-launch shadow layer %d irow vs host page rows: %d / %d match (irpp %d imain %d cprow %d); "
                        "row0 sel %s irow %s want %s", T_.specs[Ly].layer_id, int(ok.sum()), ok.numel(), irpp, imain, cprow,
                        sel[0, :6].tolist(), row[0, :6].tolist(), want[0, :6].tolist())
            n_nat = int((nat >= 0).sum(1).float().mean())
            n_got = int((sel >= 0).sum(1).float().mean())
            logger.info(
                "single-launch shadow layer %d selection (%d rows): native %d / kernel %d valid a row; overlap "
                "vs isel mean %.4f min %.4f, vs irow mean %.4f min %.4f",
                T_.specs[Ly].layer_id, rows, n_nat, n_got, *out["isel"], *out["irow"],
            )

    def shadow_compare(self, rows, hidden, ref=None, parts=None) -> None:
        """This step's head input (the kernel's xh: target layers + the final collapse and norm) against the production
        forward's ``model.norm(hidden)`` (or ``ref``: the shadow_layers check); logged."""
        torch.cuda.synchronize()
        got = self.scratch["xh"].view(torch.bfloat16).view(-1, 5120)[:rows].float()
        if ref is None:
            ref = self.base.norm(hidden.reshape(-1, 5120)[:rows])
        ref = ref.reshape(-1, 5120)[:rows].float()
        rel = float((got - ref).norm() / ref.norm().clamp_min(1e-30))
        for name in ("axn", "y1", "y2"):  # the last layer's attention / FFN outputs (shadow_layers < NL)
            r = (parts or {}).get(name)
            if r is None:
                continue
            g = self.scratch[name].view(torch.bfloat16).view(-1, 5120)[:rows].float()
            r = r.reshape(-1, 5120)[:rows].float()
            cos = torch.nn.functional.cosine_similarity(g, r, dim=1)
            ratio = g.norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)
            # the per-row agreement, and the worst columns of row 0 (a layout / weight / scale error shows there)
            d0 = (g[0] - r[0]).abs()
            top = torch.topk(d0, 4)
            logger.warning(
                "V4.1 single-launch shadow (rank %d) %s rows cos [%.4f, %.4f] norm ratio [%.4f, %.4f]; row0 worst cols %s got %s ref %s",
                self.rank, name, float(cos.min()), float(cos.max()), float(ratio.min()), float(ratio.max()),
                top.indices.tolist(), [round(float(x), 4) for x in g[0][top.indices]], [round(float(x), 4) for x in r[0][top.indices]],
            )
            logger.warning(
                "V4.1 single-launch shadow (rank %d) %s relL2 %.3e, got rms %.4f ref rms %.4f",
                self.rank, name, float((g - r).norm() / r.norm().clamp_min(1e-30)),
                float(g.pow(2).mean().sqrt()), float(r.pow(2).mean().sqrt()),
            )
        logger.warning(
            "V4.1 single-launch shadow (rank %d, %d rows): xh relL2 %.3e, got rms %.4f ref rms %.4f",
            self.rank, rows, rel, float(got.pow(2).mean().sqrt()), float(ref.pow(2).mean().sqrt()),
        )
        if not parts and hasattr(self.base, "head"):   # the head input: what the logits make of the difference
            lg = self.base.head.get_logits(got.to(torch.bfloat16)).float()
            lr = self.base.head.get_logits(ref.to(torch.bfloat16)).float()
            # the kernel's own head (head_fly over its rank's vocab shard): its logits against the same columns of lg,
            # and its global argmax (the ranks' (max, index) gathered) against its sampled ids
            n = lg.shape[0]
            vr = self.scratch["logits"].numel() // 4 // buffers.T_MAX
            kl = self.scratch["logits"].view(torch.float32).view(buffers.T_MAX, vr)[:n]
            ref_sh = lg[:, self.rank * vr : (self.rank + 1) * vr]
            k_rel = float((kl - ref_sh).norm() / ref_sh.norm().clamp_min(1e-30))
            mx, ix = kl.max(-1)
            g = get_tp_group().device_group
            mxs = [torch.empty_like(mx) for _ in range(get_tp_group().world_size)]
            ixs = [torch.empty_like(ix) for _ in range(get_tp_group().world_size)]
            torch.distributed.all_gather(mxs, mx.contiguous(), group=g)
            torch.distributed.all_gather(ixs, ix.contiguous(), group=g)
            best = torch.stack(mxs).argmax(0)
            k_arg = torch.stack(ixs).gather(0, best[None])[0] + best * vr
            sids_all = self.scratch["sids"].view(torch.int32)[:n].long()
            logger.warning(
                "V4.1 single-launch shadow kernel head (rank %d): logits shard vs head(kernel xh) relL2 %.3e; "
                "sids = argmax(kernel logits) %.4f",
                self.rank, k_rel, float((sids_all == k_arg).float().mean()),
            )
            if self.rank == 0:
                t1 = (lg.argmax(-1) == lr.argmax(-1)).float().mean()
                k5 = torch.topk(lg, 5).indices
                r5 = torch.topk(lr, 5).indices
                o5 = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(k5, r5)) / (5 * lg.shape[0])
                # the kernel's own sampled ids (greedy rows: the argmax of its head over the same xh)
                sids = self.scratch["sids"].view(torch.int32)[: lg.shape[0]].long()
                gr = self.tables.step["sparam"].view(torch.int32)[: lg.shape[0], 3] != 0
                s_own = (sids == lg.argmax(-1))[gr].float().mean() if gr.any() else torch.tensor(float("nan"))
                s_ref = (sids == lr.argmax(-1))[gr].float().mean() if gr.any() else torch.tensor(float("nan"))
                logger.warning(
                    "V4.1 single-launch shadow sampled ids (%d greedy rows): = argmax(head(kernel xh)) %.4f, "
                    "= argmax(head(native)) %.4f",
                    int(gr.sum()), float(s_own), float(s_ref),
                )
                logger.warning(
                    "V4.1 single-launch shadow logits (%d of %d rows): top-1 agree %.4f, top-5 overlap %.4f, logits relL2 %.3e",
                    lg.shape[0], rows, float(t1), o5, float((lg - lr).norm() / lr.norm().clamp_min(1e-30)),
                )

    def run_step(self, input_ids, running_bs: int, forward_context, batch) -> None:
        """The served step: one launch (target layers, head, sampling, verify, draft layers, Markov drafts)."""
        from atom.model_ops.engram.device.staging import EngramStagedRows

        if input_ids.numel() != running_bs * ROWS_PER_REQ:
            raise MonoUnsupported(f"single-launch step: {input_ids.numel()} rows for {running_bs} requests")
        # what the production forward opens and closes around its layers (runtime.v41_begin_forward / end_forward): the
        # Engram stage -- its history snapshot, each verify prefix's cursor candidate (the TentativeState staging that
        # commit_tentative picks from by the verdict's anchors) and the embedding rows' lookup on its side stream
        _w0 = time.perf_counter() if self._timer is not None else None
        md = forward_context.attn_metadata
        if os.environ.get("ATOM_MONO_SINGLE_PROWDUMP"):   # (diagnostic) this step's positions for the dump
            self._dbg_pos = md.step.positions[: input_ids.numel()].clone()
            self._dbg_md = md
        rows = md.engram_embeddings
        staged = isinstance(rows, EngramStagedRows) and not getattr(md, "staged_outside_forward", False)
        # the Engram stage (its lookups, the side-stream gather and the join) runs inside the launch's preparation graph
        # (``launch``): issued op by op here it was ~0.26 ms of host time between the verdict and the next launch
        self._eng_rows = rows if staged else None
        _w1 = time.perf_counter() if self._timer is not None else None
        dbg = os.environ.get("ATOM_MONO_SINGLE_SYNC") == "1"
        if dbg:  # bring-up: the launch's inputs, by checksum, a line a rank (they must agree across the ranks)
            torch.cuda.synchronize()
            S, n = self.tables.step, input_ids.numel()

            def ck(t):
                return int(t.contiguous().view(torch.uint8).to(torch.int64).sum())

            logger.warning(
                "V4.1 single-launch serve inputs (rank %d, %d rows): ids %d pos %d sparam %d kmeta %d kvw %d dids %d "
                "tpep %s cnt %d",
                self.rank, n, ck(input_ids), ck(md.step.positions[:n]), ck(S["sparam"][:n]), ck(S["kmeta"][: n * 4]),
                ck(S["kvw"][:n]), ck(S["dids"]), self.tables.src["tpep"].view(torch.int32).tolist(), ck(self.scratch["cnt"]),
            )
        if dbg:  # the draft context ring rows of this step's positions, before the launch (were they written?)
            cache = md.cache
            req_ = input_ids.numel() // ROWS_PER_REQ
            win = cache.geometry.window(self._layer_ids()[NL], cache.num_pages)
            pos_ = md.step.positions[: input_ids.numel()].long()
            sl_ = md.step.slots[:req_].long().repeat_interleave(ROWS_PER_REQ)
            ring_idx = int(win.ring_start) + sl_ * int(win.slot_rows) + pos_ % int(win.ring_slots)
            pool_ = cache.pool.view(torch.bfloat16).view(-1, 512)
            before = pool_[ring_idx].clone()
        # bring-up bisect of the unsynchronized serve fault: a device sync only before / only after the launch
        if os.environ.get("ATOM_MONO_SINGLE_SYNC_PRE") == "1":
            torch.cuda.synchronize()
        # join Engram's side stream BEFORE the launch: the step kernel occupies every WGP and spins on its peers, so a
        # side-stream collective (the embedding all-gather) queued behind it on one GPU while its peers' step kernels
        # wait for this GPU's is a cross-GPU circular wait; the launch also reads the embedding rows (engemb)
        self._t_rec = {} if self._timer is not None else None
        if self._t_rec is not None:
            self._timer.mark("a", self._t_rec)
            self._t_rec["pre"] = (self._t_rec["w_a"] - _w0, _w1 - _w0)   # (run_step entry -> a, of which the Engram stage)
        self.launch(input_ids, forward_context, batch)
        if self._t_rec is not None:
            self._timer.copy_st(self._t_rec, self.scratch["st"])
            self._timer.mark("z", self._t_rec)
            self._timer.step(self._t_rec, input_ids.numel())
        if os.environ.get("ATOM_MONO_SINGLE_SYNC_POST") == "1":
            torch.cuda.synchronize()
        if dbg:
            torch.cuda.synchronize()
            n = input_ids.numel()
            changed = (pool_[ring_idx] != before).any(-1)
            logger.warning("V4.1 single-launch serve draft ctx rows written by this launch (draft layer 1, by row): %s",
                           changed.int().tolist())
            vr = self.scratch["logits"].numel() // 4 // buffers.T_MAX
            kl = self.scratch["logits"].view(torch.float32).view(buffers.T_MAX, vr)[:n]
            xh = self.scratch["xh"].view(torch.bfloat16).view(-1, 5120)[:n]
            def cks(t):
                return int(t.contiguous().view(torch.uint8).to(torch.int64).sum())

            S = self.tables.step
            logger.warning(
                "V4.1 single-launch serve draft (rank %d): dh %d dlog %d rpos_d %s kmeta_d %s",
                self.rank, cks(self.scratch["dh"]), cks(self.scratch["dlog"]),
                S["rpos_d"][: n // ROWS_PER_REQ * 5].tolist(), S["kmeta_d"][: n // ROWS_PER_REQ * 5 * 4].view(-1, 4)[:, :2].tolist(),
            )
            logger.warning(
                "V4.1 single-launch serve outputs (rank %d): xh %d sids %s local argmax %s",
                self.rank, int(xh.view(torch.int16).to(torch.int64).sum()),
                self.scratch["sids"].view(torch.int32)[:n].tolist(), (kl.argmax(-1) + self.rank * vr).tolist(),
            )
            out, verdict = self.verify_outputs(running_bs)
            logger.warning(
                "V4.1 single-launch serve (rank %d, %d requests (%d scheduled), %d rows, slots %s, positions %s): "
                "outtok %s verdict %s drafts %s",
                self.rank, running_bs, md.step.scheduled_bs, input_ids.numel(), md.step.slots[:running_bs].tolist(),
                md.step.positions[: input_ids.numel() : ROWS_PER_REQ].tolist(), out.tolist(), verdict.tolist(),
                self.draft_ids(running_bs).tolist(),
            )

    def verify_outputs(self, bs: int):
        """``(output_token_ids [bs, 6] int32, -1 past the emitted tokens; verdict [3, bs] int32: accepted drafts,
        rejected drafts, anchor row)`` -- the kernel's verify (step_tail_ct3 / verify_fly), copied out of the scratch
        the next launch rewrites."""
        if not self.split:
            out = self.scratch["outtok"].view(torch.int32)[: 8 * ROWS_PER_REQ].view(8, ROWS_PER_REQ)[:bs].clone()
            verdict = self.scratch["verdict"].view(torch.int32)[:24].view(3, 8)[:, :bs].clone()
            return out, verdict
        # (split) copied on a side stream right after the first launch, so the host's wait for them (the step's
        # output D2H, ``verify_ready``) overlaps the draft launch; the caller's stream waits for the copies
        cur = torch.cuda.current_stream()
        with torch.cuda.stream(self._verify_stream):
            self._verify_stream.wait_event(self._ev_verify)
            out = self.scratch["outtok"].view(torch.int32)[: 8 * ROWS_PER_REQ].view(8, ROWS_PER_REQ)[:bs].clone()
            verdict = self.scratch["verdict"].view(torch.int32)[:24].view(3, 8)[:, :bs].clone()
            self.verify_ready.record()
        out.record_stream(cur)
        verdict.record_stream(cur)
        cur.wait_event(self.verify_ready)
        dd = os.environ.get("ATOM_MONO_SINGLE_PROWDUMP")
        if dd and getattr(self, "_ndump_prow", 0) < int(os.environ.get("ATOM_MONO_SINGLE_PROWDUMP_N", "80")):   # (diagnostic) the verification's per-row p(x), q(x), rm, token
            self._ndump_prow = getattr(self, "_ndump_prow", 0) + 1
            torch.cuda.synchronize()
            n = bs * ROWS_PER_REQ
            sc = self.scratch

            def nanrows(name, dtype, width):   # per row: any NaN / inf (the kernel's draft-side state of this step)
                t_ = sc[name].view(dtype)
                t_ = t_[: (t_.numel() // width) * width].view(-1, width).float()
                return (~torch.isfinite(t_)).any(dim=1).cpu().numpy()
            kvscan = {}
            md_ = getattr(self, "_dbg_md", None)
            if md_ is not None and os.environ.get("ATOM_MONO_SINGLE_KVSCAN") == "1":   # (diagnostic) NaN / inf per layer window ring, per request slot
                cache_ = md_.cache
                g_ = cache_.geometry
                sl_ = md_.step.slots[:bs].long()
                pool_ = cache_.pool.view(torch.bfloat16).view(-1, g_.row_bytes // 2)
                lids_ = self._layer_ids()
                wb = np.zeros((len(lids_), bs, g_.ring_slots), np.int8)
                for li, lid in enumerate(lids_):
                    win = g_.window(lid, cache_.num_pages)
                    ix = int(win.ring_start) + sl_[:, None] * int(win.slot_rows) + torch.arange(int(win.ring_slots), device=sl_.device)[None, :]
                    wb[li] = (~torch.isfinite(pool_[ix].float())).any(dim=-1).cpu().numpy()
                kvscan["win_bad"] = wb
                cb = []
                for owner, _r in g_.owners:
                    try:
                        kv_s, sc_s = cache_.compress_state(owner)
                    except Exception:
                        continue
                    cb.append(np.stack([(~torch.isfinite(kv_s[sl_])).flatten(1).any(1).cpu().numpy(),
                                        (~torch.isfinite(sc_s[sl_])).flatten(1).any(1).cpu().numpy()]))
                if cb:
                    kvscan["cstate_bad"] = np.stack(cb)   # [owner][kv, score][req]
                kvscan["slots"] = sl_.cpu().numpy()
                T_ = self.tables
                nrow = bs * ROWS_PER_REQ
                sb = []
                for Ly in sorted(T_.irow):   # rows whose selected compressed KV rows (irow, this step) hold NaN / inf
                    sel_ = T_.isel[Ly][: nrow * 512].view(nrow, 512).long()
                    row_ = T_.irow[Ly][: nrow * 512].view(nrow, 512).long().clamp(0, pool_.shape[0] - 1)
                    bad_ = (~torch.isfinite(pool_[row_].float())).any(-1) & (sel_ >= 0)
                    sb.append(bad_.any(1).cpu().numpy())
                    if bool(bad_.any()):   # which columns (compressed positions) of the first bad row
                        r0_ = int(bad_.any(1).nonzero()[0])
                        kvscan[f"selbad_cols_{Ly}"] = sel_[r0_][bad_[r0_]].cpu().numpy()
                        kvscan[f"selbad_row_{Ly}"] = np.array([r0_])
                kvscan["sel_layers"] = np.array(sorted(T_.irow))
                kvscan["sel_bad"] = np.stack(sb) if sb else np.zeros(0)
                L_ = self.K["layer4l_step_fly"].LKEYS
                bt_ = md_.step.block_tables.long()
                req_ = md_.step.batch_ids[:nrow].long().clamp_min(0)
                pos_ = md_.step.positions[:nrow].long()
                seen = set()
                for Ly in sorted(T_.irow):   # each KV owner's compressed main rows of every request: NaN columns
                    sp = T_.specs[Ly]
                    if sp.kv_owner in seen or not sp.ratio:
                        continue
                    seen.add(sp.kv_owner)
                    irpp, imain, cprow = (T_.rows[Ly][L_.index(k)] for k in ("irpp", "imain", "cprow"))
                    for q in range(bs):
                        nc = int(pos_[q * ROWS_PER_REQ + ROWS_PER_REQ - 1]) // sp.ratio + 1
                        cols = torch.arange(nc, device=bt_.device)
                        rr = bt_[req_[q * ROWS_PER_REQ], cols // irpp] * cprow + imain + cols % irpp
                        bad_ = (~torch.isfinite(pool_[rr].float())).any(-1)
                        if bool(bad_.any()):
                            kvscan[f"main_bad_{sp.kv_owner}_{q}"] = cols[bad_].cpu().numpy()
            np.savez(os.path.join(dd, f"prow_r{self.rank}_{self._ndump_prow}.npz"), **kvscan,
                     dlog_bad=nanrows("dlog", torch.float32, 32320)[:40], dh_bad=nanrows("dh", torch.bfloat16, 5120)[:48],
                     dR_bad=nanrows("dR", torch.bfloat16, 4 * 5120)[:48], dy_bad=nanrows("dy", torch.bfloat16, 5120)[:48],
                     auxs_255=(sc["auxs"][: 48 * 480].view(48, 480) == 255).any(dim=1).cpu().numpy(),
                     pos=getattr(self, "_dbg_pos", torch.zeros(1)).cpu().numpy(),
                     prow=self.scratch["prow"].view(torch.float32).reshape(-1, 8)[:n].cpu().numpy(),
                     xh_bad=nanrows("xh", torch.bfloat16, 5120)[:48],   # the target head's input rows
                     tids=self.tables.step["tids"].view(torch.int32)[:n].cpu().numpy(),   # this step's input ids
                     verdict=verdict.cpu().numpy(), out=out.cpu().numpy(),
                     dids=self.tables.step["dids"].view(torch.int32)[: bs * 5].cpu().numpy(),
                     sparam=self.tables.step["sparam"].view(torch.float32).reshape(-1, 8)[:n].cpu().numpy(),
                     qids=self.tables.state["qids"].cpu().numpy(),
                     mids=self.scratch["mids"].view(torch.int32)[: 40].cpu().numpy(),
                     qlog=(self.tables.state["qlog"].view(40, -1).cpu().numpy() if self._ndump_prow <= int(os.environ.get("ATOM_MONO_SINGLE_PROWDUMP_QLOG", "12")) else np.zeros(1)))
        return out, verdict

    def draft_ids(self, bs: int):
        """The next step's drafts [bs, 5] (the kernel's Markov block sampling, mids), as ``Drafter.propose`` returns."""
        return self.scratch["mids"].view(torch.int32)[: 8 * 5].view(8, 5)[:bs].to(torch.int64)
