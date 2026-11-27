# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1 C4 decode on gfx1250 TP4: the launch-A seam.

A MoE decoder layer whose sparse attention writes o_proj's MXFP8 input (NVFP4 K/V, the
fast o_proj) runs its post-attention part as one custom op, opaque to torch.compile:

    attention op (eager region: QKV chain, index top-k, sparse PA)
      -> ``minimax_m3_attn_tail``: o_proj -> all-reduce + residual + RMSNorm [-> MoE]

Inside the op every call decides on the host, from shapes only (so every TP rank decides
alike), between

* the fused launch ("launch A", ``launch_a_kernels.<KERNEL>``): a decode step of one
  request x 8 rows (DSpark K=7) whose sparse PA launch the attention op deferred, with the
  k5 one-shot all-reduce as the path production takes for that all-reduce. The kernel runs
  the deferred PA, o_proj and the k5 all-reduce + residual + Gemma RMSNorm in one launch
  with production's bits, on production's k5 context (inboxes, counters);
* production's calls, in production's order (the deferred PA first if there is one), for
  every other call: prefill, other row counts, other formats, no k5.

The PA deferral: while a seam layer's attention impl runs a qualifying decode step, aiter's
gfx1250 PA launch is recorded instead of launched (its arguments are pointers and ints, the
same ones the kernel then receives). A CUDA graph captures whatever the op launches, so a
graph keeps the choice of its capture; the pointers in it are the graph's own buffers.

Which kernel serves the seam is ``KERNEL`` (a module of ``launch_a_kernels``: ``available``,
``hosts(pa_traits)``, ``run(call)``); a new version of a launch is a new module, or new aiter
files behind the same module.
"""

from __future__ import annotations

import ctypes
import dataclasses
import importlib
import logging

import torch

from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")

KERNEL = "mf"  # launch_a_kernels.<KERNEL>: mf (MoE front, Stage O otherwise), stage_o, paop / atta (lose in TP4)
FALLBACK = "stage_o"  # served when KERNEL's aiter files are missing
ROWS = 8  # one request of DSpark K=7
HIDDEN = 6144

# What the op does after the all-reduce + norm (the decoder layer's production path):
TAIL_AR_NORM = 0  # fused_allreduce_gemma_rms_norm; the layer runs the MoE
TAIL_SP_MOE_FRONT = 1  # minimax_m3_sp_moe_front (ATOM_M3_SP_MOE_FRONT): AR + norm + MoE
TAIL_SP_AR_NORM = 2  # sp_fused_allreduce_gemma_rms_norm; the layer runs the MoE


@dataclasses.dataclass
class PendingPA:
    """A sparse PA launch the attention op deferred: aiter's compiled launch, its
    arguments (stream last) and traits, and the MXFP8 output it writes."""

    exe: object
    args: tuple
    traits: object
    qo_ptr: int
    os_ptr: int
    lda: int
    keep: tuple = ()  # the pa_decode call's tensors: alive until the deferred launch is issued

    def launch(self):
        _S["orig_rc"](self.exe, *self.args)


@dataclasses.dataclass
class K5:
    """Production's k5 context (``CustomAllreduce._k5_ctx``): the four inboxes, this
    rank's counters, packs a sender a buffer, rank, row cap."""

    inbox: list
    ctr: int
    slot: int
    rank: int
    max_rows: int


@dataclasses.dataclass
class LaunchACall:
    """Everything a launch-A kernel reads for one layer's call."""

    seam: "Seam"
    pa: PendingPA | None  # None for a kernel that keeps production's PA (Stage O)
    x: torch.Tensor  # E4M3 [m, 2048]: the PA's MXFP8 output, o_proj's input
    x_scale: torch.Tensor  # E8M0 [8, 64]
    residual: torch.Tensor  # bf16 [8, 6144]
    norm_w: torch.Tensor  # [6144], Gemma (the kernel adds 1)
    eps: float
    k5: K5
    ctl: torch.Tensor  # int32 [16], zeroed once, the launch resets it
    topk: object = None  # the deferred index top-k (kernels that define ``index_topk``)
    qkv: object = None  # the deferred QKV chain's operands (kernels that define ``qkv_chain``)


class Seam:
    """One decoder layer's seam: what the op reaches by name."""

    def __init__(self, name, layer):
        self.name = name
        self.layer_num = getattr(layer.self_attn, "layer_num", -1)
        self.o_proj = layer.self_attn.o_proj
        self.moe = layer.block_sparse_moe
        self.norm = layer.post_attention_layernorm
        self.attn_impl = layer.self_attn.attn.impl


_S = {
    "enabled": True,  # measurement builds only flip it (see wire dev hooks)
    "kernel": None,
    "hooked": False,
    "orig_rc": None,
    "armed": None,  # the Seam whose attention impl runs a qualifying step
    "pa_call": None,  # (qo_ptr, os_ptr, lda) of the PA call in flight
    "pa_keep": (),
    "pending": {},  # seam name -> PendingPA
    "topk": {},  # seam name -> deferred index top-k (kernels with ``index_topk``)
    "qkv": {},  # seam name -> deferred QKV chain (kernels with ``qkv_chain``)
    "traits": {},  # id(compiled launch) -> Gfx1250Traits
    "ctl": {},  # device index -> int32 [16]
    "route": {},  # (rows, mode) -> production takes the Gluon-order k5 path
    "logged": set(),
    "front": {},  # id(moe) -> (output ptr, outputs) of the MoE-front launch its layer's op just ran
    "calls": {"fused": 0, "production": 0, "pa_flushed": 0, "moe_front": 0, "moe_front_unused": 0},
}
_SEAMS: dict[str, Seam] = {}


def kernel():
    """The kernel module behind the seam, or None when its aiter files are missing."""
    if _S["kernel"] is None:
        _S["kernel"] = False
        for name in (KERNEL, FALLBACK):
            try:
                mod = importlib.import_module(f"atom.models.minimax_m3.launch_a_kernels.{name}")
            except ImportError:
                continue
            if mod.available():
                _S["kernel"] = mod
                break
    return _S["kernel"] or None


def set_moe_front(moe, out: torch.Tensor, outputs) -> None:
    """A kernel that ran the MoE front of ``moe``'s layer (routes, MXFP4 rows, shared output: ``outputs``) in
    the launch that returned ``out``; the MoE takes them when its input is ``out`` (``take_moe_front``)."""
    _S["front"][id(moe)] = (out.data_ptr(), outputs)


def take_moe_front(moe, hidden_states: torch.Tensor):
    """The MoE-front outputs of this ``moe`` call, or None: only when its layer's op just ran the front and the
    call's input is that launch's output (host-side, in the MoE's dual-stream op)."""
    ent = _S["front"].pop(id(moe), None)
    if ent is None:
        return None
    if hidden_states.data_ptr() != ent[0]:
        _S["calls"]["moe_front_unused"] += 1
        _log_once("front_unused", "MiniMax-M3 launch A: a MoE call's input is not its front launch's output; "
                  "production's MoE")
        return None
    _S["calls"]["moe_front"] += 1
    _log_once("front", "MiniMax-M3 MoE front: one-stream MoE from the front launch (%s)", getattr(moe, "prefix", "?"))
    return ent[1]


def stats() -> dict:
    return dict(_S["calls"])


def set_enabled(on: bool) -> None:
    """Measurement switch (dev builds): off sends every call to production's path."""
    _S["enabled"] = bool(on)


def _log_once(key, msg, *args):
    if key not in _S["logged"]:
        _S["logged"].add(key)
        logger.info(msg, *args)


# ---------------------------------------------------------------- construction
def bind(layer, prefix: str) -> str | None:
    """The seam name of ``layer`` (a MiniMaxM3DecoderLayer), or None when the layer keeps
    production's traced path: a MoE layer whose sparse attention writes the fast o_proj's
    MXFP8 input, TP4, gfx1250, no sequence parallelism / plugin mode, and the kernel's
    aiter files present. Per-step conditions are the op's."""
    from aiter.jit.utils.chip_info import get_gfx

    from atom.distributed.ulysses_sp import get_sp_world_size
    from atom.plugin.prepare import is_plugin_mode
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size

    attn = layer.self_attn
    if not (
        getattr(layer, "is_moe_layer", False)
        and callable(getattr(attn, "attend_mxfp8", None))
        and getattr(attn, "sparse_pa_mxfp8_out", False)
        and getattr(attn, "fast_oproj", False)
        and getattr(getattr(attn, "attn", None), "impl", None) is not None
        and get_tensor_model_parallel_world_size() == 4
        and get_sp_world_size() == 1
        and not is_plugin_mode()
        and get_gfx() == "gfx1250"
        and kernel() is not None
    ):
        return None
    name = f"{prefix}.launch_a"
    if name in _SEAMS:
        raise ValueError(f"Duplicate launch-A seam: {name}")
    seam = _SEAMS[name] = Seam(name, layer)
    if _defers_pa():
        _arm_attention(seam)
        _install_pa_hook()
    if hasattr(kernel(), "index_topk"):
        _defer_index_topk(seam)
    if hasattr(kernel(), "qkv_chain"):
        # the top-k reads the chain's index_q: a launch that runs the chain also selects
        assert hasattr(kernel(), "index_topk"), f"{KERNEL}: qkv_chain without index_topk"
        _install_qkv_hook()
    _log_once("bind", "MiniMax-M3 launch A seam (%s): C4 decode steps of 1 request x %d rows",
              kernel().__name__.rsplit(".", 1)[-1], ROWS)
    return name


def _defers_pa() -> bool:
    """Whether the kernel runs the sparse PA in its launch (paop / attA / full launch A), so
    the attention op defers it; Stage O keeps production's PA and starts at o_proj."""
    return bool(getattr(kernel(), "DEFERS_PA", True))


def _arm_attention(seam: Seam) -> None:
    """Mark the seam's attention impl while it runs a qualifying step, so its PA launch is
    deferred to the op (runs inside the attention custom op: never traced)."""
    impl = seam.attn_impl
    orig = impl.forward

    def forward(*args, **kwargs):
        if not _step_qualifies():
            return orig(*args, **kwargs)
        _flush_pending()  # nothing may stay deferred past its own layer's op
        _S["armed"] = seam
        try:
            return orig(*args, **kwargs)
        finally:
            _S["armed"] = None

    impl.forward = forward


def _defer_index_topk(seam: Seam) -> None:
    """For a kernel that selects in its launch (``index_topk``): the armed step's decode top-k
    returns the kernel's per-layer output buffers (which skip-top-k layers then read) and
    records its inputs; the op runs the production top-k into them when it does not fuse."""
    impl = seam.attn_impl
    orig = impl._cp_index_topk_decode

    def topk(*args, **kwargs):
        if _S["armed"] is seam:
            got = kernel().index_topk(seam, impl, *args, **kwargs)
            if got is not None:
                outs, inputs = got
                _S["topk"][seam.name] = (orig, args, kwargs, outs, inputs)
                return outs
        return orig(*args, **kwargs)

    impl._cp_index_topk_decode = topk


def _install_qkv_hook() -> None:
    """For a kernel that runs the QKV chain in its launch (``qkv_chain``): an armed step's
    ``mxfp8_qkv_norm_rope_nvfp4_insert`` is recorded instead of launched (rope_cache imports
    it at each call); its outputs (q_out, index_q) are the buffers the launch writes and the
    deferred top-k / PA read. Not fusing runs it first."""
    if _S.get("qkv_hooked"):
        return
    import aiter.ops.flydsl.mxfp8_qkv_rope_nvfp4_gfx1250 as qm

    orig = qm.mxfp8_qkv_norm_rope_nvfp4_insert

    def chain(*args, **kwargs):
        seam = _S["armed"]
        if seam is not None:
            inputs = kernel().qkv_chain(seam, *args, **kwargs)
            if inputs is not None:
                _S["qkv"][seam.name] = (orig, args, kwargs, inputs)
                return None
        return orig(*args, **kwargs)

    qm.mxfp8_qkv_norm_rope_nvfp4_insert = chain
    _S["qkv_hooked"] = True


def _qkv_fallback(seam: Seam) -> None:
    ent = _S["qkv"].pop(seam.name, None)
    if ent is not None:
        orig, args, kwargs, _ = ent
        orig(*args, **kwargs)


def _topk_fallback(seam: Seam) -> None:
    ent = _S["topk"].pop(seam.name, None)
    if ent is not None:
        orig, args, kwargs, outs, _ = ent
        for dst, src in zip(outs, orig(*args, **kwargs)):
            dst.copy_(src)


def _step_qualifies() -> bool:
    if not _S["enabled"] or kernel() is None:
        return False
    from atom.utils.forward_context import get_forward_context

    ctx = getattr(get_forward_context(), "context", None)
    return ctx is not None and not getattr(ctx, "is_prefill", True)


def _flush_pending() -> None:
    for name in list(_S["qkv"]):
        _qkv_fallback(_SEAMS[name])
    for name in list(_S["topk"]):
        _topk_fallback(_SEAMS[name])
    while _S["pending"]:
        _, pend = _S["pending"].popitem()
        pend.launch()
        _S["calls"]["pa_flushed"] += 1


def _install_pa_hook() -> None:
    """Wrap aiter's gfx1250 PA entry (records the MXFP8 output of an armed call) and its
    launch (defers the kernel's PA launch of an armed call)."""
    if _S["hooked"]:
        return
    import aiter.ops.flydsl.pa_decode as pad

    orig_pa, orig_rc = pad.pa_decode, pad._run_compiled
    _S["orig_rc"] = orig_rc

    def pa_decode(output, query, *args, **kwargs):
        qo, os_ = kwargs.get("quantized_output"), kwargs.get("output_scale")
        armed = (
            _S["armed"] is not None
            and qo is not None
            and os_ is not None
            and query.shape[0] == ROWS
        )
        _S["pa_call"] = (
            (qo.data_ptr(), os_.data_ptr(), qo.reshape(qo.shape[0], -1).stride(0))
            if armed
            else None
        )
        _S["pa_keep"] = (output, query, args, kwargs) if armed else ()
        try:
            return orig_pa(output, query, *args, **kwargs)
        finally:
            _S["pa_call"] = None
            _S["pa_keep"] = ()

    def run_compiled(exe, *args, **kwargs):
        seam, call = _S["armed"], _S["pa_call"]
        if seam is not None and call is not None and not kwargs:
            traits = _traits_of(exe)
            if traits is not None and kernel().hosts(traits):
                _S["pa_call"] = None  # one launch an armed call
                _S["pending"][seam.name] = PendingPA(exe, args, traits, *call, keep=_S["pa_keep"])
                return None
        return orig_rc(exe, *args, **kwargs)

    pad.pa_decode, pad._run_compiled = pa_decode, run_compiled
    _S["hooked"] = True


def _traits_of(exe):
    """The Gfx1250Traits of a compiled gfx1250 PA launch, or None."""
    t = _S["traits"].get(id(exe))
    if t is None:
        import aiter.ops.flydsl.kernels.pa_decode_kernel as pdk

        for key, compiled in pdk._GFX1250_PA_DECODE_CACHE.items():
            if compiled.get("launch") is exe:
                t = key[0] if isinstance(key, tuple) else key
                _S["traits"][id(exe)] = t
                break
    return t


# ---------------------------------------------------------------- the op
def k5_maxit() -> int:
    """Production k5's poll bound (``AITER_K5_AR_MAXIT``, 2e7): a slow but alive peer is never cut
    off (a timed-out round misaligns every later all-reduce)."""
    car = importlib.import_module("aiter.dist.device_communicators.custom_all_reduce")
    return int(getattr(car, "_K5_AR_MAXIT", 20000000))


def _k5() -> K5 | None:
    """Production's k5 context (the one its custom all-reduce launches k5 with)."""
    from aiter.dist.parallel_state import get_tp_group

    ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    ctx = int(getattr(ca, "_k5_ctx", 0) or 0) if ca is not None else 0
    if not ctx:
        return None
    # k5ar::Ctx: int rank, int64 max_rows, int64 slot, char* inbox[4], uint32* ctr
    w = [int.from_bytes(ctypes.string_at(ctx + o, 8), "little") for o in range(0, 64, 8)]
    return K5(inbox=w[3:7], ctr=w[7], slot=w[2], rank=w[0] & 0xFFFFFFFF, max_rows=w[1])


def _k5_gluon_route(residual: torch.Tensor, norm_w: torch.Tensor, mode: int) -> bool:
    try:
        return _k5_gluon_route_checked(residual, norm_w, mode)
    except Exception as e:  # an aiter without this dispatch: production's path
        _log_once("route_err", "MiniMax-M3 launch A: AR route check failed (%s); production path", e)
        _S["route"][(tuple(residual.shape), mode)] = False
        return False


def _k5_gluon_route_checked(residual: torch.Tensor, norm_w: torch.Tensor, mode: int) -> bool:
    """Whether production's all-reduce + norm of this call is the k5 one-shot kernel in
    Gluon order with bf16 output: aiter's ``fused_allreduce_rmsnorm`` dispatch (behind
    ``sp_fused_allreduce_rmsnorm`` in the sp modes), asked with the call's shapes. Shape-only,
    so cached."""
    key = (tuple(residual.shape), mode)
    got = _S["route"].get(key)
    if got is not None:
        return got
    from aiter.dist.parallel_state import get_tp_group

    # not `import aiter.dist....communicator_cuda as cc`: the `aiter.dist` attribute is torch.distributed
    cc = importlib.import_module("aiter.dist.device_communicators.communicator_cuda")

    comm = get_tp_group().device_communicator
    ca = getattr(comm, "ca_comm", None)
    x = residual  # o_proj's output has the residual's shape, dtype and layout
    n = norm_w.numel()
    total = x.numel() * x.element_size()
    ok = ca is not None and x.dim() == 2 and x.is_contiguous() and x.shape[-1] == n
    if ok and mode != TAIL_AR_NORM:
        ok = not comm.ar2s_sp_eligible(x, norm_w)
        if ok and mode == TAIL_SP_MOE_FRONT:
            ok = not comm.sp_moe_front_eligible(x, norm_w)
    if ok and getattr(comm, "gluon_ar", None) is not None:
        ok = not (total <= cc._GLUON_AR_MAX_BYTES and n >= 256 and n & (n - 1) == 0)
    if ok:
        ok = not (
            cc._AR2S_RMS_MIN_BYTES <= total <= cc._AR2S_RMS_MAX_TOTAL_BYTES
            and ca.should_ar2s_rms(x, x, norm_w, cc._AR2S_RMS_MIN_BYTES)
        )
    if ok:
        ok = not cc._use_lamport_split_ar_norm(comm, x, x, norm_w, n)
    if ok:
        lamport = ca.should_lamport_ar_rms(x, x, norm_w, n)
        gluon = cc._use_gluon_ar_norm(comm, x, x, norm_w, lamport)
        ok = bool(gluon and ca.should_k5_ar(x, x, norm_w))
    _S["route"][key] = bool(ok)
    _log_once(f"route{key}", "MiniMax-M3 launch A: post-attention AR of %s (mode %d) is k5 Gluon: %s",
              key[0], mode, bool(ok))
    return bool(ok)


def _ctl(dev: torch.device) -> torch.Tensor | None:
    c = _S["ctl"].get(dev.index)
    if c is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # allocated outside capture only (a warmup call precedes it)
        alloc = getattr(kernel(), "ctl", None)  # a kernel may want its counters elsewhere (uncached)
        c = alloc(dev) if alloc is not None else torch.zeros(16, dtype=torch.int32, device=dev)
        _S["ctl"][dev.index] = c
    return c


def _plan(seam, pend, x, x_scale, residual, norm_w, eps, mode) -> LaunchACall | None:
    if not _S["enabled"]:
        return None
    defers_pa = _defers_pa()
    if defers_pa and pend is None:
        return None
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    s2 = x_scale.reshape(m, -1)
    if not (
        kernel().rows_ok(m)
        and x.dtype == torch.float8_e4m3fn
        and (not defers_pa
             or (pend.qo_ptr, pend.os_ptr, pend.lda) == (x2.data_ptr(), s2.data_ptr(), x2.stride(0)))
        and residual.dtype == torch.bfloat16
        and tuple(residual.shape) == (m, HIDDEN)
        and residual.is_contiguous()
        and norm_w.numel() == HIDDEN
        and _k5_gluon_route(residual, norm_w, mode)
    ):
        return None
    k5 = _k5()
    ctl = _ctl(x.device)
    if k5 is None or ctl is None or k5.max_rows < m:
        return None
    topk, qkv = _S["topk"].get(seam.name), _S["qkv"].get(seam.name)
    if (hasattr(kernel(), "index_topk") and topk is None) or (hasattr(kernel(), "qkv_chain") and qkv is None):
        return None
    if topk is not None:
        _S["topk"].pop(seam.name)
        topk = topk[4]
    if qkv is not None:
        _S["qkv"].pop(seam.name)
        qkv = qkv[3]
    return LaunchACall(seam, pend, x2, s2, residual, norm_w, float(eps), k5, ctl, topk, qkv)


def minimax_m3_attn_tail(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    seam_name: str,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """o_proj on the attention's MXFP8 output -> all-reduce + residual + Gemma RMSNorm
    [-> MoE in TAIL_SP_MOE_FRONT] of the seam ``seam_name``: (output, residual_out)."""
    seam = _SEAMS[seam_name]
    _S["front"].pop(id(seam.moe), None)  # only this call's launch may hand its MoE a front
    pend = _S["pending"].pop(seam.name, None)
    call = _plan(seam, pend, x, x_scale, residual, norm_weight, eps, mode)
    if call is not None:
        out, res_out = kernel().run(call)
        _S["calls"]["fused"] += 1
        _log_once("fused", "MiniMax-M3 launch A (%s): first fused call (layer %d, %d rows)",
                  kernel().__name__.rsplit(".", 1)[-1],
                  seam.layer_num, x.reshape(-1, x.shape[-1]).shape[0])
        if mode == TAIL_SP_MOE_FRONT:
            return seam.moe.forward(out), res_out
        return out, res_out
    # production: the same calls as the layer's traced path, in its order
    _qkv_fallback(seam)
    _topk_fallback(seam)
    if pend is not None:
        pend.launch()
    _S["calls"]["production"] += 1
    from atom.model_ops.layernorm import (
        fused_allreduce_gemma_rms_norm,
        sp_fused_allreduce_gemma_rms_norm,
    )
    from atom.models.minimax_m3 import model as m3

    h = m3._o_proj_mxfp8(seam.o_proj, x, x_scale)
    if mode == TAIL_SP_MOE_FRONT:
        return seam.moe.sp_front_forward(h.contiguous(), residual, norm_weight, eps)
    if mode == TAIL_SP_AR_NORM:
        return sp_fused_allreduce_gemma_rms_norm(h, residual, seam.norm)
    return fused_allreduce_gemma_rms_norm(h, residual, seam.norm)


def _minimax_m3_attn_tail_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    seam_name: str,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="minimax_m3_attn_tail",
    op_func=minimax_m3_attn_tail,
    mutates_args=(),
    fake_impl=_minimax_m3_attn_tail_fake,
    # o_proj reads row-major E4M3 rows and contiguous E8M0 scales (as its own op)
    tags=(torch.Tag.needs_fixed_stride_order,),
)

__all__ = ["KERNEL", "LaunchACall", "bind", "kernel", "set_enabled", "set_moe_front", "stats", "take_moe_front"]
