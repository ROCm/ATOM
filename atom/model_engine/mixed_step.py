# SPDX-License-Identifier: MIT
"""ATOM_MIXED_STEP: a prefill step that also carries the running decode batch.

The scheduler builds the step's two halves as today -- a prefill batch and the
decode batch it would have run next -- and hangs the decode batch on the
prefill one (`ScheduledBatch.mixed_decode`). The runner then runs ONE target
forward over

    rows [0, P)            the prefill segment (today's prefill layout, at
                           today's buffers: its token graph replays as is)
    rows [R0, R0 + nD)     the decode segment, R0 = P rounded up to 32 (the
                           1x32 MXFP8 scale tiles), nD = the decode rows

so every row-wise layer (norms, GEMMs, router/shared expert, MoE, their quant)
reads its weights once for both. Attention and GDN run per segment, each on
today's kernels with today's metadata: the prefill segment with the metadata
the prefill builder laid out in the runner's buffers, the decode segment with
the metadata the decode builder laid out in a second set (`DecodeSide`: a
cloned `forward_vars`, a second builder, their own H2D publication).

Everything after the target forward runs as two steps would today, decode
first: the decode segment's logits, verify, rejection sampling and draft passes
(eagerly: the draft graphs read the main buffers), then the prefill's sampler
and draft passes. The runner returns both outputs, decode first, and the
scheduler post-processes them in that order -- the token/status/draft queues
see exactly the call sequence of a decode step followed by a prefill step.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging

import numpy as np
import torch

from atom.utils import CpuGpuBuffer, envs
from atom.utils import forward_context as _fc

logger = logging.getLogger("atom")

ROW_ALIGN = 32

import os as _os

# ATOM_MIXED_EAGER_DECODE=1: milestone 1's decode half (eager-shaped preparation at the exact batch, eager draft
# passes). Default: the decode half is prepared as a graphed decode step would be (its bs rung, early meta) and its
# draft span replays recordings of its own, made on the decode side's buffers (`side_drafter`).
_EAGER_DECODE = _os.environ.get("ATOM_MIXED_EAGER_DECODE", "0") == "1"
# ATOM_MIXED_DECODE_AT_END (default 0): the decode segment occupies the last rows of the step's token bucket
# (fixed addresses per bucket; but MoE then routes the pad rows between the segments: ~+20% MoE rows at c384)
# instead of right after the prefill rows, rounded up to 32 (milestone 1).
_DECODE_AT_END = _os.environ.get("ATOM_MIXED_DECODE_AT_END", "0") == "1" and not _EAGER_DECODE
# ATOM_MIXED_PIN_MAX_SEQLEN_K (default 1): the decode segment's attention runs with max_seqlen_k = max_model_len,
# as the decode graphs are recorded.
_PIN_MAX_SEQLEN_K = _os.environ.get("ATOM_MIXED_PIN_MAX_SEQLEN_K", "1") == "1" and not _EAGER_DECODE
# ATOM_MIXED_DEFER_BUILD (default 0): see forward_mixed. Experimental: +0.2-0.5 ms at c1024, but it changes the
# decode tokens vs the undeferred order (deterministically; cause not found -- not the DP row counts, not the
# state maintenance order).
_DEFER_BUILD = _os.environ.get("ATOM_MIXED_DEFER_BUILD", "0") == "1"


def round_up(n: int, m: int = ROW_ALIGN) -> int:
    return -(-n // m) * m


@dataclasses.dataclass
class MixedInfo:
    """What the attention-like regions need to run the decode segment."""

    ctx: object  # the decode half's ForwardContext
    row0: int  # first decode row in the step's tensors
    rows: int  # decode rows
    rows_pad: int  # decode rows rounded up to ROW_ALIGN
    fp: tuple | None = None  # the decode metadata's fingerprint (segment graph keys), None: no graphs
    pending: object = None  # the decode half's deferred metadata build (ATOM_MIXED_DEFER_BUILD), run on first use


def mixed_info():
    if not envs.ATOM_MIXED_STEP:
        return None
    mx = getattr(_fc.get_forward_context(), "mixed", None)
    if mx is not None and mx.pending is not None:
        pending, mx.pending = mx.pending, None
        pending(mx)
    return mx


def _run_state_ops(builder, batch):
    """`AttentionMetadataBuilder.build`'s state maintenance for `batch`, now; leaves the batch none to repeat."""
    from atom.model_engine.state_runtime import StateMaintenanceOps

    ops = batch.state_maintenance_ops
    if ops is None or ops.empty:
        return
    drain = envs.ATOM_STATE_OPS_DRAIN
    if drain == "dp":
        drain = builder.model_runner.config.parallel_config.data_parallel_size > 1
    else:
        drain = drain == "1"
    if drain and (ops.relocations or ops.checkpoint_stores or ops.checkpoint_restores):
        torch.cuda.current_stream().synchronize()
    if ops.zero_blocks:
        builder.zero_kv_blocks(ops.zero_blocks)
    if ops.relocations:
        builder.relocate_state_slots(ops.relocations)
    if ops.checkpoint_stores or ops.checkpoint_restores:
        builder.execute_paged_state_copies(ops.checkpoint_stores, ops.checkpoint_restores)
    batch.state_maintenance_ops = StateMaintenanceOps()


def _check_fix(fix):
    if fix is not None and not (fix.applied_attn and fix.applied_gdn):
        raise RuntimeError(
            "ATOM_MIXED_STEP: the decode half's builder did not apply the early fix "
            f"(attn={fix.applied_attn}, gdn={fix.applied_gdn})"
        )


@contextlib.contextmanager
def use_context(ctx):
    prev = _fc._forward_context
    _fc._forward_context = ctx
    try:
        yield
    finally:
        _fc._forward_context = prev


def _rows(t, sl):
    if t is None or not isinstance(t, torch.Tensor) or t.dim() == 0:
        return t
    return t[sl]


def _pos_rows(positions, sl):
    if positions is None:
        return None
    return positions[..., sl]


@contextlib.contextmanager
def _no_fold_request():
    """Hide an open ATOM_FOLD_ATTN_GATE request from the decode segment's
    reduce: the request's buffers are laid out over the step's rows from 0, so
    the fold falls back to the gated output + standalone quant (bitwise the
    same values) for the whole step."""
    try:
        from atom.model_ops import fold_attn_gate as fag
    except Exception:  # noqa: BLE001
        yield
        return
    req = getattr(fag._TLS, "req", None)
    fag._TLS.req = None
    try:
        yield
    finally:
        fag._TLS.req = req


# ATOM_MIXED_SEG_GRAPH (default 1 under ATOM_MIXED_STEP): the decode segment of each attention / GDN layer of a
# mixed step replays a small CUDA graph of its own instead of launching eagerly, keyed by everything it reads at a
# fixed address (its rows of the step's activations -- fixed per token bucket, since the decode rows sit at the
# bucket's end -- and the decode side's metadata buffers) and the shape it runs at. Recorded on a key's second
# sighting. The same kernels with the same arguments as the eager launches.
_SEG_GRAPH = _os.environ.get("ATOM_MIXED_SEG_GRAPH", "1") == "1"
_SEG_MIN_SEEN = int(_os.environ.get("ATOM_MIXED_SEG_GRAPH_AFTER", "2"))
# a key that keeps changing (some metadata re-allocated every step) must not grow the recordings without bound
_SEG_MAX_GRAPHS = int(_os.environ.get("ATOM_MIXED_SEG_GRAPH_MAX", "8192"))
# only steps whose prefill is at most this many tokens: above it the eager segments' host time hides under the
# prefill's GPU time and the staging copies (~0.3-0.4 ms GPU a step at c384) cost more than they save
# (c1024 recipe, GDN prefill eager, 96 decodes: the graphs cost 2-4 ms a step there, so the same bound)
_SEG_MAX_TOKENS = int(_os.environ.get("ATOM_MIXED_SEG_GRAPH_MAX_TOKENS", "1536"))


class SegGraphs:
    def __init__(self):
        self.graphs: dict = {}
        self.seen: dict = {}
        self.pool = None
        self.stream = None
        self.stats = {"replay": 0, "capture": 0, "eager": 0, "nokey": 0}

    def run(self, key, fn):
        if key is None:
            self.stats["nokey"] += 1
            fn()
            return
        g = self.graphs.get(key)
        if g is not None:
            g.replay()
            self.stats["replay"] += 1
            return
        n = self.seen[key] = self.seen.get(key, 0) + 1
        if n < _SEG_MIN_SEEN or len(self.graphs) >= _SEG_MAX_GRAPHS:
            self.stats["eager"] += 1
            fn()
            return
        from atom.utils.h2d import capturing

        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
            self.stream = torch.cuda.Stream()
        cur = torch.cuda.current_stream()
        g = torch.cuda.CUDAGraph()
        self.stream.wait_stream(cur)
        try:
            with torch.cuda.stream(self.stream), capturing():
                g.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                try:
                    fn()
                finally:
                    g.capture_end()
        except Exception:  # noqa: BLE001
            # not recordable: this key stays eager (the failed recording launched nothing)
            logger.warning("ATOM_MIXED_SEG_GRAPH: recording %s failed; eager from now on", key[:2], exc_info=True)
            self.seen[key] = -(10**9)
            self.stats["failed"] = self.stats.get("failed", 0) + 1
            cur.wait_stream(self.stream)
            fn()
            return
        cur.wait_stream(self.stream)
        g.replay()
        self.graphs[key] = g
        self.stats["capture"] += 1


SEG = SegGraphs()

# ATOM_MIXED_SEG_STAGE (default 1): a recorded decode segment reads and writes fixed staging buffers; the decode
# rows are copied in (and its outputs back) around the replay, so the recordings do not depend on where the
# decode rows sit in the step (which moves with the prefill's size).
_SEG_STAGE = _os.environ.get("ATOM_MIXED_SEG_STAGE", "1") == "1"


class _Stager:
    def __init__(self):
        self.bufs: dict = {}

    def get(self, tag, dtype, numel, device):
        b = self.bufs.get(tag)
        if b is None or b.dtype != dtype or b.numel() < numel:
            # sized generously once: a reallocation moves the address (recordings keyed on it go stale, harmlessly)
            b = self.bufs[tag] = torch.empty(max(numel, 1 << 20), dtype=dtype, device=device)
        return b


STAGE = _Stager()


def _row_extent(t):
    return 1 + sum((t.shape[i] - 1) * t.stride(i) for i in range(1, t.dim()))


def stage_rows(tag, tensors, R0, n, N, copy_in=True):
    """Copy rows [R0, R0 + n) of each group of row-indexed tensors (views of one storage with one row stride,
    leading dim N) into a staging buffer; return {name: staged view} (same strides and in-row offsets) or None if
    some tensor does not fit that picture. Tensors whose LAST dim is N (positions [3, N]) are staged by columns;
    other arguments pass through."""
    out = {}
    groups: dict = {}
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor) or t.dim() == 0:
            out[name] = t
        elif t.shape[0] == N and t.dim() >= 1 and (t.dim() == 1 or t.shape[-1] != N or t.shape[0] == N):
            groups.setdefault((t.untyped_storage().data_ptr(), t.dtype, t.stride(0)), []).append((name, t))
        elif t.dim() == 2 and t.shape[-1] == N:
            buf = STAGE.get((tag, name, "cols"), t.dtype, t.shape[0] * n, t.device)
            v = buf[: t.shape[0] * n].view(t.shape[0], n)
            if copy_in:
                v.copy_(t[:, R0 : R0 + n])
            out[name] = v
        else:
            out[name] = t
    for gi, ((_sp, dtype, s0), members) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        off0 = min(t.storage_offset() for _, t in members)
        width = max(t.storage_offset() - off0 + _row_extent(t) for _, t in members)
        if s0 <= 0 or width > s0:
            return None
        any_t = members[0][1]
        nel = any_t.untyped_storage().nbytes() // any_t.element_size()
        if off0 + (R0 + n - 1) * s0 + width > nel:
            return None
        buf = STAGE.get((tag, gi), dtype, n * s0, any_t.device)
        if copy_in:
            src = torch.as_strided(any_t, (n, width), (s0, 1), off0 + R0 * s0)
            buf[: n * s0].view(n, s0)[:, :width].copy_(src)
        for name, t in members:
            out[name] = torch.as_strided(buf, (n, *t.shape[1:]), t.stride(), t.storage_offset() - off0)
    return out


def seg_stats() -> dict:
    return dict(SEG.stats, graphs=len(SEG.graphs), keys_seen=len(SEG.seen))


def _tkey(*ts):
    out = []
    for t in ts:
        if isinstance(t, torch.Tensor):
            out.append((t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype))
        else:
            out.append(t)
    return tuple(out)


def _fingerprint(obj, depth=0):
    """What a recorded launch could have baked in from a metadata object: tensor addresses/shapes and scalars
    (max_seqlen_k excepted: pinned to max_model_len on mixed steps, as a decode graph's capture pins it)."""
    out = []
    if obj is None:
        return ()
    items = obj.items() if isinstance(obj, dict) else getattr(obj, "__dict__", {}).items()
    for k, v in sorted(items, key=lambda kv: str(kv[0])):
        if k == "max_seqlen_k":
            continue
        if isinstance(v, torch.Tensor):
            out.append((k, v.data_ptr(), tuple(v.shape)))
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out.append((k, v))
        elif isinstance(v, (list, tuple)) and len(v) <= 16 and all(
            isinstance(x, (bool, int, float, str, torch.Tensor)) or x is None for x in v
        ):
            out.append((k, _tkey(*v)))
        elif depth < 2 and (isinstance(v, dict) or hasattr(v, "__dict__")) and not callable(v):
            out.append((k, _fingerprint(v, depth + 1)))
        else:
            out.append((k, type(v).__name__, id(v)))
    return tuple(out)


_FP_LAST: dict = {}
_FP_LOGS = [0]


def _fp_debug(fp):
    """ATOM_MIXED_SEG_DEBUG=1: say which metadata entries changed between two mixed steps of one batch shape."""
    prev = _FP_LAST.get(fp[:3])
    _FP_LAST[fp[:3]] = fp
    if prev is None or prev == fp or _FP_LOGS[0] >= 20:
        return
    _FP_LOGS[0] += 1
    a, b = dict((x[0], x) for x in prev[3]), dict((x[0], x) for x in fp[3])
    diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    logger.warning("ATOM_MIXED_SEG_DEBUG: decode metadata changed at %s: %s", fp[:3], sorted(map(str, diff))[:12])


def decode_fingerprint(ctx_D):
    try:
        c = ctx_D.context
        return (
            int(c.scheduled_bs),
            int(c.running_bs),
            int(c.scheduled_tokens),
            _fingerprint(ctx_D.attn_metadata),
        )
    except Exception:  # noqa: BLE001
        logger.warning("ATOM_MIXED_STEP: no decode metadata fingerprint; segment graphs off", exc_info=True)
        return None


def attention_into_output(self, q, q_scale, k, v, positions, qkv, output, output_gate, mx):
    """`unified_attention_into_output` on a mixed step: the prefill segment
    under the step's context, then the decode segment under the decode half's."""
    ctx = _fc.get_forward_context()
    P = ctx.context.scheduled_tokens
    R0, n = mx.row0, mx.rows
    pre = slice(0, R0)
    gate = {} if output_gate is None else {"output_gate": output_gate[pre]}
    out = self.impl.forward(
        query=q[pre],
        key=k[pre],
        value=v[pre],
        position=_pos_rows(positions, pre),
        q_scale=_rows(q_scale, pre) if q_scale is not None and q_scale.shape[0] == q.shape[0] else q_scale,
        qkv=_rows(qkv, pre),
        output=output[pre],
        **gate,
    )
    if out.data_ptr() != output.data_ptr():
        output[pre].copy_(out.reshape(output[pre].shape))
    dec = slice(R0, R0 + n)
    gate = {} if output_gate is None else {"output_gate": output_gate[dec]}
    qs = _rows(q_scale, dec) if q_scale is not None and q_scale.shape[0] == q.shape[0] else q_scale

    lname = getattr(self, "layer_name", None) or id(self)
    N = q.shape[0]
    st = None
    if _SEG_GRAPH and _SEG_STAGE and mx.fp is not None:
        ins = {"q": q, "k": k, "v": v, "positions": positions, "qkv": qkv, "output_gate": output_gate}
        if q_scale is not None and q_scale.shape[0] == N:
            ins["q_scale"] = q_scale
        st = stage_rows("attn_in", ins, R0, n, N)
        sto = stage_rows("attn_out", {"output": output}, R0, n, N, copy_in=False) if st is not None else None
        if sto is None:
            st = None
    if st is not None:
        o_st = sto["output"]
        qs_st = st.get("q_scale", q_scale)
        g_st = {} if output_gate is None else {"output_gate": st["output_gate"]}

        def decode_part():
            out = self.impl.forward(
                query=st["q"], key=st["k"], value=st["v"], position=st["positions"], q_scale=qs_st,
                qkv=st["qkv"], output=o_st, **g_st,
            )
            if out.data_ptr() != o_st.data_ptr():
                o_st.copy_(out.reshape(o_st.shape))

        key = ("attn", lname, n, mx.fp, _tkey(*[st[x] for x in sorted(st)], o_st))
        with use_context(mx.ctx), _no_fold_request():
            SEG.run(key, decode_part)
        output[dec].copy_(o_st)
    else:

        def decode_part():
            out = self.impl.forward(
                query=q[dec],
                key=k[dec],
                value=v[dec],
                position=_pos_rows(positions, dec),
                q_scale=qs,
                qkv=_rows(qkv, dec),
                output=output[dec],
                **gate,
            )
            if out.data_ptr() != output[dec].data_ptr():
                output[dec].copy_(out.reshape(output[dec].shape))

        key = None
        if _SEG_GRAPH and mx.fp is not None and not _SEG_STAGE:
            key = ("attn", lname, R0, n, mx.fp, _tkey(q, k, v, positions, q_scale, qkv, output, output_gate))
        with use_context(mx.ctx), _no_fold_request():
            SEG.run(key, decode_part)
    if P < R0:
        output[P:R0].zero_()
    if R0 + n < output.shape[0]:
        output[R0 + n :].zero_()


def attention_with_output(self, q, q_scale, k, v, positions, qkv, output_gate, mx):
    """`unified_attention_with_output_base` on a mixed step."""
    output = torch.empty(
        (q.shape[0], self.impl.num_heads * self.impl.head_dim),
        dtype=torch.bfloat16,
        device=q.device,
    )
    attention_into_output(self, q, q_scale, k, v, positions, qkv, output, output_gate, mx)
    return output


def _cols(base, t):
    """(column offset, columns) of `t`, a column slice of the 2-D `base`."""
    off = (t.data_ptr() - base.data_ptr()) // base.element_size()
    return off, t.shape[1] if t.dim() == 2 else t[0].numel()


def gdn_decode_segment(base, layout, core_attn_out, layer_name, out_scale=None):
    """The decode segment of a GDN layer on a mixed step (an eager region that
    follows the layer's prefill GDN, recorded or not); nothing otherwise.

    The inputs are column slices of one projection output; the region takes
    that output whole (`base`) and re-slices it, since a recording keeps each
    argument as a reference bounded by its own view, and the fused verify
    reads q|k|v|z through one row pointer past the q|k|v view's end."""
    mx = mixed_info()
    if mx is None:
        return
    if layout is None:
        mixed_qkv, b, a, z = base
    else:
        (oq, nq), (ob, nb), (oa, na), zl = layout
        mixed_qkv = base[:, oq : oq + nq]
        b = base[:, ob : ob + nb]
        a = base[:, oa : oa + na]
        z = None
        if zl is not None:
            (oz, nz), zshape = zl
            z = base[:, oz : oz + nz].view(base.shape[0], *zshape)
    from atom.config import get_current_atom_config

    self = get_current_atom_config().compilation_config.static_forward_context[layer_name]
    R0, n, npad = mx.row0, mx.rows, mx.rows_pad
    rows = slice(R0, R0 + npad)
    kw = {}
    if z is not None:
        kw["z"] = z[rows]
    if out_scale is not None:
        # 1x32 MXFP8 scales, n32k4: a tile of 32 rows is k contiguous bytes,
        # so the segment's (32-aligned) rows own a contiguous byte range.
        k = core_attn_out.shape[1:].numel()
        kw["out_scale"] = out_scale[(R0 // ROW_ALIGN) * k : ((R0 + npad) // ROW_ALIGN) * k]
    if _SEG_GRAPH and _SEG_STAGE and mx.fp is not None and layout is not None:
        # stage the projection rows whole (the layout re-slices them), the output rows and their scale bytes
        row = base.shape[1]
        sb = STAGE.get("gdn_base", base.dtype, npad * row, base.device)[: npad * row].view(npad, row)
        sb.copy_(base[rows])
        (oq, nq), (ob, nb), (oa, na), zl = layout
        s_kw = {}
        if zl is not None:
            (oz, nz), zshape = zl
            s_kw["z"] = sb[:, oz : oz + nz].view(npad, *zshape)
        co = core_attn_out[rows]
        so = STAGE.get("gdn_out", co.dtype, co.numel(), co.device)[: co.numel()].view(co.shape)
        if out_scale is not None:
            sc = kw["out_scale"]
            ss = STAGE.get("gdn_scale", sc.dtype, sc.numel(), sc.device)[: sc.numel()].view(sc.shape)
            s_kw["out_scale"] = ss

        def decode_part():
            self.impl.forward(
                sb[:, oq : oq + nq], sb[:, ob : ob + nb], sb[:, oa : oa + na], so, layer_name, **s_kw
            )

        key = ("gdn", layer_name, npad, mx.fp, _tkey(sb, so, s_kw.get("out_scale")), layout)
        with use_context(mx.ctx):
            SEG.run(key, decode_part)
        co.copy_(so)
        if out_scale is not None:
            kw["out_scale"].copy_(s_kw["out_scale"])
        return

    def decode_part():
        self.impl.forward(
            mixed_qkv[rows], b[rows], a[rows], core_attn_out[rows], layer_name, **kw
        )

    key = None
    if _SEG_GRAPH and mx.fp is not None and not _SEG_STAGE:
        key = ("gdn", layer_name, R0, npad, mx.fp,
               _tkey(mixed_qkv, b, a, core_attn_out, kw.get("z"), kw.get("out_scale")))
    with use_context(mx.ctx):
        SEG.run(key, decode_part)


def _gdn_decode_region():
    from atom.utils.region_graph import eager_region

    return eager_region()(gdn_decode_segment)


gdn_decode_region = None


def run_gdn_decode_region(mixed_qkv, b, a, core_attn_out, layer_name, z=None, out_scale=None):
    global gdn_decode_region
    if gdn_decode_region is None:
        gdn_decode_region = _gdn_decode_region()
    base = None
    layout = None
    parts = [t for t in (mixed_qkv, b, a, z) if t is not None]
    row = mixed_qkv.stride(0)
    if (
        mixed_qkv.dim() == 2
        and all(t.untyped_storage().data_ptr() == mixed_qkv.untyped_storage().data_ptr() for t in parts)
        and all(t.stride(0) == row and t.shape[0] == mixed_qkv.shape[0] for t in parts)
    ):
        # The rows of the projection output they are column slices of.
        off = min(t.storage_offset() for t in parts)
        base = mixed_qkv.as_strided((mixed_qkv.shape[0], row), (row, 1), off)
        try:
            layout = (
                _cols(base, mixed_qkv),
                _cols(base, b),
                _cols(base, a),
                None if z is None else (_cols(base, z), tuple(z.shape[1:])),
            )
        except Exception:  # noqa: BLE001
            layout = None
    if layout is None:
        if not getattr(run_gdn_decode_region, "_warned", False):
            run_gdn_decode_region._warned = True
            logger.warning(
                "ATOM_MIXED_STEP: GDN inputs not slices of one 2-D base: base=%s b=%s a=%s z=%s",
                None if base is None else (tuple(base.shape), base.stride()),
                None if b._base is None else tuple(b._base.shape),
                None if a._base is None else tuple(a._base.shape),
                None if z is None or z._base is None else tuple(z._base.shape),
            )
        return gdn_decode_region((mixed_qkv, b, a, z), None, core_attn_out, layer_name, out_scale)
    return gdn_decode_region(base, layout, core_attn_out, layer_name, out_scale)


# ----------------------------------------------------------------------------
# The decode half's buffers
# ----------------------------------------------------------------------------


def _differ(a, b) -> bool:
    try:
        return bool(a != b)
    except Exception:  # noqa: BLE001 - arrays, objects without a scalar ==
        return a is not b


def _is_bufferish(v) -> bool:
    if isinstance(v, (torch.Tensor, CpuGpuBuffer, np.ndarray)):
        return True
    if isinstance(v, dict):
        return any(_is_bufferish(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_is_bufferish(x) for x in v)
    return False


def _empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (dict, list, tuple, set)) and len(v) == 0:
        return True
    return False


class DecodeSide:
    """A second `forward_vars` + metadata builder + H2D publication for the
    decode half of a mixed step, swapped onto the runner by `active()`."""

    def __init__(self, runner):
        self.runner = runner
        r = runner
        fv2 = {
            k: (v.clone() if isinstance(v, CpuGpuBuffer) else v)
            for k, v in r.forward_vars.items()
        }
        b1 = r.attn_metadata_builder
        runner_before = dict(r.__dict__)
        saved_fv = r.forward_vars
        r.forward_vars = fv2
        torch.set_default_device(r.device)
        try:
            b2 = type(b1)(model_runner=r)
        finally:
            torch.set_default_device(None)
            r.forward_vars = saved_fv
        # The builder's constructor may have rebound runner attributes (shared
        # step state such as ReplaySSM cursors): the runner keeps its own.
        rebound = []
        for k, v in list(r.__dict__.items()):
            if k in runner_before and runner_before[k] is not v and k != "forward_vars":
                rebound.append(k)
                r.__dict__[k] = runner_before[k]
        for k in list(r.__dict__):
            if k not in runner_before:
                rebound.append("+" + k)
        adopted, kept = [], []
        for k, v1 in b1.__dict__.items():
            if k not in b2.__dict__:
                if "stag" in k:
                    continue  # lazily created staging: b2 makes its own
                b2.__dict__[k] = v1
                adopted.append(k)
                continue
            v2 = b2.__dict__[k]
            if v1 is v2:
                continue
            if _is_bufferish(v2) and not _empty(v2):
                kept.append(k)
                continue
            if _empty(v2) and not _empty(v1):
                b2.__dict__[k] = v1
                adopted.append(k)
            elif not _is_bufferish(v1) and not _is_bufferish(v2) and _differ(v1, v2):
                b2.__dict__[k] = v1
                adopted.append(k)
        logger.info(
            "ATOM_MIXED_STEP: decode-side builder %s; adopted %s; fresh %s; runner attrs restored %s",
            type(b2).__name__, adopted, kept, rebound,
        )
        self.fv = fv2
        self.builder = b2
        # the drafter's span-graph state for the decode half (`side_drafter`)
        self.draft_state: dict = {}
        # Its own H2D publication (owner, groups, completion event).
        saved = (
            r._fv_ring, r._stage_h2d_done, r._h2d_owners, r._h2d_groups,
            r.h2d_owner, r.h2d_groups, r.attn_metadata_builder,
        )
        r._fv_ring = [fv2]
        r._stage_h2d_done = torch.cuda.Event()
        r.attn_metadata_builder = b2
        try:
            r._init_h2d_publication()
            self.h2d_owner = r.h2d_owner
            self.h2d_groups = r.h2d_groups
        finally:
            (
                r._fv_ring, r._stage_h2d_done, r._h2d_owners, r._h2d_groups,
                r.h2d_owner, r.h2d_groups, r.attn_metadata_builder,
            ) = saved
        shared = [
            k for k, v in fv2.items()
            if isinstance(v, CpuGpuBuffer) and r.forward_vars.get(k) is v
        ]
        if shared:
            logger.warning("ATOM_MIXED_STEP: buffers shared with the main side: %s", shared)

    @contextlib.contextmanager
    def active(self):
        r = self.runner
        tp = r.tokenID_processor
        saved = (
            r.forward_vars, r.h2d_owner, r.h2d_groups, r.attn_metadata_builder,
            tp.input_ids, tp.decode_src, tp.batch_info,
        )
        r.forward_vars = self.fv
        r.h2d_owner = self.h2d_owner
        r.h2d_groups = self.h2d_groups
        r.attn_metadata_builder = self.builder
        tp.input_ids = self.fv["input_ids"]
        tp.decode_src = self.fv["decode_src"]
        tp.batch_info = self.fv["batch_info"]
        try:
            yield
        finally:
            (
                r.forward_vars, r.h2d_owner, r.h2d_groups, r.attn_metadata_builder,
                tp.input_ids, tp.decode_src, tp.batch_info,
            ) = saved


@contextlib.contextmanager
def eager_drafter(drafter):
    """Draft passes without the recorded draft graphs and the fused glue: they
    read the main side's buffers at their recorded addresses."""
    if drafter is None:
        yield
        return
    saved = {
        k: drafter.__dict__[k]
        for k in ("step", "_fused_glue_deployment", "_reuse_step_buffers")
        if k in drafter.__dict__
    }
    had = set(saved)
    drafter.__dict__["step"] = None
    drafter.__dict__["_fused_glue_deployment"] = False
    drafter.__dict__["_reuse_step_buffers"] = False
    try:
        yield
    finally:
        for k in ("step", "_fused_glue_deployment", "_reuse_step_buffers"):
            if k in had:
                drafter.__dict__[k] = saved[k]
            else:
                drafter.__dict__.pop(k, None)


_MISSING = object()

# ATOM_MIXED_SIDE_CAPTURE_NOSYNC (default 1): the decode side's draft-span recordings are made as region_graph
# records (a side stream, no device synchronize / gc / empty_cache): torch.cuda.graph's synchronize drained the whole
# mixed step on every new key (16 of 67 mixed steps in a 160-request engine run: 433 ms host, ~5 ms GPU idle each).
_SIDE_CAPTURE_NOSYNC = _os.environ.get("ATOM_MIXED_SIDE_CAPTURE_NOSYNC", "1") == "1"
_CAP_STREAM = [None]


def _side_span_capture(drafter, key, args, staged0):
    """`EaglePropose._span_capture` without torch.cuda.graph's device synchronize."""
    from atom.model_ops import verify_epilogue as _ve
    from atom.utils.h2d import capturing

    if staged0 is None:
        staged0 = drafter._span_stage0(args)
        had_staged0 = False
    else:
        had_staged0 = True
    pool = drafter.__dict__.get("_span_pool")
    if pool is None:
        pool = drafter.__dict__["_span_pool"] = torch.cuda.graph_pool_handle()
    reject, anchors = args[3].clone(), args[5].clone()
    if (
        _ve.persistent_verdict(args[3])
        and _ve.persistent_verdict(args[5])
        and args[3].is_contiguous()
        and args[5].is_contiguous()
    ):
        reject, anchors = args[3], args[5]
    args = args[:3] + (reject, args[4], anchors)
    graph = torch.cuda.CUDAGraph()
    cur = torch.cuda.current_stream()
    if _CAP_STREAM[0] is None:
        _CAP_STREAM[0] = torch.cuda.Stream()
    cs = _CAP_STREAM[0]
    cs.wait_stream(cur)
    with torch.cuda.stream(cs), capturing():
        graph.capture_begin(pool=pool, capture_error_mode="thread_local")
        try:
            draft_ids = drafter._propose_fused_glue_impl(*args, staged0=staged0, eager_passes=True)
        finally:
            graph.capture_end()
    cur.wait_stream(cs)
    graph.replay()
    drafter._span_graphs[key] = dict(
        graph=graph, draft_ids=draft_ids, reject=reject, anchors=anchors, staged0=had_staged0,
    )
    drafter._span_stats["capture"] = drafter._span_stats.get("capture", 0) + 1
    return draft_ids


@contextlib.contextmanager
def side_drafter(drafter, side, context):
    """The decode half's draft passes on recordings of their own.

    The drafter's per-pass graphs and its span graphs read the main side's
    buffers at their recorded addresses. Here the span graphs (ATOM_MTP_DRAFT_
    SPAN_GRAPH) come from a dict of the decode side's, recorded (on a key's
    second sighting, as on the main side) while the decode side's buffers are
    swapped in; anything that would replay a per-pass graph runs those passes'
    code eagerly instead. Where the fused glue does not serve the step, the
    passes run as milestone 1 ran them (`eager_drafter`)."""
    from atom.spec_decode import eagle_proposer as _ep

    if (
        drafter is None
        or not _ep._SPAN_GRAPH
        or not hasattr(drafter, "_fused_glue_applies")
        or not drafter._fused_glue_applies(context)
    ):
        with eager_drafter(drafter):
            yield
        return
    d = drafter.__dict__
    keys = ("_span_graphs_d", "_span_seen_d", "_span_stats_d", "_propose_fused_glue_impl", "_span_capture")
    saved = {k: d.get(k, _MISSING) for k in keys}
    for k in keys[:3]:
        d[k] = side.draft_state.setdefault(k, {})
    impl = type(drafter)._propose_fused_glue_impl

    def eager_impl(*a, **kw):
        kw["eager_passes"] = True
        return impl(drafter, *a, **kw)

    d["_propose_fused_glue_impl"] = eager_impl
    if _SIDE_CAPTURE_NOSYNC:
        d["_span_capture"] = lambda key, args, staged0: _side_span_capture(drafter, key, args, staged0)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is _MISSING:
                d.pop(k, None)
            else:
                d[k] = v


# ----------------------------------------------------------------------------
# The mixed forward
# ----------------------------------------------------------------------------


# ATOM_MIXED_LAZY_SURFACE (default 1 under ATOM_MIXED_STEP): the prefill half's post-process surfaces the decode
# half's results (its sampled tokens and its accept/reject status, which the deferred output hands over one step
# late) only after its own launches, instead of waiting for the decode verify before them.
_LAZY_SURFACE = _os.environ.get("ATOM_MIXED_LAZY_SURFACE", "1") == "1"


# ATOM_MIXED_UNION (default 1): the mixed step is ONE step for the deferred output: its post-process
# surfaces the previous step's tokens / status / drafts (as any step does) and hands the next step the decode and
# prefill halves' results as one batch [decode seqs | prefill seqs] -- tokens, accept/reject status, drafts,
# anchor ids and the early-meta status rows concatenated. The host then never waits inside the mixed step for its
# own decode verify (milestone 1's chain, D(k) then P(k), surfaces D(k)'s results in P(k)'s post-process).
_UNION = _os.environ.get("ATOM_MIXED_UNION", "1") == "1"


class _UnionBatch:
    """What the token processor keeps of the previous batch: its requests, in order."""

    def __init__(self, req_ids):
        self.req_ids = list(req_ids)
        self.total_seqs_num = len(self.req_ids)
        self.is_dummy_run = False


class _Union:
    """The token processor's bookkeeping across the two post-processes of a mixed step (see ATOM_MIXED_UNION)."""

    def __init__(self, tp, mtp_k):
        self.tp = tp
        self.k = mtp_k
        self.d = {}

    @contextlib.contextmanager
    def decode_half(self):
        tp, d = self.tp, self.d
        q = tp.spec_status_queue
        saved_env = _os.environ.get("ATOM_MTP_VERIFY_HOST_HANDOFF")
        # the decode half's rows and status through device tensors (joined with the prefill half's), not the
        # verify epilogue's pinned host slots
        _os.environ["ATOM_MTP_VERIFY_HOST_HANDOFF"] = "0"

        def psi(batch, sampled, sync_event, sampled_logprobs=None, host_slot=None):
            assert host_slot is None and sampled_logprobs is None
            token_ids = tp.recv_async_output(tp.token_ids_cpu)
            tp.recv_logprobs()
            d["tok"] = sampled
            return _surface_prev(tp, token_ids), None

        def send(source, *, ready=None):
            d["status"] = source

        def pdi(batch, draft_token_ids):
            d["drafts"] = draft_token_ids
            token_ids = tp.recv_draft_token_ids()
            return token_ids if tp.prev_req_ids is not None else np.array([], dtype=np.int32)

        tp.prepare_sampled_ids, q.send, tp.prepare_draft_ids = psi, send, pdi
        try:
            yield
        finally:
            for o, n in ((tp, "prepare_sampled_ids"), (q, "send"), (tp, "prepare_draft_ids")):
                o.__dict__.pop(n, None)
            if saved_env is None:
                _os.environ.pop("ATOM_MTP_VERIFY_HOST_HANDOFF", None)
            else:
                _os.environ["ATOM_MTP_VERIFY_HOST_HANDOFF"] = saved_env
            d["next"] = tp.prev_token_ids
            d["status_gpu"] = getattr(tp, "_prev_status_gpu", None)

    @contextlib.contextmanager
    def prefill_half(self):
        tp, d = self.tp, self.d
        q = tp.spec_status_queue
        # the previous step's status was taken by the decode half; the one queued now is this step's, for the next
        tp.__dict__.pop("_spec_status_deferred", None)

        def psi(batch, sampled, sync_event, sampled_logprobs=None, host_slot=None):
            assert host_slot is None and sampled_logprobs is None
            d["p_req_ids"] = list(batch.req_ids)
            d["p_tok"] = sampled
            d["p_event"] = sync_event
            return {-1: 1}, None

        def send(source, *, ready=None):
            d["p_status"] = source
            d["p_ready"] = ready

        def pdi(batch, draft_token_ids):
            d["p_drafts"] = draft_token_ids
            return np.array([], dtype=np.int32)

        tp.prepare_sampled_ids, q.send, tp.prepare_draft_ids = psi, send, pdi
        try:
            yield
        finally:
            for o, n in ((tp, "prepare_sampled_ids"), (q, "send"), (tp, "prepare_draft_ids")):
                o.__dict__.pop(n, None)
            d["p_next"] = tp.prev_token_ids

    def finish(self, dec_batch, event):
        """Queue the joined results for the next step and leave the token processor pointing at them."""
        tp, d = self.tp, self.d
        tok_D = d["tok"]
        bsD = tok_D.shape[0]
        width = self.k + 1
        has_p = "p_tok" in d
        bsP = d["p_tok"].shape[0] if has_p else 0
        tok = torch.full((bsD + bsP, width), -1, dtype=tok_D.dtype, device=tok_D.device)
        tok[:bsD].copy_(tok_D.view(bsD, -1))
        if has_p:
            tok[bsD:, 0].copy_(d["p_tok"].view(bsP, -1)[:, 0])
        status = d["status"] if not has_p else torch.cat([d["status"], d["p_status"].to(d["status"].dtype)], dim=1)
        drafts = d["drafts"] if not has_p else torch.cat([d["drafts"], d["p_drafts"].to(d["drafts"].dtype)], dim=0)
        nxt = d["next"] if not has_p else torch.cat([d["next"], d["p_next"].to(d["next"].dtype)], dim=0)
        event.record()
        tp.send_to_cpu_async(tok, tp.token_ids_cpu, event)
        tp.spec_status_queue.send(status, ready=event)
        tp.draft_token_ids = drafts
        tp.pre_num_decode_token_per_seq = tp.num_spec_tokens + 1
        tp.send_draft_token_ids_to_cpu(drafts)
        tp.prev_token_ids = nxt
        req_ids = list(dec_batch.req_ids) + (d["p_req_ids"] if has_p else [])
        tp.prev_batch = _UnionBatch(req_ids)
        sg = d.get("status_gpu")
        if sg is not None:
            tp._prev_status_gpu = (status[0], status[1], len(req_ids))


def _surface_prev(tp, token_ids):
    """`prepare_sampled_ids`' surfacing of the previous batch's tokens (its deferred branch)."""
    token_id_dict = {}
    tp.prev_req_ids = None
    if tp.prev_batch is not None:
        tp.prev_req_ids = tp.prev_batch.req_ids
        lst = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        if lst and isinstance(lst[0], list):
            processed = tp._batch_process_token_ids(lst)
        else:
            processed = [(t,) for t in lst]
        token_id_dict = dict(zip(tp.prev_req_ids, processed))
    token_id_dict[-1] = 1
    return token_id_dict


class _LazySurface:
    """`tokenIDProcessor.prepare_sampled_ids` for the prefill half of a mixed step, without the wait.

    The deferred output makes the prefill half's post-process return the decode
    half's tokens (`recv_async_output`) and status (`take_deferred_spec_status`);
    both wait for the decode verify on the GPU, which runs right before. Here the
    post-process takes the decode half's queue entry without waiting and runs its
    sampling and draft launches; `resolve` then waits and fills the output: the
    same entries, the same order, only later on the host."""

    def __init__(self, tp):
        self.tp = tp
        self.pending = None
        self._orig = None

    def __enter__(self):
        tp = self.tp
        self._orig = tp.__dict__.get("prepare_sampled_ids")
        orig = tp.prepare_sampled_ids

        def lazy(batch, sampled_token_ids, sync_event, sampled_logprobs=None, host_slot=None):
            if (
                not tp.is_deferred_out
                or sampled_logprobs is not None
                or tp.prev_batch is None
                or not tp.token_ids_cpu
                or (tp.logprobs_cpu and tp.logprobs_cpu[0] is not None)
            ):
                return orig(batch, sampled_token_ids, sync_event, sampled_logprobs, host_slot=host_slot)
            entry = tp.token_ids_cpu.pop(0)  # held (with its GPU source) until `resolve` waited on it
            if tp.logprobs_cpu:
                tp.logprobs_cpu.pop(0)
            if host_slot is not None:
                tp.token_ids_cpu.append((host_slot.view, host_slot, (sampled_token_ids, None)))
                tp.logprobs_cpu.append(None)
            else:
                tp.send_to_cpu_async(sampled_token_ids, tp.token_ids_cpu, sync_event)
            tp.prev_req_ids = tp.prev_batch.req_ids
            self.pending = (entry, list(tp.prev_req_ids))
            tp.prev_batch = batch
            tp.prev_token_ids = sampled_token_ids
            return {-1: 1}, None

        tp.prepare_sampled_ids = lazy
        return self

    def __exit__(self, *exc):
        tp = self.tp
        if self._orig is None:
            tp.__dict__.pop("prepare_sampled_ids", None)
        else:
            tp.prepare_sampled_ids = self._orig
        return False

    def resolve(self, out):
        """Wait for the decode half's tokens and status; put them in `out`."""
        from atom.model_engine.output_transfer import HostSlot

        tp = self.tp
        tp.take_deferred_spec_status()
        out.num_rejected = out._copy_array_view(tp.prev_rejected_num)
        out.num_bonus = out._copy_array_view(tp.prev_bonus_num)
        if self.pending is None:
            return out
        (cpu_tensor, copy_done, _src), req_ids = self.pending
        self.pending = None
        copy_done.synchronize()
        if isinstance(copy_done, HostSlot):
            cpu_tensor = cpu_tensor.clone()
            copy_done.release()
        ids = cpu_tensor.tolist() if hasattr(cpu_tensor, "tolist") else cpu_tensor
        if ids and isinstance(ids[0], list):
            processed = tp._batch_process_token_ids(ids)
        else:
            processed = [(t,) for t in ids]
        out.req_ids = req_ids
        out.token_ids = processed
        out._req_id_to_idx = None
        return out


_COUNT = [0]


class _Phases:
    """Consecutive profiler ranges over forward_mixed's phases (host time per phase)."""

    def __init__(self):
        self._rf = None

    def __call__(self, name):
        from torch.profiler import record_function

        if self._rf is not None:
            self._rf.__exit__(None, None, None)
            self._rf = None
        if name is not None:
            self._rf = record_function(name)
            self._rf.__enter__()


def forward_mixed(r, pf_batch, dec_batch):
    """Returns (decode output, prefill output); post-process in that order."""
    from torch.profiler import record_function

    from atom.utils.forward_context import get_forward_context, reset_forward_context

    side = getattr(r, "_mixed_side", None)
    if side is None:
        side = r._mixed_side = DecodeSide(r)
    tp = r.tokenID_processor
    drafter = getattr(r, "drafter", None)
    ph = _Phases()

    # ---- the decode half's preparation, in the decode side's buffers
    ph("mx::prep_decode")
    # ATOM_MIXED_DEFER_BUILD: the decode half's metadata build (`prepare_inputs`) waits until the first decode
    # segment needs it -- the target forward's first segment (embedding, layer 0's projections and prefill GDN)
    # is then already on the GPU while the host builds.
    defer = _DEFER_BUILD and not _EAGER_DECODE and dec_batch.next_token_ids is None
    held = {}
    with side.active():
        r.h2d_owner.begin()
        ee = r.enforce_eager
        if _EAGER_DECODE:
            r.enforce_eager = True
        if defer:
            # The decode batch's state maintenance (KV block zeroing, slot relocations, checkpoint copies) keeps
            # its place before the prefill half's, as an undeferred build orders it: only the metadata waits.
            _run_state_ops(r.attn_metadata_builder, dec_batch)

            def _hold(batch, input_ids, forward_mode, *, spec_decode_indices=None):
                held["args"] = (batch, input_ids, forward_mode, spec_decode_indices)

            r.prepare_inputs = _hold
        try:
            (_ids_D, temps_D, topks_D, topps_D, greedy_D, noise_D) = r.prepare_model(dec_batch)
        finally:
            r.enforce_eager = ee
            if defer:
                r.__dict__.pop("prepare_inputs", None)
        fix = tp.early_fix
        tp.early_fix = None
        # Early meta (a graph-shaped decode half): the previous step's status is
        # taken after the target launch, as `forward` takes it.
        dec_deferred = tp.__dict__.pop("_spec_status_deferred", False)
        presence_D = r._presence_step
        greedy_ids_D = r._greedy_ids_step(dec_batch, greedy_D)
        topk_cands_D = r._topk_candidates_step(dec_batch, greedy_D, topks_D)
        if defer:
            fm_D = held["args"][2]
            ids_D = r.forward_vars["input_ids"].gpu[: fm_D.running_tokens]
            if fm_D.running_tokens > fm_D.scheduled_tokens:
                ids_D[fm_D.scheduled_tokens :].zero_()
            nr_D = (tp.num_rejected, tp.num_bonus)
            # the runner-wide DP row counts the decode half settled (the prefill half's prepare refills them)
            rt_D = np.array(r.running_tokens_per_dp_rank, copy=True)
        else:
            _check_fix(fix)
            r.h2d_owner.finish()
            ctx_D = get_forward_context()
            fm_D = ctx_D.context.forward_mode
            # the decode half's rows at the height its metadata was laid out for
            # (`running_tokens`: its bs rung x q), pad rows zeroed
            ids_D, pos_D = r._padded_inputs(fm_D)
    nD_sched = dec_batch.total_tokens_num
    nD = fm_D.running_tokens
    P = pf_batch.total_tokens_num
    R0 = round_up(P)
    nDp = round_up(nD)

    # ---- the prefill half's preparation, in the runner's buffers
    ph("mx::prep_prefill")
    r._advance_forward_vars()
    r._gate_staging_reuse()
    r._mixed_rows = R0 + nDp
    tp._mixed_defer_status = True
    try:
        (ids_P, temps_P, topks_P, topps_P, greedy_P, noise_P) = r.prepare_model(pf_batch)
    finally:
        r._mixed_rows = None
        tp._mixed_defer_status = False
    r._mark_staging_h2d_enqueued()
    ctx_P = get_forward_context()
    presence_P = r._presence_step
    fm = ctx_P.context.forward_mode
    assert fm.running_tokens >= R0 + nDp, (fm.running_tokens, R0, nDp)
    if _DECODE_AT_END:
        # the decode rows at the end of the step's height: fixed addresses per (token bucket, decode rows)
        R0 = fm.running_tokens - nDp

    # ---- one target forward over both segments
    ph("mx::target")
    ids, pos = r._padded_inputs(fm)
    ids[R0 : R0 + nD].copy_(ids_D[:nD])
    bi = r.forward_vars["batch_info"].gpu
    bi[1:2].fill_(R0 + nD)
    use_graphs = _SEG_GRAPH and not _EAGER_DECODE and fm_D.use_cudagraph and P <= _SEG_MAX_TOKENS
    pinned = {}

    def _finish_decode(mx):
        """The decode metadata (built here when deferred), its rows' positions, the pinned max_seqlen_k, and the
        segment-graph fingerprint."""
        nonlocal ctx_D
        if defer:
            prev = _fc._forward_context
            try:
                with side.active(), record_function("mx::decode_build"):
                    tp.early_fix = fix
                    tp.num_rejected, tp.num_bonus = nr_D
                    b_, ids_, fm_, sdi_ = held["args"]
                    rt_P = np.array(r.running_tokens_per_dp_rank, copy=True)
                    r.running_tokens_per_dp_rank[...] = rt_D
                    try:
                        r.prepare_inputs(b_, ids_, forward_mode=fm_, spec_decode_indices=sdi_)
                    finally:
                        # as the two prepares in order would have left them
                        r.running_tokens_per_dp_rank[...] = rt_P
                    tp.early_fix = None
                    _check_fix(fix)
                    r.h2d_owner.finish()
                    ctx_D = get_forward_context()
                    _, pos_D2 = r._padded_inputs(fm_D)
            finally:
                _fc._forward_context = prev
        else:
            pos_D2 = pos_D
        pos[..., R0 : R0 + nD].copy_(pos_D2[..., :nD])
        md = ctx_D.attn_metadata
        msk = getattr(md, "max_seqlen_k", None)
        if _PIN_MAX_SEQLEN_K and msk is not None:
            # as a decode graph's capture pins it (it sizes grids only); also what makes the segments recordable
            pinned["md"], pinned["msk"] = md, msk
            md.max_seqlen_k = r.config.max_model_len
        fp = decode_fingerprint(ctx_D) if use_graphs else None
        if fp is not None and _os.environ.get("ATOM_MIXED_SEG_DEBUG", "0") == "1":
            _fp_debug(fp)
        mx.ctx, mx.fp = ctx_D, fp

    mixed = MixedInfo(None, R0, nD, nDp, None, pending=_finish_decode)
    if not defer:
        _finish_decode(mixed)  # also before the forward: nothing waits for it
        mixed.pending = None
    ctx_P.mixed = mixed
    try:
        with record_function(f"mixed[p={P} d={dec_batch.total_seqs_num}x tok={nD} run={fm.running_tokens}]"):
            model_output = r.model(ids, pos)
        if mixed.pending is not None:  # no decode segment ran (cannot happen with attention/GDN layers)
            mixed.pending(mixed)
            mixed.pending = None
    finally:
        ctx_P.mixed = None
        if "md" in pinned:
            pinned["md"].max_seqlen_k = pinned["msk"]
    ctx_D = mixed.ctx
    bi[1:2].fill_(P)
    hidden_P = model_output[:P]
    hidden_D = model_output[R0 : R0 + nD_sched]
    if fm.running_tokens > P:
        ctx_P.context.positions = ctx_P.context.positions[..., :P].contiguous()

    # ---- the decode half, as its own step would finish
    ph("mx::post_decode")
    drafts = eager_drafter(drafter) if _EAGER_DECODE else side_drafter(drafter, side, ctx_D.context)
    union = (
        _Union(tp, dec_batch.num_spec_step)
        if (_UNION and tp.is_deferred_out and drafter is not None and not _EAGER_DECODE
            and not any(getattr(dec_batch, "return_logprobs", ()) or ())
            and not any(getattr(pf_batch, "return_logprobs", ()) or ()))
        else None
    )
    with side.active(), use_context(ctx_D), drafts, (union.decode_half() if union else contextlib.nullcontext()):
        r._presence_step = presence_D
        if dec_deferred:
            tp.prev_rejected_num, tp.prev_bonus_num = tp.recv_spec_status()
        logits_D = r._compute_logits(hidden_D, greedy_ids_D, topk_cands_D)
        # (the prefill half's deferred status is the one this postprocess sends)
        out_D = r.postprocess(
            dec_batch, logits_D, temps_D, topks_D, topps_D, greedy_D, hidden_D,
            needs_independent_noise=noise_D,
        )

    # ---- the prefill half
    ph("mx::post_prefill")
    with use_context(ctx_P):
        r._presence_step = presence_P
        if drafter is not None and not pf_batch.is_dummy_run:
            drafter.compute_draft_kv(ctx_P.context.positions, hidden_P, pf_batch.next_token_ids)
        if r._is_pure_middle_chunk(pf_batch):
            from atom.model_engine.scheduler import ScheduledBatchOutput

            out_P = ScheduledBatchOutput(
                req_ids=list(pf_batch.req_ids), token_ids=[], num_rejected=None,
                num_bonus=None, draft_token_ids=None,
            )
        else:
            logits_P = r._compute_logits(hidden_P, False, 0)
            if union is not None:
                with union.prefill_half():
                    out_P = r.postprocess(
                        pf_batch, logits_P, temps_P, topks_P, topps_P, greedy_P, hidden_P,
                        needs_independent_noise=noise_P,
                    )
                from atom.model_engine.scheduler import ScheduledBatchOutput

                out_P = ScheduledBatchOutput(
                    req_ids=[], token_ids=[], num_rejected=np.zeros(0, np.int32),
                    num_bonus=np.zeros(0, np.int32), draft_token_ids=np.array([], dtype=np.int32),
                    is_deferred_out=True,
                )
            elif _LAZY_SURFACE and tp.is_deferred_out:
                with _LazySurface(tp) as lazy:
                    out_P = r.postprocess(
                        pf_batch, logits_P, temps_P, topks_P, topps_P, greedy_P, hidden_P,
                        needs_independent_noise=noise_P,
                    )
                ph("mx::surface")
                lazy.resolve(out_P)
            else:
                tp.take_deferred_spec_status()
                out_P = r.postprocess(
                    pf_batch, logits_P, temps_P, topks_P, topps_P, greedy_P, hidden_P,
                    needs_independent_noise=noise_P,
                )
    if union is not None:
        union.finish(dec_batch, r.forward_done_event)
    ph(None)
    _COUNT[0] += 1
    if _COUNT[0] in (1, 10) or _COUNT[0] % 500 == 0:
        logger.info("ATOM_MIXED_STEP: %d mixed steps; segment graphs %s", _COUNT[0], seg_stats())
    reset_forward_context()
    r._record_forward_vars_event()
    r._record_kv_cache_ready(pf_batch)
    return out_D, out_P


_STEPLOG = {"f": None}


def steplog(batch):
    """ATOM_MX_STEPLOG: t_ns,kind(P/D/M),prefill_tokens,decode_seqs,prefill_seqs per forward (debug)."""
    import time as _t

    f = _STEPLOG["f"]
    if f is None:
        f = _STEPLOG["f"] = open(_os.environ["ATOM_MX_STEPLOG"] + ".%d.csv" % _os.getpid(), "w", buffering=1)
        import atexit

        atexit.register(f.close)
    mixed = getattr(batch, "mixed_decode", None)
    pf = batch.total_seqs_num_prefill
    kind = "M" if mixed is not None else ("P" if pf else "D")
    nd = mixed.total_seqs_num if mixed is not None else batch.total_seqs_num_decode
    f.write("%d,%s,%d,%d,%d\n" % (_t.time_ns(), kind, batch.total_tokens_num_prefill, nd, pf))

