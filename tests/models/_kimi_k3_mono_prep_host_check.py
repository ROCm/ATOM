"""MLA prep / post host check, CPU only, run by
``test_kimi_k3_mono_mla_prep_host.py`` in a subprocess (it patches
``torch.empty`` and friends to trace the step's pool allocations).

The REAL ``K3MonoDecodeRunner.forward`` (``_layer``, ``_k2``, ``_pre_o_core``,
``_post_core``, ``_prep_args``, ``_prepped_seam``, ``_seam_mirror``,
``_write_aux``) over the full 93-layer Kimi-K3 stack: ``ATOM_MONO_MLA_POST``
off / on against ``ATOM_MONO_MLA_PREP`` 0 / 1 / 2 / 3, and 1 / 3 over an
``ATOM_MONO_MLA_PREP_LAYERS`` subset (gaps between folded layers, 92
unfolded); three decode steps, one of them past the folds' 16-row width (the
unfused path).

The attention seam is the tree's own ``_apply_attn_res_impl`` (its
allocations, which always copy a strided block slice), only its triton kernel
replaced.

The kernels and the production modules are CPU stand-ins with one shared
arithmetic: K2's post stage computes what the production v-up / g_proj / gate
stand-in computes, its prep stage what the production seam computes, so any
difference between the runs is the host's -- a wrong buffer, a clobbered live
buffer, a stale read, an aux tap taken at the wrong point. Asserted bit for bit
at every step:

* every buffer the drafter reads: the aux taps, the hidden output, every MLA
  layer's KV cache, every layer's K2 output -- against flag-off, and POST on
  against POST off;
* every buffer a K2 launch is handed is still allocated when it is launched;
* the step's explicit pool allocations and releases (``torch.empty`` /
  ``empty_like`` / ``zeros`` / ``new_empty`` / a copying ``contiguous``), in
  order and size: identical to flag-off for PREP 1; for PREP 2 / 3 identical
  outside the production attention internals the stages replace.

It cannot say whether the kernels are right: that needs a GPU.
"""

import os
import sys
import weakref
from types import SimpleNamespace as NS

os.environ["ARCH"] = "gfx950"

import torch

from atom.model_ops.kimi_k3 import attention_residual as AR
from atom.models.kimi_k3.mono import runner as R
from atom.models.kimi_k3.mono.config import layer_block, num_blocks
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.stages import moe as k2b
from atom.models.kimi_k3.mono.plan import width_key
from atom.mono.runtime.aux_taps import AuxTaps

FP8 = torch.float8_e4m3fn
BF = torch.bfloat16
H = 7168
LAYERS = 93
MLA = {lid for lid in range(LAYERS) if lid % 4 == 3} | {92}
TAPS = (7, 23, 51, 67, 83)
QLEN = 6
PROJ = R.Dims(8).proj
# the MLA geometry the stand-ins use (``stages.mla_post_plan`` / the fork's prep)
HEADS, KVL, QL, ROPE, QK_NOPE = 12, 512, 1536, 64, 128
QKV_A, QH, ENTRY = QL + KVL + ROPE, QK_NOPE + ROPE, KVL + ROPE
QB = HEADS * QH
MAXPOS = 4096
NSLOTS = 512
PAGE = 4

fails = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        fails.append(what)


# ------------------------------------------------------------ allocation trace
class Trace:
    """Explicit allocations / releases, (event, bytes) in order. ``tag`` marks
    the production attention internals the prep stages replace."""

    def __init__(self):
        self.ev = []
        self.on = False
        self.tag = ""
        self.n = 0
        self.live = {}  # data_ptr -> weakref (the stand-in kernels' pointer table)

    def note(self, t):
        if self.on:
            i = self.n
            self.n += 1
            self.ev.append(("a", t.numel() * t.element_size(), self.tag, i))
            weakref.finalize(t, self.ev.append, ("f", 0, self.tag, i))
        self.live[t.data_ptr()] = weakref.ref(t)
        return t


TR = Trace()
_empty, _empty_like, _zeros = torch.empty, torch.empty_like, torch.zeros
_new_empty, _contig = torch.Tensor.new_empty, torch.Tensor.contiguous


def _w(fn):
    def f(*a, **k):
        return TR.note(fn(*a, **k))

    return f


torch.empty, torch.empty_like, torch.zeros = _w(_empty), _w(_empty_like), _w(_zeros)
torch.Tensor.new_empty = lambda self, *a, **k: TR.note(_new_empty(self, *a, **k))
torch.Tensor.contiguous = lambda self, *a, **k: (
    self if self.is_contiguous() else TR.note(_contig(self, *a, **k))
)
R.torch.cuda.is_current_stream_capturing = lambda: False
R.torch.cuda.current_stream = lambda: None
R._run_compiled = lambda launcher, *args: launcher(*args)


def reg(*ts):
    for t in ts:
        TR.live[t.data_ptr()] = weakref.ref(t)


def at(ptr, shape, dtype):
    """The registered tensor at ``ptr`` as ``shape`` (a leading-rows view)."""
    ref = TR.live.get(ptr)
    t = None if ref is None else ref()
    assert t is not None, f"no live tensor at {ptr:#x}"
    flat = t.reshape(-1) if t.is_contiguous() else t
    n = 1
    for d in shape:
        n *= d
    assert flat.dtype == dtype and flat.numel() >= n, (
        flat.dtype,
        dtype,
        flat.numel(),
        n,
    )
    return flat[:n].view(shape)


# ------------------------------------------------------------- shared arithmetic
g = torch.Generator().manual_seed(1234)


def rnd(*sh, s=1.0, dt=torch.float32):
    return (torch.randn(*sh, generator=g) * s).to(dt)


P_G = rnd(PROJ, H, s=0.02)  # attention output -> hidden
P_H = rnd(H, s=0.5)


def fp8_quant(y):
    """The per-token quant: amax * (1/448), y / scale clamped, E4M3."""
    amax = y.abs().amax(-1, keepdim=True)
    scale = amax * (1.0 / 448.0)
    inv = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    return (y * inv).clamp(-448, 448).to(FP8), scale


def seam_ref(prefix, bank, add, w):
    """The attention seam: (x8, xs, pref)."""
    ps = prefix.float() + add.float()
    cands = torch.cat([bank.float(), ps[:, None]], 1)
    rs = torch.rsqrt(cands.pow(2).mean(-1, keepdim=True) + w.eps)
    score = ((cands * rs) * w.qk.float()).sum(-1)
    wt = torch.softmax(score, -1)
    mixed = (wt[..., None] * cands).sum(1)
    y = (
        mixed
        * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + w.out_eps)
        * w.out_nw.float()
    )
    x8, xs = fp8_quant(y)
    return x8, xs, ps.to(BF)


def qa_ref(x8, xs, w):
    """q_a / kv_a and their norms: (qkv, q8, qs, kvn)."""
    qkv = ((x8.float() * xs) @ w.w_qa.float().t() * w.w_qa_s).to(BF)
    qc, kvc = qkv[:, :QL].float(), qkv[:, QL : QL + KVL].float()
    q = qc * torch.rsqrt(qc.pow(2).mean(-1, keepdim=True) + w.qa_eps) * w.qa_nw.float()
    q8, qs = fp8_quant(q)
    kvn = (
        kvc
        * torch.rsqrt(kvc.pow(2).mean(-1, keepdim=True) + w.kva_eps)
        * w.kva_nw.float()
    ).to(BF)
    return qkv, q8, qs, kvn


def rope(x, pos, w):
    c = w.cos[pos].float()[:, None, :]
    s = w.sin[pos].float()[:, None, :]
    a, b = x[..., :32], x[..., 32:]
    return torch.cat([a * c - b * s, a * s + b * c], -1)


def qb_ref(q8, qs, kvn, k_rope, pos, w):
    """q_b, the absorb, rope: (q_out [T, 12, 576] FP8, KV rows [T, 576] FP8)."""
    t = q8.size(0)
    qb = ((q8.float() * qs) @ w.w_qb.float().t() * w.w_qb_s).to(BF).view(t, HEADS, QH)
    nope = (
        torch.einsum("thd,hld->thl", qb[..., :QK_NOPE].float(), w.w_k.float()) * w.w_k_s
    )
    qr = rope(qb[..., QK_NOPE:].float(), pos, w)
    q_out = (torch.cat([nope, qr], -1) / w.q_scale).to(FP8)
    kr = rope(k_rope.float()[:, None, :], pos, w)[:, 0]
    kv = (torch.cat([kvn.float(), kr], -1) / w.k_scale).to(FP8)
    return q_out, kv


def kv_write(kv_cache, slots, kv):
    for t, s in enumerate(slots.tolist()):
        if s >= 0:
            kv_cache[s, 0] = kv[t]


def decode_ref(q_out, kv_cache, slots):
    """The latent attention output [T, 12, 512] bf16, from every KV row so far."""
    rows = []
    for t, s in enumerate(slots.tolist()):
        ctx = kv_cache[: s + 1, 0, :KVL].float().mean(0)
        rows.append(q_out[t, :, :KVL].float() * ctx)
    return torch.stack(rows).to(BF)


def post_ref(olat, x8, xs):
    """M2a's post: the core [T, PROJ] from the latent and g_proj's input."""
    a = olat.reshape(olat.size(0), -1)[:, :PROJ].float()
    gate = x8.float()[:, :PROJ] * xs
    return (a * torch.sigmoid(gate)).to(BF)


# ---------------------------------------------------------------- the model
def seam_w():
    return NS(
        eps=1e-6,
        out_eps=1e-6,
        nw=rnd(H, dt=BF),
        qk=rnd(H, s=0.05, dt=BF),
        out_nw=rnd(H, dt=BF),
    )


def mla_w(seam):
    return NS(
        seam=seam, blocks=0, q_heads=HEADS, rope_neox=True,
        w_qa=rnd(QKV_A, H, s=0.02, dt=FP8), w_qa_s=rnd(QKV_A).abs() + 0.5,
        qa_nw=rnd(QL, dt=BF), kva_nw=rnd(KVL, dt=BF), qa_eps=1e-6, kva_eps=1e-6,
        w_qb=rnd(QB, QL, s=0.05, dt=FP8), w_qb_s=rnd(QB).abs() + 0.5,
        w_k=rnd(HEADS, KVL, QK_NOPE, s=0.1, dt=FP8), w_k_s=torch.tensor([0.7]),
        cos=rnd(MAXPOS, 32, dt=BF), sin=rnd(MAXPOS, 32, dt=BF),
        k_scale=torch.tensor([0.05]), q_scale=torch.tensor([0.05]),
    )  # fmt: skip


SEAMS = {lid: seam_w() for lid in range(LAYERS)}
EMB = rnd(64, H, dt=BF)
MW = mla_w(None)  # one weight set for every MLA layer (seams per layer)
W_EXTRA = [
    MW.w_qa,
    MW.w_qa_s,
    MW.qa_nw,
    MW.kva_nw,
    MW.w_qb,
    MW.w_qb_s,
    MW.w_k,
    MW.w_k_s,
    MW.cos,
    MW.sin,
    MW.k_scale,
    MW.q_scale,
] + [t for s in SEAMS.values() for t in (s.nw, s.qk, s.out_nw)]
reg(*W_EXTRA)


def mla_for(lid, mode):
    w = NS(**vars(MW))
    w.seam, w.blocks, w.mode, w.layer_id = SEAMS[lid], layer_block(lid)[0], mode, lid
    return w


class Ctx:
    def __init__(self, rows, step):
        self.positions = torch.arange(rows, dtype=torch.int64) + 37 * step + 5
        self.slots = torch.arange(rows, dtype=torch.int64) + rows * step
        reg(self.positions, self.slots)


FC = NS(attn_metadata=None, kv_cache_data={}, context=NS(is_dummy_run=False))
R.get_forward_context = lambda: FC
R.kda_metadata = lambda fc: NS(spec_state_indices_tensor=None, num_accepted_tokens=None)


class _SeamKernel:
    """``_attn_res_fused_kernel[(T,)](...)`` on CPU: the shared seam arithmetic."""

    def __getitem__(self, grid):
        def launch(
            br,
            ps,
            sw,
            y,
            y_scale,
            hs,
            hs2,
            pref,
            ow,
            B,
            Bp,
            Hd,
            eps,
            out_eps,
            *rest,
            **kw,
        ):
            assert kw["QUANT"] and kw["OUT_NORM"] and kw["DO_ADD"]
            add = hs.float() + hs2.float() if kw["DO_ADD2"] else hs
            x8, xs, p = seam_ref(
                ps, br, add, NS(eps=eps, out_eps=out_eps, qk=sw, out_nw=ow)
            )
            y.copy_(x8)
            y_scale.copy_(xs)
            if kw["WRITE_PREF"]:
                pref.copy_(p)

        return launch


AR._attn_res_fused_kernel = _SeamKernel()
if not hasattr(AR, "triton"):
    AR.triton = NS(next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())


class SeamModule:
    """``AttnRes`` -> the tree's own ``_apply_attn_res_impl`` (its allocations,
    which always copy a strided block slice), the triton
    kernel replaced by the shared seam arithmetic."""

    def __init__(self, lid):
        self.lid = lid

    def __call__(self, prefix, block_residual, add=None, add2=None):
        s = SEAMS[self.lid]
        y, ys, pref = AR._apply_attn_res_impl(
            prefix, block_residual, s.qk, s.eps, add, out_norm_weight=s.out_nw,
            out_eps=s.out_eps, add_hidden2=add2, quant_dtype=FP8,
        )  # fmt: skip
        return (y, ys), pref


class MlaModule:
    """The production MLA module's decode path (allocations tagged ``attn``)."""

    def __init__(self, lid):
        self.lid = lid

    def _kv(self):
        return FC.kv_cache_data[f"layer_{self.lid}"].k_cache.view(NSLOTS, 1, ENTRY)

    def _q_kv(self, x, xs):
        qkv0, q80, qs0, kvn0 = qa_ref(x, xs, MW)
        qkv = torch.empty(qkv0.shape, dtype=BF)
        qkv.copy_(qkv0)
        q8 = torch.empty(q80.shape, dtype=FP8)
        q8.copy_(q80)
        qs = torch.empty(qs0.shape, dtype=torch.float32)
        qs.copy_(qs0)
        kvn = torch.empty(kvn0.shape, dtype=BF)
        kvn.copy_(kvn0)
        return q8, qs, kvn, qkv[:, QL + KVL :]

    def _rope_cache(self, q8, qs, kvn, k_rope, positions):
        q_out0, kv = qb_ref(q8, qs, kvn, k_rope, positions, MW)
        q_out = torch.empty(q_out0.shape, dtype=FP8)
        q_out.copy_(q_out0)
        kv_write(self._kv(), FC.attn_metadata.slot_mapping, kv)
        return q_out

    def _decode(self, q_out):
        o0 = decode_ref(q_out, self._kv(), FC.attn_metadata.slot_mapping)
        o = torch.empty(o0.shape, dtype=BF)
        o.copy_(o0)
        return o

    def _core(self, o, x, xs):
        c0 = post_ref(o, x, xs)
        c = torch.empty(c0.shape, dtype=BF)
        c.copy_(c0)
        return c

    def forward_pre_o(self, positions, normed):
        TR.tag = "attn"
        x, xs = normed
        o = self._decode(self._rope_cache(*self._q_kv(x, xs), positions))
        c = self._core(o, x, xs)
        TR.tag = ""
        return c

    def forward_latent(self, positions, normed):
        TR.tag = "attn"
        x, xs = normed
        o = self._decode(self._rope_cache(*self._q_kv(x, xs), positions))
        TR.tag = ""
        return o, x, xs

    def forward_prepped(
        self,
        positions,
        x,
        xs,
        q=None,
        q_scale=None,
        kv=None,
        k_rope=None,
        q_out=None,
        latent_out=False,
    ):
        TR.tag = "attn"
        if q_out is None:
            q_out = self._rope_cache(q, q_scale, kv, k_rope, positions)
        o = self._decode(q_out)
        out = o if latent_out else self._core(o, x, xs)
        TR.tag = ""
        return out


def extract(output, layer):
    """The aux tap: prefix + addends (the layer's own ``aux_hidden_state`` call
    carries a trailing block-residual slot that the tap ignores)."""
    ps = output[0].float() + sum(a.float() for a in output[1:-1] if a is not None)
    return ps.to(BF)


# ----------------------------------------------------------- the kernels
class K2Launch:
    """K2 (+ M2a post, + M2b prep) on CPU, from its packed arguments."""

    def __init__(self, key, outs):
        self.key, self.outs = key, outs

    def __call__(self, *args):
        key = self.key
        a = dict(zip(k2.abi_for(key).names, args))
        s = key.tokens
        lid = a["layer"] - R.K2_TAG0
        prefix = at(a["prefix"], (s, H), BF)
        blocks = TR.live[a["blocks"]]()
        core = at(a["core"], (s, PROJ), BF)
        if key.mla_post:
            olat = at(a["olat"], (s, HEADS, KVL), BF)
            core.copy_(
                post_ref(
                    olat, at(a["xg"], (s, H), FP8), at(a["xg_s"], (s, 1), torch.float32)
                )
            )
        dl = torch.tanh(core.float()) @ P_G
        prefix.copy_((dl if a["reset"] else prefix.float() + dl).to(BF))
        out = at(a["out"], (s, H), BF)
        bank = blocks[:s, : key.nblocks].float()
        out.copy_(
            (0.1 * torch.tanh(prefix.float() * P_H + bank.mean(1) * (lid % 7 + 1))).to(
                BF
            )
        )
        self.outs.append((lid, out.clone()))
        if key.mla_prep:
            m = lid + 1
            w = NS(
                eps=key.prep_eps,
                out_eps=key.prep_out_eps,
                qk=at(a["p_qk"], (H,), BF),
                out_nw=at(a["p_onw"], (H,), BF),
            )
            x8, xs, pref = seam_ref(prefix, blocks[:s, : key.prep_nblocks], out, w)
            at(a["pref"], (s, H), BF).copy_(pref)
            at(a["x8"], (s, H), FP8).copy_(x8)
            at(a["xs"], (s, 1), torch.float32).copy_(xs)
            assert key.mla_prep in (1, 2, 3) and m in MLA
            if key.mla_prep >= 2:
                qkv, q8, qs, kvn = qa_ref(x8, xs, MW)
                at(a["qkv"], (s, QKV_A), BF).copy_(qkv)
                at(a["q8"], (s, QL), FP8).copy_(q8)
                at(a["qs"], (s, 1), torch.float32).copy_(qs)
                at(a["kvn"], (s, KVL), BF).copy_(kvn)
            if key.mla_prep >= 3:
                pos = at(a["pos"], (s,), torch.int64)
                q_out, kv = qb_ref(q8, qs, kvn, qkv[:, QL + KVL :], pos, MW)
                at(a["qout"], (s, HEADS, ENTRY), FP8).copy_(q_out)
                # one row a slot, whatever the cache's block size
                assert a["kvc_sr"] == ENTRY, f"kvc_sr {a['kvc_sr']}"
                kvc = at(a["kvc"], (NSLOTS, 1, ENTRY), FP8)
                kv_write(kvc, at(a["slots"], (s,), torch.int64), kv)


def k1_stub(self, build, plan, prefix, delta, blocks, metadata, taps=None, tag=0):
    _, widx = layer_block(plan.layer_id)
    if delta is not None:
        prefix.copy_((prefix.float() + delta.float()).to(BF))
    if widx >= 0:
        blocks[:, widx].copy_(prefix)
    core = prefix.new_empty(prefix.size(0), PROJ)
    core.copy_((torch.tanh(prefix.float()[:, :PROJ] * 0.3)).to(BF))
    return core


# ------------------------------------------------------------- one run
def moe_w(lid):
    blocks, widx = layer_block(lid)
    z = torch.zeros(1)
    return NS(
        blocks=blocks,
        write_idx=widx,
        seam=SEAMS[lid],
        ln_eps=1e-6,
        w_o_s=z,
        w_sgu_s=None,
        w_sd_s=None,
        **{
            n: z
            for n in (
                "w_o",
                "w_gate",
                "bias",
                "w_ld",
                "w_sgu",
                "w_sd",
                "w13",
                "w13s",
                "w2",
                "w2s",
                "ln_w",
                "w_up",
            )
        },
    )


def run(mode, post, rows_steps, only=None):
    """Returns ([per-step snapshots], allocation trace)."""
    r = object.__new__(R.K3MonoDecodeRunner)
    plans = [
        NS(
            layer_id=lid,
            ffn_only=lid in MLA,
            mono=lid not in MLA,
            kda=None,
            moe=moe_w(lid),
        )
        for lid in range(LAYERS)
    ]
    layers = [
        NS(
            layer_idx=lid,
            self_attention_attn_res=SeamModule(lid),
            self_attn=MlaModule(lid),
            aux_hidden_state=lambda out, lid=lid: extract(out, NS(layer_idx=lid)),
        )
        for lid in range(LAYERS)
    ]
    emb = EMB
    outs = []

    def embed(ids):
        x = torch.empty((ids.numel(), H), dtype=BF)
        x.copy_(emb[ids % 64])
        return x

    def output_attn_res(prefix, blocks, add, add2):
        h = torch.empty_like(prefix)
        h.copy_((prefix.float() + add.float()).to(BF))
        return h, None

    r.inner = NS(
        layers=layers,
        config=NS(hidden_size=H),
        get_input_embeddings=embed,
        output_attn_res=output_attn_res,
    )
    r.plans, r.num_blocks, r.debug, r.rank = plans, num_blocks(LAYERS), False, 0
    r.tl1 = r.tl2 = None
    r.diag_off = -1
    r.dims = R.Dims(8)
    r.scratch = torch.zeros(16, dtype=torch.uint8)
    r.plain = torch.zeros(16, dtype=torch.uint8)
    r.queue = torch.zeros(k2b.QUEUE_BYTES, dtype=torch.uint8)
    r.peers = NS(addresses=torch.zeros(8), rank=0)
    r.mailboxes = NS(begin_step=lambda *a: None)
    r.ix = None
    r.aux = AuxTaps(TAPS, tuple(torch.zeros(48, H, dtype=BF) for _ in TAPS))
    r.mla_post = {}
    if post:
        heads, lora = HEADS, KVL
        r.mla_post = {
            m: NS(
                w_v=torch.zeros(heads, 128, lora, dtype=BF),
                w_v_s=torch.zeros(1),
                w_g=torch.zeros(PROJ, H, dtype=FP8),
                w_g_s=torch.zeros(1),
            )
            for m in MLA
        }
    r.mla_post_check = r.mla_prep_check = False
    r._mla_post_pending = None
    r.mla_prep, r.mla_prep_prev = {}, {}
    r._prep_pending, r._pref_hold_next = {}, None
    r.max_rows, r.device = 48, "cpu"
    r._k1 = k1_stub.__get__(r)
    if mode:
        # every M - 1 is a K2 layer (91 -> 92 included); ``only``:
        # ATOM_MONO_MLA_PREP_LAYERS
        for m in sorted(MLA if only is None else MLA & only):
            r.mla_prep[m] = mla_for(m, mode)
            r.mla_prep_prev[m - 1] = m
        R.K3MonoDecodeRunner._alloc_prep_buffers(r)
        reg(
            *[
                b
                for v in r._prep_bufs.values()
                for b in (v if isinstance(v, tuple) else (v,))
            ]
        )
    r.widths = {}
    # paged as production's (blocks of PAGE slots)
    kv = {m: torch.zeros(NSLOTS // PAGE, PAGE, 1, ENTRY, dtype=FP8) for m in MLA}
    for t in kv.values():
        reg(t)
    FC.kv_cache_data = {f"layer_{m}": NS(k_cache=t) for m, t in kv.items()}
    snaps = []
    TR.ev, TR.n = [], 0
    for step, rows in enumerate(rows_steps):
        keys = R.K3MonoDecodeRunner._k2_keys(r, rows)
        r.widths[width_key(rows, QLEN)] = NS(
            k2_keys=keys,
            k2_launchers={k: K2Launch(k, outs) for k in set(keys.values())},
        )
        c = Ctx(rows, step)
        FC.attn_metadata = NS(slot_mapping=c.slots, dtype_q=FP8)
        outs.clear()
        TR.on = True
        try:
            hidden = R.K3MonoDecodeRunner.forward(
                r, torch.arange(rows) * 7 + step, c.positions, QLEN
            )
        except AssertionError as e:  # a launch argument already released (``at``)
            TR.on = False
            return None, str(e)
        TR.on = False
        snaps.append(
            {
                "hidden": hidden.clone(),
                "aux": [b[:rows].clone() for b in r.aux.buffers],
                "kv": {m: t.clone() for m, t in kv.items()},
                "outs": list(outs),
            }
        )
        del hidden
    import gc

    gc.collect()
    return snaps, list(TR.ev)


def bits(t):
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


def same(a, b):
    return a.shape == b.shape and bool((bits(a) == bits(b)).all())


# the second step is past the folds' width: every layer on the unfused path
STEPS = (12, 20, 6)
ONLY = MLA - {7, 11, 51, 92}


def compare(tag, ref, got):
    for st, (a, b) in enumerate(zip(ref, got)):
        check(same(a["hidden"], b["hidden"]), f"{tag} step {st}: hidden output")
        check(
            all(same(x, y) for x, y in zip(a["aux"], b["aux"])),
            f"{tag} step {st}: DSpark aux taps {TAPS}",
        )
        check(
            all(same(a["kv"][m], b["kv"][m]) for m in MLA),
            f"{tag} step {st}: KV cache, all MLA layers",
        )
        if os.environ.get("DBG"):
            bad = [
                la for (la, x), (lb, y) in zip(a["outs"], b["outs"]) if not same(x, y)
            ]
            print("first differing K2 outputs", bad[:5], "of", len(a["outs"]))
        check(
            len(a["outs"]) == len(b["outs"])
            and all(
                la == lb and same(x, y)
                for (la, x), (lb, y) in zip(a["outs"], b["outs"])
            ),
            f"{tag} step {st}: every K2 output ({len(a['outs'])} layers)",
        )


base = None
# ``POSTS=1``: only the POST-on half (a quicker rerun)
POSTS = [p == "1" for p in os.environ.get("POSTS", "0,1").split(",")]
for post in POSTS:
    ref, ref_ev = run(0, post, STEPS)
    check(
        ref is not None, f"post {int(post)} mode 0: every K2 launch writes live buffers"
    )
    if base is None:
        base = ref
    else:
        compare("post 1 vs post 0, mode 0", base, ref)
    matrix = [(1, None), (1, ONLY), (2, None), (3, None)] + (
        [(3, ONLY)] if post else []
    )
    # ``MODES=3``: only those PREP sub-modes (a quicker rerun)
    want = {int(m) for m in os.environ.get("MODES", "1,2,3").split(",")}
    for mode, only in [(m, o) for m, o in matrix if m in want]:
        got, ev = run(mode, post, STEPS, only)
        tag = f"post {int(post)} mode {mode}" + (" layers subset" if only else "")
        check(
            got is not None,
            f"{tag}: every K2 launch writes live buffers{'' if got else ' -- ' + ev}",
        )
        if got is None:
            continue
        compare(tag, ref, got)
        if mode == 1:
            strip = lambda e: [(k, n) for k, n, t, _ in e]
            check(
                strip(ev) == strip(ref_ev),
                f"{tag}: pool allocations / releases == flag-off "
                f"({sum(1 for e in ref_ev if e[0] == 'a')} allocations)",
            )
        else:
            # sub-mode 2 replaces the production q_a / kv_a and norms
            keep = lambda e: [(k, n) for k, n, t, _ in e if t != "attn"]
            check(
                keep(ev) == keep(ref_ev),
                f"{tag}: pool allocations / releases outside the attention "
                "internals == flag-off",
            )

print("ALL OK" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
