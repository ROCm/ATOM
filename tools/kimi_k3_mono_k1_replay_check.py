"""K1 under ReplaySSM (``replay_cap``) against the production KDA path over
STEPS decode steps, and against K1 without it.

Each step K1 runs on fresh random inputs; the production path (the conv update,
``replayssm_sigmoid_gating_delta_rule``, the gated output norm) is fed K1's own
projection mailbox, so the two differ only past the projection. Each keeps its
own checkpoints, records and cursors, both advanced by ``replayssm_commit`` with
the step's accepted counts (random, DSpark: 1 .. QLEN). Per step: core_out,
the conv state, and after it the checkpoints and the records the step wrote.

Step 0 also runs K1 without ReplaySSM from the checkpoint (no records): its
core_out and conv state must match the replay's bit for bit.

Run on one gfx1250 GPU:
  ROWS=40,20 QLEN=5 STEPS=6 python tools/kimi_k3_mono_k1_replay_check.py
SPLIT sets ATOM_KIMI_MONO_K1_HEAD_SPLIT for every build (default auto)."""

import json
import os

import torch
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.fla_ops.replayssm import (
    replayssm_commit,
    replayssm_sigmoid_gating_delta_rule,
)
from atom.model_ops.kimi_k3.activations import rmsnorm_gated
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.k1_local_runner_gfx1250 import K1LocalRuntime
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    REPLAY_TABLE,
    K1LocalBuildSpec,
)
from atom.models.kimi_k3.mono.kernels.attention.kda_gfx1250 import (
    BETA_0,
    FA_0,
    G2_0,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

WIDTHS = [int(r) for r in os.environ.get("ROWS", "40,20").split(",")]
QLEN = int(os.environ.get("QLEN", "5"))
STEPS = int(os.environ.get("STEPS", "6"))
CAP = int(os.environ.get("CAP", "16"))
WINDOW = int(os.environ.get("WINDOW", "5"))
SPLIT = os.environ.get("SPLIT", "auto")
SENTINEL_STEP = 3  # request 0 re-prefilled before it: its cursor back to -1
FUSED, PROJ, NH, HD, LB = 6284, 1536, 12, 128, -5.0
SL = WINDOW + 2  # the conv window: width - 1 + the verify window's drafts
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
gen = torch.Generator(device=dev).manual_seed(4)
os.environ["ATOM_KIMI_MONO_K1_HEAD_SPLIT"] = SPLIT


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def bf16f(*shape, scale=1.0):
    """f32 values bf16 holds exactly: the production weights are bf16."""
    return bf16(*shape, scale=scale).float()


w = (torch.randn(FUSED, HIDDEN, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
q, s = quantize_weight_to_fp8_128x128_blockscale(
    w, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
)
qp = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, -FUSED % 128)).view(
    dtypes.fp8
)
w_in, w_in_scale = shuffle_weight(qp, layout=(16, 16)).contiguous(), s.contiguous()
in_nw = bf16(HIDDEN)
w_fb = bf16(PROJ, HD, scale=0.05)
conv_w = bf16f(3 * PROJ, 4, scale=0.3)
a_log = bf16f(NH, scale=0.5)
dt_bias = bf16f(PROJ, scale=0.5)
on_w = bf16f(HD)


def diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {
        "max_abs": float(f"{d.max().item():.3g}") if d.numel() else 0.0,
        "ref_max": float(f"{b.abs().max().item():.3g}") if d.numel() else 0.0,
        "mismatch": int((a != b).sum().item()),
        "n": a.numel(),
    }


def spec(rows, replay):
    return K1LocalBuildSpec(
        qlen=QLEN,
        delta=False,
        state_len=SL,
        eps=1e-6,
        out_eps=1e-6,
        onorm_eps=1e-5,
        lower_bound=LB,
        write_idx=-1,
        quant_core=False,
        external_seam=True,
        fp8_projection=True,
        replay_cap=CAP if replay else 0,
        replay_window=WINDOW if replay else 0,
    )


def k1_run(k1, prefix, conv, rs, st, num_acc, table=None):
    rows = prefix.size(0)
    blk1 = torch.zeros(rows, 1, HIDDEN, dtype=torch.bfloat16, device=dev)
    core = torch.zeros((rows, PROJ), dtype=torch.bfloat16, device=dev)
    k1._scratch.zero_()
    k1(
        K1LocalRuntime(
            prefix=prefix,
            delta=None,
            blocks=blk1,
            blk_sm=blk1.stride(0),
            blk_sr=blk1.stride(1),
            ares_nw=in_nw,
            ares_qk=in_nw,
            in_nw=in_nw,
            w_in=w_in,
            w_in_scale=w_in_scale,
            w_fb=w_fb,
            conv_w=conv_w,
            conv_st=conv,
            cs_seq=conv.stride(0),
            cs_dim=conv.stride(1),
            cs_tok=conv.stride(2),
            a_log=a_log,
            dt_bias=dt_bias,
            on_w=on_w,
            rstate=rs,
            rs_seq=rs.stride(0),
            st_idx=st,
            st_stride=st.stride(0),
            num_acc=num_acc,
            core_out=core,
            core_q=None,
            core_scale=None,
            replay=table,
        ),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    off, size = k1.layout["proj"]
    pairs = k1._scratch[off : off + size].view(torch.int32).view(rows, -1, 2)
    # a pair is (value, tag): the tag is the one word every pair shares
    word = 0 if pairs[..., 1].unique().numel() == 1 else 1
    proj = pairs[..., word].contiguous().view(torch.float32)
    return core, proj


def production(proj, conv, ckpt, bufs, write_pos, slot_idx, num_acc, req):
    """model.py's spec-decode (plain decode at QLEN 1) ReplaySSM step."""
    T = proj.size(0)
    p = proj.to(torch.bfloat16)
    mixed = p[:, : 3 * PROJ].contiguous()
    out_gate = p[:, G2_0 : G2_0 + PROJ]
    beta = p[:, BETA_0 : BETA_0 + NH].unsqueeze(0).contiguous()
    gate = (p[:, FA_0 : FA_0 + HD] @ w_fb.t()).view(1, T, NH, HD)
    qsl = torch.arange(req + 1, dtype=torch.int32, device=dev) * QLEN
    conv_slots = slot_idx if QLEN > 1 else slot_idx
    extra = (
        {
            "num_accepted_tokens": num_acc,
            "query_start_loc": qsl,
            "max_query_len": WINDOW,
        }
        if QLEN > 1
        else {}
    )
    q, k, v = causal_conv1d_update(
        mixed,
        conv,
        conv_w.to(torch.bfloat16),
        PROJ,
        PROJ,
        None,
        "silu",
        conv_state_indices=conv_slots,
        validate_data=False,
        **extra,
    )
    q, k, v = (x.reshape(1, T, NH, HD) for x in (q, k, v))
    out = torch.empty(T, NH, HD, dtype=torch.bfloat16, device=dev)
    replayssm_sigmoid_gating_delta_rule(
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        ckpt=ckpt,
        buf_k=bufs[0],
        buf_u=bufs[1],
        buf_g=bufs[2],
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=qsl,
        max_query_len=WINDOW,
        o=out,
        use_qk_l2norm_in_kernel=True,
        lower_bound=LB,
    )
    normed = rmsnorm_gated(
        out, on_w.to(torch.bfloat16), out_gate.reshape(T, NH, HD), 1e-5
    )
    return normed.reshape(T, PROJ)


def check(rows):
    req = rows // QLEN
    plan = Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=1
    )
    replay_k1 = plan.make_k1_local_runner(spec(rows, True)).prepare(dev)
    plain_k1 = plan.make_k1_local_runner(spec(rows, False)).prepare(dev)
    names = {"replay": replay_k1.entrypoint.name, "plain": plain_k1.entrypoint.name}
    print("built", rows, names, flush=True)
    slots = req + 3
    # the requests' checkpoint slots, not in request order
    slot_idx = (torch.randperm(slots, generator=gen, device=dev)[:req] + 0).to(
        torch.int32
    )
    conv0 = bf16(slots, 3 * PROJ, SL, scale=0.5)
    ckpt0 = torch.randn((slots, NH, HD, HD), generator=gen, device=dev) * 0.05
    # stale records: never read (each side's staging stops at the committed)
    bufs0 = [
        bf16(slots, NH, CAP, HD, scale=30.0),
        bf16(slots, NH, CAP, HD, scale=30.0),
        torch.randn((slots, NH, CAP, HD), generator=gen, device=dev) * 30.0,
    ]
    sides = {}
    for side in ("mono", "prod"):
        sides[side] = {
            "conv": conv0.clone(),
            "ckpt": ckpt0.clone(),
            "bufs": [b.clone() for b in bufs0],
            "wp": torch.full((slots,), -1, dtype=torch.int32, device=dev),
        }
    m = sides["mono"]
    table = torch.tensor(
        [
            {
                "slot_idx": slot_idx,
                "write_pos": m["wp"],
                "rbuf_k": m["bufs"][0],
                "rbuf_u": m["bufs"][1],
                "rbuf_g": m["bufs"][2],
            }[name].data_ptr()
            for name in REPLAY_TABLE
        ],
        dtype=torch.int64,
        device=dev,
    )
    # K1's slot table under ReplaySSM: column 0 (the conv's) the checkpoint
    st = torch.full((req, QLEN), -1, dtype=torch.int32, device=dev)
    st[:, 0] = slot_idx
    reports = []
    num_acc = torch.ones(req, dtype=torch.int32, device=dev)
    for step in range(STEPS):
        if step:
            num_acc = torch.randint(
                1, QLEN + 1, (req,), generator=gen, device=dev, dtype=torch.int32
            )
        if step == SENTINEL_STEP:
            for side in sides.values():
                side["wp"][slot_idx[0].long()] = -1
        for side in sides.values():
            replayssm_commit(side["wp"], slot_idx, num_acc, WINDOW, CAP)
        h = sides["mono"]["wp"][slot_idx.long()].clamp_min(0)
        assert torch.equal(h, sides["prod"]["wp"][slot_idx.long()].clamp_min(0))
        flush = h + 2 * WINDOW > CAP
        base = torch.where(flush, torch.zeros_like(h), h)
        prefix = bf16(rows, HIDDEN)
        report = {
            "rows": rows,
            "step": step,
            "h": h.tolist(),
            "flush": flush.int().tolist(),
            "acc": num_acc.tolist(),
        }
        if step == 0:
            # K1 without ReplaySSM from the checkpoints: the same step
            p_conv, p_rs = conv0.clone(), ckpt0.clone()
            p_st = st.clone()
            p_st[:, 1:] = slot_idx.unsqueeze(1) * 0 - 1
            p_core, _ = k1_run(plain_k1, prefix, p_conv, p_rs, p_st, num_acc)
        ckpt_before = m["ckpt"].clone()
        bufs_before = [b.clone() for b in m["bufs"]]
        core, proj = k1_run(replay_k1, prefix, m["conv"], m["ckpt"], st, num_acc, table)
        p = sides["prod"]
        core_p = production(
            proj, p["conv"], p["ckpt"], p["bufs"], p["wp"], slot_idx, num_acc, req
        )
        torch.cuda.synchronize()
        idx = slot_idx.long()
        report["core_out"] = diff(core, core_p)
        report["conv"] = diff(m["conv"][idx], p["conv"][idx])
        report["ckpt"] = diff(m["ckpt"][idx], p["ckpt"][idx])
        written = torch.zeros((req, CAP), dtype=torch.bool, device=dev)
        for r in range(req):
            written[r, : int(base[r]) + QLEN] = True
        for name, b in zip("kug", range(3)):
            mine = m["bufs"][b][idx].permute(0, 2, 1, 3)[written]
            ref = p["bufs"][b][idx].permute(0, 2, 1, 3)[written]
            report[f"rec_{name}"] = diff(mine, ref)
        # K1's own replay: a flushed checkpoint is its records folded in
        # (f64 on the host), any other one untouched
        rebuild, kept = [], []
        for r in range(req):
            slot = int(slot_idx[r])
            if not bool(flush[r]):
                kept.append(bool(torch.equal(m["ckpt"][slot], ckpt_before[slot])))
                continue
            state = ckpt_before[slot].double()
            for j in range(int(h[r])):
                kj = bufs_before[0][slot, :, j].double()
                uj = bufs_before[1][slot, :, j].double()
                gj = bufs_before[2][slot, :, j].double()
                state = state * gj.exp().unsqueeze(1) + uj.unsqueeze(2) * kj.unsqueeze(1)
            rebuild.append(diff(m["ckpt"][slot], state)["max_abs"])
        report["flush_vs_own_records_max_abs"] = max(rebuild) if rebuild else None
        report["unflushed_ckpt_unchanged"] = all(kept)
        if step == 0 and os.environ.get("CONV_DEBUG"):
            x = proj[:, : 3 * PROJ].to(torch.bfloat16).view(req, QLEN, -1)
            acc = num_acc.long()
            for r in range(min(req, 2)):
                slot = int(slot_idx[r])
                old = conv0[slot]
                keep = old[:, int(acc[r]) : int(acc[r]) + SL - QLEN]
                host = torch.cat([keep, x[r].t()], dim=1)
                print(
                    "conv_debug",
                    r,
                    "mono_vs_host",
                    diff(m["conv"][slot], host)["max_abs"],
                    "prod_vs_host",
                    diff(p["conv"][slot], host)["max_abs"],
                    "prod_vs_old",
                    diff(p["conv"][slot], old)["max_abs"],
                    flush=True,
                )
                for ch in (0, 1600):
                    print(
                        "conv_debug ch",
                        ch,
                        "old",
                        [round(v, 3) for v in old[ch].float().tolist()],
                        "x",
                        [round(v, 3) for v in x[r, :, ch].float().tolist()],
                        "prod",
                        [round(v, 3) for v in p["conv"][slot, ch].float().tolist()],
                        flush=True,
                    )
        if step == 0:
            report["vs_plain_core_out"] = diff(core, p_core)
            report["vs_plain_conv"] = diff(m["conv"][idx], p_conv[idx])
        print(json.dumps(report), flush=True)
        reports.append(report)
    return replay_k1, plain_k1, sides, table, reports


if __name__ == "__main__":
    keep = [check(rows) for rows in WIDTHS]
    print("done", flush=True)
