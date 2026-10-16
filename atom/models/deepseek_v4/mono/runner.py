# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One decode step of up to ``MAX_TOKENS`` tokens on the V4 layer kernel: every
decoder layer one launch (``op.Dsv4MonoKernel``), over ATOM's own state.

The kernel reads and writes ATOM's fp8 KV pool, window rows (``swa_dest_rows``),
key lists (``kv_indices_{hca,csa}``), block tables, compressor rings and the CSA
indexer's FP4 key cache in place, so a sequence can switch paths between steps.
The routed experts are ATOM's FusedMoE tensors in its gfx950 MXFP4 layout, read
in place and computed as the original path computes them (A8W4: refused
otherwise, ``baseline_refusal``); the rest of a layer (~0.1 GB a rank) is
packed from the checkpoint at install time, since ATOM post-processes those
matrices.

A step's width is padded to a launch width (1, 2, 4 or 8 samples) by repeating
row 0, whose extra outputs are dropped. A speculative verify (MTP, DSpark:
rectangular, q-bucketed or ragged) launches each request's rows as they come,
contiguous consecutive tokens, with each row's index among them (``run_pos``). A launch
width's kernels and their per-width buffers (scratch, peer memory) are built on
its first step (``WidthBuilds``); every layer of one attention variant
(compress ratio) shares them.
"""

import inspect
import logging
import time
from dataclasses import dataclass

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.deepseek_v4.mono.config import LAUNCH_WIDTHS, MAX_TOKENS
from atom.models.deepseek_v4.mono.op import Dsv4MonoKernel, Dsv4Variant
from atom.models.deepseek_v4.mono.packing import (
    pack_layer_weights,
    pack_mxfp4,
    pack_mxfp4_scales,
)
from atom.models.deepseek_v4.mono.weights import Checkpoint, load_expert, load_layer
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def baseline_refusal(experts, rows: int = MAX_TOKENS) -> str | None:
    """Why the original path's routed experts do not compute what the kernel does
    -- MXFP4 weights on MXFP8 activations (A8W4), gate / up interleaved -- on a
    decode step of up to ``rows`` rows, or None. Asked of aiter's own activation
    rule, so mono serves only a deployment whose decode it reproduces (the
    recipe's ``ATOM_MOE_GU_ITLV=1 AITER_BF16_FP8_MOE_BOUND=0``); any other keeps
    the original path, and its precision."""
    qm = experts.quant_method
    if getattr(qm, "use_triton", False) or getattr(qm, "use_triton_decode", False):
        return "routed experts on the Triton MoE path"
    if not getattr(qm, "is_guinterleave", False):
        return "routed experts' gate / up not interleaved (ATOM_MOE_GU_ITLV=1)"
    if getattr(qm, "fp8_activations", False):
        return None  # fused_moe is handed fp8 activations whatever the rows
    try:
        from aiter.fused_moe import resolve_activation_dtype
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.utility import dtypes
    except ImportError as e:
        return f"cannot tell the original path's activation format: {e}"
    for m in range(1, rows + 1):
        got = resolve_activation_dtype(
            qm.quant_type,
            experts.w13_weight.dtype,
            activation=experts.activation,
            gate_mode=GateMode.INTERLEAVE,
            M=m,
            hidden_dtype=torch.bfloat16,
            has_a1_scale=getattr(experts, "w13_input_scale", None) is not None,
        )
        if got != dtypes.fp8:
            return (
                f"the original path's {m}-row decode runs the routed experts on"
                f" {got} activations, not MXFP8 (AITER_BF16_FP8_MOE_BOUND=0)"
            )
    return None


def run_positions(seq: torch.Tensor) -> torch.Tensor:
    """Each row's index among its request's rows, ``seq`` the rows' request ids
    (a request's rows contiguous, in position order: ATOM's verify layout): the
    row minus its request's first row. int32, on the device, graph-capturable."""
    n = seq.numel()
    ar = torch.arange(n, device=seq.device)
    start = torch.ones(n, dtype=torch.bool, device=seq.device)
    start[1:] = seq[1:] != seq[:-1]
    first = torch.where(start, ar, 0).cummax(0).values
    return (ar - first).to(torch.int32)


@dataclass(frozen=True)
class AuxTaps:
    """The drafter's taps of target layers (DSpark's ``dspark_target_layer_ids``):
    layer id -> the buffer its forward hook fills with that layer's post-layer
    residual averaged over the hc copies. A mono layer does not run the module, so
    the hook does not fire and the runner fills the buffer itself."""

    by_layer: dict


def bind_aux_taps(drafter, hidden_size: int) -> AuxTaps:
    if drafter is None or not getattr(drafter, "_captures_aux", False):
        return AuxTaps({})
    draft_config = drafter.speculative_config.draft_model_hf_config
    layer_ids = tuple(
        int(i) for i in getattr(draft_config, "dspark_target_layer_ids", ())
    )
    buffers = tuple(getattr(drafter, "_aux_buffers", ()))
    if not layer_ids or len(layer_ids) != len(buffers):
        raise MonoUnsupported(
            f"{len(buffers)} aux buffers for target layers {layer_ids}"
        )
    if min(layer_ids) < 0:
        raise MonoUnsupported(f"an embedding tap (target layers {layer_ids})")
    for buf in buffers:
        if (
            buf.dim() != 2
            or buf.shape[1] != hidden_size
            or not buf.is_contiguous()
            or buf.dtype != torch.bfloat16
        ):
            raise MonoUnsupported(f"aux buffer {tuple(buf.shape)} {buf.dtype}")
    return AuxTaps(dict(zip(layer_ids, buffers)))


def _zero_abi(launcher) -> KernelAbi:
    """A launcher's parameters but ``stream``: what a compile-only call zeroes."""
    fn = getattr(launcher, "func", None) or launcher._func
    params = tuple(inspect.signature(fn).parameters)
    assert params[-1] == "stream", params
    return KernelAbi(params[:-1])


class Dsv4MonoDecodeRunner:
    """Owns the packed layer weights (bound at install, before the KV cache is
    sized) and, per launch width, the compiled kernel and its buffers.

    Construction is collective over the TP group: every rank binds and checks its
    shard, then all agree (``consensus.bind_agreed``), so a rank that refuses
    turns mono off on every rank."""

    def __init__(self, causal_lm, atom_config, spec_steps: int, drafter=None) -> None:
        tp = get_tp_group()
        self.group = tp.cpu_group
        self.tp, self.rank = tp.world_size, tp.rank_in_group
        # a verify step holds up to spec_steps + 1 rows a request (ATOM's
        # max_spec_steps: its compressor rings are C_ROWS + spec_steps rows)
        self.spec_steps = spec_steps
        self.max_seq = atom_config.max_model_len
        bind_agreed(lambda: self._bind(causal_lm, atom_config, drafter), self.group)
        self.widths = WidthBuilds(
            self._build_width,
            self._width_kernels,
            self.group,
            "DeepSeek-V4 mono",
        )
        self.calls: dict[int, int] = {}

    # ------------------------------------------------------------------ setup
    def _bind(self, causal_lm, atom_config, drafter) -> None:
        """This rank's checks and packed weights; raises ``MonoUnsupported``."""
        self.model = causal_lm.model
        # the drafter's taps of the target's layers (DSpark), which its hooks
        # would fill: a mono layer does not run the module, so the runner does
        self.aux = bind_aux_taps(drafter, atom_config.hf_config.hidden_size)
        hf = atom_config.hf_config
        self.dev = torch.device("cuda", torch.cuda.current_device())
        cus = torch.cuda.get_device_properties(self.dev).multi_processor_count
        if cus != BLOCKS:
            raise MonoUnsupported(
                f"{cus} CUs: the kernel needs all {BLOCKS} CTAs co-resident"
            )
        if self.model.enable_res_preshuffle:
            raise MonoUnsupported("the preshuffled mHC residual (gfx1250)")
        self.n_layers = hf.num_hidden_layers
        self.hc = hf.hc_mult
        # use_index_cache: a CSA layer that reuses an earlier layer's top-k runs no
        # indexer of its own, and the kernel would select for it afresh
        shared = [
            li
            for li in range(self.n_layers)
            if getattr(self.model.layers[li].attn, "skip_topk", False)
        ]
        if shared:
            raise MonoUnsupported(
                f"CSA layers {shared[:4]}.. reuse an earlier layer's index top-k"
                " (use_index_cache)"
            )
        # before anything is packed: the kernel computes the routed experts as
        # the original path does, or mono is off
        for li in range(self.n_layers):
            why = baseline_refusal(self.model.layers[li].ffn.experts)
            if why is not None:
                raise MonoUnsupported(f"layer {li}: {why}")
        ck = Checkpoint(atom_config.model)
        t0 = time.perf_counter()
        threads = torch.get_num_threads()
        torch.set_num_threads(4)  # 8 ranks x all cores oversubscribe the node
        try:
            # per layer: (LayerWeights, its packed weights, ATOM's expert bank)
            self.layers = [self._bind_layer(ck, li) for li in range(self.n_layers)]
        finally:
            torch.set_num_threads(threads)
        logger.info(
            "DeepSeek-V4 mono: rank %d packed %d layers in %.0fs",
            self.rank,
            self.n_layers,
            time.perf_counter() - t0,
        )
        # ATOM's own rope values (fp32 [positions, rope_dim / 2]), so decode
        # rotates as prefill did
        tables = {}
        self.rope_of = []
        for li in range(self.n_layers):
            rot = self.model.layers[li].attn.rotary_emb
            key = id(rot.cos_cache)
            if key not in tables:
                n = rot.cos_cache.shape[0]
                tables[key] = (
                    rot.cos_cache.float().reshape(n, -1).contiguous(),
                    rot.sin_cache.float().reshape(n, -1).contiguous(),
                )
            self.rope_of.append(tables[key])

    def _bind_layer(self, ck, li):
        W = load_layer(
            ck, li, rank=self.rank, tp=self.tp, device=self.dev, experts=False
        )
        cfg = W.cfg
        cfg.max_seq = self.max_seq
        # ATOM's indexer rotates neither queries nor keys
        cfg.indexer_hadamard = False
        cfg.validate()
        experts = self._atom_experts(li, cfg)
        if li == 0:
            self._check_expert_layout(ck, li, experts)
        # packed once, shared by every launch width's op of this layer
        packed = pack_layer_weights(W.t, experts)
        # the raw copies of what was packed are dead weight
        for name in packed:
            W.t.pop(name, None)
        torch.cuda.synchronize()
        return W, packed

    def _atom_experts(self, li, cfg):
        """Layer ``li``'s routed expert bank as ATOM holds it, viewed in the
        kernel's terms: no copy. ATOM's gfx950 MXFP4 prep shuffles weights and
        E8M0 scales in aiter's GU-interleaved layout, which the kernel reads."""
        fm = self.model.layers[li].ffn.experts
        if not (
            getattr(fm.w13_weight, "is_shuffled", False)
            and getattr(fm.w2_weight, "is_shuffled", False)
        ):
            raise MonoUnsupported(
                f"layer {li}: experts not in the shuffled MXFP4 layout"
            )
        e, inter, hidden = cfg.n_experts, cfg.inter, cfg.hidden

        def pad8(n):
            return -(-n // 8) * 8

        bank = {
            "w_ug": fm.w13_weight.data.view(torch.uint8).view(
                e, 2 * inter, hidden // 2
            ),
            "s_ug": fm.w13_weight_scale.data.view(torch.uint8).view(
                e, 2 * inter, pad8(hidden // 32)
            ),
            "w_dn": fm.w2_weight.data.view(torch.uint8).view(e, hidden, inter // 2),
            "s_dn": fm.w2_weight_scale.data.view(torch.uint8).view(
                e, hidden, pad8(inter // 32)
            ),
        }
        for name, t in bank.items():
            if not t.is_contiguous():
                raise MonoUnsupported(f"layer {li} {name}: not contiguous")
        return bank

    def _check_expert_layout(self, ck, li, bank) -> None:
        """Expert 0 of layer ``li`` from the checkpoint, laid out by the kernel's
        packer, must equal ATOM's bytes: a layout ATOM changes underneath refuses
        mono at install instead of serving wrong output."""
        ug, ugs, dn, dns = load_expert(ck, li, 0, self.rank, self.tp, self.dev)
        want = {
            "w_ug": pack_mxfp4(ug[None], True)[0],
            "s_ug": pack_mxfp4_scales(ugs[None], True)[0],
            "w_dn": pack_mxfp4(dn[None], False)[0],
            "s_dn": pack_mxfp4_scales(dns[None], False)[0],
        }
        for name, w in want.items():
            if not torch.equal(bank[name][0], w):
                raise MonoUnsupported(
                    f"layer {li} expert 0 {name}: ATOM's expert layout is not the"
                    " one the kernel reads"
                )

    def _build_width(self, S: int) -> dict:
        """The S-sample launch: a variant (kernel, scratch, peer memory) per
        compress ratio and an op per layer. Collective (the peer handshakes):
        every rank builds the same widths in the same order."""
        self._check_rings()
        variants: dict[int, Dsv4Variant] = {}
        ops = []
        for li, (W, packed) in enumerate(self.layers):
            r = W.cfg.compress_ratio
            if r not in variants:
                variants[r] = Dsv4Variant(
                    W.cfg,
                    S,
                    rank=self.rank,
                    npes=self.tp,
                    group=self.group,
                    spec_steps=self.spec_steps,
                )
            ops.append(
                Dsv4MonoKernel(
                    W,
                    S,
                    rank=self.rank,
                    npes=self.tp,
                    variant=variants[r],
                    packed=packed,
                    spec_steps=self.spec_steps,
                )
            )
        # every variant's peer buffer zeroed before the ranks agree they hold
        # the width (``WidthBuilds``): a peer's first push must not land on an
        # unzeroed buffer, nor be wiped by a late zeroing
        torch.cuda.synchronize()
        return {"variants": variants, "ops": ops}

    @staticmethod
    def _width_kernels(built: dict) -> list:
        out = []
        for var in built["variants"].values():
            out.append((var.launch, _zero_abi(var.launch)))
            out.append((var._advance, _zero_abi(var._advance)))
        return out

    def launch_width(self, rows: int) -> int | None:
        """The launch width of a ``rows``-row step, or None (not served)."""
        # with a drafter, a step of one request's verify at most (its anchor and
        # drafted tokens), launched at its own width (a pad row is a whole sample's
        # work); a wider one, more requests, keeps the original path. Without one, a
        # step pads to the next of LAUNCH_WIDTHS (fewer builds for its graphs)
        if self.spec_steps:
            return rows if rows <= self.spec_steps + 1 else None
        return next((s for s in LAUNCH_WIDTHS if s >= rows), None)

    def prepare(self, rows: int) -> bool:
        """Whether every rank holds the launch of a ``rows``-row step."""
        S = self.launch_width(rows)
        return S is not None and self.widths.prepare(S)

    # ---------------------------------------------------------------- inputs
    def _index_rows(self, md, ratio, S, n_keys, window, pos):
        """The kernel's [S, n_keys] rows for one compress class: window rows in
        [0, window), then (HCA) the compressed entries. ATOM's per-token CSR
        slice is [compressed head | window tail]; a CSA head is the indexer's
        pick, which the kernel makes itself, so only its tail is read here."""
        indptr = md.kv_indptr_hca if ratio == 128 else md.kv_indptr_csa
        flat = md.kv_indices_hca if ratio == 128 else md.kv_indices_csa
        start, end = indptr[:S].long(), indptr[1 : S + 1].long()
        pos = pos[:S].long()
        nw = torch.clamp(pos + 1, max=window)
        head = end - start - nw
        out = torch.full((S, n_keys), -1, dtype=torch.int32, device=self.dev)
        j = torch.arange(window, device=self.dev)
        src = start[:, None] + head[:, None] + j[None]
        ok = j[None] < nw[:, None]
        out[:, :window] = torch.where(ok, flat[src.clamp(max=flat.numel() - 1)], -1)
        if ratio == 128:
            j = torch.arange(n_keys - window, device=self.dev)
            src = start[:, None] + j[None]
            ok = j[None] < head[:, None]
            out[:, window:] = torch.where(ok, flat[src.clamp(max=flat.numel() - 1)], -1)
        return out

    def _state(self, li, slots, slots_in):
        attn = self.model.layers[li].attn
        comp = attn.compressor
        st = {
            "kv_state": comp.kv_state,
            "score_state": comp.score_state,
            "state_slots": slots,
            "state_slots_in": slots_in,
            "st_kv": comp.kv_state.stride(0),
        }
        if attn.indexer is not None:
            ic = attn.indexer.compressor
            st.update(
                i_kv_state=ic.kv_state,
                i_score_state=ic.score_state,
                st_i=ic.kv_state.stride(0),
                i_cache=attn.indexer.kv_cache,
                # one pool shared by every sequence: the block id alone places an entry
                i_cache_s=attn.indexer.kv_scale,
            )
        return st

    def _check_rings(self) -> None:
        """ATOM widens the compressor rings by K under MTP, as the kernel does.
        The states bind after install, so this is checked when a width is built
        (its first step, the cache bound)."""
        for li, (W, _) in enumerate(self.layers):
            comp = getattr(self.model.layers[li].attn, "compressor", None)
            if comp is not None:
                want = W.cfg.c_rows + self.spec_steps
                if comp.kv_state.shape[1] != want:
                    raise MonoUnsupported(
                        f"layer {li}: compressor state has {comp.kv_state.shape[1]}"
                        f" rows, the kernel's ring {want}"
                    )

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The final residual stream [rows, hc_mult, hidden], as the model returns."""
        md = get_forward_context().attn_metadata
        rows = positions.shape[0]
        S = self.launch_width(rows)
        built = self.widths[S]
        ops = built["ops"]
        self.calls[rows] = self.calls.get(rows, 0) + 1
        if self.rank == 0 and self.calls[rows] in (1, 10, 100, 1000, 10000):
            logger.info(
                "DeepSeek-V4 mono: decode steps by rows %s (capturing: %s)",
                self.calls,
                torch.cuda.is_current_stream_capturing(),
            )
        # A CUDA graph's pad rows (batch id -1) carry placeholder slots, rows and
        # block tables that ATOM's kernels skip. The kernel has no skip, so each
        # pad row (and each row past the step's, up to the launch width) becomes
        # a copy of row 0 -- its request's first row: run position 0 -- rewriting
        # that row's own values instead of writing through the placeholders into
        # live sequences' state. Pads come last, so none sits inside a request's
        # rows.
        bid = md.batch_id_per_q_token
        ar = torch.arange(S, device=self.dev)
        src = torch.zeros_like(ar)
        if bid is None:
            seq = ar[:rows]
            src[:rows] = ar[:rows]
        else:
            seq = bid[:rows].long()
            src[:rows] = torch.where(seq >= 0, ar[:rows], 0)

        def take(t):
            return t[src].contiguous()

        pos = take(positions[:rows].to(torch.int32))
        tokens = take(input_ids[:rows].to(torch.int32))
        seq = take(seq)
        # a fork reads the source's state (state_slot_in) and writes a fresh slot
        slots = md.state_slot_out[seq].to(torch.int32).contiguous()
        slots_in = md.state_slot_in[seq].to(torch.int32).contiguous()
        bt = md.block_tables[seq].to(torch.int32).contiguous()
        run = take(run_positions(seq[:rows] if bid is None else bid[:rows].long()))
        env = int(md.envelope_rows)
        by_ratio = {}
        for li, op in enumerate(ops):
            cfg = op.W.cfg
            r = cfg.compress_ratio
            if r not in by_ratio:
                dest = torch.stack(
                    [
                        take(md.swa_dest_rows[r][:rows].to(torch.int32)),
                        torch.zeros(S, dtype=torch.int32, device=self.dev),
                    ]
                )
                idx = self._index_rows(md, r, rows, cfg.n_keys, cfg.window, pos[:rows])
                by_ratio[r] = (dest, take(idx))
        h = self.model.embed(tokens)
        h = h.unsqueeze(1).repeat(1, self.hc, 1).contiguous()
        for li, op in enumerate(ops):
            attn = self.model.layers[li].attn
            dest, idx = by_ratio[op.W.cfg.compress_ratio]
            cos, sin = self.rope_of[li]
            h = op.forward(
                h,
                pos,
                (attn.unified_kv.view(torch.uint8), attn.unified_kv_rope),
                dest,
                idx,
                cos,
                sin,
                layer=li,
                tokens=tokens,
                block_tables=bt,
                env_rows=env,
                state=self._state(li, slots, slots_in),
                run_pos=run,
            )
            tap = self.aux.by_layer.get(li)
            if tap is not None:
                # the drafter's extract: the post-layer residual over its hc copies
                tap[:rows].copy_(h[:rows].mean(dim=1))
        for var in built["variants"].values():
            var.advance_step()
        return h[:rows]
