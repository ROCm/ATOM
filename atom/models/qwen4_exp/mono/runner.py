# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next mono decode on one GPU: the bound weights, the step's
buffers, and the forward that runs them.

A step of ``rows`` requests (one token each) is:

1. ``embed_tokens`` repeated over the four HC streams: layer 0's input ``h0``.
2. The PLE layer's hashed n-gram embedding (ATOM's ``advance_ngram_state`` and
   ``ple_embedding``, as its own forward does them): the ``emb`` row the PLE
   table points at. The rest of the PLE prelude runs in the kernels.
3. ``step_meta``: the step counter and the state tables' per-step words.
4. The 48 layer launches (``kernels.kernel.build_layer``), layer ``i`` from
   ``chain[i - 1]`` (``h0``) to ``chain[i]``, reading and writing ATOM's caches
   in place (``state``).
5. The final HC mix of ``chain[47]``: the hidden state ``compute_logits``
   takes, as the model's own forward returns it. ATOM's ``mix_fused`` mixes
   up to ``HC_ROWS_MAX`` rows one program set per row (``hc_rows``) and more
   through GEMMs, which round otherwise; the runner takes the per-row path at
   every width, so a row mixes as it would alone.

With MTP (``--method mtp``, k draft tokens) every decode step is a verify of
one request's ``k + 1`` tokens, run by the seq builds (``build_layer(...,
seq=True)``): step 3 is ``write_verify_meta``, which also gathers the PLE
window the rows read, ``commit_ple`` follows step 4, and step 5 is skipped: the
model then returns its HC streams ``[rows, hc_count, hidden]`` unmixed
(``return_hc_state``), for ``compute_logits`` and the drafter to mix.

Every buffer a step reads is allocated once per width (``prepare``) and every
per-step input is one of ATOM's persistent buffers, so a captured decode graph
replays the step unchanged.
"""

import logging
from dataclasses import dataclass

import torch

from atom.model_ops.qwen4_exp.ops.hc_fused import hc_rows
from atom.model_ops.qwen4_exp.ops.ple import advance_ngram_state
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils.forward_context import get_forward_context

from . import weights as W
from .config import MAX_ROWS
from .kernels.config import from_dict
from .kernels.kernel import LAYER_ARGS, STAGES, build_layer
from .kernels.layout import MAX_ROWS as KERNEL_ROWS
from .kernels.layout import scratch_bytes
from .kernels.stages import ple as ple_stage
from .kernels.stages import qsa as qsa_stage
from .kernels.stages.common import DELTA_STATE
from .prologue import StepPrologue
from .prologue import refusal as prologue_refusal
from .state import QSA_CACHES as CACHE_PAGES
from .state import bind_state
from .step_meta import (
    PLE_COLS,
    QSA_CACHES,
    commit_ple,
    write_step_meta,
    write_verify_meta,
)

logger = logging.getLogger("atom")

I64, BF16 = torch.int64, torch.bfloat16


@dataclass
class Tables:
    """A width's state tables (``step_meta``): ``pos`` [rows] int32, ``delta``
    [DeltaNet layers, rows, DELTA_STATE], ``ple`` [rows, ple.STATE], ``qsa``
    [QSA layers, rows, qsa.STATE] int64; each DeltaNet layer's slot 0
    addresses ``gdn_conv`` / ``gdn_rec`` and the slot strides in bytes; each
    QSA layer's cache addresses ``qsa_caches`` [QSA layers, QSA_CACHES]."""

    pos: torch.Tensor
    delta: torch.Tensor
    ple: torch.Tensor
    qsa: torch.Tensor
    gdn_conv: torch.Tensor
    gdn_rec: torch.Tensor
    qsa_caches: torch.Tensor
    conv_slot: int
    rec_slot: int
    ple_base: int
    ple_slot: int


@dataclass
class Width:
    """A ``rows``-row step: its launchers by layer kind, buffers, tables and
    each layer's kernel arguments (``LAYER_ARGS[1:]`` addresses). A verify
    step's (``seq``) also has the PLE window its rows read, ``[QKV, 9]``, and
    their snapshots, ``[rows, QKV, 9]`` (``step_meta``)."""

    rows: int
    launch: dict
    h0: torch.Tensor
    chain: torch.Tensor
    emb: torch.Tensor
    tables: Tables
    args: list
    seq: bool = False
    ple_window: torch.Tensor | None = None
    ple_snaps: torch.Tensor | None = None


class Qwen4MonoDecodeRunner:
    """Binds the weights and ATOM's caches (``MonoUnsupported`` if any is not
    as the kernels need) and runs decode steps of up to ``MAX_ROWS`` rows, or
    with MTP the verify steps of ``verify_rows`` rows."""

    max_rows = MAX_ROWS

    def __init__(self, model, atom_config, metadata_builder) -> None:
        g = from_dict(atom_config.hf_config.to_dict())
        self.geometry = g
        self.kinds = [g.kind(i) for i in range(g.layers)]
        inner = model.model
        layers = inner.layers
        if len(layers) != g.layers:
            raise MonoUnsupported(f"{len(layers)} layers loaded of {g.layers}")
        self.embed_tokens = inner.embed_tokens
        self.mixer = inner.hyper_connection_mixer
        spec = atom_config.speculative_config
        spec_tokens = 0 if spec is None else spec.num_speculative_tokens
        # a verify step's rows, or 0 without MTP
        self.verify_rows = spec_tokens + 1 if spec_tokens else 0
        if getattr(inner, "return_hc_state", False) != bool(spec_tokens):
            raise MonoUnsupported("the model returns its HC streams without MTP")
        self.state = bind_state(metadata_builder, layers, self.kinds, spec_tokens)
        self.delta_layers = [i for i, k in enumerate(self.kinds) if k != "qsa"]
        self.qsa_layers = [i for i, k in enumerate(self.kinds) if k == "qsa"]
        (self.ple_layer,) = g.ple_layers
        self.ple = layers[self.ple_layer].ple
        if self.ple is None:
            raise MonoUnsupported(f"layer {self.ple_layer} has no PLE")
        # steps 1 and 2 in one launch, else ATOM's ops
        why = prologue_refusal(self.embed_tokens, self.ple.ple_embedding, g.hc_width)
        self.prologue = None
        if why is None:
            self.prologue = StepPrologue(self.embed_tokens, self.ple.ple_embedding)
        else:
            logger.info("Qwen3.8 mono decode: steps 1 and 2 op for op: %s", why)
        self.bound = [
            W.bind_layer(layers[i], i, self.kinds[i], g.experts)
            for i in range(g.layers)
        ]
        positions = atom_config.max_model_len
        self.rope = W.rope(W.parameters(layers[self.qsa_layers[0]]), positions)

        dev = self.embed_tokens.weight.device
        self.device = dev
        self.dummy = torch.zeros(64, dtype=torch.int32, device=dev)
        scratch = scratch_bytes(KERNEL_ROWS)
        self.scratch = torch.zeros(scratch // 4, dtype=torch.int32, device=dev)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)
        # a padded row's state (``step_meta``): its writes land here and its
        # reads see only what padded rows wrote
        st = self.state
        self._nulls = {
            "conv": torch.zeros_like(st.conv[0, 0]),
            "rec": torch.zeros_like(st.rec[0, 0]),
            "ple": torch.zeros_like(st.ple_conv[0]),
            # a block table of page 0 of one-page caches
            "pages": torch.zeros(16, dtype=torch.int32, device=dev),
        }
        self.nulls = {k: v.data_ptr() for k, v in self._nulls.items()}
        self._null_caches = [
            torch.zeros(1, *CACHE_PAGES[c][1], dtype=torch.bfloat16, device=dev)
            for c in QSA_CACHES
        ]
        self.nulls["caches"] = torch.tensor(
            [t.data_ptr() for t in self._null_caches], dtype=I64, device=dev
        )
        self.widths: dict[int, Width] = {}

    # ------------------------------------------------------------ widths
    def prepare(self, rows: int) -> bool:
        """Build (on first use) the ``rows``-row step; False if mono does not
        serve that width."""
        if rows in self.widths:
            return True
        # with MTP every decode step is a verify of ``verify_rows`` rows
        seq = bool(self.verify_rows)
        served = rows == self.verify_rows if seq else 0 < rows <= self.max_rows
        if not served:
            return False
        g, dev = self.geometry, self.device
        launch = {
            k: build_layer(STAGES[k], g.experts, rows=rows, seq=seq)
            for k in set(self.kinds)
        }
        h0 = torch.zeros(rows, g.hc_width, dtype=torch.bfloat16, device=dev)
        chain = torch.zeros(
            g.layers, rows, g.hc_width, dtype=torch.bfloat16, device=dev
        )
        emb = torch.zeros(rows, g.ple_dim, dtype=torch.bfloat16, device=dev)
        window = snaps = None
        if seq:
            snaps = torch.zeros(rows, g.hc_width, PLE_COLS, dtype=BF16, device=dev)
            window = torch.zeros_like(snaps[0])
        tables = self._tables(rows, emb, window, snaps)
        args = []
        for i, b in enumerate(self.bound):
            a = dict(b.args)
            a["h_in"] = h0 if i == 0 else chain[i - 1]
            a["h_out"] = chain[i]
            if b.kind == "qsa":
                a["qsa"] = tables.qsa[self.qsa_layers.index(i)]
            else:
                a["conv_in"] = tables.delta[self.delta_layers.index(i)]
            if b.kind == "delta_ple":
                a["ple"] = tables.ple
            args.append([a.get(k, self.dummy).data_ptr() for k in LAYER_ARGS[1:]])
        self.widths[rows] = Width(
            rows, launch, h0, chain, emb, tables, args, seq, window, snaps
        )
        logger.info(
            "Qwen3.8 mono decode: %d-row %s step prepared",
            rows,
            "verify" if seq else "decode",
        )
        return True

    def _tables(self, rows: int, emb: torch.Tensor, window, snaps) -> Tables:
        """The width's tables with every entry but the per-step words filled;
        a verify step's PLE rows read ``window`` and write ``snaps[r]``."""
        st, dev = self.state, self.device
        pos = torch.zeros(rows, dtype=torch.int32, device=dev)
        delta = torch.zeros(len(self.delta_layers), rows, len(DELTA_STATE), dtype=I64)
        ple = torch.zeros(rows, len(ple_stage.STATE), dtype=I64)
        weights = self.bound[self.ple_layer].ple
        for r in range(rows):
            entries = dict(weights, emb=emb[r])
            if window is not None:
                entries |= {"conv_in": window, "conv_out": snaps[r]}
            for k, name in enumerate(ple_stage.STATE):
                if name in entries:
                    ple[r, k] = entries[name].data_ptr()
        qsa = torch.zeros(len(self.qsa_layers), rows, len(qsa_stage.STATE), dtype=I64)
        for q, i in enumerate(self.qsa_layers):
            for r in range(rows):
                entries = self.bound[i].qsa_norms | st.qsa[i]
                entries |= {"pos": pos[r], "cos": self.rope[0], "sin": self.rope[1]}
                for k, name in enumerate(qsa_stage.STATE):
                    if name in entries:
                        qsa[q, r, k] = entries[name].data_ptr()
        gdn_conv = [st.conv[st.gdn_rows[i], 0].data_ptr() for i in self.delta_layers]
        gdn_rec = [st.rec[st.gdn_rows[i], 0].data_ptr() for i in self.delta_layers]
        caches = [
            [st.qsa[i][c].data_ptr() for c in QSA_CACHES] for i in self.qsa_layers
        ]

        def nbytes(t):
            return t.numel() * t.element_size()

        return Tables(
            pos=pos,
            delta=delta.to(dev),
            ple=ple.to(dev),
            qsa=qsa.to(dev),
            gdn_conv=torch.tensor(gdn_conv, dtype=I64, device=dev),
            gdn_rec=torch.tensor(gdn_rec, dtype=I64, device=dev),
            qsa_caches=torch.tensor(caches, dtype=I64, device=dev),
            conv_slot=nbytes(st.conv[0, 0]),
            rec_slot=nbytes(st.rec[0, 0]),
            ple_base=st.ple_conv.data_ptr(),
            ple_slot=nbytes(st.ple_conv[0]),
        )

    # ------------------------------------------------------------ forward
    def embed(self, ids: torch.Tensor, w: Width) -> None:
        """Steps 1 and 2: ``w.h0`` and ``w.emb`` for tokens ``ids``."""
        md = get_forward_context().attn_metadata.ple_metadata
        if self.prologue is not None:
            self.prologue(ids, md, w.h0, w.emb)
            return
        g = self.geometry
        rows = w.rows
        e = self.embed_tokens(ids)
        w.h0.view(rows, g.hc_count, g.hidden).copy_(
            e.unsqueeze(1).expand(-1, g.hc_count, -1)
        )
        embedding = self.ple.ple_embedding
        context = advance_ngram_state(
            ids,
            md.query_start_loc,
            md.ngram_state,
            md.state_indices_in,
            md.state_indices_out,
            md.has_initial_state,
            embedding.eos_token_id,
            history_width=embedding.ngram_size - 1,
            num_accepted_tokens=md.num_accepted_tokens,
        )
        w.emb.copy_(embedding(ids, md.query_start_loc, context))

    def step_meta(self, positions: torch.Tensor, w: Width) -> None:
        """Step 3 from the step's metadata."""
        md = get_forward_context().attn_metadata
        gdn = md.gdn_metadata
        if w.seq:
            write_verify_meta(
                self.counter,
                positions,
                gdn.spec_state_indices_tensor[0],
                gdn.num_accepted_tokens,
                md.qsa_metadata.block_tables,
                w.tables,
                self.state.ple_conv,
                w.ple_window,
            )
            return
        slots_out = gdn.non_spec_state_indices_tensor
        slots_in = gdn.non_spec_state_indices_in_tensor
        if slots_in is None:
            slots_in = slots_out
        write_step_meta(
            self.counter,
            positions,
            slots_in,
            slots_out,
            md.qsa_metadata.block_tables,
            w.tables,
            self.nulls,
        )

    def layers(self, w: Width, stream=None) -> None:
        """Step 4."""
        stream = stream or torch.cuda.current_stream(self.device)
        counter, scratch = self.counter.data_ptr(), self.scratch.data_ptr()
        for i, kind in enumerate(self.kinds):
            w.launch[kind](counter, i, scratch, *w.args[i], stream=stream)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        ids = input_ids.reshape(-1)
        if positions.dim() == 2:
            # mRoPE rows (t, h, w): a text token's three are its position
            positions = positions[0]
        w = self.widths[ids.numel()]
        self.embed(ids, w)
        self.step_meta(positions, w)
        self.layers(w)
        if w.seq:
            md = get_forward_context().attn_metadata.gdn_metadata
            commit_ple(
                self.state.ple_conv, w.ple_snaps, md.spec_state_indices_tensor[0]
            )
        if self.verify_rows:
            g = self.geometry
            return w.chain[-1].view(w.rows, g.hc_count, g.hidden)
        return self.mix(w.chain[-1])

    def mix(self, h: torch.Tensor) -> torch.Tensor:
        """Step 5: ``mix_fused(h)``'s per-row path, at any width."""
        m = self.mixer
        _, mixed, _ = hc_rows(
            h,
            m.hc_norm.weight,
            m.fused_w_cat,
            m.input_mix_weight_up.weight,
            m.hc_norm.variance_epsilon,
            m.hc_count,
            m.block_inject_weight is not None,
        )
        return mixed
