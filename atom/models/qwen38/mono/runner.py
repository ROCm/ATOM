# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next decoder layers on mono's whole-layer programs: the target's 48 layers.

- ``Kernels``: one ``LayerKernel`` and its ``Buffers`` (scratch, step counter) per (kind, batch, spec, block-table
  width), shared by every layer of the kind: the layers run in order on one stream, each launch tagged ``step *
  LAYER_SLOTS + layer + 1``, and every step bumps the counter once (inside a captured graph too). Per (batch, spec,
  width): the kernels' metadata args (``step_meta.meta_buffers``). Every served shape is built at startup
  (``build``): one that does not build stops the server; a step never builds.
- ``Layer``: one decoder layer on its kind's kernel, reading and writing ATOM's pools in place. Its weights are bound
  once (Mono's own A8W4 experts in place, the rest packed from the checkpoint); a pointer table per (shape, input,
  output) is built on the first eager step that needs it, so a graph capture must follow an eager warm-up with the same
  buffers, as ATOM's does.
- ``TargetRunner``: the 48 target layers for the served verify steps; the PLE layer's PLE block runs Native first.
"""

import logging

import torch

from atom.models.qwen38.mono.config import BATCHES, SPEC
from atom.models.qwen38.mono.kernels.blocks.moe import MoEArgs
from atom.models.qwen38.mono.kernels.layer import LayerKernel
from atom.models.qwen38.mono.kernels.prep import VerifyPrep, fused_puts
from atom.models.qwen38.mono.kernels.program import Scenario
from atom.models.qwen38.mono.kernels.recipe import (
    QUARK,
    compose_layer,
    rope_tables,
)
from atom.models.qwen38.mono.kernels.weights.source import InPlace, Packed, bind
from atom.models.qwen38.mono.step_meta import (
    GDNPools,
    QSAPools,
    StepMeta,
    meta_buffers,
    target_puts,
    verify_step,
)
from atom.models.qwen38.mono.weights import load_layer, own_experts
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

AITER_LAYOUTS = ("aiter.a8w4.moe", "aiter.e8m0.moe")  # expert weights bound in place


def qsa_module(layer):
    """The QSA attention module of decoder layer ``layer`` (its caches and metadata)."""
    return next(
        m
        for m in layer.self_attn.modules()
        if hasattr(m, "bind_caches") and hasattr(m, "k_cache")
    )


class Kernels:
    """The layer kernels of one runner's kinds, their buffers and the step's metadata args."""

    def __init__(
        self, name: str, kinds: tuple[str, ...], geo, device="cuda", replay: int = 0
    ):
        self.name, self.kinds, self.geo = name, kinds, geo
        self.replay = replay  # the GDN verify's ReplaySSM record capacity (0: classic)
        self.device = torch.device(device)
        self.kernels = {}  # (kind, batch, spec, width) -> (LayerKernel, Buffers)
        self.meta = {}  # (batch, spec, width) -> step_meta.meta_buffers
        # data_ptrs of registered buffers: bound into the tables, never staged
        self.io: set[int] = set()
        self.key = None  # the current step's (batch, spec, width)
        # (batch, spec, width) built at startup (``build``)
        self.served: set[tuple] = set()
        # shapes a step asked for that were not built (warned once)
        self._unbuilt: set[tuple] = set()

    def kernel(self, kind: str, batch: int, spec: int, width: int):
        key = (kind, batch, spec, width)
        if key not in self.kernels:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"mono: kernel {key} first needed inside a graph capture"
                )
            sc = Scenario(batch, spec)
            blocks, program = compose_layer(
                kind,
                sc,
                geo=self.geo,
                table_width=width,
                family=QUARK,
                replay=self.replay,
            )
            lk = LayerKernel(program, blocks)
            self.kernels[key] = (lk, lk.kernel.buffers(self.device))
        return self.kernels[key]

    def buffers(self, batch: int, spec: int, width: int) -> dict:
        key = (batch, spec, width)
        if key not in self.meta:
            self.meta[key] = meta_buffers(
                batch, spec, width, self.geo.hc_width, self.device
            )
        return self.meta[key]

    def build(self, batch: int, spec: int, width: int) -> None:
        """Build every kind's kernel of the shape (plan, contract check, resource gate) at startup. A deployment that
        ``config_refusal`` admits is served whole, so a shape that does not build is a defect: it stops the server.
        """
        key = (batch, spec, width)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{self.name}: B{batch}S{spec} width {width} built inside a graph capture"
            )
        try:
            # (mono allocates with explicit devices and keeps host-side tensors, e.g. the dual-copy probe's, on the
            # CPU default; ATOM's model setup runs under a CUDA default)
            with torch.device("cpu"):
                for kind in self.kinds:
                    self.kernel(kind, *key)
        except Exception as e:
            raise RuntimeError(
                f"{self.name}: B{batch}S{spec} width {width} does not build ({type(e).__name__}: {e}); "
                "ATOM_MONO_ENABLE=0 serves this deployment Native"
            ) from e
        self.served.add(key)

    def serves(self, batch: int, spec: int, width: int) -> bool:
        """Whether the shape was built at startup (``build``). A step never builds: a shape that was not built runs
        Native, warned once (the block tables are sized at startup, so this does not happen in a served deployment).
        """
        key = (batch, spec, width)
        if key in self.served:
            return True
        if key not in self._unbuilt:
            self._unbuilt.add(key)
            logger.warning(
                "%s: B%dS%d width %d was not built at startup: Native",
                self.name,
                batch,
                spec,
                width,
            )
        return False

    def begin_step(self, meta: StepMeta) -> None:
        """Once per target forward, before the first layer: the metadata args and every kind's step counter in one
        launch (graph-capturable)."""
        self.key = (meta.batch, meta.spec, int(meta.block_tables.shape[1]))
        m = self.buffers(*self.key)
        bufs = [self.kernel(kind, *self.key)[1] for kind in self.kinds]
        fused_puts(meta.valid, target_puts(m, meta), [b.step for b in bufs])
        for b in bufs:
            b.launched.clear()

    @torch.inference_mode()
    def begin_step_buffers(
        self, batch: int, spec: int, width: int
    ) -> tuple[dict, list[torch.Tensor]]:
        """``begin_step`` for a caller that fills the metadata args itself (``prep.draft_meta``): the shape's buffers
        and its distinct step counters, each to be bumped once on the device."""
        self.key = (batch, spec, width)
        counters = []
        for kind in self.kinds:
            _, bufs = self.kernel(kind, *self.key)
            bufs.launched.clear()
            if all(c is not bufs.step for c in counters):
                counters.append(bufs.step)
        return self.buffers(*self.key), counters


class Layer:
    """One decoder layer on its kind's kernel: it computes the whole layer, writing ``h_out`` to ``out``. An input at a
    registered address is bound into the pointer table directly, any other is first staged (one copy).
    """

    def __init__(self, q, kernels: Kernels, experts: dict | None = None):
        self.index, self.kind, self.q = q.index, q.kind, q
        self.kernels = kernels
        # the expert blocks' args (MoEArgs names) -> Mono's own A8W4 tensors
        self.experts = experts or {}
        self.bound: dict[str, torch.Tensor] = {}
        self.tables = {}
        # the next layer: its attention-HC weight is this layer's prefetch target
        self.next: Layer | None = None
        self._pf_keep = []

    def bind(self, blocks) -> dict:
        """The blocks' weight args, checked against their layouts: the expert blocks' in place, the others packed
        from the checkpoint layer; then the kind's QSA RoPE tables (``recipe.rope_tables``).
        """
        for block in blocks:
            specs = block.weight_specs() if hasattr(block, "weight_specs") else ()
            if any(s.layout in AITER_LAYOUTS for s in specs):
                self.bound.update(bind(block, InPlace(self.experts)))
            else:
                self.bound.update(bind(block, Packed(self.q)))
        self.bound = rope_tables(blocks, self.bound, self.kind)
        return self.bound

    def _prefetch_args(self, key) -> dict:
        """``hcpf.w`` / ``hcpf.n`` (the recipe's HC weight prefetch): the next layer's attention-HC down weight and its
        byte count (0, nothing touched, for the last layer)."""
        nxt, dev = self.next, self.kernels.device
        if nxt is not None:
            nlk, _ = self.kernels.kernel(nxt.kind, *key)
            w = (nxt.bound or nxt.bind(nlk.blocks))["attn_hc.dw"]
            n = w.numel() * w.element_size()
        else:
            w, n = torch.zeros(64, dtype=torch.uint8, device=dev), 0
        cnt = torch.tensor([n], dtype=torch.int32, device=dev)
        self._pf_keep.extend([w, cnt])
        return {"hcpf.w": w, "hcpf.n": cnt}

    def _runtime_args(self, key, m: dict, pools):
        lk, _ = self.kernels.kernel(self.kind, *key)
        packed = self.bound or self.bind(lk.blocks)
        args = {}
        if "hcpf.w" in getattr(lk.program, "args", ()):
            args.update(self._prefetch_args(key))
        if isinstance(pools, GDNPools):
            pools.check_window(key[1])
            if pools.replay != self.kernels.replay:
                raise ValueError(
                    f"mono layer {self.index}: ReplaySSM records {pools.replay}, built for {self.kernels.replay}"
                )
            args.update({k: m[k] for k in ("gdn.conv_slot", "gdn.conv_off")})
            if pools.rec is None:
                args.update({k: m[k] for k in ("gdn.ssm_read", "gdn.ssm_write")})
            else:  # ReplaySSM: the checkpoint sits in the conv window's slot
                rk, ru, rg = pools.rec
                args.update(
                    {
                        "gdn.rec_k": rk,
                        "gdn.rec_u": ru,
                        "gdn.rec_g": rg,
                        "gdn.write_pos": pools.write_pos,
                    }
                )
            args.update({"gdn.conv_pool": pools.conv, "gdn.ssm_pool": pools.ssm})
        else:
            q = pools
            args.update(
                {
                    "k_cache": q.k,
                    "v_cache": q.v,
                    "index_raw": q.index_raw,
                    "index_cmp": q.index_cmp,
                }
            )
            if q.k_scale is not None:
                args.update({"k_scale": q.k_scale, "v_scale": q.v_scale})
            args.update(
                {
                    "block_table": m["block_table"],
                    "seq_lens": m["seq_lens"],
                    "rope_off": m["rope_off"],
                    "sel": m["sel"],
                }
            )
        return args, packed

    def _table(self, key, lk, h_in, out, pools):
        tkey = (key, h_in.data_ptr(), out.data_ptr())
        if tkey in self.tables:
            return self.tables[tkey]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"mono layer {self.index}: no pointer table for {tkey} (capture without warm-up?)"
            )
        m = self.kernels.buffers(*key)
        runtime = {"h_in": h_in, "h_out": out, "h1_mem": m["h1_mem"]}
        args, packed = self._runtime_args(key, m, pools)
        runtime.update(args)
        self.tables[tkey] = lk.table(packed, runtime)
        return self.tables[tkey]

    def step(self, hidden, out, batch: int, spec: int, pools) -> None:
        """One launch of the step ``begin_step`` opened (``batch`` x ``spec`` rows); ``pools``: ATOM's pools of the
        layer (``step_meta.GDNPools`` / ``step_meta.QSAPools``). CUDA-graph capturable once the table exists.
        """
        key = self.kernels.key
        if key is None or key[:2] != (batch, spec):
            raise RuntimeError("mono: begin_step was not called for this step")
        lk, bufs = self.kernels.kernel(self.kind, *key)
        h_in = hidden
        if hidden.data_ptr() not in self.kernels.io:
            h_in = self.kernels.buffers(*key)["stage"]
            h_in.copy_(hidden[: batch * spec])
        lk.kernel.launch(self._table(key, lk, h_in, out, pools), bufs, layer=self.index)


class TargetRunner:
    """The target's decoder layers for a served MTP verify step (``step_meta.verify_step``: 1-4 requests x ``SPEC``
    rows). The step's embedding and GDN metadata come from one launch (``prep.VerifyPrep``) into the first layer's
    input; the layers alternate between two registered buffers."""

    def __init__(self, model, atom_config, geo, checkpoint: str, replay: int = 0):
        from atom.models.qwen38.mono.kernels import dual

        self.model, self.geo = model, geo
        inner = model.model
        layers = inner.layers
        # the PLE layer: ATOM's PLE prelude, then the layer's delta program on its output
        self.ple_layers = frozenset(geo.ple_layers)
        with torch.device("cpu"):  # (see ``Kernels.serves``)
            logger.info(
                "Qwen mono: dual-copy mailboxes %s",
                "available" if dual.available() else "unavailable",
            )
            self.kernels = Kernels("Qwen mono", ("delta", "qsa"), geo, replay=replay)
            logger.info(
                "Qwen mono: GDN verify %s",
                f"ReplaySSM ({replay} records)" if replay else "classic",
            )
            self.layers = []
            for i in range(geo.layers):
                kind = "qsa" if geo.layer_kind(i) == "qsa" else "delta"
                q = load_layer(i, checkpoint, geo, kind, experts=False)
                experts = own_experts(i, checkpoint, geo, MoEArgs())
                self.layers.append(Layer(q, self.kernels, experts))
                logger.info("Qwen mono: layer %d (%s) loaded", i, kind)
            for a, b in zip(self.layers, self.layers[1:]):
                a.next = b
            logger.info(
                "Qwen mono: own A8W4 expert copies for %d layers", len(self.layers)
            )
            rows = SPEC * max(BATCHES)
            width = inner.hc_count * inner.config.hidden_size
            self.io = [
                torch.zeros(rows, width, dtype=torch.bfloat16, device="cuda")
                for _ in range(2)
            ]
            self.kernels.io.update(t.data_ptr() for t in self.io)
            self.prep = VerifyPrep(max(BATCHES), device="cuda")
            self.width = -(
                -int(atom_config.max_model_len) // int(atom_config.kv_cache_block_size)
            )
            self._gdn_layers, self._qsa_layers = {}, {}
            for i in range(geo.layers):
                la = getattr(layers[i], "linear_attn", None)
                if la is not None:
                    self._gdn_layers[i] = int(la.attn.layer_num)
                if getattr(layers[i], "self_attn", None) is not None:
                    self._qsa_layers[i] = qsa_module(layers[i])
            for b in BATCHES:
                self.kernels.build(b, SPEC, self.width)
        logger.info("Qwen mono: %d layers, batches %s, table width %d, served %s",
                    len(self.layers), BATCHES, self.width, sorted(self.kernels.served))  # fmt: skip

    def step_meta(
        self, input_ids, positions, intermediate_tensors, inputs_embeds
    ) -> StepMeta | None:
        """The step's metadata if Mono serves it (the embedding already written into ``io[1]``), else None."""
        if intermediate_tensors is not None or inputs_embeds is not None:
            return None
        if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 1:
            return None
        rows = input_ids.numel()
        fwd = get_forward_context()
        batch = verify_step(fwd, rows, BATCHES, replayssm=self.kernels.replay > 0)
        if batch is None:
            return None
        md = fwd.attn_metadata
        gdn = md.gdn_metadata
        if not self.kernels.serves(batch, SPEC, int(md.block_tables.shape[1])):
            return None
        idx = gdn.spec_state_indices_tensor[:batch]
        idx_in = gdn.spec_state_indices_in_tensor
        idx_in = idx if idx_in is None else idx_in[:batch]
        inner = self.model.model
        valid, prev, slot, ssm_read, rope_off = self.prep(
            input_ids[:rows], inner.embed_tokens.weight, self.io[1][:rows], gdn.num_accepted_tokens, idx, idx_in,
            batch, inner.hc_count, md.slot_mapping, SPEC, positions, md.context_lens)  # fmt: skip
        return StepMeta(
            batch,
            valid,
            md.block_tables[:batch],
            md.context_lens[:batch],
            slot,
            prev,
            ssm_read,
            idx,
            rope_off=rope_off,
        )

    def _pools(self, i: int, caches):
        g = self._gdn_layers.get(i)
        if g is not None:
            c = caches[f"layer_{g}"]
            if not self.kernels.replay:
                return GDNPools(self.geo.gdn, c.k_cache, c.v_cache)
            gdn = get_forward_context().attn_metadata.gdn_metadata
            rec = (c.replay_buf_k, c.replay_buf_u, c.replay_buf_g)
            return GDNPools(self.geo.gdn, c.k_cache, c.v_cache, rec, gdn.write_pos)
        q = self._qsa_layers.get(i)
        if q is not None:
            return QSAPools(self.geo, q.k_cache, q.v_cache, q.raw_key_cache, q.compressed_key_cache,
                            q.k_scale_cache, q.v_scale_cache)  # fmt: skip
        raise NotImplementedError(
            f"Qwen mono: layer {i} pools (GDN and QSA layers are wired)"
        )

    def forward(self, input_ids, meta: StepMeta):
        """The served step: every decoder layer one launch; the final residual streams [rows, streams, hidden]."""
        inner = self.model.model
        rows = input_ids.numel()
        # the prep wrote the embedding into io[1]; the first layer writes io[0]
        hidden = self.io[1][:rows]
        caches = get_forward_context().kv_cache_data
        self.kernels.begin_step(meta)
        flip = 0
        for i in range(inner.start_layer, inner.end_layer):
            if i in self.ple_layers:
                hidden = inner.layers[i]._apply_ple(hidden, input_ids)
            out = self.io[flip][:rows]
            flip ^= 1
            self.layers[i].step(
                hidden, out, meta.batch, meta.spec, self._pools(i, caches)
            )
            hidden = out
        return hidden.clone().view(-1, inner.hc_count, inner.config.hidden_size)
