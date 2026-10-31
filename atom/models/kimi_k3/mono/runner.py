# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One TP rank's Kimi-K3 mono decode: the bound weights, the step's buffers and
the forward that runs them.

The forward is the model's own layer loop with each eligible layer in two mono
kernels:

* **K1** (``kernels.attention.kda``) -- the attention AttnRes seam, the fused
  hidden projection, ``f_b``, the causal conv, the KDA recurrence and the gated
  output norm, leaving ``o_proj``'s input;
* **K2** (``kernels.layer``) -- ``o_proj`` and its in-kernel TP all-reduce, the
  MLP AttnRes seam, then the latent MoE (router, top-16, the MXFP4 experts, the
  shared experts, the latent all-reduce and norm, the up-projection) and the
  final all-reduce.

Layer 0 and the full-attention layers take the original modules: the kernels
have no attention of their own, so an MLA layer keeps its paged KV cache and its
own path, and layer 0 has no AttnRes block behind it to mix. ``weights.bind_model``
decides which is which; the two paths hand each other the same tensors. A
full-attention layer still runs K2 for everything after its attention
(``_pre_o_core``), unless ``bind_model`` left it on the production path.

The block-residual buffer is the one place the mono loop cannot reuse
``KimiDecoderLayer.forward``: ``AttnRes.maybe_close_block`` grows the candidates
with ``torch.cat``, while the kernels write their block into a slot of a buffer
sized for the whole stack. ``_fallback`` is that same forward with the ``cat``
replaced by a slot copy, so a fallback layer's numerics are the original path's.

ATOM carries two pending adds between layers (an MoE layer defers its routed and
its shared output separately); K2 all-reduces one MoE output, so a mono layer
returns it as the single pending and folds any two it was handed into K1's one
``delta``.
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.models.kimi_k3.mono.config import (
    MAX_ROWS,
    Dims,
    layer_block,
    num_blocks,
)
from atom.models.kimi_k3.mono.kernels import layer as k2
from atom.models.kimi_k3.mono.kernels.attention import kda as k1
from atom.models.kimi_k3.mono.kernels.stages import moe as k2b
from atom.models.kimi_k3.mono.plan import (
    _width,
    kda_metadata,
    state_slots,
    width_key,
    width_refusal,
)
from atom.models.kimi_k3.mono.weights import bind_model
from atom.mono.dsa_indexer import kernel as kix
from atom.mono.dsa_indexer.runner import IndexerRunner
from atom.mono.runtime.aux_taps import AuxTaps, bind_aux_taps
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.debug import (
    DIAG_BYTES,
    given_up_waits,
    raise_if_given_up,
    region_namer,
)
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.uncached import uncached_bytes
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# K2's mailbox tags start past every layer's K1: a step's two launches of one
# layer must not share a tag, and the model has fewer layers than this
K2_TAG0 = 128
# the DSA layers' indexer launches, past every K2's
IX_TAG0 = 2 * K2_TAG0


def inner_model(model):
    """The layer stack under whatever outer class the checkpoint registered.
    K3's text-only and vision classes both hold it one level deeper than the
    usual ``model.model``."""
    return getattr(model, "language_model", model).model


@dataclass(frozen=True)
class WidthBuild:
    """A width's kernels: K1's build key a layer with its launcher, K2's, and
    the DSA layers' indexer (one build for all of them; None when the width is
    past its bound or the indexers are not mono's)."""

    k1_keys: dict[int, k1.KdaPreBuild]
    k1_launchers: dict[k1.KdaPreBuild, object]
    k2_keys: dict[int, k2.K2Build]
    k2_launchers: dict[k2.K2Build, object]
    indexer: object | None = None

    def kernels(self) -> list:
        ix = [] if self.indexer is None else [(self.indexer, kix.ABI)]
        return (
            [(f, k1.ABI) for f in self.k1_launchers.values()]
            + [(f, k2.ABI) for f in self.k2_launchers.values()]
            + ix
        )


def ix_peer_offset(max_rows: int) -> int:
    """Where the DSA indexer's regions start in the peer buffer it shares with
    K2: past every K2 width's (up to ``max_rows``). A rank already in the next
    layer's K2 (its AR pushes from offset 0) then cannot land on hist pairs a
    peer's indexer is still polling; at a shared 0 they overlapped from 2 rows."""
    return -(-k2.peer_bytes(max_rows) // 256) * 256


def _lean_width(built: WidthBuild, spilled: list) -> WidthBuild | None:
    """``built`` with each K1 build whose launcher spilled staged a column group
    a GEMV unit (``KdaPreBuild.gemv_unit``): half the weights in flight, slower
    where ``kda.gemv_staging`` keeps two groups, so only where those do not fit.
    None when no spilled launcher is such a K1 build (a K2 or indexer spill, or
    a K1 at one group a unit already): there is nothing leaner to build."""
    lean = {
        key: replace(key, gemv_unit=1)
        for key, launcher in built.k1_launchers.items()
        if launcher in spilled and key.gemv_unit > 1
    }
    if not lean:
        return None
    k1_keys = {layer: lean.get(key, key) for layer, key in built.k1_keys.items()}
    return replace(
        built,
        k1_keys=k1_keys,
        k1_launchers={
            key: k1.build(key) if key in lean.values() else built.k1_launchers[key]
            for key in set(k1_keys.values())
        },
    )


class K3MonoDecodeRunner:
    """Binds every rank's weights and buffers, or none: a rank that cannot is a
    refusal every rank agrees on before anything collective."""

    max_rows = MAX_ROWS

    def __init__(self, model, drafter=None, metadata_builder=None) -> None:
        group = get_tp_group().cpu_group
        bind_agreed(lambda: self._bind(model, drafter, metadata_builder), group)
        peer_bytes = k2.peer_bytes(self.max_rows)
        if self.ix is not None:
            peer_bytes = max(peer_bytes, self.ix.peer_bytes)
        self.peers, self._finalizer = owned_peer_buffer(
            self,
            peer_bytes,
            group,
            self.rank,
            self.tp,
            self.device,
            plain_bytes=self.plain_bytes,
        )
        self.plain = self.peers.plain
        self.mailboxes = StepMailboxes(
            self.peers, self.scratch, self.queue, debug=self.debug
        )
        self.widths = WidthBuilds(
            self._build_width,
            WidthBuild.kernels,
            group,
            "Kimi-K3 mono",
            lean=_lean_width,
        )
        # the DSA indexers hooked last, once every rank holds a whole runner
        # (K3's layers run eagerly here: no trace before the hook)
        if self.ix is not None:
            self.ix.attach(self.scratch, self.plain, self.peers, IX_TAG0)

    # ------------------------------------------------------------------ bind
    def _bind(self, model, drafter=None, metadata_builder=None) -> None:
        self.model = model
        self.rank = get_tensor_model_parallel_rank()
        self.tp = get_tensor_model_parallel_world_size()
        self.dims = Dims(self.tp)
        self.debug = envs.ATOM_MONO_DEBUG
        self.inner = inner_model(model)
        self.plans = bind_model(self.inner, self.dims, self.rank)
        self.num_blocks = num_blocks(len(self.inner.layers))
        # a DSpark drafter taps hidden states; a plain-decode deployment has none
        self.aux = (
            AuxTaps((), ())
            if drafter is None
            else bind_aux_taps(drafter, self.inner.config.hidden_size)
        )
        first = next(p for p in self.plans if p.mono)
        self.device = first.kda.w_in.device
        self.state_len = self._state_len()
        self.replay_cap = self._replay_cap()
        self._replay_tables = {}  # layer id -> (K1's ReplaySSM table, its pointers)
        # the DSA layers' indexers: static checks every rank shares, so no
        # vote; refused, the layers keep their own and the rest stays mono
        try:
            self.ix = IndexerRunner(
                self.inner,
                metadata_builder,
                peer_offset=ix_peer_offset(self.max_rows),
                max_rows=self.max_rows,
            )
        except MonoUnsupported as why:
            logger.info("mono DSA indexer off: %s", why)
            self.ix = None
        self._allocate()

    def _state_len(self) -> int:
        """The conv state's window, a build constant: ``CONV_WIDTH - 1`` plus the
        verify window the KDA state pool was allocated for.

        Every mono layer is checked, not just the first. ``_conv_state`` and
        ``_recurrent_state`` are the only validation the KDA state pools get,
        and everything they refuse -- the channel axis, the conv dtype, the
        recurrent dtype, the window -- is a static property of a pool, the
        same on every step. So it belongs here, where an unsupported layer
        refuses the WHOLE runner and ``LazyRunner`` serves the width on the
        production path, with no mono kernel launched at all.

        Neither way of catching it later works:

        * Catching outside ``MonoDecodeModel.forward`` and replaying the step
          on the production path double-advances state. ``_conv_state`` /
          ``_recurrent_state`` hand back the KV cache itself
          (``cache.k_cache`` / ``cache.v_cache``) and ``_k1`` steps the conv
          window and the recurrence IN PLACE, so layers ``0 .. N-1`` have
          already advanced when layer N raises. The replay steps them twice:
          a crash traded for silent corruption.
        * Falling back layer by layer deadlocks. These criteria are read
          per-rank off that rank's own KV cache, and the kernels wait
          in-kernel on every peer (``dispatch.py``'s module docstring), so one
          rank dropping to production at layer N leaves its peers waiting on
          it for ever -- a crash traded for a hang, which is worse to debug.
          Making it safe would need a collective vote per layer per step,
          which is the overhead mono exists to remove.

        Build time has that vote for free: ``bind_agreed`` turns this rank's
        ``MonoUnsupported`` into a ``tp_agree(False)`` every peer follows.
        """
        lens = set()
        for plan in self.plans:
            if not plan.mono:
                continue
            lens.add(self._conv_state(plan.layer_id).size(-1))
            self._recurrent_state(plan.layer_id)
        if len(lens) != 1:
            raise MonoUnsupported(
                f"mono layers disagree on the conv window: {sorted(lens)}; "
                "`state_len` is one build constant for all of them"
            )
        return lens.pop()

    def _replay_cap(self) -> int:
        """ReplaySSM's record buffer length (its ``cache_len``), 0 without:
        read off the buffers allocated, every mono layer's checked as
        ``_state_len`` checks the state pools -- a build constant, refused for
        the whole runner. K1 indexes them as contiguous [slot][head][cap][128]:
        k and u bf16, the log gates fp32."""
        caps = set()
        for plan in self.plans:
            if not plan.mono:
                continue
            cache = get_forward_context().kv_cache_data[f"layer_{plan.layer_id}"]
            bufs = [getattr(cache, f"replay_buf_{n}", None) for n in "kug"]
            if bufs[0] is None:
                caps.add(0)
                continue
            for buf, dtype in zip(
                bufs, (torch.bfloat16, torch.bfloat16, torch.float32)
            ):
                if (
                    buf.dtype != dtype
                    or not buf.is_contiguous()
                    or tuple(buf.shape[1::2]) != (k1.NH, k1.HD)
                ):
                    raise MonoUnsupported(
                        f"layer {plan.layer_id}: a ReplaySSM record buffer is "
                        f"{tuple(buf.shape)} {buf.dtype}"
                    )
            caps.add(bufs[0].shape[2])
        if len(caps) != 1:
            raise MonoUnsupported(f"mono layers disagree on ReplaySSM: {sorted(caps)}")
        return caps.pop()

    def _recurrent_state(self, layer_id: int) -> torch.Tensor:
        """The layer's KDA recurrent state, checked.

        `config_refusal` checks `ATOM_GDN_SSM_DTYPE`, which is the *request*;
        this is the pool that was actually allocated. The recurrence reads and
        writes it in place as fp32 (see `_allocate`), so a narrower pool is
        reinterpreted rather than converted -- wrong numbers, no error. The
        conv state next door has been checked all along; this one was not,
        although the module docstring claimed both were.
        """
        cache = get_forward_context().kv_cache_data[f"layer_{layer_id}"]
        state = cache.v_cache
        if state.dtype != torch.float32:
            raise MonoUnsupported(
                f"layer {layer_id}: recurrent state {state.dtype}, the "
                "recurrence reads it in place as float32"
            )
        return state

    def _conv_state(self, layer_id: int) -> torch.Tensor:
        """The layer's conv state, with the channel dim first (the pool is
        allocated either way; the kernel reads the strides)."""
        cache = get_forward_context().kv_cache_data[f"layer_{layer_id}"]
        state = cache.k_cache
        if state.size(1) != self.dims.qkv:
            state = state.transpose(-1, -2)
        if state.size(1) != self.dims.qkv:
            raise MonoUnsupported(
                f"layer {layer_id}: conv state is {tuple(state.shape)}, no axis "
                f"of {self.dims.qkv} channels"
            )
        if state.dtype != torch.bfloat16:
            raise MonoUnsupported(f"layer {layer_id}: conv state {state.dtype}")
        return state

    def _allocate(self) -> None:
        """The scratch every launch shares, sized for the widest build. It is
        zeroed, and the peers fenced, before each step's first kernel
        (``StepMailboxes.begin_step``), and holds tagged pairs only, so a
        layer's tag is all that tells its pairs from the previous layer's."""
        # (K1's regions do not depend on the query length)
        widest_k1 = max(
            k1.scratch_bytes(key) for key in self._k1_keys(MAX_ROWS, 1).values()
        )
        widest_k2 = max(
            k2.scratch_bytes(key) for key in self._k2_keys(MAX_ROWS).values()
        )
        nbytes = max(widest_k1, widest_k2)
        if self.ix is not None:
            nbytes = max(nbytes, self.ix.scratch_bytes)
        # a debug build's per-CTA wait records sit past every kernel's regions
        # and are cleared with them at each step's start
        self.diag_off = -1
        if self.debug:
            self.diag_off = -(-nbytes // 256) * 256
            nbytes = self.diag_off + DIAG_BYTES
        # a multiple of the step zeroing's 16 B store. The step's buffers are
        # uncached (``uncached``): a hand-off's consumer reads the line at its
        # home memory, not through the producer XCD's L2 -- K1 + K2 a layer
        # -3.7 us at 1 row, -6 at 64 (2026-10-30, bits equal)
        nbytes = -(-nbytes // 16) * 16
        self.scratch = uncached_bytes(nbytes, self.device)
        # the plain words every launch hands on behind a flag (``plain_layout``):
        # out of the scratch, where a later launch's tag could match a stale
        # one; never cleared, a reader waits for the flag first. They live in
        # the peer buffer's own plain region (``PeerBuffer.plain``)
        self.plain_bytes = max(
            *(k1.plain_bytes(key) for key in self._k1_keys(MAX_ROWS, 1).values()),
            *(k2.plain_bytes(key) for key in self._k2_keys(MAX_ROWS).values()),
        )
        if self.ix is not None:
            self.plain_bytes = max(self.plain_bytes, self.ix.plain_bytes)
        # K2's ug task queue: counts from the step's clear, so out of the
        # scratch the other launches write
        self.queue = uncached_bytes(k2b.QUEUE_BYTES, self.device)
        # a plain decode's accepted-token counts (`plan.state_slots`): fixed, so
        # a captured step reads them at one address
        self.ones = torch.ones(MAX_ROWS, dtype=torch.int32, device=self.device)

    # ---------------------------------------------------------------- builds
    def _k1_keys(self, rows: int, qlen: int) -> dict[int, k1.KdaPreBuild]:
        # ReplaySSM's verify window: the step being built is a decode, its
        # metadata the builder's (a constant of the deployment)
        md = kda_metadata(get_forward_context()) if self.replay_cap else None
        window = md.replayssm_max_query_len if md is not None else 0
        keys = {
            p.layer_id: k1.KdaPreBuild(
                tokens=rows,
                qlen=qlen,
                nblocks=p.kda.blocks,
                # every mono layer is handed the previous layer's FFN output
                delta=True,
                state_len=self.state_len,
                eps=p.kda.seam.eps,
                out_eps=p.kda.seam.out_eps,
                onorm_eps=p.kda.onorm_eps,
                lower_bound=p.kda.lower_bound,
                write_idx=p.kda.write_idx,
                replay_cap=self.replay_cap,
                replay_window=window,
                **k1.gemv_staging(rows),
            )
            for p in self.plans
            if p.mono
        }
        for key in set(keys.values()):
            if k1.lds_refusal(key):
                raise MonoUnsupported(k1.lds_refusal(key))
        return keys

    def _k2_keys(self, rows: int) -> dict[int, k2.K2Build]:
        return {
            p.layer_id: k2.K2Build(
                tokens=rows,
                # the MLP seam mixes this layer's own block too, when it wrote one
                nblocks=p.moe.blocks + (p.moe.write_idx >= 0),
                eps=p.moe.seam.eps,
                out_eps=p.moe.seam.out_eps,
                ln_eps=p.moe.ln_eps,
                diag=self.debug,
            )
            for p in self.plans
            if p.moe is not None
        }

    def _build_width(self, key: int) -> WidthBuild:
        rows, qlen = _width(key)
        k1_keys = self._k1_keys(rows, qlen)
        k2_keys = self._k2_keys(rows)
        return WidthBuild(
            k1_keys=k1_keys,
            k1_launchers={k: k1.build(k) for k in set(k1_keys.values())},
            k2_keys=k2_keys,
            k2_launchers={k: k2.build(k) for k in set(k2_keys.values())},
            # past the indexer's widths the DSA layers run their own
            indexer=None if self.ix is None else self.ix.launcher(rows, qlen),
        )

    def prepare(self, rows: int, qlen: int) -> bool:
        """Whether every rank holds this step's kernels (``WidthBuilds``)."""
        if not 0 < rows <= self.max_rows or rows % qlen or width_refusal(rows):
            return False
        return self.widths.prepare(width_key(rows, qlen))

    def close(self) -> None:
        """Give the DSA indexers their own forward back and close the peer
        mappings now (once no rank can launch another step)."""
        if self.ix is not None:
            self.ix.unhook()
        self._finalizer()

    # --------------------------------------------------------------- forward
    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, qlen: int
    ) -> torch.Tensor:
        rows = input_ids.numel()
        build = self.widths[width_key(rows, qlen)]
        metadata = kda_metadata(get_forward_context())
        prefix = self.inner.get_input_embeddings(input_ids)
        # the kernels write their block into a slot, so the stack's candidates
        # are allocated once rather than grown layer by layer
        blocks = prefix.new_empty(rows, self.num_blocks, prefix.size(1))
        self.mailboxes.begin_step()
        add = add2 = None
        serving = (
            nullcontext() if self.ix is None else self.ix.hook.serving(build.indexer)
        )
        with serving:
            for plan in self.plans:
                prefix, add, add2 = self._layer(
                    build, plan, positions, prefix, blocks, add, add2, metadata
                )
        hidden, _ = self.inner.output_attn_res(prefix, blocks, add, add2)
        self._raise_on_given_up_waits(rows)
        return hidden

    def _diag_ptr(self) -> int:
        """Where K2 records a wait it gave up on, or 0 outside a debug build."""
        if self.diag_off < 0:
            return 0
        return self.scratch.data_ptr() + self.diag_off

    def _raise_on_given_up_waits(self, rows: int) -> None:
        """A debug step: raise with every K2 mailbox wait that gave up and every
        rank the step fence gave up on. Synchronizes, and a graph replay runs no
        Python -- so this is an ``--enforce-eager`` aid.

        K1 has no bounded waits: its build's ``debug`` flag turns on the
        activation taps, not the wait records, so a K1 hang shows up only as the
        fence's missing ranks."""
        if not self.debug or torch.cuda.is_current_stream_capturing():
            return
        diag = self.scratch[self.diag_off : self.diag_off + DIAG_BYTES]
        waits = given_up_waits(diag, region_namer(k2.REGIONS))
        raise_if_given_up("Kimi-K3 mono", self.rank, rows, waits, self.mailboxes)

    def _layer(self, build, plan, positions, prefix, blocks, add, add2, metadata):
        if not plan.mono and not plan.ffn_only:
            rows = prefix.size(0)
            out = self._fallback(plan, positions, prefix, blocks, add, add2)
            # a production module may hand back a tile-padded batch; the kernels
            # are built for an exact row count, so a layer that changed it would
            # mis-specialize every layer after it
            widths = {t.size(0) for t in out if t is not None}
            if widths != {rows}:
                raise MonoUnsupported(
                    f"layer {plan.layer_id} returned rows {sorted(widths)} for "
                    f"an {rows}-row step"
                )
        else:
            if plan.ffn_only:
                core, prefix = self._pre_o_core(
                    plan, positions, prefix, blocks, add, add2
                )
            else:
                delta = add if add2 is None else (add2 if add is None else add + add2)
                core = self._k1(build, plan, prefix, delta, blocks, metadata)
            # K2 updates ``prefix`` in place: on a block-write layer it restarts
            # it at the all-reduced o_proj output, which K1 has already banked
            out = (prefix, self._k2(build, plan, core, prefix, blocks), None)
        self._write_aux(plan.layer_id, out)
        return out

    def _pre_o_core(self, plan, positions, prefix, blocks, add, add2):
        """A full-attention layer that runs K2: the production
        attention up to ``o_proj``'s input, to be handed to K2 in place of K1's.

        ``bind_model`` refuses a layer that closes a block, so ``before`` mixes
        the same candidates K2's MLP seam does and nothing is banked here. The
        seam's ``prefix`` is a fresh tensor (the previous layer always hands an
        FFN output, so it takes the ``add`` branch), which K2 updates in place
        exactly as it does on a KDA layer."""
        layer, normed, prefix, _, _ = self._seam(plan, prefix, blocks, add, add2)
        core = layer.self_attn.forward_pre_o(positions, normed)
        rows = prefix.size(0)
        if (
            core.dim() != 2
            or core.size(0) < rows
            or core.size(1) != self.dims.proj
            or core.dtype != torch.bfloat16
            or core.stride() != (self.dims.proj, 1)
        ):
            # K2 reads the first ``rows`` rows of a plain [rows, proj] bf16 core
            raise MonoUnsupported(
                f"layer {plan.layer_id}: attention core {tuple(core.shape)} "
                f"{core.dtype} stride {core.stride()} for an {rows}-row step"
            )
        return core, prefix

    def _seam(self, plan, prefix, blocks, add, add2):
        """The layer's attention-side AttnRes seam, on the blocks behind it."""
        layer = self.inner.layers[plan.layer_id]
        before, write_idx = layer_block(plan.layer_id)
        normed, prefix = layer.self_attention_attn_res(
            prefix, blocks[:, :before], add, add2
        )
        return layer, normed, prefix, before, write_idx

    def _write_aux(self, layer_id: int, out: tuple) -> None:
        """This layer's aux hidden state, for a DSpark drafter tapping it.

        The drafter hooks ``layers[layer_id]`` (``Drafter.arm_aux_capture``),
        and NEITHER branch above calls that module -- ``_fallback`` reruns the
        layer's forward from its submodules -- so the hook cannot fire on a
        mono step and the drafter would keep reading whatever the last
        production step left in the buffer: every DSpark deployment would
        draft from stale features.

        The reconstruction is the layer's own ``aux_hidden_state``, called on
        this step's return rather than reimplemented, so the two cannot drift
        apart -- which is exactly what that method's docstring asks of a caller.
        ``out`` is ``(prefix, add, add2)``; the layer's own return carries the
        block residual in a fourth slot that the reconstruction ignores, and
        mono keeps that in ``blocks``.
        """
        if layer_id not in self.aux.layer_ids:
            return
        layer = self.inner.layers[layer_id]
        self.aux.write(layer_id, layer.aux_hidden_state((*out, None)))

    def _fallback(self, plan, positions, prefix, blocks, add, add2):
        """``KimiDecoderLayer.forward`` with ``maybe_close_block``'s ``cat``
        replaced by a slot copy."""
        layer, normed, prefix, before, write_idx = self._seam(
            plan, prefix, blocks, add, add2
        )
        if write_idx >= 0:
            blocks[:, write_idx].copy_(prefix)
            prefix = None
        if layer.is_linear_attn:
            attn_out = layer.self_attn(normed)
        else:
            attn_out = layer.self_attn(positions, normed)
        after = before + (write_idx >= 0)
        normed, prefix = layer.mlp_attn_res(prefix, blocks[:, :after], attn_out)
        out, shared = layer._ffn(normed)
        return prefix, out, shared

    def _k1(self, build, plan, prefix, delta, blocks, metadata, taps=None):
        """K1 -> the gated core rows (``o_proj``'s input).

        ``taps`` (check mode): ``{"x", "proj"}`` tensors for the build's own
        activation taps -- the AttnRes seam's normed row and the fused hidden
        projection. They name the stage a wrong answer starts at, which the
        outputs alone cannot.
        """
        w = plan.kda
        key = build.k1_keys[plan.layer_id]
        launcher = build.k1_launchers[key]
        if taps is not None:
            key = replace(key, debug=True)
            if key not in build.k1_launchers:
                build.k1_launchers[key] = k1.build(key)
            launcher = build.k1_launchers[key]
        state = self._conv_state(plan.layer_id)
        if state.size(-1) != key.state_len:
            raise MonoUnsupported(
                f"layer {plan.layer_id}: conv window {state.size(-1)} != the "
                f"built {key.state_len}"
            )
        rstate = self._recurrent_state(plan.layer_id)
        slots, accepted = state_slots(metadata, self.ones)
        core = prefix.new_empty(prefix.size(0), self.dims.proj)
        p = torch.Tensor.data_ptr
        args = k1.ABI.pack({
            "prefix": p(prefix), "delta": p(delta), "blocks": p(blocks),
            "blk_sm": blocks.stride(0), "blk_sr": blocks.stride(1),
            "ares_nw": p(w.seam.nw), "ares_qk": p(w.seam.qk),
            "in_nw": p(w.seam.out_nw), "w_in": p(w.w_in), "w_fb": p(w.w_fb),
            "conv_w": p(w.conv_w), "conv_st": p(state),
            "cs_seq": state.stride(0), "cs_dim": state.stride(1),
            "cs_tok": state.stride(2), "a_log": p(w.a_log),
            "dt_bias": p(w.dt_bias), "on_w": p(w.on_w),
            "rstate": p(rstate), "rs_seq": rstate.stride(0),
            "st_idx": p(slots), "st_stride": slots.stride(0),
            "num_acc": p(accepted),
            "replay": self._replay_table(plan.layer_id, metadata),
            "core_out": p(core),
            "x_dbg": 0 if taps is None else p(taps["x"]),
            "proj_dbg": 0 if taps is None else p(taps["proj"]),
            "scratch": p(self.scratch), "plain": p(self.plain),
            "layer": plan.layer_id, "tl": 0,
        })  # fmt: skip
        # the caller's stream: a FlyDSL launcher left to its default submits on
        # the NULL stream outside a graph capture
        _run_compiled(launcher, *args, torch.cuda.current_stream())
        return core

    def _replay_table(self, layer_id: int, metadata) -> int:
        """K1's ``replay`` argument (``KdaPreBuild.replay_cap``; 0 without):
        a device table of the layer's ReplaySSM pointers in ``k1.REPLAY_TABLE``
        order -- the requests' checkpoint slots and the cursors the step's
        commit moved (metadata), the layer's record buffers. Every one is a
        persistent buffer, so a layer's table is written once, on the first
        (eager) step; one that moved under a graph capture could not be
        rewritten into it, and raises."""
        if not self.replay_cap:
            return 0
        cache = get_forward_context().kv_cache_data[f"layer_{layer_id}"]
        sources = {
            "slot_idx": metadata.slot_idx,
            "write_pos": metadata.write_pos,
            "rbuf_k": cache.replay_buf_k,
            "rbuf_u": cache.replay_buf_u,
            "rbuf_g": cache.replay_buf_g,
        }
        ptrs = tuple(sources[name].data_ptr() for name in k1.REPLAY_TABLE)
        table, held = self._replay_tables.get(layer_id, (None, None))
        if held != ptrs:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"layer {layer_id}: a ReplaySSM buffer moved under a graph "
                    "capture; K1's pointer table was written for the old one"
                )
            table = torch.tensor(ptrs, dtype=torch.int64, device=self.scratch.device)
            self._replay_tables[layer_id] = (table, ptrs)
        return table.data_ptr()

    def _k2(self, build, plan, core, prefix, blocks) -> torch.Tensor:
        w = plan.moe
        key = build.k2_keys[plan.layer_id]
        out = torch.empty_like(prefix)
        p = torch.Tensor.data_ptr
        args = k2.ABI.pack({
            "core": p(core), "w_o": p(w.w_o), "prefix": p(prefix),
            "blocks": p(blocks), "blk_sm": blocks.stride(0),
            "blk_sr": blocks.stride(1), "ares_nw": p(w.seam.nw),
            "ares_qk": p(w.seam.qk), "in_nw": p(w.seam.out_nw),
            "w_gate": p(w.w_gate), "bias": p(w.bias), "w_ld": p(w.w_ld),
            "w_sgu": p(w.w_sgu), "w_sd": p(w.w_sd), "w13": p(w.w13),
            "w13s": p(w.w13s), "w2": p(w.w2), "w2s": p(w.w2s),
            "ln_w": p(w.ln_w), "w_up": p(w.w_up), "out": p(out),
            # not ``peers.kernel_args()``: that also hands over ``sym``, this
            # rank's own buffer, and K2 takes only the table -- it loads its own
            # base from entry ``rank`` of it (``load_ptr64(peers, rank)``)
            "scratch": p(self.scratch), "plain": p(self.plain),
            "queue": p(self.queue) + k2b.queue_offset(K2_TAG0 + plan.layer_id),
            "peers": self.peers.addresses.data_ptr(), "rank": self.peers.rank,
            "layer": K2_TAG0 + plan.layer_id,
            # a block-write layer's MLP seam restarts its prefix
            "reset": int(plan.moe.write_idx >= 0),
            "diag": self._diag_ptr(), "tl": 0,
        })  # fmt: skip
        _run_compiled(build.k2_launchers[key], *args, torch.cuda.current_stream())
        return out
