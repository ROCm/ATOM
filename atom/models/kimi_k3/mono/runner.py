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
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import MappingProxyType

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
from atom.models.kimi_k3.mono.kernels.stages import mla_post as k2mp
from atom.models.kimi_k3.mono.kernels.stages import mla_post_plan
from atom.models.kimi_k3.mono.kernels.stages import mla_prep as k2pr
from atom.models.kimi_k3.mono.kernels.stages import moe as k2b
from atom.models.kimi_k3.mono.plan import (
    _width,
    kda_metadata,
    state_slots,
    width_key,
    width_refusal,
)
from atom.models.kimi_k3.mono.weights import bind_mla_post, bind_mla_prep, bind_model
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

# the per-token FP8 activation the MLA prep stage writes
FP8 = torch.float8_e4m3fn
_INT_OF = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s elements as integers of their width: a bitwise comparison."""
    return t.contiguous().view(_INT_OF[t.element_size()])


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
            + [(f, k2.abi_for(key)) for key, f in self.k2_launchers.items()]
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
    where ``kda.gemv_staging`` keeps two groups, so only where those do not fit;
    and each K2 build with the MLA post / prep folds whose launcher spilled
    replaced by its layer's default build (``k2.unfolded``): those layers run
    their attention tail / seam unfolded at this width (the forward follows the
    width's K2 keys), the width stays mono. The fold builds sit at K2's
    register limit, where the backend can leave an unreferenced SGPR spill slot
    (a private segment, never accessed) that the scratch guard refuses.
    None when no spilled launcher is such a K1 or K2 build (a default K2 or
    indexer spill, or a K1 at one group a unit already): there is nothing
    leaner to build."""
    lean = {
        key: replace(key, gemv_unit=1)
        for key, launcher in built.k1_launchers.items()
        if launcher in spilled and key.gemv_unit > 1
    }
    plain = {
        key: k2.unfolded(key)
        for key, launcher in built.k2_launchers.items()
        if launcher in spilled
        and isinstance(key, k2.K2Build)
        and (key.mla_post or key.mla_prep)
    }
    if not lean and not plain:
        return None
    if plain:
        logger.info(
            "Kimi-K3 mono: %d-row K2 of layers %s without the MLA folds (their"
            " fold builds need scratch)",
            next(iter(plain)).tokens,
            sorted(layer for layer, key in built.k2_keys.items() if key in plain),
        )
    k1_keys = {layer: lean.get(key, key) for layer, key in built.k1_keys.items()}
    k2_keys = {layer: plain.get(key, key) for layer, key in built.k2_keys.items()}
    return replace(
        built,
        k1_keys=k1_keys,
        k1_launchers={
            key: k1.build(key) if key in lean.values() else built.k1_launchers[key]
            for key in set(k1_keys.values())
        },
        k2_keys=k2_keys,
        k2_launchers={
            key: built.k2_launchers.get(key) or k2.build(key)
            for key in set(k2_keys.values())
        },
    )


class K3MonoDecodeRunner:
    """Binds every rank's weights and buffers, or none: a rank that cannot is a
    refusal every rank agrees on before anything collective."""

    max_rows = MAX_ROWS
    # ``ATOM_MONO_MLA_POST`` (``_bind_mla_post``): none until bound
    mla_post: Mapping = MappingProxyType({})
    _mla_post_pending = None
    # ``ATOM_MONO_MLA_PREP`` (``_bind_mla_prep``): MLA layer M -> its prep,
    # K2 layer M - 1 -> M; none until bound
    mla_prep: Mapping = MappingProxyType({})
    mla_prep_prev: Mapping = MappingProxyType({})
    _prep_pending: dict = MappingProxyType({})
    _pref_hold_next = None
    _positions = None

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
        # before ``_allocate``: an ``mla_post`` K2 key adds scratch (its pairs)
        self._bind_mla_post()
        self._bind_mla_prep()
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
        # the MLA post stage up to its width bound; wider steps run those
        # layers' attention tail unfused (``_pre_o_core``)
        post = self.mla_post if rows <= mla_post_plan.MAX_ROWS else {}
        return {
            p.layer_id: k2.K2Build(
                tokens=rows,
                # the MLP seam mixes this layer's own block too, when it wrote one
                nblocks=p.moe.blocks + (p.moe.write_idx >= 0),
                eps=p.moe.seam.eps,
                out_eps=p.moe.seam.out_eps,
                ln_eps=p.moe.ln_eps,
                diag=self.debug,
                mla_post=p.layer_id in post,
                **self._prep_key(p.layer_id, rows),
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
        self._prep_pending = {}
        self._pref_hold_next = None
        # ``ATOM_MONO_MLA_PREP=3``: the KV-cache write needs the positions
        self._positions = positions
        # the production prefix's pool block (``_seam_mirror``), released where
        # the production prefix is: when a later layer returns a new prefix
        hold = None
        with serving:
            for plan in self.plans:
                res = self._layer(
                    build, plan, positions, prefix, blocks, add, add2, metadata
                )
                if self._pref_hold_next is not None or res[0] is not prefix:
                    # a folded seam's mirror block replaces the held one, and any
                    # other new prefix releases it, as production releases its
                    # previous prefix there
                    hold, self._pref_hold_next = self._pref_hold_next, None
                prefix, add, add2 = res
                del res
        hidden, _ = self.inner.output_attn_res(prefix, blocks, add, add2)
        del hold
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
            post = keep = None
            if (
                plan.ffn_only
                and plan.layer_id in self.mla_post
                and build.k2_keys[plan.layer_id].mla_post
            ):
                core, prefix, post, keep = self._post_core(
                    plan, positions, prefix, blocks, add, add2
                )
            elif plan.ffn_only:
                core, prefix = self._pre_o_core(
                    plan, positions, prefix, blocks, add, add2
                )
            else:
                delta = add if add2 is None else (add2 if add is None else add + add2)
                core = self._k1(build, plan, prefix, delta, blocks, metadata)
            # K2 updates ``prefix`` in place: on a block-write layer it restarts
            # it at the all-reduced o_proj output, which K1 has already banked
            k2_args = (build, plan, core, prefix, blocks)
            ffn = self._k2(*k2_args) if post is None else self._k2(*k2_args, post)
            out = (prefix, ffn, None)
            # ``keep`` holds what the launch was handed pointers to until now
            del keep
            if self._mla_post_pending is not None:
                pending, self._mla_post_pending = self._mla_post_pending, None
                self._mla_post_compare(*pending)
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
        # the mirror lives as long as the production ``normed``
        layer, normed, prefix, _mirror, attn = self._prepped_seam(
            plan, prefix, blocks, add, add2
        )
        if attn is not None:
            core = layer.self_attn.forward_prepped(positions, *normed, **attn)
        else:
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

    def _bind_mla_post(self) -> None:
        """``ATOM_MONO_MLA_POST=1``: the full-attention layers whose v-up, g_proj
        and output gate run as the first stage of their own K2 launch
        (``kernels.stages.mla_post``) -- 4 launches fewer a layer, no new launch
        and no grid barrier. Off by default; then ``mla_post`` is empty and every
        key, launch and argument is as before. A layer the stage cannot serve
        exactly (a DSA layer's sparse MLA, a tail outside the FP8 recipe) keeps
        its unfused tail (``_pre_o_core``), as ``bind_model`` keeps a layer it
        refuses on the production path. The refusals are static properties of
        the model, the same on every rank; the stage is rank-local besides, so
        its K2 talks to its peers exactly as the unfused build does."""
        self.mla_post: dict = {}
        self.mla_post_check = envs.ATOM_MONO_MLA_POST_CHECK
        self._mla_post_pending = None
        if not envs.ATOM_MONO_MLA_POST:
            return
        for p in self.plans:
            if not p.ffn_only:
                continue
            layer = self.inner.layers[p.layer_id]
            try:
                self.mla_post[p.layer_id] = bind_mla_post(layer, p.layer_id, self.dims)
            except MonoUnsupported as why:
                logger.info(
                    "Kimi-K3 mono: layer %d keeps its unfused MLA tail (%s)",
                    p.layer_id,
                    why,
                )
        if self.rank == 0:
            logger.info(
                "Kimi-K3 mono: ATOM_MONO_MLA_POST: v-up / g_proj / gate in K2 on "
                "%d MLA layers",
                len(self.mla_post),
            )

    def _post_core(self, plan, positions, prefix, blocks, add, add2):
        """``_pre_o_core`` with the attention stopped at the MLA decode's latent
        output: K2's post stage makes ``core`` from it and g_proj's FP8 input.
        Returns the buffer the stage writes, the seam's prefix and the stage's
        kernel arguments, and the tensors those arguments point into: they must
        stay allocated until the launch (a caching allocator hands a released
        block to the next allocation, under graph capture across graphs)."""
        post = self.mla_post[plan.layer_id]
        # the mirror lives as long as the production ``normed``
        layer, normed, prefix, _mirror, attn = self._prepped_seam(
            plan, prefix, blocks, add, add2
        )
        if attn is not None:
            x, xs = normed
            olat = layer.self_attn.forward_prepped(
                positions, x, xs, latent_out=True, **attn
            )
        else:
            olat, x, xs = layer.self_attn.forward_latent(positions, normed)
        rows = prefix.size(0)
        heads, lora = post.w_v.shape[0], post.w_v.shape[2]
        if (
            olat.dim() != 3
            or tuple(olat.shape) != (rows, heads, lora)
            or olat.dtype != torch.bfloat16
            or olat.stride(1) != lora
            or olat.stride(2) != 1
            or x.dtype != post.w_g.dtype
            or tuple(x.shape) != (rows, post.w_g.shape[1])
            or not x.is_contiguous()
            or xs is None
            or xs.dtype != torch.float32
            or xs.numel() != rows
            or not xs.is_contiguous()
            or not prefix.is_contiguous()
        ):
            raise MonoUnsupported(
                f"layer {plan.layer_id}: MLA post inputs latent "
                f"{tuple(olat.shape)} {olat.dtype} {olat.stride()}, x "
                f"{tuple(x.shape)} {x.dtype}, scale "
                f"{None if xs is None else (tuple(xs.shape), xs.dtype)} for a "
                f"{rows}-row step"
            )
        core = prefix.new_empty(rows, self.dims.proj)
        p = torch.Tensor.data_ptr
        args = {
            "olat": p(olat), "olat_sr": olat.stride(0), "w_v": p(post.w_v),
            "w_v_s": p(post.w_v_s), "xg": p(x), "xg_s": p(xs),
            "w_g": p(post.w_g), "w_g_s": p(post.w_g_s),
        }  # fmt: skip
        if self.mla_post_check and not torch.cuda.is_current_stream_capturing():
            self._mla_post_pending = (plan.layer_id, layer, olat, x, xs, core)
        # ``_mirror``: the production ``normed``'s allocations live as long as
        # the production ``normed`` (x, xs) does, past the K2 launch
        return core, prefix, args, (olat, x, xs, _mirror)

    def _mla_post_compare(self, layer_id, layer, olat, x, xs, core) -> None:
        """``ATOM_MONO_MLA_POST_CHECK=1`` (eager GPU validation): the production
        tail -- ``_v_up_proj``, g_proj, ``attn * sigmoid(gate)`` -- on the same
        inputs, against the core the K2 post stage wrote; rank 0 logs the bf16
        ulp statistics. Synchronizes."""
        sa = layer.self_attn
        av = sa.attn.impl._v_up_proj(olat).reshape(olat.size(0), -1)
        gate = sa.g_proj(x, xs)
        ref = av * torch.sigmoid(gate)
        a = core.view(torch.int16).int()
        b = ref.view(torch.int16).int()
        a = torch.where(a < 0, -(a & 0x7FFF), a)
        b = torch.where(b < 0, -(b & 0x7FFF), b)
        d = (a - b).abs()
        if self.rank == 0:
            logger.info(
                "MLA post check layer %d: %d / %d bf16 values differ, max %d ulp, "
                "max |diff| %.3e (max |ref| %.3e)",
                layer_id,
                int((d > 0).sum()),
                d.numel(),
                int(d.max()),
                float((core.float() - ref.float()).abs().max()),
                float(ref.float().abs().max()),
            )

    def _bind_mla_prep(self) -> None:
        """``ATOM_MONO_MLA_PREP=1``: the MLA layers M whose attention seam (and
        input RMSNorm + per-token FP8 quant) runs as the tail stage of K2(M-1)
        (``kernels.stages.mla_prep``, P0). Needs M-1 to run K2. Off by default;
        then every key, launch and argument is as before. A layer it cannot
        serve keeps its production seam (as ``bind_model`` keeps a refused
        layer); the refusals are static, the same on every rank."""
        self.mla_prep = {}
        self.mla_prep_prev = {}
        self.mla_prep_check = envs.ATOM_MONO_MLA_PREP_CHECK
        raw = envs.ATOM_MONO_MLA_PREP.strip() or "0"
        if raw == "0":
            return
        if raw not in ("1", "2", "3"):
            raise MonoUnsupported(
                f"ATOM_MONO_MLA_PREP={raw}: sub-modes are 1 (the seam, P0), 2 (+ the "
                "q_a / kv_a GEMV and norms, P1 / P2) and 3 (+ q_b, the absorb, rope "
                "and the FP8 KV-cache write, P3 .. P5)"
            )
        mode = int(raw)
        # GPU bisect aid: fold only these MLA layers (comma list; default every one)
        only = envs.ATOM_MONO_MLA_PREP_LAYERS.strip()
        only = {int(v) for v in only.split(",") if v.strip()} if only else None
        by_id = {p.layer_id: p for p in self.plans}
        for p in self.plans:
            if not p.ffn_only or (only is not None and p.layer_id not in only):
                continue
            prev = by_id.get(p.layer_id - 1)
            if prev is None or prev.moe is None:
                continue
            try:
                prep = bind_mla_prep(
                    self.inner.layers[p.layer_id], p.layer_id, self.dims, mode
                )
                nb = prev.moe.blocks + (prev.moe.write_idx >= 0)
                if not 1 <= prep.blocks <= nb:
                    raise MonoUnsupported(
                        f"layer {p.layer_id}: its seam mixes {prep.blocks} blocks, "
                        f"K2 of layer {prev.layer_id} stages {nb}"
                    )
            except MonoUnsupported as why:
                logger.info(
                    "Kimi-K3 mono: layer %d keeps its production seam (%s)",
                    p.layer_id,
                    why,
                )
                continue
            self.mla_prep[p.layer_id] = prep
            self.mla_prep_prev[prev.layer_id] = p.layer_id
        if not self.mla_prep:
            return
        self._alloc_prep_buffers()
        if self.rank == 0:
            logger.info(
                "Kimi-K3 mono: ATOM_MONO_MLA_PREP=%d: the prep of %d MLA layers in "
                "the preceding K2 (layers %s)",
                mode,
                len(self.mla_prep),
                sorted(self.mla_prep),
            )

    def _alloc_prep_buffers(self) -> None:
        """The prep stage's outputs, allocated once here -- outside every CUDA
        graph's memory pool -- and used as ``[:rows]`` views by every width.

        A step's pool allocations then stay exactly the flag-off ones:
        ``_prep_args`` takes nothing from the pool, and ``_prepped_seam``
        replays the production seam's own allocations (``_seam_mirror``).

        ``pref`` alternates between two buffers by the MLA layer's ordinal:
        K2(M-1) reads prefix(M-1) -- the previous MLA layer's ``pref``, updated
        in place since -- while it writes ``pref`` (M). x8 / xs are dead by the
        next MLA layer's K2(M-1)."""
        n, dev = self.max_rows, self.device
        hidden = self.inner.config.hidden_size

        def e(*shape, dtype):
            return torch.empty(*shape, dtype=dtype, device=dev)

        self._prep_bufs = {
            "pref": (
                e(n, hidden, dtype=torch.bfloat16),
                e(n, hidden, dtype=torch.bfloat16),
            ),
            "x8": e(n, hidden, dtype=FP8),
            "xs": e(n, 1, dtype=torch.float32),
        }
        if max(w.mode for w in self.mla_prep.values()) >= 2:
            self._prep_bufs.update(
                qkv=e(n, k2pr.QKV_A, dtype=torch.bfloat16),
                q8=e(n, k2pr.QL, dtype=FP8),
                qs=e(n, 1, dtype=torch.float32),
                kvn=e(n, k2pr.KVL, dtype=torch.bfloat16),
            )
        if max(w.mode for w in self.mla_prep.values()) >= 3:
            heads = {w.q_heads for w in self.mla_prep.values()}
            if len(heads) != 1:
                raise MonoUnsupported(
                    f"MLA q_out heads {sorted(heads)}: one buffer serves every layer"
                )
            # zeros: the padded heads past the 12 a rank owns stay 0, as the
            # production q_out's (P4 writes the 12 every step)
            self._prep_bufs.update(
                qb=e(n, k2pr.QB, dtype=torch.bfloat16),
                q_out=torch.zeros(n, heads.pop(), k2pr.ENTRY, dtype=FP8, device=dev),
            )
            # a dummy run writes no KV-cache row (nor does production there)
            self._neg_slots = torch.full((n,), -1, dtype=torch.int64, device=dev)
        self._prep_ord = {m: i for i, m in enumerate(sorted(self.mla_prep))}

    def _seam_mirror(
        self, prefix: torch.Tensor, block_residual: torch.Tensor, add=None, add2=None
    ) -> tuple:
        """The production attention seam's pool allocations, in its order and
        sizes (``attention_residual._apply_attn_res_impl``: ``br``, ``y``,
        ``y_scale``, the addends' copies, ``pref``; ``br`` and the copies
        released when it returns), with no kernel. Returns ``(y, y_scale)`` for
        the caller to keep alive as long as the production ``normed``; ``pref``
        waits in ``_pref_hold_next`` for the forward loop to keep as long as the
        production prefix. A change to that function's allocations must change
        this one with it."""
        dev = prefix.device
        rows = prefix.size(0)
        br = None
        if not block_residual.is_contiguous():
            br = torch.empty(
                block_residual.shape, dtype=block_residual.dtype, device=dev
            )
        ps = prefix.contiguous()
        y = torch.empty((rows, prefix.size(1)), dtype=FP8, device=dev)
        y_scale = torch.empty((rows, 1), dtype=torch.float32, device=dev)
        hs = add.contiguous() if add is not None else None
        hs2 = add2.contiguous() if add2 is not None else None
        self._pref_hold_next = torch.empty_like(ps) if add is not None else None
        del br, hs, hs2
        return y, y_scale

    def _prep_key(self, layer_id: int, rows: int) -> dict:
        """K2 build-key fields for layer ``layer_id`` (an M - 1) at ``rows``, or
        none (past the stage's width the next layer keeps its own seam; so does
        a layer whose P0 sums do not fit K2's ``red`` at this width, e.g. 16 rows
        on 7+ blocks: ``build`` would refuse the whole width)."""
        m = self.mla_prep_prev.get(layer_id)
        if m is None or rows > k2pr.MAX_ROWS:
            return {}
        w = self.mla_prep[m]
        if not k2pr.red_fits(rows, w.blocks + 1):
            return {}
        key = {
            "mla_prep": w.mode,
            "prep_nblocks": w.blocks,
            "prep_eps": w.seam.eps,
            "prep_out_eps": w.seam.out_eps,
        }
        if w.mode >= 2:
            key.update(qa_eps=w.qa_eps, kva_eps=w.kva_eps)
        if w.mode >= 3:
            key.update(rope_neox=w.rope_neox, rope_maxpos=w.cos.numel() // 32)
        return key

    def _prep_args(self, layer_id: int, prefix: torch.Tensor) -> dict:
        """The prep stage's arguments of K2(``layer_id``) for MLA layer
        ``layer_id`` + 1; its outputs (``[:rows]`` of ``_alloc_prep_buffers``'
        buffers, nothing from the graph pool) kept for ``_prepped_seam``."""
        m = self.mla_prep_prev[layer_id]
        w = self.mla_prep[m]
        rows = prefix.size(0)
        bufs = self._prep_bufs
        if rows > bufs["x8"].size(0) or prefix.size(1) != bufs["x8"].size(1):
            raise MonoUnsupported(
                f"layer {m}: a {tuple(prefix.shape)} prefix, the prep buffers "
                f"{tuple(bufs['x8'].shape)}"
            )
        out = {
            "pref": bufs["pref"][self._prep_ord[m] % 2][:rows],
            "x8": bufs["x8"][:rows],
            "xs": bufs["xs"][:rows],
        }
        p = torch.Tensor.data_ptr
        args = dict.fromkeys(k2pr.ARGS, 0)
        args.update(
            pref=p(out["pref"]), x8=p(out["x8"]), xs=p(out["xs"]),
            p_nw=p(w.seam.nw), p_qk=p(w.seam.qk), p_onw=p(w.seam.out_nw),
        )  # fmt: skip
        out["attn"] = None
        if w.mode >= 2:
            qkv, q8, qs, kvn = (bufs[n][:rows] for n in ("qkv", "q8", "qs", "kvn"))
            args.update(
                w_qa=p(w.w_qa), w_qa_s=p(w.w_qa_s), qkv=p(qkv), q8=p(q8), qs=p(qs),
                kvn=p(kvn), qa_nw=p(w.qa_nw), kva_nw=p(w.kva_nw),
            )  # fmt: skip
            # ``forward_prepped``'s inputs: ``_mla_inputs``' results
            out["qkv"] = qkv
            out["p2"] = {"q": q8, "q_scale": qs, "kv": kvn}
            out["attn"] = {**out["p2"], "k_rope": qkv[:, k2pr.QL + k2pr.KVL :]}
        if w.mode >= 3:
            args.update(self._prep_args3(m, w, rows, out))
        self._prep_pending[m] = out
        return args

    def _prep_args3(self, m: int, w, rows: int, out: dict) -> dict:
        """Sub-mode 3's arguments for MLA layer ``m`` (P3 .. P5): q_b, W_K, the
        rope tables, the layer's FP8 KV cache, the step's slots and positions;
        ``out`` takes what ``forward_prepped`` and the check read."""
        fc = get_forward_context()
        meta = fc.attn_metadata
        if meta.dtype_q != FP8:
            raise MonoUnsupported(
                f"layer {m}: q_out dtype {meta.dtype_q}, the stages write {FP8}"
            )
        kv_cache = fc.kv_cache_data[f"layer_{m}"].k_cache
        if kv_cache.element_size() != 1 or not kv_cache.is_contiguous():
            raise MonoUnsupported(
                f"layer {m}: KV cache {kv_cache.dtype} stride {kv_cache.stride()}, "
                "the stages write FP8 rows of a contiguous cache"
            )
        # the production kernel's view (block, offset) and one row a slot (the
        # stages' address: slot * 576 B, whatever the block size)
        kv3 = kv_cache.view(kv_cache.shape[0], -1, k2pr.ENTRY)
        kvrows = kv_cache.view(-1, k2pr.ENTRY)
        slots = meta.slot_mapping.reshape(-1)
        if fc.context.is_dummy_run:
            slots = self._neg_slots
        pos = self._positions.reshape(-1)
        if (
            slots.dtype != torch.int64
            or slots.numel() < rows
            or not slots.is_contiguous()
            or pos.dtype != torch.int64
            or pos.numel() < rows
            or not pos.is_contiguous()
        ):
            raise MonoUnsupported(
                f"layer {m}: slot mapping {slots.dtype} {tuple(slots.shape)} / "
                f"positions {pos.dtype} {tuple(pos.shape)} for a {rows}-row step"
            )
        # the tokens' own (the kernels read ``rows`` of each)
        slots, pos = slots[:rows], pos[:rows]
        bufs = self._prep_bufs
        qb, q_out = bufs["qb"][:rows], bufs["q_out"][:rows]
        out.update(
            qb=qb, slots=slots, pos=pos, kv3=kv3, kvrows=kvrows, attn={"q_out": q_out}
        )
        p = torch.Tensor.data_ptr
        return {
            "w_qb": p(w.w_qb), "w_qb_s": p(w.w_qb_s), "w_k": p(w.w_k),
            "w_k_s": p(w.w_k_s), "qb": p(qb), "qout": p(q_out),
            "qout_sr": q_out.stride(0), "kvc": p(kv_cache), "kvc_sr": kvrows.stride(0),
            "slots": p(slots), "pos": p(pos), "cosc": p(w.cos), "sinc": p(w.sin),
            "k_sc": p(w.k_scale), "q_sc": p(w.q_scale),
        }  # fmt: skip

    @torch.no_grad()
    def _mla_prep_compare(self, layer_id, layer, prefix, blocks, add, add2, pend):
        """``ATOM_MONO_MLA_PREP_CHECK=1`` (eager GPU validation): the production
        seam (and, sub-mode 2, fused_qkv_a_proj and the q_a / kv_a norm + quant)
        on the same inputs against what K2(M-1)'s prep stages wrote; rank 0 logs
        how many values differ. P2 is rerun on P1's own output and must show 0;
        sub-mode 3: ``_mla_prep_compare3``. Synchronizes."""

        def ndiff(a, b):
            return int((_bits(a) != _bits(b)).sum())

        (x8r, xsr), prefr = layer.self_attention_attn_res(prefix, blocks, add, add2)
        seam = (
            f"seam x8 {ndiff(pend['x8'], x8r)}/{x8r.numel()} "
            f"xs {ndiff(pend['xs'], xsr)} pref {ndiff(pend['pref'], prefr)}"
        )
        msgs = [seam]
        if "p2" in pend:
            from atom.models.deepseek_v2 import _fuse_rmsnorm_quant

            sa = layer.self_attn
            qkv = pend["qkv"]
            ref = sa.fused_qkv_a_proj(pend["x8"], pend["xs"])
            msgs.append(f"P1 qkv {ndiff(qkv, ref)}/{ref.numel()} (GEMM order)")
            # P2 on P1's own output: aiter's kernel, operation for operation
            (q8e, qse), _, kve, _ = _fuse_rmsnorm_quant(
                qkv[:, : k2pr.QL], sa.q_a_layernorm.weight, sa.q_a_layernorm.eps,
                qkv[:, k2pr.QL : k2pr.QL + k2pr.KVL], sa.kv_a_layernorm.weight,
                sa.kv_a_layernorm.eps, None, dtype_quant=sa.qknorm_dtype,
                shuffle=False, scale_shuffle_padding=False, group_size=128,
                quant_type=sa.qknorm_quant_type_value, output_unquantized_inp1=False,
                transpose_scale=True,
            )  # fmt: skip
            a = pend["p2"]
            msgs.append(
                f"P2 q8 {ndiff(a['q'], q8e)} qs {ndiff(a['q_scale'], qse)} "
                f"kv {ndiff(a['kv'], kve)} (must be 0)"
            )
        if "qb" in pend:
            msgs.append(self._mla_prep_compare3(sa, pend, kve, ndiff))
        if self.rank == 0:
            logger.info("MLA prep check layer %d: %s", layer_id, "; ".join(msgs))

    def _mla_prep_compare3(self, sa, pend, kve, ndiff) -> str:
        """Sub-mode 3's part of ``_mla_prep_compare``: q_b_proj on the stage's
        own ``q8`` / ``q_scale`` (P3, GEMM order), then the production absorb,
        rope and KV-cache write on the stages' own ``qb`` / ``kv`` / ``k_rope``.
        The KV rows and q_out's rope part must match bit for bit; q_out's nope
        part differs by the absorb's summation order. Leaves the production
        kernel's KV bytes in place (equal when the check passes)."""
        from atom.model_ops.attention_mla import (
            _aiter_triton_fp8_bmm,
            fused_qk_rope_concat_and_cache_mla,
        )

        impl = sa.attn.impl
        p2 = pend["p2"]
        qbr = impl.q_proj(p2["q"], p2["q_scale"])
        p3 = f"P3 qb {ndiff(pend['qb'], qbr)}/{qbr.numel()} (GEMM order)"
        qb = pend["qb"].view(-1, k2pr.HEADS, k2pr.QH)
        ql = _aiter_triton_fp8_bmm(
            qb[..., : k2pr.QK_NOPE].transpose(0, 1), impl.W_K, impl.W_K_scale,
            group_size=128, transpose_bm=True,
        )  # fmt: skip
        slots, kv3, kvrows = pend["slots"], pend["kv3"], pend["kvrows"]
        live = slots >= 0
        ours = kvrows[slots.clamp(min=0)].clone()
        q_out = pend["attn"]["q_out"]
        q_ref = torch.zeros_like(q_out)
        fused_qk_rope_concat_and_cache_mla(
            ql, qb[..., k2pr.QK_NOPE :], kve,
            pend["qkv"][:, k2pr.QL + k2pr.KVL :], kv3, q_ref[:, : k2pr.HEADS],
            slots, impl._k_scale, impl._q_scale, pend["pos"],
            impl.rotary_emb.cos_cache, impl.rotary_emb.sin_cache,
            is_neox=impl.rotary_emb.is_neox_style, is_nope_first=True,
            compute_all_q_rope=False,
        )  # fmt: skip
        theirs = kvrows[slots.clamp(min=0)]
        q_live = q_out[live][:, : k2pr.HEADS]
        r_live = q_ref[live][:, : k2pr.HEADS]
        return p3 + (
            f"; P5 KV rows {ndiff(ours[live], theirs[live])}/{ours[live].numel()} "
            f"(must be 0), q_out rope "
            f"{ndiff(q_live[..., k2pr.KVL :], r_live[..., k2pr.KVL :])} (must be 0), "
            f"q_out nope {ndiff(q_live[..., : k2pr.KVL], r_live[..., : k2pr.KVL])} "
            "(absorb order)"
        )

    def _prepped_seam(self, plan, prefix, blocks, add, add2):
        """``_seam``, or -- when K2(M-1) already ran this layer's prep -- its
        results: ``(layer, normed, prefix, mirror, attn)`` with ``normed`` the
        per-token FP8 activation and its scale, ``prefix`` the stage's ``pref``,
        ``mirror`` the production seam's pool allocations, to hold as long as
        the production path holds its own, then (sub-mode 2) ``attn``, the
        ``forward_prepped`` inputs; the last two None without the stage."""
        pend = None
        if plan.layer_id in self._prep_pending:
            pend = self._prep_pending.pop(plan.layer_id)
        if pend is None:
            layer, normed, prefix, _, _ = self._seam(plan, prefix, blocks, add, add2)
            return layer, normed, prefix, None, None
        layer = self.inner.layers[plan.layer_id]
        before, _ = layer_block(plan.layer_id)
        if self.mla_prep_check and not torch.cuda.is_current_stream_capturing():
            self._mla_prep_compare(
                plan.layer_id, layer, prefix, blocks[:, :before], add, add2, pend
            )
        mirror = self._seam_mirror(prefix, blocks[:, :before], add, add2)
        return layer, (pend["x8"], pend["xs"]), pend["pref"], mirror, pend["attn"]

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

    def _k2(self, build, plan, core, prefix, blocks, post=None) -> torch.Tensor:
        """``post`` (an ``mla_post`` build): the post stage's arguments, ``core``
        then being the buffer the stage writes and o_proj reads."""
        w = plan.moe
        key = build.k2_keys[plan.layer_id]
        out = torch.empty_like(prefix)
        p = torch.Tensor.data_ptr
        args = k2.abi_for(key).pack({
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
            **(post or ({} if not key.mla_prep else dict.fromkeys(k2mp.ARGS, 0))),
            **(self._prep_args(plan.layer_id, prefix) if key.mla_prep else {}),
        })  # fmt: skip
        _run_compiled(build.k2_launchers[key], *args, torch.cuda.current_stream())
        return out
