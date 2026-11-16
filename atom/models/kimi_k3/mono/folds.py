# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The host side of K3 mono's MLA folds, mixed into ``K3MonoDecodeRunner``.

Two folds move an MLA layer's glue into K2 launches the runner already makes
(``ATOM_MONO_MLA_POST`` / ``ATOM_MONO_MLA_PREP``, both on by default):

* **post** (``kernels.stages.mla_post``): the layer's v-up BMM, g_proj and
  output gate as the first stage of its own K2, from the MLA decode's latent
  output (``_post_core``);
* **prep** (``kernels.stages.mla_prep``): the layer's attention seam, q_a /
  kv_a, q_b, the absorb, rope and FP8 KV-cache write as the tail stages of the
  PRECEDING layer's K2; the layer then starts at its MLA decode from the
  stages' ``q_out`` (``_prepped_seam``).

Here: which layers fold (``_bind_mla_post`` / ``_bind_mla_prep``, static
refusals every rank shares), their K2 build-key fields (``_fold_key``) and
launch arguments (``_fold_args``), the prep stages' buffers, and the
production seam's pool allocations replayed (``_seam_mirror``), so a folded
step's CUDA-graph pool layout matches the unfused one's outside the production
attention internals the stages replace (the q_a / kv_a / q_b temporaries it
skips). A layer or width a fold
cannot serve keeps the unfused path; with both flags ``0`` every key, launch
and argument is the unfused path's.
"""

import logging
from collections.abc import Mapping
from types import MappingProxyType

import torch

from atom.models.kimi_k3.mono.config import layer_block
from atom.models.kimi_k3.mono.kernels.stages import mla_post as k2mp
from atom.models.kimi_k3.mono.kernels.stages import mla_post_plan
from atom.models.kimi_k3.mono.kernels.stages import mla_prep as k2pr
from atom.models.kimi_k3.mono.weights import bind_mla_post, bind_mla_prep
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# the per-token FP8 activation the MLA prep stage writes
FP8 = torch.float8_e4m3fn


class MlaFolds:
    """The MLA post / prep folds of a ``K3MonoDecodeRunner`` (its ``plans``,
    ``inner``, ``dims``, ``rank``, ``max_rows``, ``device`` and ``_seam``)."""

    # ``ATOM_MONO_MLA_POST`` (``_bind_mla_post``): none until bound
    mla_post: Mapping = MappingProxyType({})
    # ``ATOM_MONO_MLA_PREP`` (``_bind_mla_prep``): MLA layer M -> its prep,
    # K2 layer M - 1 -> M; none until bound
    mla_prep: Mapping = MappingProxyType({})
    mla_prep_prev: Mapping = MappingProxyType({})
    _prep_pending: Mapping = MappingProxyType({})
    _pref_hold_next = None
    _positions = None

    def _fold_step(self, positions: torch.Tensor) -> None:
        """A step's start: no prep pending, no mirror block held, and the
        positions the prep stages' rope and KV-cache write read."""
        self._prep_pending = {}
        self._pref_hold_next = None
        self._positions = positions

    def _fold_key(self, layer_id: int, rows: int) -> dict:
        """K2(``layer_id``)'s build-key fields at ``rows``: its own post stage
        up to the stage's width (a wider step runs the layer's attention tail
        unfused, ``_pre_o_core``) and the next layer's prep (``_prep_key``)."""
        post = layer_id in self.mla_post and rows <= mla_post_plan.MAX_ROWS
        return {"mla_post": post, **self._prep_key(layer_id, rows)}

    def _fold_args(self, key, layer_id: int, prefix: torch.Tensor, post) -> dict:
        """K2(``layer_id``)'s arguments past ``k2.ABI`` for build ``key``: the
        post stage's (``post``, from ``_post_core``; zeros in a prep build
        without it), then the prep stages' (``_prep_args``)."""
        if not key.mla_prep:
            return post or {}
        zeros = dict.fromkeys(k2mp.ARGS, 0)
        return {**(post or zeros), **self._prep_args(layer_id, prefix)}

    def _bind_mla_post(self) -> None:
        """``ATOM_MONO_MLA_POST`` (on by default): the full-attention layers whose
        v-up, g_proj and output gate run as the first stage of their own K2
        launch (``kernels.stages.mla_post``) -- 4 launches fewer a layer, no new
        launch and no grid barrier. Off (``0``), ``mla_post`` is empty and every
        key, launch and argument is the unfused path's. A layer the stage cannot
        serve exactly (a DSA layer's sparse MLA, a tail outside the FP8 recipe)
        keeps its unfused tail (``_pre_o_core``), as ``bind_model`` keeps a
        layer it refuses on the production path. The refusals are static
        properties of the model, the same on every rank; the stage is rank-local
        besides, so its K2 talks to its peers exactly as the unfused build
        does."""
        self.mla_post: dict = {}
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
                "Kimi-K3 mono: v-up / g_proj / gate in K2 on %d MLA layers",
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
        layer, normed, prefix, _mirror, q_out = self._prepped_seam(
            plan, prefix, blocks, add, add2
        )
        if q_out is not None:
            x, xs = normed
            olat = layer.self_attn.forward_prepped(x, xs, q_out, latent_out=True)
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
        # ``_mirror``: the production ``normed``'s allocations live as long as
        # the production ``normed`` (x, xs) does, past the K2 launch
        return core, prefix, args, (olat, x, xs, _mirror)

    def _bind_mla_prep(self) -> None:
        """``ATOM_MONO_MLA_PREP`` (on by default): the MLA layers M whose prep
        -- the attention seam with the input RMSNorm + per-token FP8 quant (P0),
        fused_qkv_a_proj and the q_a / kv_a RMSNorms (P1 / P2), q_b, the absorb,
        rope and the FP8 KV-cache write (P3 .. P5) -- runs as the tail stages of
        K2(M-1) (``kernels.stages.mla_prep``). Needs M-1 to run K2. Off
        (``0``), every key, launch and argument is the unfused path's. A layer
        it cannot serve keeps its production seam (as ``bind_model`` keeps a
        refused layer); the refusals are static, the same on every rank."""
        self.mla_prep = {}
        self.mla_prep_prev = {}
        if not envs.ATOM_MONO_MLA_PREP:
            return
        by_id = {p.layer_id: p for p in self.plans}
        for p in self.plans:
            if not p.ffn_only:
                continue
            prev = by_id.get(p.layer_id - 1)
            if prev is None or prev.moe is None:
                continue
            try:
                prep = bind_mla_prep(
                    self.inner.layers[p.layer_id], p.layer_id, self.dims
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
                "Kimi-K3 mono: the prep of %d MLA layers in the preceding K2 "
                "(layers %s)",
                len(self.mla_prep),
                sorted(self.mla_prep),
            )

    def _alloc_prep_buffers(self) -> None:
        """The prep stage's outputs, allocated once here -- outside every CUDA
        graph's memory pool -- and used as ``[:rows]`` views by every width.

        A step's pool allocations then match the flag-off ones outside the
        production attention internals the stages replace (the q_a / kv_a /
        q_b temporaries a folded step skips): ``_prep_args`` takes nothing from
        the pool, and ``_prepped_seam`` replays the production seam's own
        allocations (``_seam_mirror``).

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
            "qkv": e(n, k2pr.QKV_A, dtype=torch.bfloat16),
            "q8": e(n, k2pr.QL, dtype=FP8),
            "qs": e(n, 1, dtype=torch.float32),
            "kvn": e(n, k2pr.KVL, dtype=torch.bfloat16),
            "qb": e(n, k2pr.QB, dtype=torch.bfloat16),
        }
        heads = {w.q_heads for w in self.mla_prep.values()}
        if len(heads) != 1:
            raise MonoUnsupported(
                f"MLA q_out heads {sorted(heads)}: one buffer serves every layer"
            )
        # zeros: the padded heads past the 12 a rank owns stay 0, as the
        # production q_out's (P4 writes the 12 every step)
        self._prep_bufs["q_out"] = torch.zeros(
            n, heads.pop(), k2pr.ENTRY, dtype=FP8, device=dev
        )
        # a dummy run writes no KV-cache row (nor does production there)
        self._neg_slots = torch.full((n,), -1, dtype=torch.int64, device=dev)
        self._prep_ord = {m: i for i, m in enumerate(sorted(self.mla_prep))}

    def _seam_mirror(
        self, prefix: torch.Tensor, block_residual: torch.Tensor, add=None, add2=None
    ) -> tuple:
        """The production attention seam's pool allocations, in its order and
        sizes (``attention_residual._apply_attn_res_impl``: ``br`` -- only for a
        block bank strided along H; a slice of the bank is read in place --,
        ``y``, ``y_scale``, the addends' copies, ``pref``; ``br`` and the copies
        released when it returns), with no kernel. Returns ``(y, y_scale)`` for
        the caller to keep alive as long as the production ``normed``; ``pref``
        waits in ``_pref_hold_next`` for the forward loop to keep as long as the
        production prefix. A change to that function's allocations must change
        this one with it (``test_seam_mirror_allocates_as_the_production_seam``
        fails otherwise)."""
        dev = prefix.device
        rows = prefix.size(0)
        br = None
        if block_residual.stride(-1) != 1:
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
        return {
            "mla_prep": True,
            "prep_nblocks": w.blocks,
            "prep_eps": w.seam.eps,
            "prep_out_eps": w.seam.out_eps,
            "qa_eps": w.qa_eps,
            "kva_eps": w.kva_eps,
            "rope_neox": w.rope_neox,
            "rope_maxpos": w.cos.numel() // 32,
        }

    def _prep_args(self, layer_id: int, prefix: torch.Tensor) -> dict:
        """The prep stages' arguments of K2(``layer_id``) for MLA layer ``m`` =
        ``layer_id`` + 1: its weights, its FP8 KV cache, the step's slots and
        positions, and the stages' outputs (``[:rows]`` of
        ``_alloc_prep_buffers``' buffers, nothing from the graph pool), the ones
        layer ``m`` reads kept for ``_prepped_seam``."""
        m = self.mla_prep_prev[layer_id]
        w = self.mla_prep[m]
        rows = prefix.size(0)
        bufs = self._prep_bufs
        if rows > bufs["x8"].size(0) or prefix.size(1) != bufs["x8"].size(1):
            raise MonoUnsupported(
                f"layer {m}: a {tuple(prefix.shape)} prefix, the prep buffers "
                f"{tuple(bufs['x8'].shape)}"
            )
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
        # one row a slot (the stages' address: slot * 576 B, whatever the
        # block size)
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
        o = {n: v[:rows] for n, v in bufs.items() if n != "pref"}
        pref = bufs["pref"][self._prep_ord[m] % 2][:rows]
        self._prep_pending[m] = {
            "pref": pref, "x8": o["x8"], "xs": o["xs"], "q_out": o["q_out"],
        }  # fmt: skip
        p = torch.Tensor.data_ptr
        return {
            "pref": p(pref), "x8": p(o["x8"]), "xs": p(o["xs"]),
            "p_nw": p(w.seam.nw), "p_qk": p(w.seam.qk), "p_onw": p(w.seam.out_nw),
            "w_qa": p(w.w_qa), "w_qa_s": p(w.w_qa_s), "qkv": p(o["qkv"]),
            "q8": p(o["q8"]), "qs": p(o["qs"]), "kvn": p(o["kvn"]),
            "qa_nw": p(w.qa_nw), "kva_nw": p(w.kva_nw),
            "w_qb": p(w.w_qb), "w_qb_s": p(w.w_qb_s), "w_k": p(w.w_k),
            "w_k_s": p(w.w_k_s), "qb": p(o["qb"]), "qout": p(o["q_out"]),
            "qout_sr": o["q_out"].stride(0), "kvc": p(kv_cache),
            "kvc_sr": kvrows.stride(0), "slots": p(slots), "pos": p(pos),
            "cosc": p(w.cos), "sinc": p(w.sin), "k_sc": p(w.k_scale),
            "q_sc": p(w.q_scale),
        }  # fmt: skip

    def _prepped_seam(self, plan, prefix, blocks, add, add2):
        """``_seam``, or -- when K2(M-1) already ran this layer's prep -- its
        results: ``(layer, normed, prefix, mirror, q_out)`` with ``normed`` the
        per-token FP8 activation and its scale, ``prefix`` the stage's ``pref``,
        ``mirror`` the production seam's pool allocations, to hold as long as
        the production path holds its own, then ``q_out``, the MLA decode's
        input (``forward_prepped``); the last two None without the stage."""
        pend = None
        if plan.layer_id in self._prep_pending:
            pend = self._prep_pending.pop(plan.layer_id)
        if pend is None:
            layer, normed, prefix, _, _ = self._seam(plan, prefix, blocks, add, add2)
            return layer, normed, prefix, None, None
        layer = self.inner.layers[plan.layer_id]
        before, _ = layer_block(plan.layer_id)
        mirror = self._seam_mirror(prefix, blocks[:, :before], add, add2)
        return layer, (pend["x8"], pend["xs"]), pend["pref"], mirror, pend["q_out"]
