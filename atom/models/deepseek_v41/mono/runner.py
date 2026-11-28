# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""One TP rank's V4.1 mono decode: the bound weights, the step's buffers, and the
forward that runs them.

The forward is the model's own layer loop with each layer in two mono
kernels: K1 (``kernels.attn_pre``) from the attention seam to the rotated query
and the window row; K2 (``kernels.layer_post``) K2a (``kernels.attn_post``: a
selecting layer's indexer, the sparse decode attention, the output projections,
their all-reduce (in-kernel, over this runner's peer buffer), the FFN seam and
ffn_norm) and K2b (``kernels.moe``: the MoE and its all-reduce) in one launch.
``ATOM_MONO_CHECK`` launches K2a and K2b apart, to compare each. Between K1
and K2 of a compressing layer its compressor runs as one launch
(``compress_launch``; the original module where that launch is not built for
the layer or the step); before an Engram layer the original Engram runs.
``draft_runner`` runs the DSpark draft's layers on the same kernels.
"""

import logging
import math
import os
import struct
import threading
from dataclasses import dataclass, field, replace
from functools import partial

import torch
from aiter.dist.parallel_state import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.config import get_current_atom_config
from atom.distributed.group_agreement import group_agree
from atom.model_ops.deepseek_v41.candidate_table import candidate_block_table
from atom.model_ops.deepseek_v41.index_plane import FP4_MFMA
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState, initial_pre_mix
from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.peer_embedding import shared_peer_embedding
from atom.model_ops.v4_kernels.paged_decode import LOG2E
from atom.model_ops.verify_ids import IdVerifiers
from atom.models.deepseek_v41.config import AttentionMode
from atom.models.deepseek_v41.dspark import DeepseekV41DSpark
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.check import (
    check_front,
    check_index_score,
    check_moe,
    check_post,
    check_selection,
    decode_keys,
)
from atom.models.deepseek_v41.mono.compress_launch import MonoCompressor
from atom.models.deepseek_v41.mono.config import MAX_MOE_ROWS, MAX_ROWS
from atom.models.deepseek_v41.mono.draft_ctx_launch import MonoDraftContext
from atom.models.deepseek_v41.mono.engram_launch import MonoEngram
from atom.models.deepseek_v41.mono.epoch_begin import EpochMailboxes
from atom.models.deepseek_v41.mono.head_launch import bind_lm_head
from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import col_gather as cg
from atom.models.deepseek_v41.mono.kernels import compress as cp
from atom.models.deepseek_v41.mono.kernels import debug as kernel_debug
from atom.models.deepseek_v41.mono.kernels import draft_ctx as dc
from atom.models.deepseek_v41.mono.kernels import engram as eg
from atom.models.deepseek_v41.mono.kernels import layer as kl
from atom.models.deepseek_v41.mono.kernels import layer_pair as lp
from atom.models.deepseek_v41.mono.kernels import layer_run as lr
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.models.deepseek_v41.mono.kernels import moe as k2b
from atom.models.deepseek_v41.mono.kernels.dims import UG_PART, Dims, kvrow_bytes
from atom.models.deepseek_v41.mono.kernels.index_query import (
    QS_BYTES,
    QS_M_TILES,
)
from atom.models.deepseek_v41.mono.kernels.moe_shape import EXPERTS, TOPK
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_bytes, plain_layout
from atom.models.deepseek_v41.mono.kernels.token_par import own_rows
from atom.models.deepseek_v41.mono.numa_plan import numa_table
from atom.models.deepseek_v41.mono.step_meta import KMETA, write_step_meta
from atom.models.deepseek_v41.mono.weights import bind_model
from atom.mono.lm_head import kernel as lk
from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.layout import NUMA_PAGE
from atom.mono.runtime.aux_taps import AuxTaps, bind_aux_taps
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.debug import DIAG_BYTES, given_up_waits, raise_if_given_up
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.numa import scratch_word
from atom.mono.runtime.step_begin import ZERO_ALIGN
from atom.mono.runtime.timeline import LayerTimeline
from atom.mono.runtime.uncached import uncached_bytes
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# token parallel from this many rows (and a multiple of the TP size), by TP
# size: the sublayers' all-reduces become reduce-scatters, the per-token stages
# run on each rank's own tokens. Below it the gather's extra hop costs more than
# a rank's few tokens save; at TP 2 it never paid (12-48 rows: a K2 equal or
# 0.5-4.9 us slower), so a TP size not here runs without it
TOK_PAR_FROM = {4: 12}
# ... where the layer kernel serves the step (``_layer_capable``): below it the
# layer kernel's seams fold a rank's columns instead (``_shard``)
LAYER_TOK_PAR_FROM = {4: 24}
# the widths below which the layer kernel serves an index-only layer
# (``_fuses``): the layer build with the indexer uses scratch at TP4 from 24
# rows, at TP8 at 30 (25..29, no served width, never built: kept out); a TP
# size not here runs those layers as K1 then K2
INDEX_LAYER_BELOW = {4: 24, 8: 25}

# K2a's and K2b's mailbox tags start past every layer's K1 and K2a
POST_TAG0 = 64
MOE_TAG0 = 128
# the step's epoch word (an int64, padded to one 16-byte unit) before the queue words
EPOCH_WORD_BYTES = 16
# the step's row buffers, allocated for the widest step (``_allocate``,
# ``_place_index_outputs``) and narrowed to a step's rows (``_use_rows``): each
# with its row dim
_ROW_BUFFERS = (
    ("res_embed", 1), ("embedded", 0), ("pre_init", 1), ("res_attn", 1), ("res_ffn", 1),
    ("gates_attn", 1), ("gates_ffn", 1),
    ("ffn_normed", 1), ("ffn_out", 1), ("q", 1), ("normed", 1), ("qr", 1),
    ("qr_scale", 1), ("ring_rel", 0), ("key_meta", 0), ("ilog", 0), ("ibmax", 0),
    ("isel", 0), ("irow", 0), ("icand", 0), ("no_bound", 0), ("iq", 0),
    ("iq_scale", 0), ("iw", 0),
)  # fmt: skip
# the MoE's input: every rank's rows under DP attention (``LayerPostBuild.ep``)
_MOE_ROW_BUFFERS = ("ffn_normed",)


def _rows(x, dim, s):
    """``x`` (a tensor, or a tuple / dict of them) narrowed to s rows."""
    if isinstance(x, tuple):
        return tuple(_rows(v, dim, s) for v in x)
    if isinstance(x, dict):
        return {k: _rows(v, dim, s) for k, v in x.items()}
    return x.narrow(dim, 0, s)


@dataclass(frozen=True)
class RowsBuild:
    """The kernels of an S-row step: K1's build a layer and their launchers,
    and K2 (or, with ``check``, K2a and K2b apart)."""

    keys: list
    launchers: dict
    layer_post: object = None
    # ``layer_post``'s build key (``_lean_rows`` rebuilds it)
    post_key: object = None
    attn_post: object = None
    moe: object = None
    # K1 key -> the layer kernel of K1 and K2 in one launch (``_fuses``)
    layers: dict = field(default_factory=dict)
    # (pair) K1 key -> the pair kernel of two such layers
    # (``layer_pair``), where ``ATOM_MONO_PAIR`` asks for one (``layer_pair.pair_mode``)
    pairs: dict = field(default_factory=dict)
    # (run kernel) K1 key -> the run kernel of a run of such layers
    # (``layer_run``), where ``ATOM_MONO_RUN`` asks for one
    runs: dict = field(default_factory=dict)
    # a sharded step's column gathers (``_gather_cols``): ``col_gather.State``
    # -> its launcher
    gathers: dict = field(default_factory=dict)
    # the compressing layers' compressor (``MonoCompressor``): ratio -> its
    # launcher
    compress: dict = field(default_factory=dict)
    # the final norm and LM head (``MonoLmHead``), None where not served
    head: object = None
    # a sharded step's Engram gate (``MonoEngram``), None where not served
    engram: object = None
    # the draft's context KV write (``MonoDraftContext``), None where not served
    ctx: object = None


# the layers whose attention norm has other readers (the compressor, the indexer)
_FEEDS_INDEX = (AttentionMode.FULL, AttentionMode.REINDEX)


def _draft_of(drafter):
    """The draft model under a proposer (None: no drafter), unwrapped from the
    mono draft's routing wrapper (``MonoDraftModel.model``)."""
    if drafter is None:
        return None
    draft = drafter.model
    return getattr(draft, "model", draft)


def f32_bits(x: float) -> int:
    """``x`` rounded to fp32, as the int32 of its bits (a kernel's Int32 arg)."""
    return struct.unpack("<i", struct.pack("<f", x))[0]


def _filled_apart(vals, device: int):
    """(run kernel) ``vals`` (int64) in a new device tensor, copied and waited
    for on a thread and stream of its own: callable inside a thread_local
    capture without adding to it. -> (host, device) tensors."""
    out = {}

    def work():
        try:
            torch.cuda.set_device(device)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                host = torch.tensor(vals, dtype=torch.int64).pin_memory()
                dev = torch.empty(len(vals), dtype=torch.int64, device=torch.device("cuda", device))
                dev.copy_(host, non_blocking=True)
            stream.synchronize()
            out["got"] = (host, dev)
        except BaseException as e:  # noqa: BLE001 -- re-raised on the caller's thread
            out["err"] = e

    th = threading.Thread(target=work, name="v41-run-table")
    th.start()
    th.join()
    if "err" in out:
        raise out["err"]
    return out["got"]


def _staging_in_line(metadata):
    """The step's Engram staging to run on the step's own stream (its lookup
    in line), None to keep its side stream. The forward joins the side stream
    before its first layer (the all-gather spins on CUs: beside a kernel that
    needs every CU resident it deadlocks), so the side stream overlaps
    nothing, while its fork and join, captured, are 2-parent graph nodes a
    replay pays as idle before each. In line,
    the fork, the join and the ``done`` records are same-stream no-ops. Not
    where the step was staged outside the forward (the plugin's metadata
    builder) or where taking a layer's rows gathers on the main stream itself
    (``fallback_done``)."""
    rows = getattr(metadata, "engram_embeddings", None)
    staging = getattr(rows, "staging", None)
    if (
        staging is None
        or getattr(metadata, "staged_outside_forward", False)
        or staging.fallback_done
        or not isinstance(getattr(staging, "stream", None), torch.cuda.Stream)
    ):
        return None
    return staging


class V41MonoDecodeRunner:
    """Binds every rank's weights and buffers, or none: a rank that cannot is a
    refusal every rank agrees on (``group_agree``) before anything collective.

    A step is up to ``max_rows`` rows, whatever requests they belong to (a
    row's window comes from its own batch id, ``write_step_meta``); the buffers
    hold the widest and every width has its own kernel builds. The rows, the
    routing, the K1 builds and where RoPE positions come from are the target's
    here; ``draft_runner.DraftMonoRunner`` sets its own."""

    # the draft runner's: a full step beginning every step (``epoch_begin``)
    always_full_begin = False

    max_rows = MAX_ROWS
    experts, topk = EXPERTS, TOPK
    ug_groups = 2
    # the compressing layers' compressor in one launch (``compress_launch``);
    # None: the original compressor
    compressor = None
    # the final norm and LM head in one launch (``head_launch``); None: the
    # original ``compute_logits``
    lm_head = None
    # a sharded step's Engram gate in one launch (``engram_launch``); None:
    # the column gather and the original
    engram = None
    # the draft's context KV write after the step in one launch
    # (``draft_ctx_launch``); None: the draft's ``write_context_kv``
    draft_ctx = None
    # the launches a rank binds or leaves to the original path on its own
    # (None: not served): served on every rank of the group or on none
    optional_launches = ("lm_head", "engram", "draft_ctx")
    # the embedding read from every rank's vocab shard (``_bind_embedding``);
    # None: the module's own
    embed_peer = None
    # the ModelRunner's forward output buffer [tokens, hidden]: the step's
    # collapsed hidden state is gathered into it (``hidden_out``), so the
    # runner's copy of the forward's output into it is a no-op; None: a
    # buffer of the runner's own
    outputs = None
    # below this many rows an ug task takes one FP4 group, not ug_groups: a
    # small step's tasks end closer together. The outputs are the same to the
    # bit either way
    fine_ug_below = 4

    # ... and past one token tile up to this many, where every token's x is
    # resident (``moe.ug_x_resident``: the tasks read it in place, more of
    # them balance better)
    fine_ug_tiled_max = 22

    # ... and wherever ``ug_groups`` would leave a slot fewer ug tasks than
    # this: an expert's slice on a rank narrows as TP grows while the CTAs
    # streaming it do not, so fewer, larger tasks leave CTAs idle and each
    # task bound by one CTA's streaming rate
    min_ug_parts = 8

    @classmethod
    def ug_groups_at(cls, rows: int, tok_par: bool, inter: int) -> int:
        """An ug task's FP4 groups for a MoE of ``rows`` rows (``tok_par``: a
        token-parallel step's) whose experts are ``inter`` columns a rank: one
        below ``fine_ug_below``, where every token's x is resident up to
        ``fine_ug_tiled_max``, and where ``ug_groups`` would give a slot fewer
        than ``min_ug_parts`` tasks; else ``ug_groups``."""
        fine = (
            rows < cls.fine_ug_below
            or (k2b.ug_x_resident(rows, tok_par) and rows <= cls.fine_ug_tiled_max)
            or inter // (UG_PART * cls.ug_groups) < cls.min_ug_parts
        )
        return 1 if fine else cls.ug_groups

    # the index plane is FP4: K1 quantizes the indexer query for, and K2a
    # scores with, ``index_score_fp4``
    index_fp4 = False
    index_layout = "fp8"

    def __init__(
        self, model, drafter, max_model_len: int, index_fp4: bool, outputs,
        sampling, draft_context: bool = True,
    ) -> None:  # fmt: skip
        # the forward's output buffer; the step's staged (temperatures, top-p)
        # the mono LM head draws by, None where the owner verifies otherwise
        # (no head bound); whether the owner routes the draft's context write
        # here (``write_context``; else no launch bound for it)
        self.outputs, self.sampling = outputs, sampling
        self.draft_context = draft_context
        # the indexer's widest bound: a ratio-1 layer sees every position. No
        # token's bound may exceed it (positions stay below max_model_len): its
        # selection would never reach a final level nor publish IRDY
        self.index_bound_max = ip.cdiv(max_model_len, ip.CHUNK) * ip.CHUNK
        self.index_fp4 = index_fp4
        self.index_layout = (
            get_current_atom_config().index_cache_layout if index_fp4 else "fp8"
        )
        self._bind_agreed(lambda: self._bind(model, drafter))

    def _bind_agreed(self, bind) -> None:
        """``bind()`` on every rank, then, once every rank agreed it did, the
        collective peer buffer (``peer_layout``: the MoE's rows, every rank's
        under DP attention)."""
        group = self._group().cpu_group
        bind_agreed(bind, group)
        # a launch one rank serves alone waits for ever on the others' pushes
        for name in self.optional_launches:
            if not group_agree(getattr(self, name) is not None, group):
                setattr(self, name, None)
        if self.lm_head is not None:
            # its own peer buffer (served only without DP attention: the
            # group is the TP group)
            self.lm_head.open_peers(group, self.rank)
        plain = plain_layout(self.max_rows, len(self.selecting), self.tp)
        self.peers, self._finalizer = owned_peer_buffer(
            self,
            peer_bytes(self.max_rows * self.dp, self.moe_dims),
            group,
            self.rank,
            self.moe_dims.ranks,
            self.scratch.device,
            plain_bytes=sum(n for _, n in plain.values()),
        )
        self._place_index_outputs(plain)
        self._bind_embedding(group)
        self._full = {name: getattr(self, name) for name, _ in _ROW_BUFFERS}
        # the step's beginning (``epoch_begin``): the epoch-tagged hand-offs
        # need no zeroing once a draft pass has run since the last step; a
        # debug build (its wait records zeroed each step) and the draft
        # runner (``always_full_begin``) take the full beginning every step
        self.mailboxes = EpochMailboxes(
            self.peers,
            self.scratch,
            *(buf for buf, _ in self._launch_scratch()),
            debug=self.debug,
            epoch_word=self.epoch_word,
            always_full=self.debug or self.always_full_begin,
        )
        self.widths = WidthBuilds(
            self._rows_build, self._row_kernels, group, "V4.1 mono",
            lean=self._lean_rows,
        )  # fmt: skip

    def _place_index_outputs(self, plain: dict) -> None:
        """Each selecting layer's pool rows (the attention's keys), the
        candidate blocks and the split rows' answers, in the peer buffer's
        never-zeroed part at one offset on every rank (``plain_layout``): a
        token-parallel step's indexer writes a row's into every rank's
        (``index_score._st_out``), a split row's answer goes to every rank's
        merge (``index_score._put_answer``)."""
        words = self.peers.plain.view(torch.int32)

        def region(name, *shape):
            off, n = plain[name]
            return words[off // 4 : (off + n) // 4].view(*shape)

        irow = region("irow", len(self.selecting), self.max_rows, ip.TOPK)
        # every layer sharing a selection shares its KV owner
        # (``index_score._selected_rows``)
        self.irow = dict(zip(self.selecting, irow))
        self.icand = region("icand", self.max_rows, ip.TOPK_BLOCKS)
        self.ixsel = region("ixsel", self.max_rows, self.tp, ip.TOPK, 2)

    def prepare(self, rows: int) -> bool:
        """Whether every rank holds the ``rows``-row kernels (``WidthBuilds``)."""
        return 0 < rows <= self.max_rows and self.widths.prepare(rows)

    def _bind(self, model, drafter) -> None:
        self.model = model
        self._bind_rank()
        self.layers = bind_model(model, self.dims, self.moe_tp)
        # without a drafter no layer taps its residual
        self.aux = (
            AuxTaps((), ())
            if drafter is None
            else bind_aux_taps(
                drafter,
                model.config.hidden_size,
                DeepseekV41DSpark._target_layer_input,
            )
        )
        taps = len(self.aux.buffers)
        # every layer's residual column-sharded (``_shard``), gathered by column
        # where a reader needs every column (``_gather_cols``: one
        # ``col_gather`` launch, the taps' planes included)
        self.shards = self.tp > 1 and self.dp == 1 and taps <= cg.MAX_TAPS
        for rope in (model.window_rope, model.global_rope):
            for table in (rope.cos_cache, rope.sin_cache):
                if table.dtype != torch.bfloat16 or table.shape[-1] != k1.HALF:
                    raise MonoUnsupported(
                        f"rope table {table.dtype} {tuple(table.shape)}"
                    )
        self.check = envs.ATOM_MONO_CHECK
        if self.check and self.dp > 1:
            # its K2b alone would wait for rows no K2a of this launch publishes
            raise MonoUnsupported("ATOM_MONO_CHECK under DP attention")
        # check mode compares K2a's and K2b's outputs: two launches
        self.split_k2 = self.check
        from_rows = LAYER_TOK_PAR_FROM if self._layer_capable() else TOK_PAR_FROM
        self.tok_par_from = from_rows.get(self.tp, self.max_rows + 1)
        # DP attention gathers no rows: a rank's step is its own
        if self.dp == 1 and not self._custom_gather_fits(
            (self.max_rows // self.tp, 1 + taps, k1.HIDDEN)
        ):
            logger.warning(
                "V4.1 mono: token parallel is off, its row gather has no custom "
                "all-gather (the NCCL one cannot be captured)"
            )
            self.tok_par_from = self.max_rows + 1
        self.debug = envs.ATOM_MONO_DEBUG
        tl_prefix = envs.ATOM_MONO_TIMELINE
        dev = torch.device("cuda", torch.cuda.current_device())
        # the layer kernel's (``_layer``) beside the two launches' of the layers
        # it does not serve
        post_points = (
            (("k2a", k2a.TL_POINTS), ("k2b", k2b.TL_POINTS))
            if self.check
            else (("k2", k2.TL_POINTS), ("layer", kl.TL_POINTS))
        )
        self.timelines = (
            {
                name: LayerTimeline(
                    f"{tl_prefix}_{name}", len(model.layers), points, self.rank, dev
                )
                for name, points in (("k1", k1.TL_POINTS), *post_points)
            }
            if tl_prefix
            else {}
        )
        self.timeline = bool(self.timelines)
        # the layers that select (FULL / REINDEX): each its selection buffer
        self.selecting = [
            spec.layer_id
            for spec, block in zip(model.topology, model.layers)
            if block.attn.indexer is not None
        ]
        if not self.check:
            # check mode runs the original compressor and head, the reference
            self.compressor = MonoCompressor.bind(
                model, self.max_rows, self.kv_fp8, dev
            )
            if self.dp == 1 and (verifiers := self._id_verifiers(dev)) is not None:
                # DP attention shards the head over the DP ranks instead; the
                # launch picks a verify's ids, none without a drafter
                self.lm_head = bind_lm_head(
                    model, self.tp, self.max_rows, verifiers, lambda: self.step_ids
                )
            self.engram = MonoEngram.bind(model, self.tp, self.max_rows, dev)
            if self.draft_context:
                self.draft_ctx = MonoDraftContext.bind(
                    _draft_of(drafter), self.tp, self.max_rows, dev
                )
        self._allocate(model.config)

    def _id_verifiers(self, dev):
        """The head's verifies on its ids (``head_launch``), its draws by the
        step's staged sampling; None without a drafter or that sampling (its
        owner verifies otherwise)."""
        spec = get_current_atom_config().speculative_config
        if spec is None or self.sampling is None:
            return None
        temps, top_ps = self.sampling
        return IdVerifiers.from_spec(spec, dev, lambda: temps, lambda: top_ps)

    def index_query(self):
        """K1's indexer query as the original scorer takes it quantized: e4m3,
        or the FP4 (values, packed scales) of one-row sequences."""
        if not self.index_fp4:
            return self.iq
        rows = self.iq.shape[0]
        scales = self.iq_scale.view(rows, 1, 1, ip.DIM // 32, 16, QS_M_TILES)
        if self.index_layout == FP4_MFMA:
            # Debug/reference scorer only: K1 and K2 keep their query ABI.
            natural = scales[:, 0, 0, :, :, : ip.HEADS // 16].permute(0, 3, 2, 1)
            scales = natural.reshape(rows, 1, 32, 2, 2).permute(0, 4, 2, 3, 1)
            scales = scales.contiguous().view(rows, 1, 2, 32, ip.HEADS // 16)
        return (
            self.iq.view(rows, 1, *self.iq.shape[1:]),
            scales,
        )

    @staticmethod
    def _group():
        """The ranks one mono step spans: the TP group, or under DP attention
        (``config_refusal`` let DP > 1 through only there, at TP 1) the DP
        group, which the MoE is expert parallel over."""
        tp = get_tensor_model_parallel_world_size()
        dp = get_dp_group()
        return dp if tp == 1 and dp.world_size > 1 else get_tp_group()

    @property
    def kv_fp8(self) -> bool:
        """The pool's rows are MXFP8 (``V41PoolGeometry.kv_dtype`` fp8): the
        attention dequantizes them, K1 writes its window row so."""
        return get_current_atom_config().kv_cache_dtype == "fp8"

    def _bind_rank(self) -> None:
        """This rank, the TP size every build is for and DP attention's ranks,
        the MoE expert parallel or TP over them (``config_refusal`` refused the
        ones the kernels are not built for)."""
        self.tp = get_tensor_model_parallel_world_size()
        group = self._group()
        self.dp = group.world_size // self.tp
        ep = get_current_atom_config().enable_expert_parallel
        # ATOM's EP group: the experts over every TP and DP rank
        self.ep = self.tp * self.dp if ep else 1
        self.moe_tp = 1 if ep else self.dp
        self.rank = (
            group.rank_in_group if self.dp > 1 else get_tensor_model_parallel_rank()
        )
        # attention's and the shared expert's widths, and under expert parallel
        # the routed experts a rank holds; the MoE's (its widths, its ranks)
        self.dims = Dims(self.tp, self.ep)
        self.moe_dims = Dims(self.tp * self.moe_tp, self.ep, self.moe_tp > 1)
        # the MoE runs every rank's rows: its widest build bounds a rank's step
        self.max_rows = min(self.max_rows, MAX_MOE_ROWS // self.dp)

    def _custom_gather_fits(self, widest: tuple[int, ...]) -> bool:
        """Whether an all-gather of a rank's bf16 ``widest`` (the widest step's)
        takes aiter's custom one (``ATOM_USE_CUSTOM_ALL_GATHER``, the group's
        ``ca_comm`` and its size limit): the same on every rank."""
        if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
            return False
        group = get_tp_group()
        ca = getattr(group.device_communicator, "ca_comm", None)
        if ca is None or ca.disabled:
            return False
        return bool(
            ca.should_custom_ag(
                torch.empty(widest, dtype=torch.bfloat16, device="meta")
            )
        )

    def _tok_par(self, s: int) -> bool:
        """Whether an s-row step runs token parallel (``tok_par_from``); never
        in check mode, which compares each kernel's whole rows, nor with K2
        split, whose separate K2b would read rows K2a's owner never gathered.
        DP attention owns its rows already (``LayerPostBuild.dp_ranks``)."""
        return (
            self.dp == 1
            and not self.check
            and not self.split_k2
            and s >= self.tok_par_from
            and s % self.tp == 0
        )

    def _shard(self, s: int) -> bool:
        """Whether an s-row step's seams fold a rank's columns (``shards``):
        not token parallel (rows split instead), nor in check mode, which
        compares each kernel's whole outputs, nor at a TP size whose slices a
        rank are not whole gate blocks (``attn_pre.seam_shardable``)."""
        return (
            self.shards
            and k1.seam_shardable(self.tp)
            and not self.split_k2
            and not self._tok_par(s)
        )

    def _own_rows(self, s: int) -> tuple[int, int]:
        """(first row, rows) of an s-row step this rank's indexer selects
        (``index_score._own_rows``)."""
        return own_rows(s, self.tp, self.rank) if self._tok_par(s) else (0, s)

    def _split_over(self, s: int) -> tuple[int, int]:
        """(ranks, this rank) an s-row step's FULL rows may split over
        (``index_plan.row_shares``): the TP group, or none where a rank selects
        its own rows (token parallel) or ranks are DP attention's."""
        if self._tok_par(s) or self.dp > 1:
            return 1, 0
        return self.tp, self.rank

    @staticmethod
    def _lean_rows(built: RowsBuild, spilled: list) -> RowsBuild | None:
        """``WidthBuilds``' leaner build of a width whose spilled kernels are
        layer kernels or K2: without the layer kernels, those layers run K1
        and K2 apart (their builds are the width's anyway, ``_layer``); K2
        rebuilt with its split rows' answers staged (``staged_answer``). None
        where another kernel spilled, or K2 did in a build that splits no row
        (staging would rebuild the same kernel): the width is refused."""
        post, key = built.layer_post, built.post_key
        if post in spilled:
            if key is None or key.staged_answer or not key.index_split:
                return None
            key = replace(key, staged_answer=True)
            post = k2.build_layer_post(key)
        rest = [f for f in spilled if f is not built.layer_post]
        # (pair) a spilled pair kernel dropped: its layers run apart
        pairs = {k: f for k, f in built.pairs.items() if f not in rest}
        rest = [f for f in rest if f not in built.pairs.values()]
        # (run kernel) a spilled run kernel dropped: its layers run apart
        runs = {k: f for k, f in built.runs.items() if f not in rest}
        rest = [f for f in rest if f not in built.runs.values()]
        layers = {k: f for k, f in built.layers.items() if f not in rest}
        if len(built.layers) - len(layers) != len(rest):
            return None
        return replace(
            built, layers=layers, pairs=pairs, runs=runs, layer_post=post, post_key=key
        )

    def _layer_blocks(self) -> list:
        """(spec, block) of every layer this runner runs, in order (the
        draft's: its stages, ``DraftMonoRunner``)."""
        return list(zip(self.model.topology, self.model.layers))

    def _layer_keys(self, s: int, numa: bool = True) -> list:
        """K1's build of every layer for an s-row step (``numa`` False: before
        the step's dual table, which these builds' layouts decide)."""
        # layer 0 starts from the embeddings and an Engram layer settles first:
        # neither seam has an owed post to fold
        return [
            k1.AttnPreBuild(
                tokens=s,
                fold=spec.layer_id != 0 and block.engram is None,
                embed_in=spec.layer_id == 0,
                feeds_index=spec.mode in _FEEDS_INDEX,
                index_fp4=self.index_fp4,
                aux=spec.layer_id in self.aux.layer_ids,
                tp=self.tp,
                timeline=self.timeline,
                diag_off=self.diag_off,
                tok_par=self._tok_par(s),
                kv_fp8=self.kv_fp8,
                numa=self._numa(s) if numa else "",
                shard=self._shard(s),
            )
            for spec, block in zip(self.model.topology, self.model.layers)
        ]

    def _numa(self, s: int) -> str:
        """The s-row step's regions kept in two copies, one on each I/O die
        (``numa_plan``), in the dual area from ``numa_start``: none without
        one, nor token parallel (its seam regions per owner), DP attention
        (its routes in peer regions) or K2 split (K2b's scratch its own
        launch's)."""
        if self.numa_start < 0 or self._tok_par(s) or self.dp > 1 or self.split_k2:
            return ""
        return numa_table(self._numa_layouts(s), self.numa_start)[0]

    def _numa_layouts(self, s: int) -> tuple:
        """The s-row step's scratch layouts: K1's and K2's, and every layer
        kernel build of the step's (``_fused_builds``: an indexing or
        compressing one places its flags past a plain one's)."""
        fused = self._fused_builds(s, numa=False).values()
        return (
            k1.scratch_layout(s),
            k2.scratch_layout(self._post_key(s, False)),
            *(layout for build in fused for layout in kl.scratch_layouts(build)),
        )

    def _fused_builds(self, s: int, numa: bool = True) -> dict:
        """The layer kernel builds an s-row step runs, keyed (K1 key, the
        compressor's ratio): one a distinct layer that fuses (``_fuses``).
        ``numa`` False: before the step's dual table."""
        if not self._layer_serves(s):
            return {}
        return {
            (key, ratio): self._layer_build(s, key, numa, ratio)
            for (spec, block), key in zip(
                self._layer_blocks(), self._layer_keys(s, numa)
            )
            if self._fuses(s, spec, key, block)
            for ratio in (self._compress_ratio(spec, block),)
        }

    def _pair_mode(self) -> int:
        """``ATOM_MONO_PAIR`` for the target's runner (0: off;
        the draft's runner never pairs)."""
        if type(self) is not V41MonoDecodeRunner or self._run_mode():
            return 0
        return lp.pair_mode()

    def _run_mode(self) -> int:
        """(run kernel) ``ATOM_MONO_RUN`` for the target's runner (0: off; the
        draft's runner never runs one)."""
        if type(self) is not V41MonoDecodeRunner:
            return 0
        return lr.run_mode()

    def _run_builds(self, s: int, numa: bool = True) -> dict:
        """(run kernel) K1 key -> the run build of each ordinary layer build
        of an s-row step (``layer_run.runnable``)."""
        mode = self._run_mode()
        if not mode or s > lr.RUN_MAX_TOKENS:
            return {}
        return {
            key: lr.RunBuild(build, mode == 2, mode == 3)
            for (key, ratio), build in self._fused_builds(s, numa).items()
            if not ratio and lr.runnable(build)
        }

    def _pair_builds(self, s: int, numa: bool = True) -> dict:
        """(pair) K1 key -> the pair build of each pairable
        ordinary layer build of an s-row step (``layer_pair.pairable``)."""
        mode = self._pair_mode()
        if not mode:
            return {}
        if s > lp.PAIR_MAX_TOKENS:
            return {}
        return {
            key: lp.PairBuild(build, mode)
            for (key, ratio), build in self._fused_builds(s, numa).items()
            if not ratio and lp.pairable(build)
        }

    def _layer_capable(self) -> bool:
        """Whether the layer kernel (K1 and K2 in one launch) covers this
        runner's layers at some width: not under the check's split K2 or DP
        attention, where every layer runs K1 then K2."""
        return not (self.split_k2 or self.dp > 1)

    def _layer_serves(self, s: int) -> bool:
        """Whether an s-row step's layers may run as one launch (``_fuses``),
        token parallel or not."""
        return self._layer_capable()

    def _layer_build(self, s: int, pre, numa: bool = True, ratio: int = 0):
        """The layer kernel over K1 build ``pre`` at s rows: K2's build with
        the indexer where K1 feeds one, else without; ``ratio``: a compressing
        layer's compressor in the launch (``LayerBuild.compress_ratio``)."""
        post = self._post_key(s, numa)
        if not pre.feeds_index:
            post = replace(post, index_bound_max=0, index_fp4=False, index_mfma=False)
        if not ratio:
            return kl.LayerBuild(
                pre=pre, post=post, engram_host=self._hosts_engram(s, pre)
            )
        eps, k_eps = self.compressor.eps[ratio]
        return kl.LayerBuild(
            pre=pre, post=post, compress_ratio=ratio, compress_eps=eps,
            compress_k_eps=k_eps,
        )  # fmt: skip

    def _candidate_table(self, spec, step, cache) -> None:
        """After a candidate-producing layer (its K2a's indexer wrote
        ``icand``): the REINDEX layers' block table over its candidates."""
        if not spec.produces_candidates:
            return
        self.candidate_table = candidate_block_table(
            self.icand,
            step.block_tables,
            step.batch_ids,
            cache.geometry.index_blocks_per_page(spec.ratio),
            step.visible[spec.ratio],
            rows_per_block=ip.BLOCK_ROWS,
        )

    def _compress_ratio(self, spec, block) -> int:
        """The ratio a layer's compressor runs at inside its layer kernel, 0
        for a layer that compresses nothing (or one the launch does not
        serve: ``MonoCompressor.ratio_of``)."""
        if not spec.ratio or block.attn.compressor is None or self.compressor is None:
            return 0
        return self.compressor.ratio_of(spec.layer_id)

    def _fuses(self, s: int, spec, key, block) -> bool:
        """Whether a layer runs K1 and K2 as one launch (``kernels.layer``):
        nothing runs between its K1 and its K2 (its K1 folds the owed post or,
        layer 0 and an Engram layer, takes a settled residual) but a
        compressor the launch carries (``_compress_ratio``: a step it is not
        built for runs them apart, ``_layer``). A layer with an indexer runs
        its scorer in the launch (a candidate producer's table follows it,
        ``_candidate_table``); ``feeds_index`` without an indexer is a
        compressor's alone."""
        compresses = bool(spec.ratio) and block.attn.compressor is not None
        indexes = block.attn.indexer is not None
        carried = (
            self._compress_ratio(spec, block) > 0
            # the compressor reads every row's normed
            and not self._tok_par(s)
        )
        return (
            self._layer_serves(s)
            and (not compresses or carried)
            and key.feeds_index == indexes
            and (not indexes or s < INDEX_LAYER_BELOW.get(self.tp, 0))
        )

    def _post_key(self, s: int, numa: bool = True) -> k2.LayerPostBuild:
        return k2.LayerPostBuild(
            tokens=s,
            index_bound_max=self.index_bound_max,
            index_fp4=self.index_fp4,
            index_mfma=self.index_layout == FP4_MFMA,
            experts=self.experts,
            topk=self.topk,
            tp=self.tp,
            timeline=self.timeline,
            diag_off=self.diag_off,
            # the MoE's rows decide it (every rank's under DP attention)
            ug_groups=self.ug_groups_at(
                s * self.dp, self._tok_par(s), self.moe_dims.inter_real
            ),
            tok_par=self._tok_par(s),
            ep=self.ep,
            moe_tp=self.moe_tp,
            kv_fp8=self.kv_fp8,
            numa=self._numa(s) if numa else "",
            shard=self._shard(s),
        )

    def _rows_build(self, s: int) -> RowsBuild:
        """The s-row step's kernels."""
        keys = self._layer_keys(s)
        launchers = {key: k1.build_attn_pre(key) for key in set(keys)}
        post_key = self._post_key(s)
        gathers = {
            state: cg.build_col_gather(key) for state, key in self._gathers(s).items()
        }
        if self.split_k2:
            key_a, key_b = post_key.halves()
            return RowsBuild(
                keys,
                launchers,
                attn_post=k2a.build_attn_post(key_a),
                moe=k2b.build_moe(key_b),
                gathers=gathers,
            )
        # a compressing layer's build carries its compressor: keyed apart from
        # an index-only layer's of the same K1 key
        layers = {
            key: kl.build_layer(build) for key, build in self._fused_builds(s).items()
        }
        pairs = {
            key: lp.build_layer_pair(build)
            for key, build in self._pair_builds(s).items()
        }
        runs = {
            key: lr.build_layer_run(build) for key, build in self._run_builds(s).items()
        }
        compress = (
            {}
            if self.compressor is None
            else {
                ratio: cp.build_compress(key)
                for ratio, key in self.compressor.builds(s).items()
            }
        )
        return RowsBuild(
            keys,
            launchers,
            layer_post=k2.build_layer_post(post_key),
            post_key=post_key,
            layers=layers,
            pairs=pairs,
            runs=runs,
            gathers=gathers,
            compress=compress,
            head=None if self.lm_head is None else self.lm_head.build(s),
            engram=self._engram_build(s),
            ctx=None if self.draft_ctx is None else self.draft_ctx.build(s),
        )

    def _bind_embedding(self, group) -> None:
        """The model's embedding as one gather from every rank's vocab shard
        (``PeerEmbedding``) where it is vocab-parallel over this runner's
        group (TP > 1, no DP attention) with bf16 rows; else, or where the
        ranks refuse the exchange, the module's own. Collective: every rank
        reaches it alike."""
        embed = getattr(self.model, "embed", None)
        if not (
            isinstance(embed, VocabParallelEmbedding)
            and embed.tp_size == self.tp > 1
            and self.dp == 1
            and embed.weight.dtype == torch.bfloat16
        ):
            return
        try:
            peer = shared_peer_embedding(embed, group)
        except MonoUnsupported as why:
            # every rank sees the same refusal (``PeerView`` gathers each
            # rank's): every rank keeps the module's embedding, mono serves on
            logger.warning("V4.1 mono: peer embedding off, the module's: %s", why)
            return
        self.embed_peer = peer

    def _embed(self, embed, ids: torch.Tensor) -> torch.Tensor:
        """``embed(ids)`` [tokens, hidden]: the peer gather into ``embedded``
        where bound (``_bind_embedding``), the same rows."""
        if self.embed_peer is None:
            return embed(ids)
        return self.embed_peer(ids, self.embedded[: ids.numel()])

    def _launch_scratch(self) -> list:
        """The bound launches' own flags, zeroed with the step's others:
        (buffer, its bytes at an s-row step) of the compressor and the draft
        context write."""
        out = []
        if self.compressor is not None:
            out.append((self.compressor.scratch, cp.scratch_bytes))
        if self.draft_ctx is not None:
            out.append((self.draft_ctx.scratch, partial(dc.scratch_bytes, tp=self.tp)))
        return out

    def write_context(self, taps, positions, cache, step) -> bool:
        """The draft's context KV of the step this runner just served, in
        one launch (``draft_ctx_launch``), True if it ran: its mailboxes are
        the step's (zeroed at its begin), so only right after that step."""
        if self.draft_ctx is None:
            return False
        build = self.widths[positions.numel()]
        return self.draft_ctx.launch(
            build.ctx, taps, positions, cache, step, self._peer_args()
        )

    def _gathers(self, s: int) -> dict:
        """An s-row step's column gathers (``_gather_cols``), none unsharded:
        the settle before an Engram layer, if any, and the step's end."""
        if not self._shard(s):
            return {}
        out = {
            cg.State.COLLAPSE: cg.ColGatherBuild(
                rows=s, taps=len(self.aux.buffers), state=cg.State.COLLAPSE, tp=self.tp
            )
        }
        if any(block.engram is not None for block in self.model.layers):
            out[cg.State.SETTLE] = cg.ColGatherBuild(
                rows=s, taps=0, state=cg.State.SETTLE, tp=self.tp
            )
        return out

    @staticmethod
    def _row_kernels(build: RowsBuild) -> list:
        """Every kernel of ``build``, each with its ABI."""
        kernels = [(launcher, k1.ABI) for launcher in build.launchers.values()]
        kernels += [(launcher, kl.ABI) for launcher in build.layers.values()]
        kernels += [(launcher, lp.ABI) for launcher in build.pairs.values()]
        kernels += [(launcher, lr.ABI) for launcher in build.runs.values()]
        kernels += [(launcher, cg.ABI) for launcher in build.gathers.values()]
        kernels += [(launcher, cp.ABI) for launcher in build.compress.values()]
        if build.head is not None:
            kernels.append((build.head, lk.ABI))
        if build.engram is not None:
            kernels += [(launcher, eg.ABI) for launcher in build.engram.values()]
        if build.ctx is not None:
            kernels.append((build.ctx, dc.ABI))
        if build.layer_post is not None:
            return kernels + [(build.layer_post, k2.ABI)]
        return kernels + [(build.attn_post, k2a.ABI), (build.moe, k2b.ABI)]

    def _use_rows(self, s: int) -> None:
        """This step is s rows: its kernels, and every row buffer narrowed."""
        if not 0 < s <= self.max_rows:
            raise ValueError(f"{s} rows is not a step of this runner")
        self.build = self.widths[s]
        for name, dim in _ROW_BUFFERS:
            rows = s * self.dp if name in _MOE_ROW_BUFFERS else s
            setattr(self, name, _rows(self._full[name], dim, rows))

    def _allocate(self, config) -> None:
        dev = torch.device("cuda", torch.cuda.current_device())
        s, hc, h = self.max_rows, config.hc_mult, config.hidden_size
        bf = torch.bfloat16
        # explicit dtypes: the runner is built inside a warmup forward, whose
        # default dtype is the model's (bf16)
        f32 = torch.float32

        def plain(*shape, dtype):
            return torch.empty(*shape, device=dev, dtype=dtype)

        def hand(*shape, dtype):
            """K1's output for K2: uncached under the layer kernel, whose readers
            on other XCDs must not hit a line an earlier step left in their L2."""
            if not self._layer_capable():
                return plain(*shape, dtype=dtype)
            n = math.prod(shape) * dtype.itemsize
            return uncached_bytes(-(-n // 16) * 16, dev)[:n].view(dtype).view(*shape)

        def gates(alloc):
            return (
                alloc(1, s, hc, dtype=f32),  # pre
                alloc(1, s, hc, dtype=f32),  # post
                alloc(1, s, hc, hc, dtype=f32),  # comb
            )

        # each seam's residual and gates: the attention seam reads the FFN seam's
        # and the FFN seam the attention seam's, so one of each suffices
        self.res_attn = hand(1, s, hc, h, dtype=bf)
        self.res_ffn = plain(1, s, hc, h, dtype=bf)
        # (pair) the pair's own hand-off of its body a's FFN
        # seam residual and gates to body b's attention seam (read on other
        # XCDs in the launch): uncached, the pair's alone -- every single
        # layer keeps the cached ``res_ffn`` / ``gates_ffn``
        pairs = self._pair_mode() > 0 or self._run_mode() > 0
        self.res_pair = hand(1, s, hc, h, dtype=bf) if pairs else None
        # layer 0's residual, its streams written from the embedding by its
        # seam (``AttnPreBuild.embed_in``)
        self.res_embed = hand(1, s, hc, h, dtype=bf)
        # the step's embedding rows (``PeerEmbedding``: one gather from every
        # rank's vocab shard), layer 0's input
        self.embedded = plain(s, h, dtype=bf)
        self.gates_attn, self.gates_ffn = gates(hand), gates(plain)
        self.gates_pair = gates(hand) if pairs else None
        # the layer kernel's window rows of this step (``kernels.layer``), laid
        # out as the pool's planes
        self.kvrow = (
            hand(kvrow_bytes(s, self.kv_fp8), dtype=torch.uint8)
            if self._layer_capable()
            else None
        )
        # the embedding's pre gates, constant: the first layer reads them
        self.pre_init = initial_pre_mix((1, s), hc, dev)
        self.ffn_normed = torch.empty(1, s * self.dp, h, device=dev, dtype=bf)
        # the FFN output, reduced: the next layer's owed post
        self.ffn_out = torch.empty(1, s, h, device=dev, dtype=bf)
        # the step's collapsed hidden state (``_gather_cols``, ``_gather_rows``)
        out = self.outputs
        self.hidden_out = (
            out[:s]
            if out is not None and out.dtype == bf and out.shape[1:] == (h,)
            else torch.empty(s, h, device=dev, dtype=bf)
        )
        self.q = torch.empty(1, s, self.dims.heads, k1.HEAD_DIM, device=dev, dtype=bf)
        self.normed = torch.empty(1, s, h, device=dev, dtype=bf)
        self.qr = torch.empty(1, s, k1.Q_RANK, device=dev, dtype=torch.float8_e4m3fn)
        self.qr_scale = torch.empty(
            1, s, k1.Q_RANK // 32, device=dev, dtype=torch.float8_e8m0fnu
        )
        self.ring_rel = torch.empty(s, device=dev, dtype=torch.int32)
        # each token's window metadata (``attention._keys``)
        self.key_meta = torch.empty(s, KMETA, device=dev, dtype=torch.int32)
        # the indexer: its logits and block maxima, each selecting layer's
        # selection, a zero bound for a layer without keys from a selection,
        # and each selection's bound this step (its pool rows and the candidate
        # blocks live in the peer buffer: ``_place_index_outputs``)
        width = self.index_bound_max
        self.ilog = torch.empty(s, width, device=dev, dtype=f32)
        self.ibmax = torch.empty(s, ip.blocks(width), device=dev, dtype=f32)
        # each selecting layer's lists' anchors in front of its selection
        # (``index_plan.anchor_bytes``, read and written by its K2a's selects
        # step after step; zero: a first step's windows miss)
        assert s <= ip.ANCHOR_ROWS, (s, ip.ANCHOR_ROWS)
        anchor_words = ip.anchor_bytes(width) // 4

        def selection():
            words = torch.zeros(
                anchor_words + s * ip.TOPK, device=dev, dtype=torch.int32
            )
            return words[anchor_words:].view(s, ip.TOPK)

        self.isel = {layer: selection() for layer in self.selecting}
        self.no_bound = torch.zeros(s, device=dev, dtype=torch.int32)
        # the FP4 score stage's FULL-layer tasks a ratio, then each row's share
        # (``index_plan.fill_shares``), each step's written by ``write_step_meta``
        self.walk_plans = (
            {
                ratio: torch.zeros(BLOCKS + s, 4, device=dev, dtype=torch.int32)
                for ratio in sorted(
                    {
                        spec.ratio
                        for spec, block in zip(self.model.topology, self.model.layers)
                        if block.attn.indexer is not None
                        and spec.candidate_source is None
                    }
                )
            }
            if self.index_fp4
            else {}
        )
        # ``index_score._sel_rows`` reads a candidate producer's bound off its
        # own plan row: whole only at a ratio that never splits (the FP4 path's
        # target; the draft model has no topology)
        if self.index_fp4:
            assert not any(
                spec.produces_candidates and ip.splits(self.tp, spec.ratio)
                for spec in getattr(self.model, "topology", ())
            ), "V4.1 mono: a candidate producer's ratio splits its plan rows"
        self.bounds = {}
        self.candidate_table = None
        # the indexer's query, its scales and the scaled head weights: e4m3 and
        # a scale a head, or E2M1 and E8M0s in the FP4 scorer's packed order
        # (its padding never written: zero, as ``pack_q_scales`` pads)
        # K1's hand-off to K2a's scorer: under the layer kernel within one launch
        # (``hand``)
        heads, dim = config.index_n_heads, config.index_head_dim
        if self.index_fp4:
            self.iq = hand(s, heads, dim // 2, dtype=torch.uint8)
            # the FP4 scales' padding is never written: zero
            self.iq_scale = hand(s, QS_BYTES, dtype=torch.uint8).zero_()
        else:
            self.iq = hand(s, heads, dim, dtype=torch.float8_e4m3fn)
            self.iq_scale = hand(s, heads, dtype=f32)
        self.iw = hand(s, heads, dtype=f32)
        self.diag_off = -1
        self.numa_start, self.numa_word = -1, 0
        # every width's: a narrower step's builds may place their regions
        # further out than the widest's (token parallel from some width on)
        nbytes = max(self._scratch_need(w) for w in range(1, s + 1))
        # a debug build's wait records: past every kernel's regions, cleared
        # with them every step
        if self.debug:
            self.diag_off = nbytes
            nbytes += DIAG_BYTES
        self._allocate_scratch(nbytes, dev)
        # each width's zeroing (``_zero_extents``), once the peer buffer exists
        self._extents = {}

    def _scratch_need(self, s: int) -> int:
        """The step scratch's regions an s-row step's kernels use."""
        post_key = self._post_key(s)
        key_a, key_b = post_key.halves()
        return max(
            k1.scratch_bytes(s),
            k2a.scratch_bytes(key_a),
            k2b.scratch_bytes(key_b),
            k2.scratch_bytes(post_key),
            # the layer kernel's hand-over flags past both, an indexing or
            # compressing build's past a plain one's
            *(kl.scratch_bytes(b) for b in self._fused_builds(s, numa=False).values()),
            # (pair) a pair build's PND past them
            *(lp.pair_scratch_bytes(b) for b in self._pair_builds(s, numa=False).values()),
            # (run kernel) a run build's PND past them
            *(lr.run_scratch_bytes(b) for b in self._run_builds(s, numa=False).values()),
        )

    def _zero_extents(self, s: int):
        """What an s-row step zeroes (``StepMailboxes.begin_step``): the peer
        data and the scratch ranges its kernels' regions lie in -- every one a
        pair of this step may land in, none past them (a wider step zeroes its
        own wider ranges before it reads them)."""
        got = self._extents.get(s)
        if got is not None:
            return got
        align = lambda n: -(-n // ZERO_ALIGN) * ZERO_ALIGN
        base = self.scratch.data_ptr()
        # the debug words lie past every width's regions in the same scratch,
        # the queue words right before them: one range through them all
        # (``step_begin.MAX_SCRATCH`` ranges a step)
        end = (
            self.diag_off + DIAG_BYTES if self.diag_off >= 0 else self._scratch_need(s)
        )
        ranges = [(self.queue_words.data_ptr(), k2b.QUEUE_BYTES + align(end))]
        if self.numa_start >= 0 and self._numa(s):
            end = numa_table(self._numa_layouts(s), self.numa_start)[1]
            ranges.append((base + self.numa_start, align(end - self.numa_start)))
        for buf, nbytes in self._launch_scratch():
            ranges.append((buf.data_ptr(), align(nbytes(s))))
        peer = max(peer_bytes(s * self.dp, self.moe_dims), peer_bytes(s, Dims(self.tp)))
        got = self._extents[s] = (min(peer, self.peers.bytes.nbytes), ranges)
        return got

    def _allocate_scratch(self, nbytes: int, dev) -> None:
        """The step scratch: ``nbytes`` of regions, then the dual area every
        width's two-copy regions share (``_numa``), from a page boundary so that
        one calibration of it (``numa.scratch_word``) holds for every width.
        The word is each rank's own (its pages' homes), a kernel argument
        (``_tail_args``), not a build's constant: every rank and every start
        runs the same binaries. Right before the regions, the
        launches' queue words (``k2b.QUEUES``): outside every build's layout,
        which the step's layer builds lay out apart."""
        qbytes = k2b.QUEUE_BYTES
        # the step's epoch word right before the queue words (``epoch_begin``
        # stores each new epoch there; every launch reads it, ``epoch`` arg)
        eb = EPOCH_WORD_BYTES
        start = -(-nbytes // NUMA_PAGE) * NUMA_PAGE
        self.numa_start = start
        widths = [w for w in range(1, self.max_rows + 1) if self._numa(w)]
        area = max(
            (numa_table(self._numa_layouts(w), start)[1] - start for w in widths),
            default=0,
        )
        if not area:
            self.numa_start = -1
            raw = uncached_bytes(eb + qbytes + nbytes, dev)
            self.epoch_word = raw[:eb]
            self.queue_words, self.scratch = raw[eb : eb + qbytes], raw[eb + qbytes :]
            return
        # a page more: the regions from the first page boundary past the
        # queue words
        raw = uncached_bytes(eb + qbytes + start + area + NUMA_PAGE, dev)
        shift = eb + qbytes + -(raw.data_ptr() + eb + qbytes) % NUMA_PAGE
        self.epoch_word = raw[shift - qbytes - eb : shift - qbytes]
        self.queue_words = raw[shift - qbytes : shift]
        self.scratch = raw[shift : shift + start + area]
        self.numa_word = scratch_word(self.scratch[start:], area, "V4.1 mono")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``_forward``, the Engram staging (if any) on the step's own stream
        meanwhile (``_staging_in_line``)."""
        staging = _staging_in_line(get_forward_context().attn_metadata)
        if staging is None:
            return self._forward(input_ids, positions)
        side, staging.stream = staging.stream, torch.cuda.current_stream()
        try:
            return self._forward(input_ids, positions)
        finally:
            staging.stream = side

    def _forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        model = self.model
        metadata = get_forward_context().attn_metadata
        cache, step = metadata.cache, metadata.step
        if cache.packed:
            raise MonoUnsupported("packed KV cache")
        rows = input_ids.numel()
        self._use_rows(rows)
        # the drafts the head's verdict checks (``head``)
        self.step_ids = input_ids
        self.mailboxes.begin_step(
            self._zero_extents(rows), (self.queue_words.data_ptr(), k2b.QUEUE_BYTES)
        )
        hidden = self._embed(model.embed, input_ids).view(1, rows, -1)
        model.begin_forward(hidden, None)
        # Engram staging's side-stream all-gather spins on CUs until every
        # rank joins it: beside a kernel that needs every CU resident it
        # deadlocks, so it ends before the first one
        rows_staged = metadata.engram_embeddings
        engram_joined = getattr(rows_staged, "stage", None) is not None
        if engram_joined:
            rows_staged.join()
        # the Engram layers a host layer projected this step
        self._engram_hosted = set()
        self._engram_joined = engram_joined
        write_step_meta(
            step,
            cache.geometry,
            cache.geometry.window(0, cache.num_pages),
            self.ring_rel,
            self.key_meta,
            self.walk_plans,
            # a token-parallel step's indexer scores this rank's rows alone
            self._own_rows(rows),
            *self._split_over(rows),
        )
        # layer 0 reads the embedding as every stream (``embed_in``): the
        # state's residual is it, unexpanded, until layer 0's seam writes the
        # streams to ``res_embed``
        state = SinglePassHCState(hidden, self.pre_init)
        # a sharded layer (``_shard``) leaves only this rank's columns of the
        # residual and its output, and of its drafter tap; token parallel, only
        # its own rows of every column (gathered at the end)
        shards = self._shard(rows)
        sharded = False
        items = list(zip(model.topology, model.layers, self.layers, self.build.keys))
        paired = self._paired(rows, items)
        # (run kernel) first layer -> its run's length
        runs = self._runs(rows, items)
        skip = False
        skip_to = 0
        for i, (spec, block, weights, key) in enumerate(items):
            if skip:
                # (pair) run as the previous layer's pair's body b
                skip = False
                continue
            if i < skip_to:
                # (run kernel) run as an iteration of an earlier layer's run
                continue
            rope = model.global_rope if spec.ratio else model.window_rope
            fuses = self._fuses(rows, spec, key, block)
            if block.engram is not None:
                state = self._engram(
                    rows, spec, block, state, metadata, sharded, engram_joined
                )
            # this layer leaves its columns only, as every one past it
            sharded = shards
            if fuses and i in runs and self._layer_run(
                items[i : i + runs[i]], state, cache, step
            ):
                pre, post, comb = self.gates_ffn
                state = SinglePassHCState(self.res_ffn, pre, self.ffn_out, post, comb)
                skip_to = i + runs[i]
                continue
            if fuses and i in paired and self._layer_pair(
                items[i], items[i + 1], state, cache, step
            ):
                pre, post, comb = self.gates_ffn
                state = SinglePassHCState(self.res_ffn, pre, self.ffn_out, post, comb)
                skip = True
                continue
            if fuses and self._layer(
                spec, block, weights, key, state, cache, step, rope
            ):
                self._candidate_table(spec, step, cache)
                pre, post, comb = self.gates_ffn
                state = SinglePassHCState(self.res_ffn, pre, self.ffn_out, post, comb)
                continue
            residual, gates = self._front(spec, key, weights, state, cache, step, rope)
            if self.check:
                check_front(
                    self,
                    spec.layer_id,
                    block,
                    # the reference takes the expanded streams
                    (
                        SinglePassHCState.from_embeddings(
                            state.residual, model.config.hc_mult
                        )
                        if key.embed_in
                        else state
                    ),
                    rope,
                    cache,
                    step,
                    residual,
                    gates,
                )
            if block.attn.indexer is not None:
                if self.compressor is None or not self.compressor.launch(
                    self.build.compress, rows, spec, cache, step, rope, self.normed,
                    self.epoch_word.data_ptr(),
                ):
                    block.attn._compress_batch(self.normed, cache, step, rope)
                if self.check:
                    # the original scorer, the reference the selection is held to
                    check_index_score(self, block.attn.indexer, cache, step)
            post = (spec, block, weights, residual, gates, step, cache, rope)
            if self.split_k2:
                self._attn_post(*post)
                if not self.check:
                    self._moe(spec.layer_id, weights)
            else:
                self._layer_post(*post)
            self._candidate_table(spec, step, cache)
            if self.check:
                if block.attn.indexer is not None:
                    check_selection(self, spec, step)
                # the original index build, for the reference attention alone
                keys = decode_keys(cache, spec, step)
                check_post(self, block, keys, residual, gates, rope, cache, step)
                self._moe(spec.layer_id, weights)
                check_moe(self, block)
            pre, post, comb = self.gates_ffn
            state = SinglePassHCState(self.res_ffn, pre, self.ffn_out, post, comb)
        if sharded:
            # every column of the hidden state and of the drafter's taps
            hidden = self._gather_cols(
                rows, state, cg.State.COLLAPSE, len(model.topology)
            )
        else:
            hidden = state.collapse()
        if self._tok_par(rows):
            hidden = self._gather_rows(hidden, rows)
        model.end_forward(hidden, None)
        self._raise_on_given_up_waits(rows)
        for timeline in self.timelines.values():
            timeline.step_done(rows)
        return hidden.squeeze(0)

    def _engram_host_of(self) -> dict:
        """host layer id -> the Engram layer whose projection it hosts:
        layer 0 (its own launch, ``embed_in``) the first's; the first Engram
        layer (its own launch: it takes a settled residual) the second's."""
        if self.engram is None:
            return {}
        ids = sorted(self.engram.blocks)
        out = {0: ids[0]} if ids and ids[0] > 0 else {}
        if len(ids) > 1:
            out[ids[0]] = ids[1]
        return out

    def _hosts_engram(self, s: int, pre) -> bool:
        """Whether the layer build over K1 key ``pre`` at s rows hosts an
        Engram projection: layer 0's (``embed_in``) and the first Engram
        layer's (no fold, no index), on a sharded step the Engram launch
        serves."""
        how = None if self.engram is None else self._engram_rows(s)
        if how is None or not how[0] or not self._engram_host_of():
            return False
        return pre.embed_in or (not pre.fold and not pre.feeds_index)

    def _engram_host_args(self, layer_id, key) -> dict:
        """A host layer's compressor-slot arguments for its Engram
        projection, recorded in ``_engram_hosted`` (the Engram launch then
        runs its gate alone); {} where it hosts none this step."""
        target = self._engram_host_of().get(layer_id)
        if target is None or not self._hosts_engram(key.tokens, key):
            return {}
        if kl.engram_host_build(self._layer_build(key.tokens, key)) is None:
            return {}
        how = self._engram_rows(key.tokens)
        got = self.engram.host_args(
            target, get_forward_context().attn_metadata, t0=how[1],
            joined=getattr(self, "_engram_joined", False),
        )  # fmt: skip
        if got is None:
            return {}
        self.__dict__.setdefault("_engram_hosted", set()).add(target)
        return got

    def _engram_rows(self, s: int):
        """(split, first row, rows) of an s-row step's Engram launch: a
        sharded step's every row on the rank's columns; a token-parallel
        step's own rows; at TP 1 every row; None under DP attention."""
        if self._shard(s):
            return True, 0, s
        if self._tok_par(s):
            t0, n = own_rows(s, self.tp, self.rank)
            return False, t0, n
        return (False, 0, s) if self.tp == 1 and self.dp == 1 else None

    def _engram_build(self, s: int):
        rows = None if self.engram is None else self._engram_rows(s)
        if rows is None:
            return None
        split, _, n = rows
        return self.engram.build(n, split)

    def _engram(self, rows, spec, block, state, metadata, sharded, joined=False):
        """An Engram layer's gated residual (settled, owing nothing): one launch
        (``engram_launch``) where built, else the original on every column
        (gathered first where sharded). ``joined``: the step's stream already
        waited on the Engram staging's (``forward``)."""
        how = None if self.engram is None else self._engram_rows(rows)
        if how is not None and self.engram.launch(
            self.build.engram, spec.layer_id, state, metadata,
            self._peer_args(), t0=how[1], joined=joined,
            hosted=spec.layer_id in getattr(self, "_engram_hosted", ()),
        ):  # fmt: skip
            return SinglePassHCState(self.engram.gated(rows), state.pre_mix)
        if sharded:
            # the original reads the settled residual's every column
            state = self._gather_cols(rows, state, cg.State.SETTLE, spec.layer_id)
        state = state.settle()
        return SinglePassHCState(
            block.engram_forward(state.residual, None, None), state.pre_mix
        )

    def sample_token_ids(self, hidden: torch.Tensor):
        """The verify of a step of ``hidden``'s rows (``forward``'s, eager or
        replayed), the final norm and LM head in one launch
        (``MonoLmHead.launch``) of that width's build: (output ids, spec
        status); None where the width has no head launch."""
        head = self.widths[hidden.shape[0]].head
        if head is None or not hidden.is_contiguous():
            return None
        return self.lm_head.launch(head, hidden)

    def _gather_rows(self, hidden, rows):
        """A token-parallel step's every row from its owner: the collapsed
        hidden state and the drafter's taps, one all-gather (each rank holds
        its own tokens' rows only)."""
        n = rows // self.tp
        lo = self.rank * n
        taps = [buf[lo : lo + n] for buf in self.aux.buffers]
        mine = torch.stack([hidden[0, lo : lo + n], *taps], dim=1)
        every = get_tp_group().all_gather(mine, use_custom=True, dim=0)
        for k, buf in enumerate(self.aux.buffers):
            buf[:rows].copy_(every[:, k + 1])
        return self.hidden_out[:rows].copy_(every[:, 0]).unsqueeze(0)

    def _gather_cols(self, rows, state, mode, call):
        """A sharded step's every column from its owner (``_shard``: a rank's
        layers write their own columns only), one ``col_gather`` launch on
        ``state`` (which owes its post): ``SETTLE`` -> the state settled in
        place (an Engram layer's input); ``COLLAPSE`` -> the step's collapsed
        hidden state [1, rows, dim], the drafter's taps gathered with it.
        ``call``: this gather's place in the step (its pairs' tag)."""
        p = torch.Tensor.data_ptr
        peers = self._peer_args()
        args = {
            "res": p(state.residual), "sub": p(state.pending),
            "post": p(state.post_mix), "comb": p(state.combination),
            "pre": p(state.pre_mix), "out": p(self.hidden_out), "call": call,
            "sym": peers["sym"], "peers": peers["peers"], "rank": peers["rank"],
            "epoch": peers["epoch"],
        }  # fmt: skip
        taps = self.aux.buffers if mode == cg.State.COLLAPSE else ()
        for k, name in enumerate(cg.TAPS):
            args[name] = p(taps[k]) if k < len(taps) else 0
        _run_compiled(
            self.build.gathers[mode], *cg.ABI.pack(args), torch.cuda.current_stream()
        )
        if mode == cg.State.SETTLE:
            return SinglePassHCState(state.residual, state.pre_mix)
        return self.hidden_out[:rows].unsqueeze(0)

    def _front(self, spec, key, weights, state, cache, step, rope):
        """K1 -> (the residual after the owed post, the seam's (pre, post, comb));
        the query, the window row and the indexer's inputs land in this runner's
        buffers."""
        layer_id = spec.layer_id
        args, res_out, gates = self._front_args(
            layer_id, key, weights, state, cache, step, rope
        )
        # an indexer layer's bounds: its indexer stages run on a step that
        # scores a row (``index_query.step_queried``)
        bound = self._selection_bound(spec, step) if key.feeds_index else self.no_bound
        args["ibound"] = bound.data_ptr()
        args.update(self._tail_args())
        args["layer"] = layer_id
        args["tl"] = self._timeline_ptr("k1", layer_id)
        # the caller's stream: a FlyDSL launcher left to its default submits on the
        # NULL stream, outside a graph capture
        _run_compiled(
            self.build.launchers[key], *k1.ABI.pack(args), torch.cuda.current_stream()
        )
        return res_out, gates

    def _paired(self, rows: int, items) -> set:
        """(pair) The layers that run with the next one as a pair
        (``layer_pair``): greedily, two consecutive layers of one pairable
        build, neither an Engram layer nor compressing nor producing candidates."""
        pairs = self.build.pairs
        if not pairs:
            return set()

        def ok(item):
            spec, block, _, key = item
            return (
                key in pairs and block.engram is None
                and self._fuses(rows, spec, key, block)
                and self._compress_ratio(spec, block) == 0
                and not spec.produces_candidates
            )  # fmt: skip

        out, i = set(), 0
        while i + 1 < len(items):
            if items[i][3] == items[i + 1][3] and ok(items[i]) and ok(items[i + 1]):
                out.add(i)
                i += 2
            else:
                i += 1
        return out

    def _layer_pair(self, item_a, item_b, state, cache, step) -> bool:
        """(pair) Layers a and b (consecutive, one build) in one
        launch (``layer_pair``): b's state is a's outputs (the runner's FFN
        seam buffers). False (nothing run) where either has no layer kernel."""
        model = self.model
        spec_a, block_a, weights_a, key = item_a
        spec_b, block_b, weights_b, _ = item_b
        rope_a = model.global_rope if spec_a.ratio else model.window_rope
        rope_b = model.global_rope if spec_b.ratio else model.window_rope
        got_a = self._layer_args(spec_a, block_a, weights_a, key, state, cache, step, rope_a)
        if got_a is None:
            return False
        # body a's FFN seam writes the pair's own uncached residual and
        # gates, body b's attention seam reads them (``res_pair`` / ``gates_pair``)
        p = torch.Tensor.data_ptr
        pre, post, comb = self.gates_pair
        got_a[1].update({
            "res_out": p(self.res_pair), "pre_out": p(pre), "post_out": p(post),
            "comb_out": p(comb),
        })  # fmt: skip
        state_b = SinglePassHCState(self.res_pair, pre, self.ffn_out, post, comb)
        got_b = self._layer_args(spec_b, block_b, weights_b, key, state_b, cache, step, rope_b)
        if got_b is None:
            return False
        args = {f"a_{n}": v for n, v in got_a[1].items()}
        args.update({f"b_{n}": v for n, v in got_b[1].items()})
        _run_compiled(
            self.build.pairs[key], *lp.ABI.pack(args), torch.cuda.current_stream()
        )
        return True

    def _runs(self, rows: int, items) -> dict:
        """(run kernel) First layer -> length of each run of consecutive
        layers of one runnable build (``layer_run``), none compressing nor
        producing candidates; an Engram layer only as a run's first (its Engram
        launch runs before the run's). A run of one too: every ordinary layer
        runs the run kernel, one code object."""
        runs = self.build.runs
        if not runs:
            return {}

        def ok(item):
            spec, block, _, key = item
            return (
                key in runs and self._fuses(rows, spec, key, block)
                and self._compress_ratio(spec, block) == 0
                and not spec.produces_candidates
            )  # fmt: skip

        out, i = {}, 0
        while i < len(items):
            if not ok(items[i]):
                i += 1
                continue
            j = i + 1
            while (
                j < len(items) and items[j][3] == items[i][3] and ok(items[j])
                and items[j][1].engram is None
            ):  # fmt: skip
                j += 1
            # (diagnostic) ``ATOM_MONO_RUN_MAXLEN``: runs cut to at most this many layers
            cap = int(os.environ.get("ATOM_MONO_RUN_MAXLEN", "0")) or j - i
            for k in range(i, j, cap):
                out[k] = min(cap, j - k)
            i = j
        return out

    def _layer_run(self, items, state, cache, step) -> bool:
        """(run kernel) A run of layers (consecutive, one build) in one launch
        (``layer_run``): each layer's arguments a row of the run's table, its
        state the previous one's outputs -- inside the run the uncached
        ``res_pair`` / ``gates_pair``, the last layer's ``res_ffn`` /
        ``gates_ffn`` (``_layer_args``' own). False (nothing run) where a layer
        has no layer kernel."""
        model = self.model
        key = items[0][3]
        n = len(items)
        p = torch.Tensor.data_ptr
        pre, post, comb = self.gates_pair
        rows_args = []
        st = state
        for k, (spec, block, weights, _) in enumerate(items):
            rope = model.global_rope if spec.ratio else model.window_rope
            got = self._layer_args(spec, block, weights, key, st, cache, step, rope)
            if got is None:
                return False
            args = got[1]
            if k < n - 1:
                args.update({
                    "res_out": p(self.res_pair), "pre_out": p(pre), "post_out": p(post),
                    "comb_out": p(comb),
                })  # fmt: skip
                st = SinglePassHCState(self.res_pair, pre, self.ffn_out, post, comb)
            rows_args.append(args)
        tab = self._run_table(rows_args)
        seen = self.__dict__.setdefault("_runs_logged", set())
        span = (items[0][0].layer_id, n, key.tokens)
        if span not in seen:
            seen.add(span)
            logger.info(
                "V4.1 mono run kernel (ATOM_MONO_RUN=%d): layers %d..%d in one launch, %d rows",
                self._run_mode(), span[0], span[0] + n - 1, span[2],
            )  # fmt: skip
        tail = self._tail_args()
        _run_compiled(
            self.build.runs[key],
            *lr.ABI.pack({"ltab": p(tab), "nlay": n, **{t: tail[t] for t in lr.ABI.names[2:]}}),
            torch.cuda.current_stream(),
        )
        return True

    def _run_table(self, rows_args) -> torch.Tensor:
        """(run kernel) The run's argument table on the device (``layer_run``:
        ``RUN_ARGS`` int64 a layer, ``layer.ABI`` order), one per distinct
        content, never rewritten. Outside a capture: an async copy in the
        stream from its own pinned host copy. Inside one (a key the warmup
        forward did not have: the capture's step buffers): filled at once by a
        helper thread on a stream of its own -- the capture is thread_local
        (``region_graph``), so the graph holds no copy node and the table holds
        the capture's values before any replay."""
        vals = tuple(int(a[name]) for a in rows_args for name in lr.RUN_ARG_NAMES)
        tabs = self.__dict__.setdefault("_run_tabs", {})
        capturing = torch.cuda.is_current_stream_capturing()
        got = tabs.get(vals)
        last = self.__dict__.setdefault("_run_last", {})
        span = (len(rows_args), rows_args[0]["layer"])
        if got is None and capturing and span in last and len(last[span]) == len(vals):
            # which arguments a capture's step changes (logged once a run)
            names = sorted({
                f"{lr.RUN_ARG_NAMES[k % lr.RUN_ARGS]}" for k, (a, b) in enumerate(zip(vals, last[span])) if a != b
            })  # fmt: skip
            logger.info("V4.1 mono run kernel: capture table of layers %d.. (+%d) differs from the warmup's in %s",
                        span[1], span[0] - 1, names)  # fmt: skip
        last[span] = vals
        if got is None:
            if capturing:
                got = tabs[vals] = [*_filled_apart(vals, torch.cuda.current_device()), True]
            else:
                host = torch.tensor(vals, dtype=torch.int64).pin_memory()
                dev = torch.empty_like(host, device=torch.device("cuda", torch.cuda.current_device()))
                dev.copy_(host, non_blocking=True)
                got = tabs[vals] = [host, dev, False]
        got[2] = got[2] or capturing
        if len(tabs) > 4096 and not capturing:
            # eager steps with new buffers each: drop the tables no graph holds
            for k in [k for k, v in tabs.items() if not v[2] and v is not got]:
                del tabs[k]
        return got[1]

    def _layer(self, spec, block, weights, key, state, cache, step, rope) -> bool:
        """``_layer_args`` launched."""
        got = self._layer_args(spec, block, weights, key, state, cache, step, rope)
        if got is None:
            return False
        launcher, args = got
        _run_compiled(
            launcher,
            *kl.ABI.pack(args),
            torch.cuda.current_stream(),
        )
        return True

    def _layer_args(self, spec, block, weights, key, state, cache, step, rope):
        """K1 and K2 in one launch (``kernels.layer``), a compressing layer's
        compressor between them; K2's outputs as ``_layer_post``'s. False (and
        nothing run) where the width has no such build (it spilled:
        ``_lean_rows``) or this step is one the compressor is not built for
        (``MonoCompressor.layer_args``): the caller runs them apart."""
        layer_id = spec.layer_id
        ratio = self._compress_ratio(spec, block)
        launcher = self.build.layers.get((key, ratio))
        if launcher is None:
            return None
        cargs = {name: 0 for name in kl.COMPRESS_ARGS}
        if ratio:
            cargs = self.compressor.layer_args(key.tokens, spec, cache, step, rope)
            if cargs is None:
                return None
        else:
            cargs.update(self._engram_host_args(layer_id, key))
        own, residual, gates = self._front_args(
            layer_id, key, weights, state, cache, step, rope
        )
        args = {f"k1_{name}": value for name, value in own.items()}
        args.update(
            self._attn_post_args(
                spec, block, weights, residual, gates, step, cache, rope
            )
        )
        args.update(self._moe_args(weights))
        args.update(self._tail_args())
        args["kvrow"] = self.kvrow.data_ptr()
        args.update(cargs)
        args["layer"] = layer_id
        args["post_layer"] = POST_TAG0 + layer_id
        args["moe_layer"] = MOE_TAG0 + layer_id
        args["queue_words"] = self._queue_words(layer_id)
        args["tl"] = self._timeline_ptr("layer", layer_id)
        return launcher, args

    def _front_args(self, layer_id, key, weights, state, cache, step, rope):
        """K1's own arguments (its ABI up to the shared tail) -> (them, the
        residual it writes, its gates)."""
        if (state.pending is not None) != key.fold:
            raise MonoUnsupported(f"layer {layer_id}: owed post is not as built")
        if key.embed_in:
            res_out = self.res_embed
        else:
            res_out = self.res_attn if key.fold else state.residual
        pre, post, comb = self.gates_attn
        aux = self.aux.buffers[self.aux.layer_ids.index(layer_id)] if key.aux else None
        p = torch.Tensor.data_ptr
        fold = key.fold
        args = {
            "res_in": p(state.residual), "pend": p(state.pending) if fold else 0,
            "post_in": p(state.post_mix) if fold else 0,
            "comb_in": p(state.combination) if fold else 0, "pre_in": p(state.pre_mix),
            "hc_fn": p(weights["hc_attn_fn"]), "hc_scale": p(weights["hc_attn_scale"]),
            "hc_base": p(weights["hc_attn_base"]),
            "attn_w": p(weights["attn_norm.weight"]),
            "wqkv": p(weights["attn.wqkv_a.weight"]),
            "wqkv_s": p(weights["attn.wqkv_a.weight_scale"]),
            "qn_w": p(weights["attn.q_norm.weight"]),
            "kvn_w": p(weights["attn.kv_norm.weight"]),
            "wqb": p(weights["attn.wq_b.weight"]),
            "wqb_s": p(weights["attn.wq_b.weight_scale"]),
            "cos": p(rope.cos_cache), "sin": p(rope.sin_cache),
            "pos": p(self._rope_positions(cache, step)),
            "ring": p(cache.ring_view(layer_id)[0]) if cache.kv_fp8 else p(cache.pool),
            "ring_scale": p(cache.ring_view(layer_id)[1]) if cache.kv_fp8 else 0,
            "ring_rows": p(self.ring_rel),
            "ring_off": cache.geometry.window(layer_id, cache.num_pages).ring_start,
            "res_out": p(res_out), "post_out": p(post), "comb_out": p(comb),
            "pre_out": p(pre), "q_out": p(self.q), "normed": p(self.normed),
            "qr_out": p(self.qr), "qrs_out": p(self.qr_scale),
            "aux": 0 if aux is None else p(aux),
        }  # fmt: skip
        args.update(self._index_projection_args(key, weights))
        return args, res_out, (pre, post, comb)

    def _index_projection_args(self, key, weights) -> dict:
        """K1's indexer projections: FULL / REINDEX builds only."""
        names = ("iq_w", "iq_ws", "iw_w", "iq_out", "iqs_out", "iw_out")
        if not key.feeds_index:
            return dict.fromkeys(names, 0)
        p = torch.Tensor.data_ptr
        return {
            "iq_w": p(weights["attn.indexer.wq_b.weight"]),
            "iq_ws": p(weights["attn.indexer.wq_b.weight_scale"]),
            "iw_w": p(weights["attn.indexer.weights_proj.weight"]),
            "iq_out": p(self.iq), "iqs_out": p(self.iq_scale),
            "iw_out": p(self.iw),
        }  # fmt: skip

    def _layer_post(self, spec, block, weights, residual, gates, step, cache, rope):
        """K2: K2a's work, then K2b's, in one launch."""
        layer_id = spec.layer_id
        args = k2.ABI.pack({
            **self._attn_post_args(spec, block, weights, residual, gates, step, cache, rope),
            **self._moe_args(weights),
            **self._tail_args(),
            "layer": POST_TAG0 + layer_id, "moe_layer": MOE_TAG0 + layer_id,
            "queue_words": self._queue_words(layer_id),
            "tl": self._timeline_ptr("k2", layer_id),
        })  # fmt: skip
        _run_compiled(self.build.layer_post, *args, torch.cuda.current_stream())

    def _attn_post(self, spec, block, weights, residual, gates, step, cache, rope):
        """K2a: the decode sparse attention over this layer's keys (its class's
        ``key_meta``, the selection of its top-k owner, the window), the output
        projections and their all-reduce, folded into the residual by the FFN
        seam -> ``res_ffn``, the FFN gates, ``ffn_normed``."""
        layer_id = spec.layer_id
        args = k2a.ABI.pack({
            **self._attn_post_args(spec, block, weights, residual, gates, step, cache, rope),
            **self._tail_args(),
            # its own tags: K1's regions of this layer share the scratch
            "layer": POST_TAG0 + layer_id,
            "tl": self._timeline_ptr("k2a", layer_id),
        })  # fmt: skip
        _run_compiled(self.build.attn_post, *args, torch.cuda.current_stream())

    def _tail_args(self) -> dict:
        """The layer launches' shared tail: the step scratch, its dual area's
        calibration word (``numa_word``: this rank's, measured at run time,
        so no build depends on it), the peer buffer."""
        p = torch.Tensor.data_ptr
        return {
            "scratch": p(self.scratch),
            "numa_word": self.numa_word,
            **self._peer_args(),
        }

    def _peer_args(self) -> dict:
        """The peer buffer's kernel arguments and the step's epoch word, which
        every launch tagging its hand-offs reads (``kernels.mailbox.epoch_of``)."""
        return {**self.peers.kernel_args(), "epoch": self.epoch_word.data_ptr()}

    def _queue_words(self, layer_id: int) -> int:
        """Layer ``layer_id``'s K2b queue words (``k2b.queue_offset``)."""
        return self.queue_words.data_ptr() + k2b.queue_offset(MOE_TAG0 + layer_id)

    def _attn_post_args(self, spec, block, weights, residual, gates, step, cache, rope):
        """K2a's own arguments."""
        layer_id = spec.layer_id
        p = torch.Tensor.data_ptr
        pre, post, comb = gates
        pre_out, post_out, comb_out = self.gates_ffn
        attn = block.attn
        geometry = cache.geometry
        meta = self.key_meta
        index = self._selection_args(spec, attn, cache, step)
        owner = spec.topk_owner if spec.ratio else None
        return {
            **index,
            "q": p(self.q),
            "pool": p(cache.main_view(spec.kv_owner)[0]) if cache.kv_fp8 else p(cache.pool),
            "pool_scale": p(cache.main_view(spec.kv_owner)[1]) if cache.kv_fp8 else 0,
            "window": p(cache.ring_view(layer_id)[0]) if cache.kv_fp8 else 0,
            "window_scale": p(cache.ring_view(layer_id)[1]) if cache.kv_fp8 else 0,
            # a layer without a selection reads a dummy row it never uses
            "sel": p(meta if owner is None else self.irow[owner]),
            "topk": 1 if owner is None else ip.TOPK,
            "sbound": p(self.no_bound if owner is None else self.bounds[owner]),
            # a window layer's keys are all ring rows: it reads no block table
            "kmeta": p(meta),
            "table": p(step.block_tables) if spec.ratio else p(meta),
            "rows_per_page": geometry.rows_per_page(spec.ratio or 1),
            "page_rows": geometry.rows_per_page(spec.ratio or 1) if cache.kv_fp8 else geometry.page_bytes // geometry.row_bytes,
            "main_off": geometry.main_offset(spec.kv_owner) if spec.ratio else 0,
            "ring_off": cache.main_view(spec.kv_owner)[0].shape[0] if cache.kv_fp8 else geometry.window(layer_id, cache.num_pages).ring_start,
            "ring_slots": geometry.window(layer_id, cache.num_pages).ring_slots,
            "sink": p(attn.attn_sink),
            "qk_scale": f32_bits(attn.softmax_scale * LOG2E),
            "pos": p(self._rope_positions(cache, step)),
            "cos": p(rope.cos_cache), "sin": p(rope.sin_cache),
            "woa": p(weights["attn.wo_a.weight"]),
            "woa_s": p(weights["attn.wo_a.weight_scale"]),
            "wob": p(weights["attn.wo_b.weight"]),
            "wob_s": p(weights["attn.wo_b.weight_scale"]),
            "res_in": p(residual), "post_in": p(post), "comb_in": p(comb),
            "pre_in": p(pre), "hc_fn": p(weights["hc_ffn_fn"]),
            "hc_scale": p(weights["hc_ffn_scale"]), "hc_base": p(weights["hc_ffn_base"]),
            "ffn_w": p(weights["ffn_norm.weight"]), "res_out": p(self.res_ffn),
            "post_out": p(post_out), "comb_out": p(comb_out), "pre_out": p(pre_out),
            "normed": p(self.ffn_normed),
        }  # fmt: skip

    def _moe(self, layer_id, weights) -> None:
        """K2b: router, routed and shared experts and their all-reduce ->
        ``ffn_out``."""
        key_a, _ = self._post_key(self.ffn_normed.shape[1]).halves()
        args = k2b.ABI.pack({
            "x": self.ffn_normed.data_ptr(), **self._moe_args(weights),
            **self._tail_args(), "layer": MOE_TAG0 + layer_id,
            "queue_words": self._queue_words(layer_id),
            "tl": self._timeline_ptr("k2b", layer_id),
            # where the merged launch's layout puts K2b's regions
            "scratch": self.scratch.data_ptr() + k2.moe_base(key_a),
        })  # fmt: skip
        _run_compiled(self.build.moe, *args, torch.cuda.current_stream())

    def _moe_args(self, weights) -> dict:
        """K2b's own arguments but its input."""
        p = torch.Tensor.data_ptr
        return {
            "gate_w": p(weights["ffn.gate.weight"]),
            "bias": p(weights["ffn.gate.e_score_correction_bias"]),
            "w13": p(weights["ffn.experts.w13_weight"]),
            "w13_s": p(weights["ffn.experts.w13_weight_scale"]),
            "w2": p(weights["ffn.experts.w2_weight"]),
            "w2_s": p(weights["ffn.experts.w2_weight_scale"]),
            "sgu": p(weights["ffn.shared_experts.gate_up_proj.weight"]),
            "sgu_s": p(weights["ffn.shared_experts.gate_up_proj.weight_scale"]),
            "sw2": p(weights["ffn.shared_experts.w2.weight"]),
            "sw2_s": p(weights["ffn.shared_experts.w2.weight_scale"]),
            "out": p(self.ffn_out),
        }  # fmt: skip

    def _raise_on_given_up_waits(self, rows: int) -> None:
        """A debug build's eager step: raise with every mailbox wait that gave up
        (a graph replay runs no Python: run it with ``--enforce-eager``)."""
        if not self.debug or torch.cuda.is_current_stream_capturing():
            return
        regions = {
            *k1.scratch_layout(rows),
            *k2.scratch_layout(self._post_key(rows)),
        }
        diag = self.scratch[self.diag_off : self.diag_off + DIAG_BYTES]
        waits = given_up_waits(diag, kernel_debug.region_names(regions))
        raise_if_given_up("V4.1 mono", self.rank, rows, waits, self.mailboxes)

    def _rope_positions(self, cache, step) -> torch.Tensor:
        """Each row's position for RoPE (int64 [S])."""
        return cache.rope_positions(step)

    def _timeline_ptr(self, kernel, layer_id) -> int:
        timeline = self.timelines.get(kernel)
        return timeline.ptr(layer_id) if timeline is not None else 0

    def _selection_bound(self, spec, step) -> torch.Tensor:
        """A selecting layer's column bound a row (K2a's ``ibound``, which K1's
        indexer stages read too): its candidates' on a REINDEX layer, else the
        step's visible rows at its ratio."""
        if spec.candidate_source is not None:
            return self.candidate_table[1]
        return step.visible[spec.ratio]

    def _selection_args(self, spec, attn, cache, step) -> dict:
        """K2a's indexer arguments: a selecting layer's scorer inputs, its bound
        (the one ``index_plan`` derives every cell from, recorded for the layers
        that reuse its selection) and outputs; dummies elsewhere."""
        p = torch.Tensor.data_ptr
        dummy = p(self.no_bound)
        args = {
            "iact": 0, "iq": dummy, "iw": dummy, "plane": dummy, "iqs": dummy,
            "pscale": dummy, "iplan": dummy, "itab": dummy,
            "itab_stride": 0, "itab_len": 1, "ibatch": dummy, "ishift": 0,
            "ibound": dummy, "ilog": p(self.ilog),
            "lstride": self.ilog.shape[1], "isel": dummy, "icand": dummy, "ilift": 0,
            "iprod": 0, "ibmax": p(self.ibmax), "bstride": self.ibmax.shape[1],
            "icout": dummy, "irow": dummy, "ixsel": dummy,
        }  # fmt: skip
        if attn.indexer is None:
            return args
        reindex = spec.candidate_source is not None
        shift = 0
        bound = self._selection_bound(spec, step)
        if reindex:
            table = self.candidate_table[0]
        elif self.index_fp4:
            # the request's PAGE table, an entry 2 ** shift pages of the plane
            per_page = cache.geometry.index_blocks_per_page(spec.ratio)
            assert per_page & (per_page - 1) == 0, per_page
            shift = per_page.bit_length() - 1
            table = step.block_tables
        else:
            table = cache.unit_tiles(step, spec.ratio)
        batch_ids = step.batch_ids
        assert batch_ids.dtype == torch.int32 and batch_ids.stride(0) == 1
        self.bounds[spec.layer_id] = bound
        units = cache.index_units[spec.kv_owner]
        if self.index_fp4 and units.layout != self.index_layout:
            raise ValueError("V4.1 Mono index build and cache layouts differ")
        if self.index_fp4:
            args.update({
                "iqs": p(self.iq_scale), "pscale": p(units.scales),
                "iplan": dummy if reindex else p(self.walk_plans[spec.ratio]),
            })  # fmt: skip
        args.update({
            "iact": 1, "iq": p(self.iq), "iw": p(self.iw),
            "plane": p(units.values), "itab": p(table),
            "itab_stride": table.stride(0), "itab_len": table.shape[1],
            "ibatch": p(batch_ids), "ishift": shift,
            "ibound": p(bound), "isel": p(self.isel[spec.layer_id]),
            "icand": p(self.icand), "ilift": int(reindex),
            "iprod": int(spec.produces_candidates), "icout": p(self.icand),
            "irow": p(self.irow[spec.layer_id]), "ixsel": p(self.ixsel),
        })  # fmt: skip
        return args
