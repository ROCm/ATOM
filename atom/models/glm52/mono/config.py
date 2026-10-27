# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The deployments GLM-5.2 mono serves, and why any other is refused.

The kernels themselves live in ``atom/models/glm52/mono/kernels`` and are built
for one shard geometry at a time (``glm5_tp_config``); this module is the one
place that decides whether a deployment yields that geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.glm52.mono.kernels.geometry_config import (
    EPS,
    GLM5_AGENTX_BATCHES,
    GLM5_GRAPH_BATCHES,
    GLM5_QUERY_LENGTHS,
    glm5_tp_config,
)
from atom.models.glm52.mono.kernels.layout import INDEX_SEG, index_cp_world
from atom.models.utils import is_neox_rope_style
from atom.mono.plan.shard import Shard
from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs

# the model's full widths
Q_HEADS = 64
EXPERT_INTER = 2048  # moe_intermediate_size

# the TP sizes the kernels are built for
SUPPORTED_TP = (4,)
# the batches a build covers, and the batches a step is routed at: only one and
# two were measured faster than the production path, so only they are served
GRAPH_BATCHES = GLM5_GRAPH_BATCHES
AGENTX_BATCHES = GLM5_AGENTX_BATCHES
# the served decode widths: no speculation at all (1), and an MTP verify of
# the anchor token plus 3/4/5 drafts (4, 5, 6) -- sourced from the kernel's own
# declaration so a width added there cannot silently fail to reach here
QUERY_LENGTHS = GLM5_QUERY_LENGTHS

# the loaded checkpoint must match this shard geometry exactly; ``index_topk``
# is not part of ``LayerConfig`` (the indexer lives outside the mono kernels),
# so it is checked against the fixed width the kernels were built for
EXPECTED_INDEX_TOPK = 2048
EXPECTED_SCORING_FUNC = "sigmoid"
EXPECTED_TOPK_METHOD = "noaux_tc"

# What the fused TP4 indexer (``Glm5TP4MonoKernel(with_indexer=True)``) is
# configured for here -- see ``atom/models/glm52/mono/kernels/tp4_op.py``.
# 1048576 -- GLM-5.2's own declared ``max_position_embeddings``, so the fused
# indexer no longer refuses a default deployment.
#
# This is a capacity, and since the selector became a merge tree it is no
# longer an LDS quantity. The tree ranks a FIXED ``INDEX_SEG`` (16384) keys in
# LDS per leaf and reaches past that by STACKING leaves, not by growing one:
# at GLM's ``topk`` 2048 the fan is ``16384 // 2048 == 8``, so two merge levels
# cover ``16384 * 8 * 8 == 1048576`` exactly, with no LDS beyond a depth-0
# build's (``glm/layout.py``; ``tests/model_ops/test_glm5_index_tree.py``
# pins the depth at each capacity). ``index_levels`` derives the depth from
# this constant as a Python int and the levels unroll statically, so build
# time does not grow with it either -- depths 0/1/2 compiled in 37.1/32.2/37.4s.
#
# What raising it does cost is **global** scratch and the scoring work itself.
# ``index_scores`` is ``samples * index_max_seq * 8`` bytes per width, so
# across the eight served widths {1,2,4,5,6,8,10,12} the pool grows from
# 6 MiB at 16384 to 384 MiB at 1048576. The scoring cost scales with context
# exactly as the external indexer's already does -- this buys capacity, not
# speed, and a short request still only scores its own ``bound`` keys.
#
# 1048576 still divides evenly by ``INDEX_KEYS_PER_TASK`` (64) and by
# ``FUSED_INDEX_PAGE_SIZE`` (16), which ``glm/kernel.py`` asserts.
#
# The index page size is not independently configurable: it shares
# ``block_tables`` with ATOM's KV cache, so it must equal
# ``kv_cache_block_size`` exactly, not merely divide it. That is now the
# binding constraint on a default deployment -- ``--block-size 16``.
FUSED_INDEX_MAX_SEQ = 1048576
FUSED_INDEX_PAGE_SIZE = 16


def fused_index_capacity(max_model_len: int) -> int:
    """The ``index_max_seq`` a deployment should BUILD at.

    ``FUSED_INDEX_MAX_SEQ`` is the ceiling the fused indexer *supports*; it is
    not what every deployment should compile against. The distinction is not
    cosmetic, because the scoring work map is enumerated from the compiled
    capacity and not from the request:

        layout.py:448
        tasks += [("index_score",
                   samples * ceil(capacity / INDEX_KEYS_PER_TASK))]

    The keys each task scores are bounded by ``row_position(s) + 1``, so a
    short request does no extra *scoring* -- but the tasks themselves are
    enumerated, dispatched and retired regardless. Building a 16 k deployment
    against a 1 M capacity therefore inflates that table 64x (1 536 -> 98 304
    tasks per fused layer at S=6; 27 648 -> 1 769 472 per step across the 18
    full layers) to do identical work, of which at ISL 8192 only 0.78 % has
    any key at all. Measured: fused cost +11.14 % against the external
    indexer at a 1 M build and -0.09 % at a 16 k build, same hardware, same
    requests.

    So track the deployment instead. A run that caps ``--max-model-len`` pays
    for what it caps; a genuine 1 M deployment pays for 1 M, which is the
    bargain it asked for.

    Rounded UP to a whole ``INDEX_SEG`` leaf and floored at one leaf: the
    merge tree's fan is ``index_seg(cap) // topk``, so a capacity under
    ``topk`` would give a fan of 0 and ``index_levels`` would not terminate.
    A whole number of leaves also keeps the fan at 8 and the depth at
    ``ceil(log8(cap / INDEX_SEG))``. ``INDEX_SEG`` is a multiple of both
    ``INDEX_KEYS_PER_TASK`` (64) and ``FUSED_INDEX_PAGE_SIZE`` (16), which
    ``glm/kernel.py`` asserts, so every value this returns is legal.
    """

    leaves = max(1, -(-int(max_model_len) // INDEX_SEG))
    return min(FUSED_INDEX_MAX_SEQ, leaves * INDEX_SEG)


@dataclass(frozen=True)
class Dims:
    """A TP rank's share of GLM-5.2's widths."""

    tp: int

    def __post_init__(self):
        shard = Shard(self.tp)
        shard.split(Q_HEADS, "attention heads")
        shard.split(EXPERT_INTER, "expert intermediate")

    @property
    def local_heads(self) -> int:
        return Q_HEADS // self.tp

    @property
    def inter(self) -> int:
        return EXPERT_INTER // self.tp


def native_config(
    *, samples: int, tp_size: int, kv_cache_dtype: str, mtp: bool, query_length: int
):
    """The measured scaled-FP4 GLM-5.2 shard geometry for this step, or None.

    Query length 1 is no speculation at all -- ``atom_config.speculative_config
    is None``, so ``mtp`` is always False for it -- and every other served
    width is an MTP verify, so ``mtp`` must be True for it. The two are
    mutually exclusive: a query length of 1 arriving with an MTP config is
    incoherent (there is no such deployment) and is refused here explicitly,
    rather than left to fall through whatever the membership test below
    happens to do.
    """

    if tp_size not in SUPPORTED_TP or kv_cache_dtype != "fp8":
        return None
    if query_length not in QUERY_LENGTHS or mtp != (query_length != 1):
        return None
    if samples % query_length or samples // query_length not in GRAPH_BATCHES:
        return None
    return glm5_tp_config(tp_size)


def model_geometry_refusal(hf_config, shard, tp_size: int) -> str | None:
    """Why the loaded checkpoint is not the GLM-5.2 the kernels are built for."""
    expected = {
        "hidden_size": shard.hidden,
        "num_attention_heads": shard.local_heads * tp_size,
        "q_lora_rank": shard.q_lora,
        "kv_lora_rank": shard.kv_lora,
        "qk_rope_head_dim": shard.pe_dim,
        "qk_nope_head_dim": shard.nope_dim,
        "v_head_dim": shard.v_dim,
        "n_routed_experts": shard.n_experts,
        "num_experts_per_tok": shard.top_k,
        "n_shared_experts": shard.num_shared_experts,
        "index_topk": EXPECTED_INDEX_TOPK,
        "routed_scaling_factor": shard.route_scale,
        "scoring_func": EXPECTED_SCORING_FUNC,
        "topk_method": EXPECTED_TOPK_METHOD,
        "norm_topk_prob": True,
        "rms_norm_eps": EPS,
    }
    for name, value in expected.items():
        if getattr(hf_config, name, None) != value:
            return f"{name}={getattr(hf_config, name, None)!r}"
    if hf_config.moe_intermediate_size != shard.inter * tp_size:
        return "expert width"
    # Production routes this MoE with GROUPED top-k -- `deepseek_v2.py` passes
    # `use_grouped_topk=True, num_expert_group=config.n_group,
    # topk_group=config.topk_group` -- which scores each group by the sum of
    # its best two experts, keeps the best `topk_group` groups and zeroes every
    # expert outside them before taking the top-k (aiter `ops/topk.py`).
    # `route_top8` (`kernels/kernel.py`) is a plain global top-8 over all
    # N_EXPERTS: `wave_umax_dpp` reduces across every lane and there is no group
    # structure in it anywhere.
    #
    # The two agree only where the masking is a no-op. With every group kept,
    # `masked_fill` leaves `scores_for_choice` untouched and production's top-k
    # IS the global one over `sigmoid + bias` -- the same key `route_top8`
    # sorts on. With a group left out, mono picks a different expert set
    # whenever the global top 8 straddles it: different output, no error, no
    # fallback. `topk_method` is pinned to `noaux_tc` above, which is the
    # grouped method, so this is reachable rather than theoretical.
    #
    # Not an `expected` entry: those compare for equality, and what matters
    # here is the RELATION between the two, plus the absent-reads-as-ungrouped
    # default that a bare equality check cannot express.
    n_group = getattr(hf_config, "n_group", None) or 1
    topk_group = getattr(hf_config, "topk_group", None) or 1
    if topk_group < n_group:
        return f"grouped top-k (n_group={n_group}, topk_group={topk_group})"
    # The kernel rotates adjacent pairs unconditionally (`kernel.py`: x0, x1 at
    # `pr*2` and `pr*2 + 1`), i.e. interleaved / GPT-J, with no NeoX branch, for
    # both the MLA rope and the fused indexer's own, independent one. A
    # checkpoint that asks for NeoX would get a different position encoding with
    # no error. Deliberately NOT `expected` entries: that dict compares for
    # equality, while what decides the layout is how production RESOLVES the
    # flag, and the two defaults are opposites -- a missing or null
    # `rope_interleave` is interleaved and fine, a missing or null
    # `indexer_rope_interleave` is NeoX and unservable. Hence the shared
    # resolver rather than a local reimplementation of either rule.
    if is_neox_rope_style(hf_config, "rope_interleave", default_is_neox=False):
        return f"rope_interleave={getattr(hf_config, 'rope_interleave', None)!r}"
    if is_neox_rope_style(hf_config, "indexer_rope_interleave", default_is_neox=True):
        return (
            "indexer_rope_interleave="
            f"{getattr(hf_config, 'indexer_rope_interleave', None)!r}"
        )
    return None


def _is_gfx950() -> bool:
    """True on gfx950, and on no device at all -- a contract check builds
    compile-only and never binds one, as ``_cu_count`` does for K3."""
    import torch

    if not torch.cuda.is_available():
        return True
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    return arch.startswith("gfx950")


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use GLM-5.2 mono, or None."""
    hf = atom_config.hf_config
    spec = atom_config.speculative_config
    is_mtp = spec is not None and spec.method == "mtp"
    query_length = 1 if spec is None else spec.num_speculative_tokens + 1
    tp_size = atom_config.tensor_parallel_size
    geometry = native_config(
        samples=query_length if query_length > 1 else 4,
        tp_size=tp_size,
        kv_cache_dtype=atom_config.kv_cache_dtype,
        mtp=is_mtp,
        query_length=query_length,
    )
    checks = [(envs.ATOM_MONO_ENABLE, "switched off")]
    checks += common_refusals(
        atom_config, Dims, SUPPORTED_TP, refuse_tr_hybrid=False
    ) + [
        (geometry is not None, "native geometry"),
        (getattr(hf, "model_type", None) == "glm_moe_dsa", "model type"),
        (not atom_config.enable_dp_attention, "DP attention"),
        (not is_plugin_mode(), "plugin mode"),
        (not envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV, "shuffled MLA cache"),
        (envs.ATOM_MLA_PAGE_SIZE == 1, "segmented MLA cache"),
        # the kernel is built for gfx950's 163,840 B LDS budget and the runner
        # always asks for native FP4 MFMA; another arch would be admitted here
        # and fail at compile/launch instead of falling back
        (_is_gfx950(), "not gfx950"),
    ]
    if geometry is not None:
        why = model_geometry_refusal(hf, geometry, tp_size)
        checks.append((why is None, why))
    return first_refusal(checks)


def fused_indexer_refusal(atom_config) -> str | None:
    """Why the fused TP4 indexer cannot serve this deployment, or None.

    Checked independently of ``config_refusal``: a deployment GLM-5.2 mono
    already serves (``config_refusal(atom_config) is None``) may still refuse
    the fused indexer alone, in which case the "full" layers fall back to the
    existing external-indexer mono path (``Glm52MonoRunner._refresh_indexer``
    plus ``Glm5MonoKernel(with_indexer=False)``), which this refusal must
    never disturb.

    ``max_model_len`` is checked here, statically, and this is the ONLY place
    ``FUSED_INDEX_MAX_SEQ`` is enforced. An earlier design enforced it per
    step instead (``contract.fused_indexer_step_refusal``, deleted), on the
    reasoning that the capacity is a kernel property rather than a deployment
    one and that a long deployment should still get the fused path for its
    short requests. That reasoning is sound but **unimplementable on the path
    GLM-5.2 actually serves**: ``--cudagraph-mode`` defaults to ``FULL``
    (``arg_utils.py:144,326``) and ``_get_engine_kwargs`` always passes it, so
    the whole forward is captured into one graph per ``(bs, max_seqlen_q)``
    and replayed (``model_runner.py:3208``). A per-step Python branch there
    cannot exist: reading ``positions`` on the host needs a ``.item()``, which
    raises ``hipErrorStreamCaptureUnsupported`` during capture, and even if it
    did not, capture would bake whichever branch the dummy capture inputs
    chose into every later replay.

    A request past the capacity is refused as a deployment, not handled in
    the kernel: scoring would silently truncate the context (CP off) or spin
    on a mailbox nobody publishes to (CP on). ``indexer_cp_refusal`` requires
    this refusal to have passed, so that second case is gated here and only
    here.

    The cost is real and is the deliberate trade: a deployment that does not
    cap ``--max-model-len`` at or under ``FUSED_INDEX_MAX_SEQ`` gets the
    external-indexer mono path for every request, including short ones.

    What widening the window now takes is NOT what it used to take. The
    selector is a merge tree over fixed 16384-key leaves, so one CTA serves
    any context and ``INDEX_LOCAL_LDS_CAPACITY`` must stay at 16384 -- it is
    the leaf size, not a cap. Raising ``FUSED_INDEX_MAX_SEQ`` alone deepens
    the tree (``index_levels``: 0 up to 16384, 1 to 131072, 2 to 1048576) at
    unchanged LDS, and does not turn ``TILED_INDEXER`` on, which is now the
    CP feature alone. What it does change, and what has to be re-run before
    it moves, is: the ``index_scores`` plane is still linear in
    ``index_max_seq``, so global scratch grows with it; the scoring work per
    step grows with it; and no end-to-end decode has been run at any depth
    above 0 -- selection parity on GPU and a CPU property test are all that
    cover the tree today, with no performance number of any kind.
    """
    checks = [
        (envs.ATOM_GLM52_FUSED_INDEXER, "fused indexer switched off"),
        (
            atom_config.max_model_len <= FUSED_INDEX_MAX_SEQ,
            (
                f"fused indexer capacity is {FUSED_INDEX_MAX_SEQ} tokens, but "
                f"max_model_len={atom_config.max_model_len}; cap "
                f"--max-model-len at {FUSED_INDEX_MAX_SEQ} to use it"
            ),
        ),
        (
            atom_config.index_cache_dtype == "fp8",
            (
                f"fused indexer requires index_cache_dtype=fp8, got "
                f"{atom_config.index_cache_dtype!r}"
            ),
        ),
        (
            atom_config.kv_cache_block_size == FUSED_INDEX_PAGE_SIZE,
            (
                f"fused indexer index page size must equal "
                f"kv_cache_block_size={FUSED_INDEX_PAGE_SIZE}, got "
                f"{atom_config.kv_cache_block_size}"
            ),
        ),
    ]
    return first_refusal(checks)


def indexer_cp_refusal(atom_config) -> str | None:
    """Why the fused indexer's context parallelism cannot serve this, or None.

    Checked independently of ``fused_indexer_refusal`` and strictly narrower
    than it: CP is a property of a build the fused indexer is already making,
    so a refusal here leaves the fused indexer running exactly as it does with
    CP off, and ``fused_indexer_refusal`` returning non-None makes this moot.

    The conditions come from the vendored kernel's own contract -- FlyDSL
    ``kernels/monokernel/doc/glm-indexer-cp.md``, "CP algorithm and input
    contract" -- and only the ones ATOM can actually vary are checked:

    - **CP group size.** Each rank owns a contiguous, 64-token-aligned shard of
      the query's live causal context; the group is chosen by
      ``index_cp_world(npes, samples, index_max_seq)``, which is the TP size
      except for TP8's single-query medium contexts. A group of one is not
      context parallelism, so TP1 is refused by name rather than silently
      building a CP kernel that shards nothing. (``SUPPORTED_TP`` is ``(4,)``
      today, so this is a guard against a future TP, not a live case.)
    - **``index_max_seq`` a positive multiple of 64 that accommodates every
      query.** The multiple-of-64 half is checked here against
      ``FUSED_INDEX_MAX_SEQ``; the "accommodates every query" half is already
      enforced by ``fused_indexer_refusal``'s ``max_model_len`` check, which
      this refusal requires to have passed.
    - **``dcp_size == 1``.** Redundant with ``common_refusals``' "decode CP"
      entry, which ``config_refusal`` already applies, but named here because
      the kernel contract names it: the two are independent mechanisms and a
      future relaxation of the former must not silently enable CP.

    Two further contract items are **invariants of ATOM's fused path, not
    checks**, and are recorded here rather than tested because there is no
    deployment knob that can break them:

    - *Index caches remain replicated.* ATOM allocates the index plane per
      rank from the same ``MlaKvPool`` geometry on every rank, and every rank
      writes the identical rows.
    - *Identical indexer weights and inputs on every rank.*
      ``weights.indexer_weights`` reads ``indexer.wk_weights_proj`` and
      ``indexer.wq_b`` whole, with no rank slicing, so the 32 index heads are
      replicated rather than TP-sharded; ``enable_dp_attention`` -- the one
      thing that would desynchronise the inputs -- is already refused by
      ``config_refusal``.
    """
    if not envs.ATOM_GLM52_INDEXER_CP:
        # Returned before any field of ``atom_config`` is read. With the switch
        # off this must cost nothing and inspect nothing, so that adding a
        # condition here can never make an otherwise-fine deployment fail.
        return "fused indexer CP switched off"
    fused_why = fused_indexer_refusal(atom_config)
    tp_size = atom_config.tensor_parallel_size
    checks = [
        (fused_why is None, f"fused indexer off ({fused_why})"),
        (
            index_cp_world(tp_size, 1, FUSED_INDEX_MAX_SEQ) > 1,
            f"fused indexer CP needs a CP group above one rank, got TP{tp_size}",
        ),
        (
            atom_config.decode_context_parallel_size == 1,
            (
                f"fused indexer CP requires dcp_size=1, got "
                f"{atom_config.decode_context_parallel_size}"
            ),
        ),
        (
            FUSED_INDEX_MAX_SEQ > 0 and FUSED_INDEX_MAX_SEQ % 64 == 0,
            (
                f"fused indexer CP requires index_max_seq a positive multiple "
                f"of 64, got {FUSED_INDEX_MAX_SEQ}"
            ),
        ),
    ]
    return first_refusal(checks)
