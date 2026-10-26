# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The deployments Kimi-K3 mono serves, and why any other is refused.

Every width a kernel loops over, places tasks by or lays a region out with is a
build-time constant derived from ``Dims`` (``atom.mono.plan.shard``), so a TP
size either yields every width exactly or is refused by name before anything
loads.

A refusal here is what keeps a gap between the kernels' assumptions and ATOM's
production decode from becoming a wrong answer. The gaps, and the check that
closes each one:

* **paged KV / block tables** -- the kernels run KDA layers only, whose state is
  a slot-indexed conv / recurrent pool, not a paged cache. The MLA layers keep
  the original path and their paged cache untouched (``runner`` routes per
  layer), so there is nothing to refuse: mono never reads a block table.
* **batched rows across requests** -- a step is ``requests x qlen`` rows and the
  KDA stage places one task per (request, head, v-chunk). Rows of different
  query lengths cannot share a build, so only a *uniform* spec-verify step is
  served (``step_rows``), capped at ``MAX_ROWS``.
* **MTP / spec width** -- K1 reads a per-token snapshot slot
  ``st_idx[request][token]``; ReplaySSM keeps that table's shape but leaves only
  column 0 live, so it is refused rather than read wrong.
* **TP sharding** -- ``SUPPORTED_TP``; ``Dims`` refuses any size that does not
  divide every width.
* **weight layouts** -- ``weights.py`` checks each tensor it binds.
* **KV dtype** -- the conv state must be bf16 and the recurrent state fp32.
  ``ATOM_GDN_SSM_DTYPE`` is checked here, but that is the *request*; the pools
  that were actually allocated are checked in ``runner._conv_state`` and
  ``runner._recurrent_state`` -- at build for the first mono layer, and again
  where each step packs them.
"""

from dataclasses import dataclass

from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.shard import Shard, ShardError
from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.utils import envs

# ---------------------------------------------------------------- model widths

HIDDEN = 7168
# the KDA mixer: 96 heads of 128, the hidden projections fused into one GEMM
KDA_HEADS = 96
HEAD_DIM = 128
CONV_WIDTH = 4
# the latent MoE
EXPERTS = 896
TOPK = 16
MOE_INTER = 3072  # a routed expert's intermediate width
SHARED_EXPERTS = 2
SHARED_INTER = MOE_INTER * SHARED_EXPERTS  # the shared branch's
LATENT = 3584  # routed_expert_hidden_size
# the attention-residual block: a block closes every ATTN_RES_BLOCK layers
ATTN_RES_BLOCK = 12
# the SiTUv2 activation's constants, as the checkpoint carries them
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

# the GEMV task tile the in-projection is padded up to
GEMV_ROWS = 16


@dataclass(frozen=True)
class Dims:
    """A TP rank's share of K3's widths."""

    tp: int

    def __post_init__(self):
        shard = Shard(self.tp)
        shard.split(KDA_HEADS, "KDA heads")
        shard.split(MOE_INTER, "routed intermediate")
        shard.split(SHARED_INTER, "shared intermediate")
        shard.split(HIDDEN, "latent up rows")
        shard.split(LATENT, "latent rows")
        shard.split(EXPERTS, "experts")
        if self.heads * HEAD_DIM % GEMV_ROWS:
            raise ShardError(f"a rank's projection at TP {self.tp} is not 16-row")

    @property
    def heads(self) -> int:
        """The rank's KDA heads."""
        return KDA_HEADS // self.tp

    @property
    def proj(self) -> int:
        """One of q / k / v / g on this rank."""
        return self.heads * HEAD_DIM

    @property
    def qkv(self) -> int:
        """The conv'd part of the fused projection."""
        return 3 * self.proj

    @property
    def fused(self) -> int:
        """The fused hidden projection's logical rows: q | k | v | g, then the
        two tails (f_a is one head's width, beta one value a head)."""
        return 4 * self.proj + HEAD_DIM + self.heads

    @property
    def nproj(self) -> int:
        """... rounded up to the GEMV's task tile. The rows past ``fused`` are
        zero weights: they are computed and discarded."""
        return -(-self.fused // GEMV_ROWS) * GEMV_ROWS

    @property
    def inter(self) -> int:
        """A routed expert's intermediate width on this rank."""
        return MOE_INTER // self.tp

    @property
    def sh_inter(self) -> int:
        """The shared branch's."""
        return SHARED_INTER // self.tp

    @property
    def up_n(self) -> int:
        """The latent up-projection's output rows this rank owns: the result is
        all-reduced, not gathered, so each rank holds a disjoint row band of a
        weight the model keeps replicated."""
        return HIDDEN // self.tp


# the TP sizes the kernels are built for. The widths above are baked into the
# kernels' task placement at TP8 (12 heads a rank, 384 / 768 intermediate, 896
# latent-up rows); another size compiles different kernels, which do not exist.
SUPPORTED_TP = (8,)
# Rows a step serves at most. A token's bf16 x row is staged in a CTA's LDS, so
# the row count is bounded by LDS, not by anything the model asks for: the whole
# 7168-wide row at 16 rows wants 229,632 B against gfx950's 163,840. The GEMV
# stages a *column slab* at a time instead (``kernels.attention.kda.slab_plan``,
# ``kernels.stages.gemv.slabs``), which is what would let a step be wider than
# 8. Both kernels still take one slab -- i.e. the whole row, exactly as before
# -- at every row count up to 8.
#
# **16 rows is correct and faster; the default stays at 8 anyway, pending a
# task-level accuracy run.** Three classes of 8-row assumption have been found
# and fixed:
#
#   * ``MAX_U = 8 * TOPK`` sized the expert union while ``load_route`` wrote an
#     unclamped slot -- an out-of-bounds write past 8 rows.
#   * ``kof`` indexed ``slot * 8 + t``, where the 8 was the *token* stride, so
#     slot n's entries aliased into slot n+1.
#   * Three stages that own 128 columns and give one thread to each
#     ``(token, column pair)`` -- ``attention/back.stage_reduce``,
#     ``stages/moe.stage_lnorm``, ``stages/moe.stage_reduce`` -- ran a single
#     pass over ``s * 64`` work items against ``THREADS``, which covers only
#     ``s <= 8``. Past that the block ran out of threads and those tokens were
#     never computed at all. They are strided now, as DeepSeek-V4.1's
#     equivalents already were; this was a porting omission, not a design limit.
#
# All of them evaluate to the old constants at ``s == 8``, so the 8-row build is
# byte-identical (code-object sha256 unchanged at S=1, 4 and 8) -- at ``s <= 8``
# the added loops have one constexpr iteration and ``tid``'s known range folds
# the second guard away, so the compiler deletes it.
#
# Measured at 16 rows (bs=2 x MTP7): target forward 23.393 -> 20.248 ms vs stock
# production, 318.0 -> 365.0 output tok/s. The win *shrinks* with width --
# -25.7% at 8 rows against a matched baseline, -17.4% at 16 -- because K1's GEMV
# issues the same MFMAs at both widths while the staging does not. Nothing above
# 16 is measured; do not extrapolate the trend.
#
# The default stays 8 because **no task-level accuracy run exists at any
# width**, and K2's latent MoE carries a real, unsettled precision cost
# (rel_L2 1.7e-1 .. 3.0e-1: the expert intermediate is bf16 in the reference and
# MXFP4 here). Raising the cap also arms mono for every CUDA-graph capture size
# inside it, so it is a decision to make on accuracy evidence, not on speed.
#
MAX_ROWS = 8
# tokens a request contributes at most (the conv rollback window the kernel
# tiles for)
MAX_QLEN = 16


def layer_block(layer_id: int) -> tuple[int, int]:
    """``(blocks the AttnRes mixes before this layer, the block this layer
    writes or -1)``. A block closes every ``ATTN_RES_BLOCK`` layers, so layer
    ``k`` mixes ``ceil(k / 12)`` prior blocks and layer 12 j writes block j."""
    writes = layer_id % ATTN_RES_BLOCK == 0
    return -(-layer_id // ATTN_RES_BLOCK), (
        layer_id // ATTN_RES_BLOCK if writes else -1
    )


def num_blocks(layers: int) -> int:
    """The block-residual buffer's width for a model of ``layers`` layers."""
    return -(-layers // ATTN_RES_BLOCK)


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use K3 mono decode, or None."""
    hf = getattr(atom_config.hf_config, "text_config", atom_config.hf_config)
    linear = getattr(hf, "linear_attn_config", None) or {}
    spec = atom_config.speculative_config
    checks = [(envs.ATOM_MONO_ENABLE, "switched off")]
    # `refuse_tr_hybrid` is left at its default True, which refuses
    # TR_HYBRID. That is NOT what a CLI launch gets: `EngineArgs.cudagraph_mode`
    # defaults to "FULL", so mono arms from the CLI without `--cudagraph-mode`
    # being passed. The `None -> TR_HYBRID` default in `Config.__post_init__`
    # is reached only by a config built without `arg_utils`, i.e. plugin
    # mode, which this refusal
    # then turns mono off for. `recipes/Kimi-K3.md` says the same.
    #
    # Structurally K3 could drop this refusal the way GLM-5.2 does: the
    # installer wraps the *uncompiled* `KimiK3ForConditionalGeneration` and the
    # runner drives `inner.layers` one at a time, so the compiled
    # `KimiLinearModel.forward` never runs underneath mono and no token graph
    # of it can conflict. It is kept only because K3 mono has never been
    # run under TR_HYBRID -- every measurement to date used FULL -- and arming
    # an untested path is worse than requiring one documented flag. Drop it
    # when there is a TR_HYBRID run to point at, not before.
    checks += common_refusals(atom_config, Dims, SUPPORTED_TP) + [
        # the kernels are one CTA a CU with every CTA resident: a spin wait on
        # another CTA only makes progress while that CTA runs
        (
            _cu_count() in (None, BLOCKS),
            f"{_cu_count()} CUs, the kernels need {BLOCKS}",
        ),
        # a connector's copy kernels run on another stream and can hold CUs,
        # which wedges a kernel that waits on every one of them. The field is a
        # dict that defaults to empty, so this is a truthiness test, as every
        # other reader of it in ATOM does: `is None` refuses every deployment.
        (
            not getattr(atom_config, "kv_transfer_config", None),
            "KV connector",
        ),
        # the KDA recurrence reads and writes fp32 state in place; a narrower
        # pool would be read as fp32 rather than converted
        (
            envs.ATOM_GDN_SSM_DTYPE == "fp32",
            f"ATOM_GDN_SSM_DTYPE={envs.ATOM_GDN_SSM_DTYPE}",
        ),
        # K1 resumes a request from ``st_idx[request][token]``, a snapshot slot
        # a token. ReplaySSM keeps that table's shape but leaves only column 0
        # live, rebuilding the rest from its (k, u, g) records. It defaults ON
        # whenever spec decode is used -- which mono needs -- so without the
        # override every step would be declined one at a time instead.
        (
            envs.ATOM_ENABLE_REPLAYSSM is False,
            "ReplaySSM (mono needs ATOM_ENABLE_REPLAYSSM=0)",
        ),
        # the model's own widths, which the kernels bake
        ((hidden := getattr(hf, "hidden_size", None)) == HIDDEN, f"hidden {hidden}"),
        (
            (heads := getattr(hf, "linear_num_key_heads", None)) == KDA_HEADS,
            f"{heads} KDA heads",
        ),
        (
            (dim := getattr(hf, "linear_key_head_dim", None)) == HEAD_DIM,
            f"KDA head dim {dim}",
        ),
        (
            (conv := linear.get("short_conv_kernel_size")) == CONV_WIDTH,
            f"conv width {conv}",
        ),
        # without it the merged projection puts beta where this one puts g
        (bool(linear.get("use_full_rank_gate")), "no full-rank gate"),
        # a build constant: the kernel's forget gate is lower-bounded, not free
        (linear.get("gate_lower_bound") is not None, "no KDA gate lower bound"),
        (
            (block := getattr(hf, "attn_res_block_size", None)) == ATTN_RES_BLOCK,
            f"attn_res_block_size {block}",
        ),
        ((n := getattr(hf, "num_experts", None)) == EXPERTS, f"{n} experts"),
        ((k := getattr(hf, "num_experts_per_token", None)) == TOPK, f"top-{k}"),
        (
            (li := getattr(hf, "moe_intermediate_size", None)) == MOE_INTER,
            f"routed intermediate {li}",
        ),
        (
            (sh := getattr(hf, "num_shared_experts", None)) == SHARED_EXPERTS,
            f"{sh} shared experts",
        ),
        (
            (lat := getattr(hf, "routed_expert_hidden_size", None)) == LATENT,
            f"latent hidden {lat}",
        ),
        (bool(getattr(hf, "latent_moe_use_norm", False)), "no latent MoE norm"),
        (
            float(getattr(hf, "routed_scaling_factor", 1.0)) == 1.0,
            "routed scaling factor != 1",
        ),
        (getattr(hf, "num_expert_group", 1) == 1, "expert groups"),
        # The route stage bakes the router in, it does not read these fields:
        # it scores with sigmoid and renormalizes the pre-bias sigmoids by an
        # IEEE ``1 / sum`` (``kernels/stages/moe.py``), while the production
        # path hands both to ``FusedMoE`` from the config (``model.py``).
        # Today's shipped ``moonshotai/Kimi-K3`` is exactly sigmoid +
        # renormalize, so these are LATENT refusals -- nothing is miscomputed
        # right now. They are here because a checkpoint that flipped either
        # field would be served by the wrong router with no error at all: a
        # softmax checkpoint scored as sigmoid, or unnormalized weights
        # normalized, is a silently different model, not a crash.
        (
            (act := getattr(hf, "moe_router_activation_func", None)) == "sigmoid",
            f"router activation {act!r}",
        ),
        (getattr(hf, "moe_renormalize", None) is True, "router does not renormalize"),
        (
            float(getattr(hf, "activation_situ_beta", 0.0) or 0.0) == SITU_BETA,
            "SiTU beta",
        ),
        (
            float(getattr(hf, "activation_situ_linear_beta", 0.0) or 0.0)
            == SITU_LINEAR_BETA,
            "SiTU linear beta",
        ),
        # a step is a uniform spec verify: without a drafter there is none
        (spec is not None, "no speculative decoding"),
        # ...and the contract is DSpark specifically, as DeepSeek-V4.1's mono
        # already states it (`deepseek_v41/mono/config.py`); another method
        # would be admitted here on a path never built or measured for it
        (spec is None or spec.method == "dspark", "not DSpark"),
        (
            spec is None or 0 < spec.num_speculative_tokens + 1 <= MAX_ROWS,
            f"speculative width past the {MAX_ROWS}-row buffers",
        ),
    ]
    return first_refusal(checks)


def _cu_count() -> int | None:
    """This device's CUs, or None where there is no device (a contract check
    builds compile-only and never binds one)."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
