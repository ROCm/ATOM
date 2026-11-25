# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sequence-parallel prefill for MiniMax-M3 under pure TP.

From ``SP_MIN_TOKENS`` tokens (``SP_MIN_UBATCH_TOKENS`` per TBO micro-batch) each
rank keeps ``rows`` tokens of the residual stream. Per layer the two bf16 TP
all-reduces become an all-to-all of the MXFP8 attention output (each rank applies
the whole o_proj to its rows) and a reduce-scatter of the FFN output, each
followed by an all-gather of the next layer's input; norms and quants run on the
shard. Under TBO every phase is one micro-batch hand-off on the comm stream.

Port of vLLM #63491 (``vllm/models/minimax_m3/amd/sp_prefill.py``).
"""

import logging
from dataclasses import dataclass

import torch
from torch import nn

from atom.utils import envs

logger = logging.getLogger("atom")

# TP4 isolated prefill breaks even near 4096 tokens.
SP_MIN_TOKENS = envs.ATOM_M3_SP_MIN_TOKENS
# Every micro-batch of a TBO step needs this many tokens: their hand-offs
# alternate, so both take one path; below it the phases overlap too little.
SP_MIN_UBATCH_TOKENS = envs.ATOM_M3_SP_MIN_UBATCH_TOKENS
SP_MAX_TOKENS = 65536
# The FlyDSL MoE prefill chain's smallest step (`MIN_PREFILL_TOKENS`).
_CHAIN_MIN_TOKENS = 3072
_ROW_ALIGN = 32
_FP8 = torch.float8_e4m3fn


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


class _Coll:
    """RCCL collectives of one phase on ``comm`` / ``stream``."""

    def __init__(self, comm, stream, world: int):
        self.comm = comm
        self.stream = stream
        self.world = world

    def all_gather(self, outs: list[torch.Tensor], ins: list[torch.Tensor]) -> None:
        """Each ``[rows, ...]`` shard into its ``[padded, ...]`` output."""
        self.comm.group_start()
        for i, o in zip(ins, outs):
            self.comm.all_gather(_u8(o), _u8(i.contiguous()), stream=self.stream)
        self.comm.group_end()

    def reduce_scatter(self, out: torch.Tensor, x: torch.Tensor) -> None:
        """``[padded, H]`` partial sums -> this rank's ``[rows, H]`` sums."""
        self.comm.reduce_scatter(out, x, stream=self.stream)

    def all_to_all(self, outs: list[torch.Tensor], ins: list[torch.Tensor]) -> None:
        """Row block r of each input to rank r; row block r of each output from
        rank r."""
        w = self.world
        self.comm.group_start()
        for x, o in zip(ins, outs):
            xb, ob = _u8(x), _u8(o)
            n = xb.numel() // w
            for r in range(w):
                self.comm.send(xb[r * n : (r + 1) * n], r, stream=self.stream)
                self.comm.recv(ob[r * n : (r + 1) * n], r, stream=self.stream)
        self.comm.group_end()


class _PullColl:
    """The same hand-offs as `_Coll` over aiter's pull collectives (XGMI reads of
    peer buffers). Anything that does not fit the pull buffer or its size rules
    goes to ``fallback`` (the RCCL `_Coll`), so a call never fails because of it.
    The MXFP8 attention-output all-to-all is zero copy: the quant writes into the
    pull buffer and one kernel unpacks it into the o_proj layout."""

    def __init__(self, pull, fallback: "_Coll", world: int, staged: bool = False):
        self.pull = pull
        self.fallback = fallback
        self.world = world
        # all-gather, reduce-scatter and the plain all-to-all copy their inputs
        # into the pull buffer first; that copy costs more than the pull saves
        # (measured: 1.07-1.09x RCCL), so they stay on RCCL unless asked for.
        self.staged = staged

    def _fields(self, ins, scale: int = 1):
        """Byte (offset, size) of each input staged back to back, or None."""
        plan, off = [], 0
        for x in ins:
            nbytes = x.numel() * x.element_size()
            if nbytes % (8 * scale) or not x.is_contiguous():
                return None
            plan.append((off, nbytes))
            off += nbytes
        return plan if off <= self.pull.max_bytes else None

    def _stage(self, ins, plan):
        self.pull.acquire()
        for x, (off, nbytes) in zip(ins, plan):
            self.pull.buffer(x.shape, x.dtype, off).copy_(x)

    def all_gather(self, outs: list[torch.Tensor], ins: list[torch.Tensor]) -> None:
        plan = self._fields(ins) if self.staged else None
        if plan is None:
            return self.fallback.all_gather(outs, ins)
        self._stage(ins, plan)
        self.pull.all_gather([(o, n, _u8(out)) for (o, n), out in zip(plan, outs)])

    def all_to_all(self, outs: list[torch.Tensor], ins: list[torch.Tensor]) -> None:
        plan = self._fields(ins, self.world) if self.staged else None
        if plan is None:
            return self.fallback.all_to_all(outs, ins)
        self._stage(ins, plan)
        self.pull.all_to_all([(o, n, _u8(out)) for (o, n), out in zip(plan, outs)])

    def reduce_scatter(self, out: torch.Tensor, x: torch.Tensor) -> None:
        if (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.numel() % self.world == 0
            and x.numel() * 2 <= self.pull.max_bytes
            and x.data_ptr() == self.pull.buffer(x.shape, x.dtype).data_ptr()
        ):
            # already in the pull buffer (_pull_rs_buffer): pull straight from it
            return self.pull.reduce_scatter(out, x.numel())
        if (
            not self.staged
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.numel() * 2 > self.pull.max_bytes
            or x.numel() % self.world
        ):
            return self.fallback.reduce_scatter(out, x)
        self.pull.acquire()
        self.pull.buffer(x.shape, x.dtype).copy_(x)
        self.pull.reduce_scatter(out, x.numel())

    def can_a2a_mxfp8(self, padded: int, d: int) -> bool:
        return padded * (d + d // 32) <= self.pull.max_bytes and d % 256 == 0

    def a2a_mxfp8(self, quant, x: torch.Tensor, rows: int):
        """``quant(x, out=(codes, scales))`` into the pull buffer, then the unpack
        all-to-all: (x_q [rows, w*d], x_s [rows, w*d/32]) for o_proj_full."""
        padded, d = x.shape
        codes = self.pull.buffer((padded, d), _FP8)
        scales = self.pull.buffer(
            (padded, d // 32), torch.float8_e8m0fnu, byte_offset=padded * d
        )
        self.pull.acquire()
        quant(x, out=(codes, scales))
        out_q = torch.empty((rows, self.world * d), dtype=_FP8, device=x.device)
        out_s = torch.empty(
            (rows, self.world * d // 32), dtype=torch.float8_e8m0fnu, device=x.device
        )
        self.pull.all_to_all_mxfp8(_u8(out_q), _u8(out_s), rows, d)
        return out_q, out_s


_pull_state: dict = {}


def _pull():
    """The process's `PullCollective` over the TP group, or None (not wanted, not
    available, or the group is not one it handles). Built on first use."""
    if envs.ATOM_M3_SP_COMM not in ("pull", "pull_all"):
        return None
    if "pull" in _pull_state:
        return _pull_state["pull"]
    pull = None
    try:
        from aiter.ops.triton.comms.pull_collective import PullCollective

        tp = _tp()
        if tp.world_size in (2, 4):
            hidden = 6144
            max_bytes = envs.ATOM_M3_SP_PULL_MAX_TOKENS * hidden * 2
            pull = PullCollective(
                tp.cpu_group,
                max_bytes,
                torch.device("cuda", torch.cuda.current_device()),
            )
    except Exception as exc:  # noqa: BLE001 aiter without the module, no IPC, ...
        logger.warning("SP pull collectives off, using RCCL: %r", exc)
    logger.info("MiniMax-M3 SP hand-offs: %s", "pull" if pull else "RCCL")
    _pull_state["pull"] = pull
    return pull


def _pull_rs_buffer(sp: "SeqShard", hidden: int):
    """The FFN partial's output buffer: the pull buffer itself, so the reduce-scatter
    pulls from it without a stage copy; None (plain tensor, RCCL) where pull is off,
    TBO runs, or the partial does not fit."""
    from atom.utils.tbo.ubatching import tbo_active

    pull = _pull()
    if pull is None or tbo_active():
        return None
    n = sp.padded * hidden
    if n * 2 > pull.max_bytes or n % sp.world:
        return None
    pull.acquire()
    return pull.buffer((sp.padded, hidden), torch.bfloat16)


def _tp():
    from aiter.dist.parallel_state import get_tp_group

    return get_tp_group()


def _run_phase(fn):
    """``fn(coll)`` as one phase: under TBO a hand-off on this micro-batch's
    communicator and the comm stream, else on the TP group's pynccl
    communicator and the current stream. What ``fn`` returns is what it made
    on the comm stream for later compute (`tbo_on_comm` holds it there)."""
    from atom.utils.tbo.ubatching import (
        tbo_active,
        tbo_current_ubatch_id,
        tbo_get_comm_stream,
        tbo_get_ubatch_tp_comm,
        tbo_on_comm,
    )

    world = _tp().world_size
    if tbo_active():
        comm = tbo_get_ubatch_tp_comm(tbo_current_ubatch_id())
        return tbo_on_comm(lambda: fn(_Coll(comm, tbo_get_comm_stream(), world)))
    comm = _tp().device_communicator.pynccl_comm
    coll = _Coll(comm, torch.cuda.current_stream(), world)
    pull = _pull()
    if pull is None:
        return fn(coll)
    return fn(_PullColl(pull, coll, world, staged=envs.ATOM_M3_SP_COMM == "pull_all"))


@dataclass(frozen=True)
class SeqShard:
    num_tokens: int
    rows: int
    world: int
    rank: int

    @property
    def padded(self) -> int:
        return self.rows * self.world

    def shard(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of the full ``[num_tokens, ...]`` ``x`` (zero padded)."""
        start = self.rank * self.rows
        n = min(self.rows, max(0, self.num_tokens - start))
        if n == self.rows:
            return x[start : start + n].contiguous()
        out = x.new_zeros((self.rows, *x.shape[1:]))
        if n:
            out[:n].copy_(x[start : start + n])
        return out

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """``[num_tokens, ...]`` -> ``[padded, ...]``, the padding rows zero."""
        if x.shape[0] == self.padded:
            return x
        out = x.new_empty((self.padded, *x.shape[1:]))
        out[: x.shape[0]].copy_(x)
        out[x.shape[0] :].zero_()
        return out


def _smallest_ubatch_tokens(num_tokens: int) -> int:
    """This micro-batch's token count, or its partner's when smaller: both take
    one path, so both must decide on the same number."""
    from atom.utils.tbo.ubatching import _get_current_tbo_context

    partner = _get_current_tbo_context().partner
    if partner is None or partner.forward_context is None:
        return num_tokens
    other = partner.forward_context.context.running_tokens
    return min(num_tokens, int(other))


def plan(num_tokens: int) -> SeqShard | None:
    from atom.utils.tbo.ubatching import tbo_active

    world = _tp().world_size
    if world == 1 or num_tokens == 0 or num_tokens > SP_MAX_TOKENS:
        return None
    if tbo_active():
        smallest = _smallest_ubatch_tokens(num_tokens)
        if smallest < max(SP_MIN_UBATCH_TOKENS, _CHAIN_MIN_TOKENS):
            return None
    elif num_tokens < max(SP_MIN_TOKENS, _CHAIN_MIN_TOKENS):
        return None
    rows = (num_tokens + world - 1) // world
    rows = (rows + _ROW_ALIGN - 1) // _ROW_ALIGN * _ROW_ALIGN
    return SeqShard(num_tokens, rows, world, _tp().rank_in_group)


# Model glue, called from model.py.


def wanted() -> bool:
    """Whether to build the SP weights (``o_proj_full``)."""
    from aiter.dist.parallel_state import get_pp_group

    return (
        envs.ATOM_M3_SP_PREFILL
        and _tp().world_size > 1
        and get_pp_group().world_size == 1
    )


def make_o_proj_full(attn: nn.Module, quant_config, prefix: str):
    """An unsharded copy of ``attn.o_proj`` (+3.1 GB per rank at TP4) whose
    parameters load with the sharded ones, or None where SP cannot run."""
    from atom.model_ops.linear import ReplicatedLinear

    if not wanted():
        return None
    o_proj = attn.o_proj
    full = ReplicatedLinear(
        attn.total_num_heads * attn.head_dim,
        attn.hidden_size,
        bias=False,
        quant_config=quant_config,
        prefix=prefix,
    )
    for name, p in o_proj.named_parameters(recurse=False):
        fp = getattr(full, name, None)
        loader = getattr(p, "weight_loader", None)
        if fp is None or loader is None or not hasattr(fp, "weight_loader"):
            continue

        def both(param, loaded, *args, _loader=loader, _fp=fp):
            _loader(param, loaded, *args)
            _fp.weight_loader(_fp, loaded)

        p.weight_loader = both
    return full


def _native_mxfp8(lin: nn.Module | None) -> bool:
    """``lin`` takes pre-quantized E4M3 + group32 E8M0 input (`quantize_fp8`)."""
    if lin is None:
        return False
    weight = getattr(lin, "weight", None)
    return (
        getattr(lin, "native_a8_group_rows", None) in (1, 32)
        and weight is not None
        and weight.dtype == _FP8
    )


class SharedFull(nn.Module):
    """An unsharded copy of a layer's MXFP8 shared expert (``MiniMaxM3MLP``) that
    runs on this rank's own rows (+3.1 GiB per rank at TP4 for 57 layers)."""

    def __init__(
        self,
        shared: nn.Module,
        hidden: int,
        intermediate: int,
        quant_config,
        prefix: str,
    ):
        super().__init__()
        from atom.model_ops.linear import ReplicatedLinear

        self.gate_up_proj = ReplicatedLinear(
            hidden,
            2 * intermediate,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = ReplicatedLinear(
            intermediate,
            hidden,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.swiglu_alpha = shared.swiglu_alpha
        self.swiglu_beta = shared.swiglu_beta
        self.swiglu_limit = shared.swiglu_limit
        # The TP shard's parameters load the checkpoint; the same tensors fill the
        # unsharded copy (gate then up, each ``rows`` of the 2 * intermediate).
        for src, dst in (
            (shared.gate_up_proj, self.gate_up_proj),
            (shared.down_proj, self.down_proj),
        ):
            for name, p in src.named_parameters(recurse=False):
                fp = getattr(dst, name, None)
                loader = getattr(p, "weight_loader", None)
                if (
                    fp is None
                    or loader is None
                    or not hasattr(fp, "weight_loader_process")
                ):
                    continue

                def both(param, loaded, *args, _loader=loader, _fp=fp, **kw):
                    shard_id = args[0] if args else kw.get("loaded_shard_id")
                    _loader(param, loaded, *args, **kw)
                    if shard_id is None:
                        _fp.weight_loader_process(_fp.data, loaded)
                    else:
                        rows = loaded.shape[0]
                        _fp.weight_loader_process(
                            _fp.data.narrow(0, shard_id * rows, rows), loaded
                        )

                p.weight_loader = both

    def forward(self, x: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        from atom.model_ops.swiglu_oai import swiglu_oai_split

        gate_up = self.gate_up_proj(x, x_scale=x_scale)
        h = swiglu_oai_split(
            gate_up,
            alpha=self.swiglu_alpha,
            beta=self.swiglu_beta,
            limit=self.swiglu_limit,
        )
        return self.down_proj(h)


def make_shared_full(shared: nn.Module | None, config, quant_config, prefix: str):
    """``SharedFull`` of ``shared`` where owner-local shared experts can run, else None."""
    if shared is None or not wanted() or not envs.ATOM_M3_SP_SHARED_LOCAL:
        return None
    if getattr(shared.gate_up_proj, "output_sizes", None) != getattr(
        shared.gate_up_proj, "_ckpt_output_sizes", None
    ) or getattr(shared.down_proj, "_k_padded", False):
        return None  # a padded shard cannot be copied into the unsharded layout
    return SharedFull(
        shared,
        config.hidden_size,
        config.intermediate_size * config.n_shared_experts,
        quant_config,
        prefix,
    )


def _layer_supports_sp(layer: nn.Module) -> bool:
    attn = layer.self_attn
    linears = [attn.qkv_proj, attn.o_proj, getattr(attn, "o_proj_full", None)]
    if not layer.is_moe_layer:
        linears.append(layer.mlp.gate_up_proj)
    return (
        all(_native_mxfp8(lin) for lin in linears)
        and not layer._forward_pre_hooks
        and not getattr(attn.attn, "per_token_fp8_out", False)
    )


def plan_model(model: nn.Module, num_tokens: int) -> SeqShard | None:
    """The token sharding of a sequence-parallel forward of ``model``, or None."""
    from atom.utils.forward_context import get_forward_context

    if not envs.ATOM_M3_SP_PREFILL or getattr(model, "_sp_supported", True) is False:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    ctx = get_forward_context()
    if ctx is None or ctx.context is None or not ctx.context.is_prefill:
        return None
    sp = plan(num_tokens)
    if sp is None:
        return None
    if not hasattr(model, "_sp_supported"):
        layers = model.layers[model.start_layer : model.end_layer]
        model._sp_supported = wanted() and all(_layer_supports_sp(x) for x in layers)
        n_chain = 0
        for x in layers:
            x._sp_flydsl_moe = x.is_moe_layer and _flydsl_moe_ok(x.block_sparse_moe)
            n_chain += x._sp_flydsl_moe
        logger.info(
            "MiniMax-M3 sequence-parallel prefill (from %d tokens, TBO micro-batches "
            "from %d): %s; FlyDSL MXFP4 MoE chain on %d of %d MoE layers",
            SP_MIN_TOKENS,
            SP_MIN_UBATCH_TOKENS,
            "on" if model._sp_supported else "off",
            n_chain,
            sum(x.is_moe_layer for x in layers),
        )
    return sp if model._sp_supported else None


class GatheredLayerOutput(tuple):
    """``(full,)``: a layer's post-layer hidden state, gathered by SP, as its
    forward hooks receive it (`MiniMaxM3DecoderLayer.aux_hidden_state`)."""


def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from atom.model_ops.blockscale import quantize_fp8

    return quantize_fp8(x)


def _input_phase(sp: SeqShard, layer, embeddings: torch.Tensor):
    """First input norm + MXFP8 quant on this rank's rows -> gathered codes,
    scales, residual."""
    hidden = embeddings.shape[1]
    dev = embeddings.device
    codes = torch.empty((sp.padded, hidden), dtype=_FP8, device=dev)
    scales = torch.empty(
        (sp.padded, hidden // 32), dtype=torch.float8_e8m0fnu, device=dev
    )

    def run(coll):
        r = sp.shard(embeddings)
        x = layer.input_layernorm.forward_cuda(r)
        x_q, x_s = _quant(x)
        coll.all_gather([codes, scales], [x_q, x_s])
        return r

    residual = _run_phase(run)
    return codes, scales, residual


def _o_proj_all_to_all(attn, attn_output: torch.Tensor, sp: SeqShard, coll):
    """This rank's heads of every token -> the whole o_proj of this rank's rows,
    through an all-to-all of the MXFP8 codes + scales."""
    from atom.models.minimax_m3.sp_layout import a2a_unpack

    d = attn_output.shape[1]
    w, rows = sp.world, sp.rows
    if isinstance(coll, _PullColl) and coll.can_a2a_mxfp8(sp.padded, d):
        from atom.model_ops.blockscale import quantize_fp8

        x_q, x_s = coll.a2a_mxfp8(quantize_fp8, sp.pad(attn_output), rows)
        return attn.o_proj_full(x_q, x_scale=x_s)
    send_q, send_s = _quant(sp.pad(attn_output))
    recv_q = torch.empty_like(send_q)
    recv_s = torch.empty_like(send_s)
    coll.all_to_all([recv_q, recv_s], [send_q, send_s])
    # block r: rank r's heads of this rank's rows -> [rows, w * d], heads in rank order
    x_q, x_s = a2a_unpack(recv_q, recv_s, w, rows)
    return attn.o_proj_full(x_q, x_scale=x_s)


def _expert_tables(moe):
    """(w13, w2, w13 scale, w2 scale) of the routed experts, aiter-preshuffled,
    and the factor their output still needs (FHMoE folds it into the weights)."""
    experts = moe.experts
    if moe.fhmoe is not None:
        return moe.fhmoe.routed_tables, 1.0
    tables = (
        experts.w13_weight,
        experts.w2_weight,
        experts.w13_weight_scale,
        experts.w2_weight_scale,
    )
    return tables, moe.routed_scaling_factor


def _flydsl_moe_ok(moe) -> bool:
    """``moe`` runs as the FlyDSL MXFP4 chain on gathered MXFP4 codes + routing:
    single-group MXFP4 experts with the OAI SwiGLU (1.702 / 7.0), no biases or
    pads, the shared expert an unfused MXFP8 MLP (or none)."""
    from aiter import QuantType, dtypes

    from atom.model_ops.minimax_m3.flydsl_moe.moe_mxfp4.prefill import (
        supports_shapes,
    )

    experts = moe.experts
    method = experts.quant_method
    shared = moe.shared_experts
    checks = [
        (envs.ATOM_M3_SP_FLYDSL_MOE, "ATOM_M3_SP_FLYDSL_MOE=0"),
        (not moe.fuse_shared_experts, "shared expert fused into the MXFP4 tables"),
        (
            getattr(method, "quant_type", None) == QuantType.per_1x32,
            f"quant {getattr(method, 'quant_type', None)}",
        ),
        (not getattr(method, "hidden_pad", 0), "hidden pad"),
        (not getattr(method, "intermediate_pad", 0), "intermediate pad"),
        (getattr(experts, "w13_bias", None) is None, "w13 bias"),
        (getattr(experts, "w2_bias", None) is None, "w2 bias"),
        (getattr(experts, "expert_map", None) is None, "expert map"),
        (getattr(experts, "ep_size", 1) == 1, "expert parallel"),
        (
            getattr(experts, "swiglu_limit", None) == 7.0,
            f"swiglu limit {getattr(experts, 'swiglu_limit', None)}",
        ),
        (
            getattr(experts, "custom_routing_function", None) is None,
            "custom routing",
        ),
        (
            not getattr(experts, "apply_router_weight_on_input", False),
            "router weight on input",
        ),
        (
            shared is None or _native_mxfp8(shared.gate_up_proj),
            "shared expert not native MXFP8",
        ),
    ]
    why = next((w for ok, w in checks if not ok), None)
    if why is None:
        (w13, w2, _, _), _ = _expert_tables(moe)
        routed = moe.gate.weight.shape[0]
        if w13.dtype not in (dtypes.fp4x2, torch.uint8):
            why = f"expert weights {w13.dtype}"
        elif w13.shape[0] != routed:
            why = f"{w13.shape[0]} expert rows for {routed} routed experts"
        elif not supports_shapes(w2.shape[1], w2.shape[2] * 2):
            why = f"shape H {w2.shape[1]} / I {w2.shape[2] * 2}"
    if why is not None:
        logger.info("SP FlyDSL MoE chain off for this layer: %s", why)
    return why is None


def _route(moe, x: torch.Tensor) -> torch.Tensor:
    """Top-k of this rank's rows, packed ``[ids | weights as int32]``."""
    experts = moe.experts
    k = experts.top_k
    weights, ids = experts.quant_method.select_experts(experts, x, moe.gate(x))
    return torch.cat(
        [ids[:, :k].to(torch.int32), weights[:, :k].float().view(torch.int32)], dim=1
    )


@dataclass
class FfnInput:
    """A layer's gathered FFN input: MXFP8 codes + scales (dense MLP, shared
    expert), bf16 rows (MoE without the chain, or the check), MXFP4 codes +
    scales and routing (the chain)."""

    codes: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    rows: torch.Tensor | None = None
    fp4: tuple[torch.Tensor, torch.Tensor] | None = None
    routing: torch.Tensor | None = None
    # SHARED_LOCAL: this rank's own rows' MXFP8 codes + scales (not gathered)
    local: tuple[torch.Tensor, torch.Tensor] | None = None


def _shared_local(moe, chain: bool) -> bool:
    """The shared expert runs on this rank's own rows with its full weights, so
    the MXFP8 rows are not gathered (only the MXFP4 rows + routing are)."""
    from atom.utils.tbo.ubatching import tbo_active

    return bool(
        chain
        and getattr(moe, "shared_full", None) is not None
        and not envs.ATOM_M3_SP_MOE_CHECK
        and not tbo_active()
    )


_side_stream: torch.cuda.Stream | None = None


def _owner_rows(sp: SeqShard) -> tuple[int, int]:
    """This rank's first row of the padded layout and how many of its rows are real."""
    start = sp.rank * sp.rows
    return start, min(sp.rows, max(0, sp.num_tokens - start))


def _shared_owner_rows_side(sp: SeqShard, moe, inp: FfnInput, out: torch.Tensor):
    """Owner-local shared expert on a side stream next to the routed chain: the full
    shared MLP of this rank's own rows, added to those rows of the TP partial
    ``out``. Returns the callback that waits for and adds the result."""
    start, n = _owner_rows(sp)
    if n == 0:
        return lambda: None
    codes, scales = inp.local
    cur = torch.cuda.current_stream()
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    ready = torch.cuda.Event()
    ready.record(cur)
    done = torch.cuda.Event()
    with torch.cuda.stream(_side_stream):
        _side_stream.wait_event(ready)
        shared = moe.shared_full(codes[:n], scales[:n])
        codes.record_stream(_side_stream)
        scales.record_stream(_side_stream)
        done.record(_side_stream)

    def finish():
        cur.wait_event(done)
        shared.record_stream(cur)
        out[start : start + n] += shared

    return finish


def _post_attention_phase(sp: SeqShard, layer, attn_output, residual):
    """o_proj, post-attention norm and the FFN input (+ routing) on this rank's
    rows -> the gathered `FfnInput`, residual."""
    hidden = residual.shape[1]
    dev = residual.device
    moe = layer.block_sparse_moe if layer.is_moe_layer else None
    chain = moe is not None and layer._sp_flydsl_moe
    local = _shared_local(moe, chain)
    need_fp8 = moe is None or (chain and moe.shared_experts is not None and not local)
    need_rows = (moe is not None and not chain) or (chain and envs.ATOM_M3_SP_MOE_CHECK)
    inp = FfnInput()
    if need_fp8:
        inp.codes = torch.empty((sp.padded, hidden), dtype=_FP8, device=dev)
        inp.scales = torch.empty(
            (sp.padded, hidden // 32), dtype=torch.float8_e8m0fnu, device=dev
        )
    if need_rows:
        inp.rows = torch.empty((sp.padded, hidden), dtype=residual.dtype, device=dev)
    if chain:
        inp.fp4 = (
            torch.empty((sp.padded, hidden // 2), dtype=torch.uint8, device=dev),
            torch.empty((sp.padded, hidden // 32), dtype=torch.uint8, device=dev),
        )
        k = moe.experts.top_k
        inp.routing = torch.empty((sp.padded, 2 * k), dtype=torch.int32, device=dev)
    norm = layer.post_attention_layernorm

    def run(coll):
        from atom.model_ops.minimax_m3.flydsl_moe.moe_mxfp4.prefill import (
            mxfp4_quant,
        )

        h = _o_proj_all_to_all(layer.self_attn, attn_output, sp, coll)
        x, r = norm.forward_cuda(h, residual)
        outs, ins = [], []
        if need_fp8:
            outs += [inp.codes, inp.scales]
            ins += list(_quant(x))
        elif local:
            inp.local = _quant(x)
        if need_rows:
            outs.append(inp.rows)
            ins.append(x)
        if chain:
            outs += [*inp.fp4, inp.routing]
            ins += [*mxfp4_quant(x, None, None), _route(moe, x)]
        coll.all_gather(outs, ins)
        return r

    residual = _run_phase(run)
    return inp, residual


_CHAIN_WARM_UP_TOKENS = (4096, 16384, 32768)
_chain_warmed = False


def _moe_chain(sp: SeqShard, moe, inp: FfnInput) -> torch.Tensor:
    """The FlyDSL MXFP4 MoE (+ shared MLP) of every token -> its TP partial."""
    from atom.model_ops.minimax_m3.flydsl_moe.moe_mxfp4.prefill import (
        MIN_PREFILL_TOKENS,
        a4w4_prefill_moe,
    )

    global _chain_warmed
    t = sp.num_tokens
    (w13, w2, w13_s, w2_s), scale = _expert_tables(moe)
    k = inp.routing.shape[1] // 2
    q, s = inp.fp4
    assert t >= MIN_PREFILL_TOKENS, t

    def chain(n: int, out: torch.Tensor | None, **fuse) -> torch.Tensor:
        return a4w4_prefill_moe(
            (q[:n], s[:n]),
            w13,
            w13_s,
            w2,
            w2_s,
            inp.routing[:n, k:].view(torch.float32),
            inp.routing[:n, :k],
            hidden_size=w2.shape[1],
            intermediate_size=w2.shape[2] * 2,
            num_experts=w13.shape[0],
            out=out,
            **fuse,
        )

    if not _chain_warmed:
        _chain_warmed = True
        # the one-row ``add`` builds the same (scale + add) reduce the steps run
        dummy = torch.zeros((1, w2.shape[1]), dtype=torch.bfloat16, device=q.device)
        for n in _CHAIN_WARM_UP_TOKENS:
            if n <= sp.padded and n != t:
                chain(n, None, scale=scale, add=dummy)
    out = _pull_rs_buffer(sp, w2.shape[1])
    if out is None:
        out = torch.empty(
            (sp.padded, w2.shape[1]), dtype=torch.bfloat16, device=q.device
        )
    # The routed scale and the shared expert's output ride on the chain's final
    # reduce (``reduce * scale + shared``) rather than two passes over ``out``.
    finish = None
    shared, row0 = None, 0
    if inp.local is not None:
        if envs.ATOM_M3_SP_SHARED_LOCAL_SIDE:
            # launched first: it overlaps the routed chain on a side stream
            finish = _shared_owner_rows_side(sp, moe, inp, out)
        else:
            row0, n = _owner_rows(sp)
            if n > 0:
                codes, scales = inp.local
                shared = moe.shared_full(codes[:n], scales[:n])
    elif moe.shared_experts is not None:
        shared = moe.shared_experts(inp.codes[:t], x_scale=inp.scales[:t])
    chain(t, out[:t], scale=scale, add=shared, add_row0=row0)
    out[t:].zero_()
    if finish is not None:
        finish()
    return out


_checked: set[int] = set()


def _check_moe(lid: int, moe, inp: FfnInput, out: torch.Tensor, t: int) -> None:
    if lid in _checked:
        return
    _checked.add(lid)
    ref = moe(inp.rows[:t]).float().flatten()
    got = out[:t].float().flatten()
    cos = float((ref @ got) / (ref.norm() * got.norm() + 1e-12))
    logger.info(
        "SP FlyDSL MoE check layer %d (%d tokens): cos %.6f vs the regular MoE",
        lid,
        t,
        cos,
    )


def _ffn(sp: SeqShard, layer, inp: FfnInput, lid: int) -> torch.Tensor:
    """The MoE / dense MLP of every token -> its TP partial ``[padded, H]``."""
    t = sp.num_tokens
    if not layer.is_moe_layer:
        return sp.pad(layer.mlp(inp.codes[:t], x_scale=inp.scales[:t]))
    moe = layer.block_sparse_moe
    if not layer._sp_flydsl_moe:
        return sp.pad(moe(inp.rows[:t]))
    out = _moe_chain(sp, moe, inp)
    if inp.rows is not None:
        _check_moe(lid, moe, inp, out, t)
    return out


def _ffn_out_phase(sp: SeqShard, model, partial, residual, nxt, want_full):
    """Reduce-scatter of the FFN partial, then ``nxt``'s input norm + quant (the
    final norm after the last layer) on this rank's rows and the gather; with
    ``want_full`` the post-layer hidden state is gathered too."""
    hidden = partial.shape[1]
    dev = partial.device
    full = (
        torch.empty((sp.padded, hidden), dtype=partial.dtype, device=dev)
        if want_full
        else None
    )
    scales = None
    if nxt is not None:
        codes = torch.empty((sp.padded, hidden), dtype=_FP8, device=dev)
        scales = torch.empty(
            (sp.padded, hidden // 32), dtype=torch.float8_e8m0fnu, device=dev
        )
    else:
        codes = torch.empty((sp.padded, hidden), dtype=partial.dtype, device=dev)

    def run(coll):
        h = torch.empty((sp.rows, hidden), dtype=partial.dtype, device=dev)
        coll.reduce_scatter(h, partial)
        if nxt is not None:
            x, r = nxt.input_layernorm.forward_cuda(h, residual)
            x_q, x_s = _quant(x)
            outs, ins = [codes, scales], [x_q, x_s]
        else:
            normed, r = model.norm.forward_cuda(h, residual)
            outs, ins = [codes], [normed]
        if full is not None:
            outs.append(full)
            ins.append(r)
        coll.all_gather(outs, ins)
        return r

    residual = _run_phase(run)
    return codes, scales, residual, full


def _fire_output_hooks(layer: nn.Module, positions, full: torch.Tensor) -> None:
    out = GatheredLayerOutput((full,))
    for hook in list(layer._forward_hooks.values()):
        hook(layer, (positions,), out)


def forward(model, positions: torch.Tensor, hidden_states: torch.Tensor, sp: SeqShard):
    """Sequence-parallel forward of ``MiniMaxM3Model`` from the embeddings to the
    final norm."""
    layers = model.layers[model.start_layer : model.end_layer]
    t = sp.num_tokens
    aux_layers = model.aux_hidden_state_layers
    aux_hidden_states: list[torch.Tensor] = []
    if model.start_layer in aux_layers:
        aux_hidden_states.append(hidden_states.clone())
    codes, scales, residual = _input_phase(sp, layers[0], hidden_states)
    for idx, layer in enumerate(layers):
        lid = model.start_layer + idx
        attn = layer.self_attn
        qkv = attn.qkv_proj(codes[:t], x_scale=scales[:t])
        attn_output = attn.attend(positions, qkv).reshape(t, -1)
        inp, residual = _post_attention_phase(sp, layer, attn_output, residual)
        partial = _ffn(sp, layer, inp, lid)
        nxt = layers[idx + 1] if idx + 1 < len(layers) else None
        tap = bool(layer._forward_hooks)
        eagle = nxt is not None and lid + 1 in aux_layers
        codes, scales, residual, full = _ffn_out_phase(
            sp, model, partial, residual, nxt, tap or eagle
        )
        if full is not None:
            full = full[:t]
            if tap:
                _fire_output_hooks(layer, positions, full)
            if eagle:
                aux_hidden_states.append(full)
    hidden = codes[:t]
    if aux_hidden_states:
        return hidden, aux_hidden_states
    return hidden
