"""ATOM_QWEN_MONO_MOE: run a Qwen3.5 sparse MoE block and the norm after it as one persistent FlyDSL kernel.

The kernel (mono_moe2.MonoMoE, all-reduce build) covers what is between two all-reduces of a TP4 decode layer:
router GEMM (+ the shared-expert gate column), top-8 softmax, MXFP4 routed experts, the bf16 shared expert, the
sigmoid-gated add, the 4-rank all-reduce, the residual add (in place) and the Gemma RMSNorm of the next layer's
input_layernorm (or the model's final norm). The block hands the norm's outputs on as `MoENormed`, which that norm
passes through.

It takes up to `TMAX` (128) rows: decode steps of at most 25 sequences at MTP4. Other steps (prefill, larger
batches) run the native chain. The verify graphs are captured at every batch size up to 32, so the rows it runs
are never padding.

ATOM_QWEN_MONO_MOE_CHECK=1: every eager call also runs the native chain on a copy of the residual and logs the
relative differences (rank 0).
"""
import ctypes
import logging
import os
from types import SimpleNamespace

import torch

from atom.model_ops.moe_tail_norm import MoENormed

logger = logging.getLogger("atom")

_CHECK = os.getenv("ATOM_QWEN_MONO_MOE_CHECK", "0") == "1"
_HIDDEN, _EXPERTS, _INTER = 2048, 256, 128


class _Peers:
    """The all-reduce inbox: cached hipMalloc memory IPC-mapped by every rank (as aiter's Gluon all-reduce)."""

    def __init__(self, dev, cpu_group, nbytes):
        from aiter.dist.device_communicators.custom_all_reduce import IPCBufferPool, _loaded_hip_runtime

        self.pool = IPCBufferPool(dev, cpu_group)
        inbox = self.pool.create("mono_moe_inbox", nbytes, raw_cached=True)
        err = _loaded_hip_runtime().hipMemset(ctypes.c_void_p(inbox.data_ptr), 0, ctypes.c_size_t(nbytes))
        assert err == 0, f"hipMemset of the mono MoE inbox failed: {err}"
        torch.cuda.synchronize()
        self.addresses = torch.tensor(self.pool.open_peer_buffers("mono_moe_inbox"), dtype=torch.int64)
        torch.distributed.barrier(group=cpu_group)


class _Runner:
    """One kernel instance a process: its flag words, scratch buffers and inbox serve every layer and graph (the
    launches run one after another on the stream)."""

    def __init__(self):
        from aiter.dist.parallel_state import get_tp_group

        from . import mono_moe2

        self.m2 = mono_moe2
        tp = get_tp_group()
        self.cpu_group = tp.cpu_group
        self.rank = tp.rank_in_group
        dev = torch.device("cuda", torch.cuda.current_device())
        self.peers = _Peers(dev, self.cpu_group, mono_moe2.PEER_BYTES)
        self.mm = mono_moe2.MonoMoE(dev, peers=self.peers, rank=self.rank)
        # the normed outputs, alternating by layer: a layer's output is read by the next layer's mixer while the
        # next MoE writes the other buffer
        self.xo = [torch.zeros(mono_moe2.TMAX, _HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(2)]
        self.compiled = False
        self.checks = 0

    def precompile(self, L, x, residual):
        """Compile both builds on every rank, then meet: a rank still compiling while its peers wait in the
        all-reduce would trip their 2 s spin deadline."""
        self.mm.precompile(L, x, res=residual, xo=self.xo[0][: x.shape[0]])
        torch.distributed.barrier(group=self.cpu_group)
        self.compiled = True
        logger.info("ATOM_QWEN_MONO_MOE: fused MoE kernels compiled (rank %d)", self.rank)


_RUNNER: _Runner | None = None


def _runner() -> _Runner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = _Runner()
    return _RUNNER


def _layer_weights(block, norm):
    L = getattr(block, "_mono_weights", None)
    if L is None:
        ex = block.experts
        L = SimpleNamespace(
            gate=block.gate.weight,
            w13=ex.w13_weight,
            s13=ex.w13_weight_scale,
            w2=ex.w2_weight,
            s2=ex.w2_weight_scale,
            sh_gu=block.shared_expert.gate_up_proj.weight,
            sh_down=block.shared_expert.down_proj.weight,
            norm_w=norm.weight,
        )
        for name, shape in (("gate", (_EXPERTS + 1, _HIDDEN)), ("sh_gu", (2 * _INTER, _HIDDEN)),
                            ("sh_down", (_HIDDEN, _INTER)), ("norm_w", (_HIDDEN,))):
            t = getattr(L, name)
            assert tuple(t.shape) == shape and t.dtype == torch.bfloat16, (name, t.shape, t.dtype)
        assert L.w13.shape[0] == _EXPERTS and L.w13.numel() * L.w13.element_size() == _EXPERTS * 2 * _INTER * _HIDDEN // 2, (
            "w13", L.w13.shape, L.w13.dtype)
        assert L.w2.numel() * L.w2.element_size() == _EXPERTS * _HIDDEN * _INTER // 2, ("w2", L.w2.shape, L.w2.dtype)
        for t in (L.gate, L.w13, L.s13, L.w2, L.s2, L.sh_gu, L.sh_down, L.norm_w):
            assert t.is_contiguous()
        object.__setattr__(block, "_mono_weights", L)
    return L


def mono_moe_forward(block, hidden_states: torch.Tensor, residual: torch.Tensor):
    """The block + its next norm as one kernel: `MoENormed(normed, residual)` (residual updated in place), or None
    when this step stays on the native chain."""
    from . import mono_moe2

    T = hidden_states.shape[0]
    if T == 0 or T > mono_moe2.TMAX or torch.compiler.is_compiling():
        return None
    from atom.utils.forward_context import get_forward_context

    context = get_forward_context().context
    if context is None or context.is_prefill:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if _RUNNER is None and capturing:
        # never set up inside a capture (IPC + compile): this graph keeps the native chain
        logger.warning("ATOM_QWEN_MONO_MOE: first call inside a graph capture; native MoE for this graph")
        return None
    run = _runner()
    norm = block.mono_norm
    L = _layer_weights(block, norm)
    x = hidden_states.contiguous()
    if not run.compiled:
        run.precompile(L, x, residual)
    xo = run.xo[block.mono_parity][:T]
    if _CHECK and not capturing:
        ref_res = residual.clone()
        ref_out, ref_res = norm(block.native_moe_forward(x), ref_res)
    run.mm(L, x, res=residual, xo=xo)
    if _CHECK and not capturing:
        def rel(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()

        run.checks += 1
        if run.rank == 0 and (run.checks <= 200 or run.checks % 1000 == 0):
            logger.info("ATOM_QWEN_MONO_MOE_CHECK layer %d T %d: out rel %.3e residual rel %.3e (call %d)",
                        block.mono_layer, T, rel(xo, ref_out), rel(residual, ref_res), run.checks)
    return MoENormed(xo, residual)


def enable_mono_moe(layers, final_norm) -> int:
    """Hand each TP4 sparse MoE block the norm that reads its output (the next layer's input_layernorm or the
    final norm). Returns how many blocks run fused."""
    from atom.models.qwen3_next import Qwen3NextSparseMoeBlock

    n = 0
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, Qwen3NextSparseMoeBlock):
            continue
        consumer = getattr(layers[i + 1], "input_layernorm", None) if i + 1 < len(layers) else final_norm
        if (
            consumer is None
            or not getattr(consumer, "fused_allreduce", False)
            or getattr(consumer, "use_fused_quant", False)
            or mlp.tp_size != 4
            or mlp.shared_expert is None
            or mlp.is_rocm_aiter_fusion_shared_expert_enabled
            or mlp.combine_tail_norm is not None
            or mlp.defer_gated_sum
            or getattr(layer, "layer_scale", False)
        ):
            continue
        # not a submodule: the norm belongs to the next layer / the model
        object.__setattr__(mlp, "mono_norm", consumer)
        mlp.mono_parity = i % 2
        mlp.mono_layer = i
        n += 1
    logger.info("ATOM_QWEN_MONO_MOE: %d MoE blocks run the fused kernel (decode, <= 128 rows)", n)
    return n
