# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 mono kernels of a Kimi-K3 DSpark draft layer's dense MLP and
o_proj: production's bits in fewer launches.

``ATOM_KIMI_MONO_DRAFT_OPROJ_AR``: the attention's o_proj -- its activation
quant, its GEMM and its TP all-reduce -- as one launch
(``draft_proj_ar_gfx1250``). ``ATOM_KIMI_MONO_DRAFT_MLP``: the MLP's
down_proj likewise, behind production's gate_up_proj and SiLU * mul, or
behind ``draft_gate_up_gfx1250`` (both in one launch) at the widths
``ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS`` gives it. Steps wider than
``ATOM_KIMI_MONO_DRAFT_AR_ROWS`` keep production's projections: every rank
pushes its whole partial to every rank in one hop, which costs more than
production's two-shot all-reduce once the partial is large.

The projection kernel waits in-kernel on every TP rank, so every decision
that picks it is one all ranks take alike: a layer's binding and the peers'
banks are voted on over the TP group's CPU group, a width's kernels are
built by every rank or by none (``WidthBuilds``), and a step's refusals --
a dummy or prefill step, an input the kernels do not read, a width first
seen while a CUDA graph is being captured -- depend only on what every rank
sees the same. A refused layer, width or step runs production's kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.utils import envs

HIDDEN = 7168
GROUPS_X = HIDDEN // 128
MAX_ROWS = 64
KINDS = ("o_proj", "down", "gate_up")


def _logger():
    import logging

    return logging.getLogger("atom")


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


@dataclass(frozen=True)
class DraftProj:
    """A row-parallel projection as ``draft_proj_ar_gfx1250`` reads it, in
    place: the rank's preshuffled FP8 ``[HIDDEN, k]`` and its E8M0 block
    scales."""

    prefix: str
    w: torch.Tensor
    s: torch.Tensor
    k: int


@dataclass(frozen=True)
class DraftMlp:
    """A draft MLP's gate_up_proj (the rank's ``[2 inter, HIDDEN]`` FP8,
    gate rows first) and its down_proj."""

    prefix: str
    w_gu: torch.Tensor
    s_gu: torch.Tensor
    inter: int
    down: DraftProj


def _e8m0_fp8_gemm(linear, name: str) -> None:
    """Refused unless ``linear`` runs the preshuffled per_1x128 E8M0 FP8
    GEMM on an activation it quantizes itself, with no bias and no pad."""
    from aiter import QuantType, dtypes

    _need(getattr(linear, "bias", None) is None, f"{name} has a bias")
    _need(
        getattr(linear, "native_a8_group_rows", None) is None
        and linear.quant_type.value == QuantType.per_1x128.value
        and bool(getattr(linear, "blockscale_e8m0_scale", False))
        and envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE,
        f"{name} is not the preshuffled per_1x128 E8M0 FP8 GEMM",
    )
    _need(
        getattr(linear, "params_dtype", None) == dtypes.fp8,
        f"{name}: activations as {getattr(linear, 'params_dtype', None)}",
    )
    _need(
        getattr(linear, "input_scale", None) is None,
        f"{name} has a static activation scale",
    )
    _need(
        not getattr(linear, "is_output_padded", False)
        and not getattr(linear, "_k_padded", False),
        f"{name} is padded",
    )


def bind_proj(linear, name: str, tp: int) -> DraftProj:
    """A row-parallel projection's tensors. Refused where production would
    compute anything but dynamic per-1x128 E8M0 quant, the preshuffled FP8
    GEMM and the TP all-reduce: another quant scheme or layout, a bias, a
    static activation scale, a padded K or N, a linear that does not reduce
    its output, a shard width the kernel does not take."""
    from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
    from atom.models.kimi_k3.mono.weights import _block_fp8, _param

    _e8m0_fp8_gemm(linear, name)
    _need(
        getattr(linear, "tp_dim", None) == 1
        and bool(getattr(linear, "reduce_results", False)),
        f"{name} is not a row-parallel linear reducing its output",
    )
    k = int(_param(linear, name).shape[1])
    w, s = _block_fp8(linear, name, (HIDDEN, k))
    why = lay.refusal(1, k, tp)
    _need(why is None, f"{name}: {why}")
    return DraftProj(name, w, s, k)


def bind_mlp(mlp, tp: int) -> DraftMlp:
    """A draft MLP's tensors. Refused where the post-attention norm does not
    hand gate_up_proj the (E4M3, transposed per-1x128 E8M0) pair, where
    gate_up_proj or down_proj is not the kernels' GEMM, or where the
    activation is not the plain SiLU * mul."""
    from aiter import QuantType, dtypes

    from atom.model_ops.activation import SiluAndMul
    from atom.models.kimi_k3.mono.weights import _block_fp8, _param

    name = mlp.prefix
    norm = getattr(mlp, "_input_norm", None)
    _need(norm is not None, f"{name}: no input norm bound")
    _need(
        norm.use_fused_quant
        and not norm.fused_allreduce
        and norm.quant_type.value == QuantType.per_1x128.value
        and norm.params_dtype == dtypes.fp8
        and norm._aiter_transpose_scale
        and norm._aiter_e8m0_scale,
        f"{name}: the input norm does not hand gate_up_proj its E8M0 FP8 row",
    )
    gate_up = mlp.gate_up_proj
    _e8m0_fp8_gemm(gate_up, f"{name} gate_up_proj")
    rows = int(_param(gate_up, f"{name} gate_up_proj").shape[0])
    _need(rows % 256 == 0, f"{name} gate_up_proj has {rows} rows")
    inter = rows // 2
    w_gu, s_gu = _block_fp8(gate_up, f"{name} gate_up_proj", (rows, HIDDEN))
    act = mlp.act_fn
    _need(
        type(act) is SiluAndMul and not act.fused_quant,
        f"{name}: an activation other than SiLU * mul",
    )
    down = bind_proj(mlp.down_proj, f"{name} down_proj", tp)
    _need(down.k == inter, f"{name} down_proj takes {down.k}, not {inter}")
    return DraftMlp(name, w_gu, s_gu, inter, down)


def _hazards(launcher) -> tuple[str, ...]:
    from atom.models.kimi_k3.mono.runner_gfx1250 import _resident_hazards

    return _resident_hazards(launcher)


class DraftMlpRunner:
    """A device's draft kernels, a build a (kind, width), and the banks and
    epoch counters the projection kernels share: the draft's layers run one
    after another on one stream, every rank the same launches in the same
    order."""

    def __init__(self, device: torch.device, tp: int, rank: int, group) -> None:
        from atom.mono.runtime.widths import WidthBuilds

        self.device, self.tp, self.rank, self.group = device, tp, rank, group
        self.peers = None
        self._mapped = False
        # the layers' shard widths: one build serves every layer
        self.k_oproj: int | None = None
        self.inter: int | None = None
        self.kernels: dict = {}
        self.refused: set = set()
        # gate_up's split-K exchange, allocated with its first build
        self.partials = None
        self.widths = WidthBuilds(
            self._build,
            lambda built: [(built.launcher, built.abi)],
            group,
            "Kimi-K3 DSpark draft MLP",
            hazards=_hazards,
        )

    def _refuse(self, why: str) -> None:
        if why not in self.refused:
            self.refused.add(why)
            _logger().warning(
                "Kimi-K3 DSpark draft MLP: production kernels for %s", why
            )

    def banks(self):
        """The projection kernels' receive banks on every rank and their
        epoch counters, mapped once (collective past one rank); None where
        the TP group has no symm_mem transport."""
        if not self._mapped:
            self._mapped = True
            from atom.models.kimi_k3.mono.kernels.stages import (
                draft_mlp_layout as lay,
            )
            from atom.models.kimi_k3.mono.symm_peers import SymmPeers, map_symm_peers

            nbytes = lay.peer_bytes(self.tp, MAX_ROWS)
            if self.tp == 1:
                buffer = torch.zeros(nbytes, dtype=torch.uint8, device=self.device)
                self.peers = SymmPeers(
                    buffer,
                    torch.tensor(
                        [buffer.data_ptr()], dtype=torch.int64, device=self.device
                    ),
                    torch.zeros(
                        lay.EPOCH_COUNTERS, dtype=torch.int32, device=self.device
                    ),
                )
            else:
                self.peers = map_symm_peers(
                    nbytes, lay.EPOCH_COUNTERS, self.group, self.tp, self.device
                )
        return self.peers

    def _build(self, key: int):
        """The (kind, width) kernel: the projection where production's GEMM
        sums K in one chain, gate_up at the split the dispatch picks."""
        from atom.models.kimi_k3.mono import draft_dispatch
        from atom.models.kimi_k3.mono.kernels.stages import (
            draft_gate_up_gfx1250 as gk,
        )
        from atom.models.kimi_k3.mono.kernels.stages import draft_proj_ar_gfx1250 as dp

        kind, rows = KINDS[key // 1000], key % 1000
        if kind == "gate_up":
            from atom.models.kimi_k3.mono.kernels.stages import (
                draft_mlp_layout as lay,
            )

            split = draft_dispatch.mxfp8_split_k(
                rows, 2 * self.inter, HIDDEN, "gate_up_proj"
            )
            kernel = gk.build_draft_gate_up(rows, self.inter, split)
            if self.partials is None:
                self.partials = torch.empty(
                    lay.gate_up_partial_bytes(self.inter),
                    dtype=torch.uint8,
                    device=self.device,
                )
            return kernel
        k = self.k_oproj if kind == "o_proj" else self.inter
        split = draft_dispatch.mxfp8_split_k(rows, HIDDEN, k, kind)
        _need(split == 1, f"{rows} rows: {kind}'s GEMM splits K in {split}")
        from atom.utils.poll_isa import bound_build

        # a build whose polls issue their loads together is read as compiled
        # and kept only if it re-reads every line
        return bound_build(
            lambda together: dp.build_draft_proj_ar(
                rows, self.tp, k, poll_together=together
            ),
            f"draft {kind}",
            _logger(),
        )

    def kernel(self, kind: str, rows: int):
        """The (kind, width) kernel every rank holds, or None: a width the
        ranks refused, one past ``ATOM_KIMI_MONO_DRAFT_AR_ROWS`` (the
        projections) or ``ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS`` (gate_up), or
        one first seen while a CUDA graph is captured (None then, until an
        eager step builds it)."""
        key = (kind, rows)
        if key in self.kernels:
            return self.kernels[key]
        widest = (
            envs.ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS
            if kind == "gate_up"
            else envs.ATOM_KIMI_MONO_DRAFT_AR_ROWS
        )
        if rows > widest:
            # a width the switches leave to production: the same on every
            # rank, so no vote
            self.kernels[key] = None
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        width = rows + 1000 * KINDS.index(kind)
        built = self.widths[width] if self.widths.prepare(width) else None
        self.kernels[key] = built
        _logger().info(
            "Kimi-K3 DSpark draft MLP: %s at %d rows on %s",
            kind,
            rows,
            "production's kernels" if built is None else built.name,
        )
        return built

    def launch_proj(self, kernel, layer: DraftProj, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.size(0), HIDDEN), dtype=torch.bfloat16, device=x.device)
        p = torch.Tensor.data_ptr
        args = kernel.abi.pack({
            "x": p(x), "w": p(layer.w), "w_scale": p(layer.s), "out": p(out),
            "peers": p(self.peers.peers), "epochs": p(self.peers.epochs),
            "rank": self.rank,
        })  # fmt: skip
        kernel.launcher(*args, stream=torch.cuda.current_stream())
        return out

    def launch_gate_up(self, kernel, layer: DraftMlp, x8, x_scale) -> torch.Tensor:
        act = torch.empty(
            (x8.size(0), layer.inter), dtype=torch.bfloat16, device=x8.device
        )
        p = torch.Tensor.data_ptr
        args = kernel.abi.pack({
            "x": p(x8), "x_scale": p(x_scale), "w": p(layer.w_gu),
            "w_scale": p(layer.s_gu), "act": p(act), "partials": p(self.partials),
            "gu": 0,
        })  # fmt: skip
        kernel.launcher(*args, stream=torch.cuda.current_stream())
        return act


_RUNNERS: dict = {}


def runner_for(device: torch.device) -> DraftMlpRunner:
    key = (device.type, device.index)
    if key not in _RUNNERS:
        from aiter.dist.parallel_state import get_tp_group

        group = get_tp_group()
        tp = int(group.world_size)
        _RUNNERS[key] = DraftMlpRunner(
            device, tp, int(group.rank_in_group), group.cpu_group if tp > 1 else None
        )
    return _RUNNERS[key]


def _bind(module, slot: str, switch: str, device, local_bind, what: str):
    """``module``'s binding (``slot``), made on its first eager step: every
    rank binds and maps the banks or none does. A refusal turns the
    module's attempts (``switch``) off. Collective: every rank reaches it
    at the same call."""
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        runner = runner_for(device)
        found = {}

        def bind():
            found["layer"] = local_bind(runner)

        bind_agreed(bind, runner.group)
        bind_agreed(
            lambda: _need(
                runner.banks() is not None,
                "no symm_mem banks on the TP group (AITER_CUSTOM_AR_USE_SYMM_MEM)",
            ),
            runner.group,
        )
    except MonoUnsupported as why:
        setattr(module, switch, False)
        _logger().warning(
            "Kimi-K3 DSpark draft MLP: %s keeps the production kernels (%s)", what, why
        )
        return None
    setattr(module, slot, found["layer"])
    return found["layer"]


def _decode_step() -> bool:
    from atom.utils.forward_context import get_forward_context

    context = get_forward_context().context
    return not (context.is_dummy_run or context.is_prefill)


def proj_step_refusal(layer: DraftProj, x) -> str | None:
    """Why this step's projection is not the kernel's (None: it is)."""
    if not _decode_step():
        return "not a decode step"
    if (
        not isinstance(x, torch.Tensor)
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or x.size(1) != layer.k
        or not x.is_contiguous()
    ):
        return (
            f"{layer.prefix}: an activation of {getattr(x, 'dtype', None)} "
            f"{tuple(getattr(x, 'shape', ()))}"
        )
    if not 1 <= x.size(0) <= MAX_ROWS:
        return f"{x.size(0)} rows"
    return None


def mlp_step_refusal(x):
    """Why this step's MLP is not the kernels' (None: it is). Else the
    norm's ``(x, x_scale)``."""
    from aiter import dtypes

    if not _decode_step():
        return "not a decode step", None
    if not isinstance(x, tuple) or len(x) != 2:
        return "an input norm output that is not an (FP8, scale) pair", None
    x8, x_scale = x
    rows = x8.size(0) if x8.dim() == 2 else -1
    if not 1 <= rows <= MAX_ROWS:
        return f"{rows} rows", None
    if (
        x8.dtype != dtypes.fp8
        or tuple(x8.shape) != (rows, HIDDEN)
        or not x8.is_contiguous()
        or x_scale.dtype != dtypes.fp8_e8m0
        or x_scale.numel() != rows * GROUPS_X
    ):
        return f"an input row of {x8.dtype} {tuple(x8.shape)}", None
    return None, (x8, x_scale)


def forward_oproj(linear, x):
    """The draft's o_proj on the projection kernel: its all-reduced output,
    or None where the layer, the width or the step is production's."""

    def local_bind(runner):
        layer = bind_proj(linear, linear.prefix, runner.tp)
        if runner.k_oproj is None:
            runner.k_oproj = layer.k
        _need(layer.k == runner.k_oproj, f"{linear.prefix}: K {layer.k}")
        return layer

    layer = linear.mono_proj
    if layer is None:
        layer = _bind(
            linear, "mono_proj", "_try_mono_proj", linear.weight.device, local_bind,
            linear.prefix,
        )  # fmt: skip
        if layer is None:
            return None
    runner = runner_for(layer.w.device)
    why = proj_step_refusal(layer, x)
    if why is not None:
        runner._refuse(why)
        return None
    kernel = runner.kernel("o_proj", x.size(0))
    if kernel is None:
        return None
    return runner.launch_proj(kernel, layer, x)


def forward_mlp(mlp, x):
    """``K3DSparkMLP.forward`` on the draft kernels: its all-reduced output,
    or None where the layer or the step is production's. gate_up_proj and
    the activation run production's kernels where the width has no gate_up
    kernel, down_proj where it has no projection kernel."""
    from atom.models.dspark_draft import linear_out

    def local_bind(runner):
        layer = bind_mlp(mlp, runner.tp)
        if runner.inter is None:
            runner.inter = layer.inter
        _need(layer.inter == runner.inter, f"{mlp.prefix}: {layer.inter} wide")
        return layer

    layer = mlp.mono_mlp
    if layer is None:
        layer = _bind(
            mlp, "mono_mlp", "_try_mono_mlp", mlp.down_proj.weight.device,
            local_bind, mlp.prefix,
        )  # fmt: skip
        if layer is None:
            return None
    runner = runner_for(layer.w_gu.device)
    why, found = mlp_step_refusal(x)
    if why is not None:
        runner._refuse(why)
        return None
    x8, x_scale = found
    rows = x8.size(0)
    gate_up = runner.kernel("gate_up", rows)
    down = runner.kernel("down", rows)
    if gate_up is None and down is None:
        return None
    if gate_up is not None:
        act = runner.launch_gate_up(gate_up, layer, x8, x_scale)
    else:
        act = mlp.act_fn(linear_out(mlp.gate_up_proj(x8, x_scale)))
    if down is not None:
        return runner.launch_proj(down, layer.down, act)
    return linear_out(mlp.down_proj(act))
