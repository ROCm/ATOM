# SPDX-License-Identifier: MIT
"""One draft forward, warmed and optionally captured per captured size.

Kept out of ``drafter.py`` so it stays importable without a GPU AITER build:
the invariants below -- which batch a pass may run at, staging with a padded
tail, the pad contract -- are pure torch, and CI has no aiter. A flavor's own
passes still live with the flavor; only the machine is here.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from atom.utils import envs

if TYPE_CHECKING:
    from atom.config import Config


def _owned(value: Any, staged: tuple[torch.Tensor, ...] = ()) -> Any:
    """A copy the graph pool cannot re-issue.

    Outputs that alias declared staged inputs already live outside the graph
    pool and can be handed back directly. Clone every other tensor because the
    pool may re-issue its storage on a later replay.
    """
    if isinstance(value, torch.Tensor):
        if any(value.data_ptr() == tensor.data_ptr() for tensor in staged):
            return value
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_owned(v, staged) for v in value)
    return value


@dataclass(frozen=True)
class StagedInput:
    """One staged input's type: everything but the leading batch axis."""

    shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.int32


@dataclass
class DraftGraph:
    """One draft forward: declared by the flavor, bound to storage here.

    Named for what it is FOR, not for what it always holds -- a pass that
    declines to pad holds no graph at all (EPLB, eager, the separate-draft
    flavor), because a graph is one shape and only padding pins the shape.

    Three capabilities, each needing strictly more than the last:

    * **warmup** -- run it once per ``capture_sizes`` entry at startup, paying its
      per-shape JIT there rather than mid-serve. Needs nothing.
    * **pad** -- run at the batch the step settled on, which the startup sweep
      warmed, rather than at whatever this step's batch happens to be. Needs
      ``inputs``: fixed addresses for the fabricated rows to land in.
    * **capture** -- record it. Needs pad.

    The pass is ``epilogue ∘ forward``. Warmup always runs both, so nothing
    serving runs is left cold -- narrowing the warmup to the capturable part is
    how the LM head silently stopped being warmed once. Capture takes ``forward``
    alone, or both when ``capture_epilogue``; either way there is exactly one
    place a graph is replayed, so a flavor cannot put the seam in the wrong spot.

    ``warmup_inputs`` makes that startup batch plausible, since warming on zeros
    compiles a shape no real batch draws.

    Everything here counts SEQUENCES, in the target's own words:
    ``scheduled_bs`` is what this rank was handed, ``running_bs`` is the
    DP-agreed batch a recording holds, and ``capture_sizes`` is the ladder
    the startup sweep warms.
    """

    forward: Callable[..., Any]  # (running_bs, **staged) -> Any
    epilogue: Callable[..., Any] | None = None  # (fwd_out, running_bs, **staged)
    capture_epilogue: bool = False
    inputs: "Mapping[str, StagedInput]" = field(default_factory=dict)
    warmup_inputs: Callable[..., None] | None = None

    _buffers: dict[str, torch.Tensor] = field(init=False, default_factory=dict)
    _cuda_graphs: dict[int, tuple] = field(init=False, default_factory=dict)

    @property
    def name(self) -> str:
        """Which pass this is, for the assertions below.

        Taken from the forward rather than declared, so it cannot drift and it
        points at the exact function. A declared one would have to be unique
        across the repo on its own, and the words available -- "block", "step"
        -- are the three most overloaded in it.
        """
        return self.forward.__qualname__

    def bind(self, config: "Config", device) -> "DraftGraph":
        """Give the declared pass its storage. Once, at drafter init.

        Buffers are allocated now rather than on first use: a capture bakes
        their addresses, and one born on the capture step would come out of the
        graph's own private pool.
        """
        assert self.inputs, (
            f"{self.name} stages nothing, so a fabricated row has no buffer "
            f"to land in -- declare no pass rather than an unpaddable one"
        )
        self._buffers = {
            role: torch.zeros(
                (config.max_num_seqs, *staged.shape),
                dtype=staged.dtype,
                device=device,
            )
            for role, staged in self.inputs.items()
        }
        return self

    @property
    def will_capture(self) -> bool:
        """Whether warmup also captures. The name `envs.ATOM_DRAFT_CUDAGRAPH`
        points at, and the only thing that decides it."""
        return envs.ATOM_DRAFT_CUDAGRAPH

    def label(self, scheduled_bs: int, running_bs: int) -> str:
        """``bs=<scheduled>/<running>``, plus ``graph`` when this step replays.

        One convention, one implementation: "did the draft get into a graph" is
        then a single grep across every flavor, rather than each flavor writing
        the same string and the answer holding by coincidence. Reads the same
        predicate `run` does, so the mark cannot claim a replay that did not
        happen.
        """
        return f"bs={scheduled_bs}/{running_bs}" + (
            " graph" if self.is_captured(running_bs) else ""
        )

    def stage(
        self, running_bs: int, srcs: "Mapping[str, torch.Tensor]"
    ) -> dict[str, torch.Tensor]:
        """Copy every input into its fixed buffer, widening the batch to ``running_bs``.

        How many rows are real comes from the sources, never from the caller: a
        count that cannot disagree with the tensor it describes.
        """
        counts = {t.shape[0] for t in srcs.values()}
        assert len(counts) <= 1, (
            f"{self.name} was handed batches of {sorted(counts)}; they "
            f"describe one step, so one of them is not this step's"
        )
        return {
            role: self._stage_one(role, running_bs, src) for role, src in srcs.items()
        }

    def buffer(self, role: str, bs: int) -> torch.Tensor:
        """Return the fixed-storage view a producer may fill for the real rows.

        A following :meth:`stage` still owns padding ``bs`` to ``running_bs``.
        The caller must not mutate this view between that stage and the pass
        consuming it.
        """
        assert role in self._buffers, (
            f"{self.name} has no staged input {role!r}; "
            f"declared roles are {sorted(self._buffers)}"
        )
        buf = self._buffers[role]
        assert 0 <= bs <= buf.shape[0], (
            f"{self.name} requested {bs} rows from draft input {role!r}, "
            f"whose capacity is {buf.shape[0]}"
        )
        return buf[:bs]

    def _stage_one(self, role: str, running_bs: int, src: torch.Tensor | None):
        """Copy ``src`` into this input's fixed buffer, tail-repeating its last row.

        Every input repeats the same last index, so a pad row is a coherent copy
        of the last real one and only redoes work that row already does. Zeros
        instead faulted 8/8 ranks at the first padded decode step, and repeating
        one input alone -- an incoherent mix -- faulted too.

        ``src=None`` hands back the view unwritten: warmup wants a well-formed
        batch, not a particular one.
        """
        buf = self._buffers[role]
        out = buf[:running_bs]
        if src is None:
            return out
        assert src.dtype == buf.dtype, (
            f"draft input '{role}' of {self.name} arrived as {src.dtype}, but "
            f"its storage is {buf.dtype}; a capture has that address baked"
        )
        assert src.ndim == buf.ndim, (
            f"draft input '{role}' of {self.name} arrived as {tuple(src.shape)}, "
            f"whose leading axis is not the batch; this pass stages "
            f"{tuple(buf.shape[1:])} per sequence. MRoPE positions ([3, N]) are "
            f"the case that reaches here"
        )
        scheduled_bs = src.shape[0]
        dst = out[:scheduled_bs]
        assert src.shape == dst.shape, (
            f"draft input '{role}' of {self.name} arrived as {tuple(src.shape)}, "
            f"but its fixed storage expects {tuple(dst.shape)}"
        )
        # A producer may have written straight into this exact fixed-storage
        # view. Do not turn that into a self-copy launch.
        if src.data_ptr() != dst.data_ptr() or src.stride() != dst.stride():
            dst.copy_(src)
        if running_bs > scheduled_bs:
            out[scheduled_bs:].copy_(src[-1])  # broadcasts; no expand needed
        return out

    def is_captured(self, running_bs: int) -> bool:
        """Whether a recording exists at this batch -- and so whether this step
        replays, the two having become one question.

        Nothing about the step enters. A recording holds its collective at one
        width, so the decision has to be one the whole DP group agrees on, and
        ``running_bs`` was reduced in ``sync_dp_metadata`` before it got here.
        That is what a second term used to buy: ``is_dummy_run`` and
        ``is_prefill`` are both per-rank, and either splits the group into two
        widths (measured -- V4-Flash-DSpark tp8 + DPA hung 8/8 on the first
        real decode). Neither is reachable from an agreed batch, so neither is
        available to get wrong. Safe on a dummy and on a prefill step for one
        reason: ``Drafter.warmup_draft_graphs`` records on a real decode-shaped
        context, and this pass is that shape whatever the target just did.
        """
        return running_bs in self._cuda_graphs

    @property
    def _to_capture(self) -> Callable[..., Any]:
        """What a graph holds -- and so what a replay stands in for."""
        return self._forward_and_epilogue if self.capture_epilogue else self.forward

    def run(self, running_bs: int, **staged) -> Any:
        """Replay this batch's graph, then whatever it did not cover.

        The batch is the whole argument. `running_bs` is the reduction's own
        answer rounded on a ladder every rank shares, so every rank answers
        alike and nothing about the step needs to reach here -- see
        :meth:`is_captured`.

        What leaves here is a VALUE. A replay hands back the tensors the
        capture allocated, and the pool re-issues those addresses to sizes
        captured later -- a reference does not hold them the way the eager
        allocator's does. Only ``capture_epilogue`` can put a survivor in
        there; the eager path allocates normally and needs no copy.
        """
        captured = (
            self._cuda_graphs.get(running_bs) if self.is_captured(running_bs) else None
        )
        if captured is None:
            out = self._to_capture(running_bs, **staged)
        else:
            graph, out = captured
            graph.replay()
            if self.capture_epilogue:
                out = _owned(out, tuple(staged.values()))
        return (
            out
            if self.capture_epilogue
            else self._run_epilogue(out, running_bs, **staged)
        )

    def _forward_and_epilogue(self, running_bs: int, **staged) -> Any:
        return self._run_epilogue(
            self.forward(running_bs, **staged), running_bs, **staged
        )

    def _run_epilogue(self, out: Any, running_bs: int, **staged) -> Any:
        return (
            out if self.epilogue is None else self.epilogue(out, running_bs, **staged)
        )

    @torch.inference_mode()
    def warmup(self, running_bs: int, *, pool=None, stream=None):
        """Run the pass once at one captured size, paying its per-shape JIT here."""
        staged = {role: self._stage_one(role, running_bs, None) for role in self.inputs}
        if self.warmup_inputs is not None:
            self.warmup_inputs(running_bs, **staged)
        # forward AND epilogue, so nothing serving runs is left cold.
        self._forward_and_epilogue(running_bs, **staged)
        if not self.will_capture:
            return pool
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # thread_local, not the "global" default: global invalidates a capture
        # when ANY thread makes an unsafe HIP call, and the NCCL watchdog polls
        # `hipEventQuery` every ~100ms. Same reasoning as `cuda_graph.py`'s.
        with torch.cuda.graph(
            graph, pool, stream=stream, capture_error_mode="thread_local"
        ):
            out = self._to_capture(running_bs, **staged)
        self._cuda_graphs[running_bs] = (graph, out)
        return pool or graph.pool()
