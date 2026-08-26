# SPDX-License-Identifier: MIT
"""One draft forward, warmed and optionally captured per ``graph_bs``.

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

    * **warmup** -- run it once per ``graph_bs`` at startup, paying its
      per-shape JIT there rather than mid-serve. Needs nothing.
    * **pad** -- run at the batch the target just ran, which the startup sweep
      warmed, rather than at whatever this step's batch happens to be. Needs
      ``inputs`` (fixed addresses to pad into) and ``pads`` (an assertion that
      the fabricated rows reach nothing).
    * **capture** -- record it. Needs pad.

    The pass is ``epilogue ∘ forward``. Warmup always runs both, so nothing
    serving runs is left cold -- narrowing the warmup to the capturable part is
    how the LM head silently stopped being warmed once. Capture takes ``forward``
    alone, or both when ``capture_epilogue``; either way there is exactly one
    place a graph is replayed, so a flavor cannot put the seam in the wrong spot.

    ``warmup_inputs`` makes that startup batch plausible, since warming on zeros
    compiles a shape no real batch draws.

    Everything here counts SEQUENCES, in the same three words the target uses:
    ``bs`` is real, ``pad_bs`` is what actually runs, ``graph_bs`` is the
    captured set.
    """

    forward: Callable[..., Any]  # (pad_bs, **staged) -> Any
    epilogue: Callable[..., Any] | None = None  # (fwd_out, pad_bs, **staged)
    capture_epilogue: bool = False
    inputs: "Mapping[str, StagedInput]" = field(default_factory=dict)
    pads: bool = False
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
        assert not self.pads or self.inputs, (
            f"{self.name} says it may pad but stages nothing, so there is no "
            f"buffer for the fabricated rows to land in"
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
        """Whether warmup also captures. One gate for every flavor, since the
        graph is made by the shared ``warmup``."""
        return self.pads and envs.ATOM_DRAFT_CUDAGRAPH

    def target_pad_bs(self, bs: int, context) -> int:
        """The batch the target just ran at, or ``bs`` when it pinned none.

        Never a batch the drafter picks. ``forward_mode.effective_bs`` IS what
        the target ran: ``model_runner`` builds the attention metadata with it,
        so every per-sequence buffer already reaches that far -- ``slot_mapping``
        -1, ``context_lens`` 0, ring slots published -- and a pad row fabricates
        nothing the backend has not already accounted for, on any backend and
        without the drafter knowing each one's convention. It is a warmed shape
        by construction too: on the cudagraph path ``ForwardMode.decide`` picks
        it out of ``graph_bs``, which is the list the startup sweep warms.

        Not ``context.graph_bs``, which agrees except where DSpark's ragged path
        overwrites it with a token-derived count (``_dspark_ragged_moe_graph_bs``)
        that is neither captured nor the batch the metadata was built at.

        Off the cudagraph path every eager branch of ``decide`` returns
        ``effective_bs == scheduled_bs``, so this answers ``bs`` and nothing is
        padded -- which is the only safe answer, since nobody sized the target's
        metadata past the real batch there.

        An empty batch is never widened: a pad row is a copy of the last real
        one, and there is no last real one to copy. Declining the padding rather
        than the pass is deliberate -- under DP every rank still has to reach the
        draft's collectives.
        """
        mode = context.forward_mode
        if not bs or not self.pads or context.is_dummy_run:
            return bs
        if mode is None or not mode.use_cudagraph:
            return bs
        return max(bs, int(mode.effective_bs))

    def label(self, real_bs: int, pad_bs: int) -> str:
        """``bs=<real>/<pad>`` plus a trailing ``graph`` on a replay.

        One convention, one implementation: "did the draft get into a graph" is
        then a single grep across every flavor, rather than each flavor writing
        the same string and the answer holding by coincidence.
        """
        return f"bs={real_bs}/{pad_bs}" + (" graph" if self.is_captured(pad_bs) else "")

    def stage(
        self, pad_bs: int, srcs: "Mapping[str, torch.Tensor]"
    ) -> dict[str, torch.Tensor]:
        """Copy every input into its fixed buffer, widening the batch to ``pad_bs``.

        The only way in, because ``pads`` is the part of the contract nothing
        else can check: a pass that lies about it still runs and still returns
        the right shape -- its fabricated rows just land in another sequence's
        KV.

        How many rows are real comes from the sources, not from the caller: a
        count that cannot disagree with the tensor it describes.
        """
        counts = {t.shape[0] for t in srcs.values()}
        assert len(counts) <= 1, (
            f"{self.name} was handed batches of {sorted(counts)}; they "
            f"describe one step, so one of them is not this step's"
        )
        bs = counts.pop() if counts else pad_bs
        assert pad_bs == bs or self.pads, (
            f"{self.name} was asked for {pad_bs} rows against {bs} real ones, "
            f"but never said its fabricated rows are inert"
        )
        return {role: self._stage_one(role, pad_bs, src) for role, src in srcs.items()}

    def _stage_one(self, role: str, pad_bs: int, src: torch.Tensor | None):
        """Copy ``src`` into this input's fixed buffer, tail-repeating its last row.

        Every input repeats the same last index, so a pad row is a coherent copy
        of the last real one and only redoes work that row already does. Zeros
        instead faulted 8/8 ranks at the first padded decode step, and repeating
        one input alone -- an incoherent mix -- faulted too.

        ``src=None`` hands back the view unwritten: warmup wants a well-formed
        batch, not a particular one.
        """
        buf = self._buffers[role]
        out = buf[:pad_bs]
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
        bs = src.shape[0]
        out[:bs].copy_(src)
        if pad_bs > bs:
            out[bs:].copy_(src[-1])  # copy_ broadcasts; no expand needed
        return out

    def is_captured(self, pad_bs: int) -> bool:
        return pad_bs in self._cuda_graphs

    @property
    def _to_capture(self) -> Callable[..., Any]:
        """What a graph holds -- and so what a replay stands in for."""
        return self._forward_and_epilogue if self.capture_epilogue else self.forward

    def run(self, pad_bs: int, **staged) -> Any:
        """Replay this batch size's graph, then whatever it did not cover."""
        captured = self._cuda_graphs.get(pad_bs)
        if captured is None:
            out = self._to_capture(pad_bs, **staged)
        else:
            graph, out = captured
            graph.replay()
        return (
            out if self.capture_epilogue else self._run_epilogue(out, pad_bs, **staged)
        )

    def _forward_and_epilogue(self, pad_bs: int, **staged) -> Any:
        return self._run_epilogue(self.forward(pad_bs, **staged), pad_bs, **staged)

    def _run_epilogue(self, out: Any, pad_bs: int, **staged) -> Any:
        return out if self.epilogue is None else self.epilogue(out, pad_bs, **staged)

    @torch.inference_mode()
    def warmup(self, pad_bs: int, *, pool=None, stream=None):
        """Run the pass once at one ``graph_bs``, paying its per-shape JIT here."""
        staged = {role: self._stage_one(role, pad_bs, None) for role in self.inputs}
        if self.warmup_inputs is not None:
            self.warmup_inputs(pad_bs, **staged)
        # forward AND epilogue, so nothing serving runs is left cold.
        self._forward_and_epilogue(pad_bs, **staged)
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
            out = self._to_capture(pad_bs, **staged)
        self._cuda_graphs[pad_bs] = (graph, out)
        return pool or graph.pool()
