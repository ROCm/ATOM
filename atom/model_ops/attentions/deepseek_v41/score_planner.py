# SPDX-License-Identifier: MIT
"""The index scorer's step layout: where each row's logits go in the packed
buffer, band by band. A step planner (`add_step_planner`): laid out on the
host into fixed-address buffers published with the step's metadata, so a
forward reads views of them and copies nothing of its own."""

import torch

from atom.model_ops.deepseek_v41.score_workspace import PackedRows
from atom.utils import CpuGpuBuffer

from .metadata import StepPlan, visible_buffer_name


def _name(ratio, field):
    return f"v41_score_{field}_{ratio}"


class ScorePlanner:
    """Lays a step's rows out in `workspace`'s packed logits at each of
    `ratios`. `max_tokens` bounds a step's rows, which the buffers are sized
    for."""

    def __init__(self, workspace, ratios, max_tokens):
        self.workspace, self.ratios = workspace, tuple(ratios)
        self._names = tuple(
            _name(ratio, field)
            for ratio in self.ratios
            for field in ("offsets", "block_offsets")
        )
        self._max_tokens = max_tokens

    def buffers(self, device, publication_group):
        return {
            name: CpuGpuBuffer(
                self._max_tokens,
                dtype=torch.int32,
                device=device,
                pin_memory=torch.device(device).type != "cpu",
                publication_group=publication_group,
            )
            for name in self._names
        }

    def __call__(self, buffers, rows):
        """Lay the staged rows out: each row's logits and block maxima start,
        and each ratio's bands (`ScoreWorkspace.pack`) on the host."""
        bands = {
            _name(ratio, "bands"): self.workspace.pack(
                buffers[visible_buffer_name(ratio)].np[:rows],
                buffers[_name(ratio, "offsets")].np,
                buffers[_name(ratio, "block_offsets")].np,
            )
            for ratio in self.ratios
        }
        return StepPlan(dict.fromkeys(self._names, rows), bands)


def score_layout(step, ratio):
    """`step`'s rows at `ratio` laid out in the packed logits (`PackedRows`);
    None when no `ScorePlanner` laid them out."""
    bands = step.planned_host.get(_name(ratio, "bands"))
    if bands is None:
        return None
    return PackedRows(
        step.visible[ratio],
        step.planned[_name(ratio, "offsets")],
        step.planned[_name(ratio, "block_offsets")],
        bands,
    )
