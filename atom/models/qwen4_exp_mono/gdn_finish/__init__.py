"""Qwen q4 only: complete staged replay publication before returning output."""

from .device import replay_unpublished
from .finish import finish


def forward(module, z, **args):
    replay_unpublished(**args)
    x, scale, _ = finish(
        args["o"],
        z,
        module.norm.weight,
        module.norm.eps,
        args["buf_k"],
        args["buf_g"],
        args["write_pos"],
        args["slot_idx"],
        args["cu_seqlens"],
    )
    return module.out_proj(x, x_scale=scale)
