# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter's A8W4 MoE expert layout (gfx950), as Atom holds its MXFP4 experts in memory.

Atom's ``Mxfp4MoEMethod`` (``atom/model_ops/moe.py``) with MXFP8 activations
(``is_guinterleave``) creates, per layer, zero-filled tensors with the intermediate ``I``
padded to ``P`` (a multiple of 256) and every expert ``e`` (a fused shared expert is the
last slot):

- ``w13`` [E, 2 P, H / 2]: gate rows at ``0 .. I``, up rows at ``P .. P + I``;
- ``w13_scale`` [E, 2 P, H / 32] (E8M0, zeros in the padding);
- ``w2`` [E, H, P / 2]: K (the intermediate) padded with zero codes;
- ``w2_scale`` [E, H, P / 32];

then shuffles them in place with aiter's ``moe_shuffle_weight`` / ``moe_shuffle_scale``
(gfx950, ``is_guinterleave=True``). After that, for an expert's [N, K] weight (``N = 2 P``
for ``w13``, ``H`` for ``w2``) in **16-row blocks** ``b`` and **128-K steps** ``s``
(``S = K / 128`` steps; ``w13``'s blocks interleave gate and up: block ``2 j`` is gate rows
``16 j ..``, block ``2 j + 1`` up rows ``16 j ..``):

- weights: unit ``(b, s)`` is 1 KiB at byte ``(b * S + s) * 1024``; lane ``l``'s 16 bytes at
  ``16 l`` hold row ``l % 16`` of the block, K ``32 (l // 16) .. + 31`` (two codes a byte, the
  even element low). That is ``gemv.pack_weight``'s unit, so the MFMA reads it unchanged;
- scales: the code of (block ``b``, step ``s``, lane ``l``) is byte ``2 (s % 2) + b % 2`` of
  dword ``((b // 2) * S / 2 + s // 2) * 64 + l``: a dword per lane holds two blocks x two
  steps. The tensor is 2D ``[E * N, K / 32]`` uint8 (row-major in memory, as above).

Expert ``e``'s weights start at byte ``e * N * K / 2`` and its scales at ``e * N * K / 32``.
This module builds that layout from row-major MXFP4 (``atom_experts``) and checks it (the
``aiter.a8w4.moe`` / ``aiter.e8m0.moe`` layouts of ``weights.source``).
"""

import torch

from atom.models.qwen38.mono.kernels.weights.layer import MXLinear

PAD_ALIGN = 256  # Atom pads the intermediate (and hidden) to a multiple of 256
FP4X2 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)


def padded(n: int) -> int:
    return -(-n // PAD_ALIGN) * PAD_ALIGN


# ------------------------------------------------------------------ shuffles (aiter's, gfx950)


def shuffle_weight(w: torch.Tensor, gate_up: bool) -> torch.Tensor:
    """aiter ``shuffle_weight(layout=(16, 16), is_guinterleave=True)`` of [E, N, K / 2] bytes."""
    e, n, kb = w.shape
    w = w.view(torch.uint8)
    if gate_up:
        x = w.reshape(e, 2, n // 32, 16, kb // 64, 4, 16).permute(0, 2, 1, 4, 5, 3, 6)
    else:
        x = w.reshape(e, n // 16, 16, kb // 64, 4, 16).permute(0, 1, 3, 4, 2, 5)
    return x.contiguous().reshape(e, n, kb)


def shuffle_scale(s: torch.Tensor, experts: int, gate_up: bool) -> torch.Tensor:
    """aiter ``shuffle_scale(is_guinterleave=True)`` of [E * N, K / 32] codes (K / 32 % 8 == 0)."""
    rows, kg = s.shape
    n = rows // experts
    s = s.view(torch.uint8)
    if kg % 8:
        raise ValueError(
            f"shuffle_scale: K / 32 = {kg} must be a multiple of 8 (pad K to 256)"
        )
    if gate_up:
        x = s.reshape(experts, 2, n // 32, 16, kg // 8, 2, 4).permute(
            0, 2, 4, 6, 3, 5, 1
        )
    else:
        x = s.reshape(experts, n // 32, 2, 16, kg // 8, 2, 4).permute(
            0, 1, 4, 6, 3, 5, 2
        )
    return x.contiguous().reshape(rows, kg)


# ------------------------------------------------------------------ Atom's tensors from row-major MXFP4


def atom_experts(
    gate: MXLinear,
    up: MXLinear,
    down: MXLinear,
    experts: int,
    shared: tuple[MXLinear, MXLinear, MXLinear] | None,
) -> dict[str, torch.Tensor]:
    """The four tensors Atom holds for one layer's experts (fused shared expert last), from
    row-major MXFP4 stacked per expert: ``gate`` / ``up`` [E * I, H / 2] (codes [E * I, H / 32]),
    ``down`` [E * H, I / 2]; ``shared`` (gate, up, down) of one expert, or None.
    -> ``w13``, ``w13_scale``, ``w2``, ``w2_scale`` (weights in ``float4_e2m1fn_x2`` where torch
    has it, carrying ``is_shuffled = True``, as Atom's)."""
    inter = gate.packed.shape[0] // experts
    hidden = gate.packed.shape[1] * 2
    mats = [(gate.packed.view(experts, inter, -1), gate.codes.view(experts, inter, -1))]
    mats += [(up.packed.view(experts, inter, -1), up.codes.view(experts, inter, -1))]
    mats += [
        (down.packed.view(experts, hidden, -1), down.codes.view(experts, hidden, -1))
    ]
    if shared is not None:
        mats = [
            (torch.cat((p, s.packed[None])), torch.cat((c, s.codes[None])))
            for (p, c), s in zip(mats, shared, strict=True)
        ]
    slots, p_, dev = mats[0][0].shape[0], padded(inter), gate.packed.device
    w13 = torch.zeros(slots, 2 * p_, hidden // 2, dtype=torch.uint8, device=dev)
    s13 = torch.zeros(slots, 2 * p_, hidden // 32, dtype=torch.uint8, device=dev)
    for half, (p, c) in enumerate(mats[:2]):
        w13[:, half * p_ : half * p_ + inter] = p
        s13[:, half * p_ : half * p_ + inter] = c
    w2 = torch.zeros(slots, hidden, p_ // 2, dtype=torch.uint8, device=dev)
    s2 = torch.zeros(slots, hidden, p_ // 32, dtype=torch.uint8, device=dev)
    w2[:, :, : inter // 2] = mats[2][0]
    s2[:, :, : inter // 32] = mats[2][1]
    out = {
        "w13": shuffle_weight(w13, True).view(FP4X2),
        "w13_scale": shuffle_scale(s13.reshape(-1, hidden // 32), slots, True),
        "w2": shuffle_weight(w2, False).view(FP4X2),
        "w2_scale": shuffle_scale(s2.reshape(-1, p_ // 32), slots, False),
    }
    out["w13"].is_shuffled = True
    out["w2"].is_shuffled = True
    return out


# ------------------------------------------------------------------ the bind-time checks (layouts in source.py)


def unit_mask(w: torch.Tensor, expert: int) -> torch.Tensor:
    """[N / 16, K / 128] bool: which 16 x 128 units of ``expert`` (in the shuffled layout) hold a
    nonzero byte."""
    _, n, kb = w.shape
    return (w[expert].view(torch.uint8).reshape(n // 16, kb // 64, 1024) != 0).any(-1)


def scale_mask(s: torch.Tensor, experts: int, expert: int) -> torch.Tensor:
    """[N / 16, K / 128] bool: which units of ``expert`` have a nonzero E8M0 code (any lane) in
    the shuffled scale layout."""
    rows, kg = s.shape
    n = rows // experts
    # pair, k1, l, kp, np
    x = (
        s[expert * n : (expert + 1) * n]
        .view(torch.uint8)
        .reshape(n // 32, kg // 8, 64, 2, 2)
    )
    nz = (x != 0).any(2)  # [pair, k1, kp, np]
    return nz.permute(0, 3, 1, 2).reshape(n // 16, kg // 4)


def used_units(
    gate_up: bool, n: int, k: int, n_stored: int, k_stored: int, device="cpu"
) -> torch.Tensor:
    """The units a used [n, k] weight occupies in its padded [n_stored, k_stored] slot (on ``device``,
    explicitly: Atom loads under ``torch.set_default_device("cuda")``)."""
    m = torch.zeros(n_stored // 16, k_stored // 128, dtype=torch.bool, device=device)
    # gate_up: the interleaved used blocks come first
    m[: -(-n // 16), : -(-k // 128)] = True
    return m
