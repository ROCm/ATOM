# SPDX-License-Identifier: Apache-2.0
# Overlay for p1/sp-local-shared (ATOM_P1_SP_SHARED_LOCAL): the FULL (unsharded, I 3072) MXFP8 shared expert of
# MiniMax-M3.1 on a rank's own rows, taken unchanged from amd-kernels PR #5 (Inferact/amd-kernels 399431b,
# moe/mxfp4_h6144_i1536_e128_top4: shared_dense.shared_dense_full + its 4-wave GEMMs, files copied verbatim; see
# PR5_HEAD). The PR's full copy does not depend on the TP degree (gate/up N 6144 K 6144, down N 6144 K 3072), so the
# TP2 package's kernels serve TP4. Only the weight-assembly helpers below are new: the full expert is rebuilt from the
# four TP4 shards in the i768 entry's layout (a bit-exact permutation of the checkpoint bytes; no re-quantization).
from dataclasses import dataclass

import torch

H, I_FULL, I_TP4 = 6144, 3072, 768


def weight_layout(w):
    """Row-major e4m3 [N, K] -> AITER shuffle_weight(.., (16, 16)): 16-row blocks of [K / 64, 4, 16 rows, 16 B]."""
    n, k = w.shape
    return w.view(torch.uint8).reshape(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()


def scale_layout(s):
    """Row-major e8m0 [N, K / 32] -> AITER shuffle_scale_a16w4(.., 1, False)."""
    n, g = s.shape
    return s.view(torch.uint8).reshape(n // 32, 2, 16, g // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous()


@dataclass
class SharedFull:
    """The full shared expert in the kernels' layout (``shared_dense_full`` reads w13, s13, w2, s2)."""

    w13: torch.Tensor  # [384, 96, 4, 16, 16] u8 (gate rows 0..3071, up rows 3072..6143)
    s13: torch.Tensor  # [192, 24, 4, 16, 2, 2] u8
    w2: torch.Tensor  # [384, 48, 4, 16, 16] u8
    s2: torch.Tensor  # [192, 12, 4, 16, 2, 2] u8
    full: bool = True

    @classmethod
    def from_row_major(cls, gate, gate_s, up, up_s, down, down_s):
        """Checkpoint tensors: gate/up e4m3 [3072, 6144] + e8m0 [3072, 192], down e4m3 [6144, 3072] + [6144, 96]."""
        w13 = torch.cat([gate.view(torch.uint8), up.view(torch.uint8)])
        s13 = torch.cat([gate_s.view(torch.uint8), up_s.view(torch.uint8)])
        return cls(weight_layout(w13), scale_layout(s13), weight_layout(down), scale_layout(down_s))

    @classmethod
    def from_tp4_shards(cls, w13s, s13s, w2s, s2s):
        """The four TP4 ranks' shards in the i768 entry's layout (``SharedWeights``: w13 [96, 96, 4, 16, 16], s13
        [48, 24, 4, 16, 2, 2], w2 [384, 12, 4, 16, 16], s2 [192, 3, 4, 16, 2, 2]; rank order) -> the full expert,
        byte for byte ``from_row_major`` of the checkpoint (rank r holds gate/up rows r*768.. and down K columns
        r*768..; 16-row blocks / 32-row scale groups / 64-column K blocks never straddle two shards)."""
        u8 = lambda t: t.view(torch.uint8)
        g = [u8(t)[:48] for t in w13s] + [u8(t)[48:] for t in w13s]
        gs = [u8(t)[:24] for t in s13s] + [u8(t)[24:] for t in s13s]
        return cls(torch.cat(g), torch.cat(gs), torch.cat([u8(t) for t in w2s], dim=1),
                   torch.cat([u8(t) for t in s2s], dim=1).contiguous())


def shared_expert_full(codes, scales, w, *, stream=None, keep=None):
    """bf16 [n, 6144] of the full shared expert on MXFP8 codes u8 [n, 6144] + row-major e8m0 u8 [n, 192] (the MoE
    preparation's shared operands). ``stream`` / ``keep``: PR #5's side-stream rule (buffers from the current
    stream's pool, held in ``keep`` until the caller's join)."""
    from .shared_dense import shared_dense_full

    if keep is not None and stream is not None and stream != torch.cuda.current_stream():
        keep.extend((codes, scales))
    return shared_dense_full(codes, scales, w, stream=stream, keep=keep)


# one row count per FULL_CONFIGS tier (TP4 own rows reach 8..8192)
FULL_WARM_UP_ROWS = (32, 1281, 2561, 4097)


def warm_up(w, device=None):
    """Compile every full-expert variant (before serving; local, no collective)."""
    dev = w.w13.device if device is None else device
    for n in FULL_WARM_UP_ROWS:
        z = torch.zeros((n, H), dtype=torch.uint8, device=dev)
        shared_expert_full(z, torch.full((n, H // 32), 127, dtype=torch.uint8, device=dev), w)
    torch.cuda.synchronize(dev)
