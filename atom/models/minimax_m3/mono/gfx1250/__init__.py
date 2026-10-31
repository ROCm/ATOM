# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 mono decode on gfx1250 (MI455X): the gfx950 kernels' tasks, scratch
layout and numerical contracts on 16 waves of 32 and WMMA.

What the kernels read on gfx1250 and how it meets WMMA (8-bit / FP4 operand:
lane l holds row / column l % 16 at k 32 (s // 16) + 16 (l // 16) + s % 16):

- PTPC FP8 linears (qkv, o_proj): ``shuffle_weight(layout=(16, 16))``, a 64-k
  chunk kc of row group rg is the 512 B blocks 2 kc, 2 kc + 1 at
  (rg K / 32 + 2 kc + c) 512; lane l's 16 B of each at l 16 is its WMMA A operand.
- MXFP4 experts (``ATOM_MOE_GU_ITLV``): w13 rows interleaved gate / up
  (GUGU, row 2 i gate i, 2 i + 1 up i) then the same (16, 16) shuffle on packed
  bytes, so a 128-k chunk is the FP4 A operand of one scaled 16x16x128 WMMA and
  a 16-row tile holds 8 gate / up pairs. Scales ``shuffle_scale_n32k4``: expert
  e, row n, k-block kb at byte e N K / 32 + (n // 32) K + (kb // 4) 128 +
  (n % 32) 4 + kb % 4, one dword a row and 128-k step (the WMMA scale VGPR).
  Activations go in as MXFP8 (a8w4), as gfx1250's unfused path quantizes them.
"""
