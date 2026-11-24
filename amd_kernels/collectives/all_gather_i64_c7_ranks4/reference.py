# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Exact four-rank gather of int64 rows [local requests, 7].

`src` decodes as source rank = src // src_stride, local row = src % src_stride.
Only referenced source rows must be valid; padded inputs may contain anything.
Input rows and maps can change on every graph replay. No reduction, casting
or sampling arithmetic occurs. Output is contiguous [output requests, 7].
"""
import torch

def reference(local,src,src_stride):
    rows=[local[int(i)//src_stride][int(i)%src_stride] for i in src]
    return torch.stack(rows) if rows else local[0].new_empty((0,7))
