# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Pack, gather and reorder DSpark sampler output without arithmetic on tokens.

Each of four ranks provides up to 256 local requests, 1-8 token columns,
int32 sampled/rejected counts, optional int32 post-mask NaNs per logit row,
and an exclusive row prefix `cu`. The output has eight int64 token columns,
int32 counts and optional int32 per-request NaN sums. Missing token columns
are padded with zero, as the sampler expects. Unused sampled-token suffix bytes are
copied unchanged; only valid prefixes are meaningful to the sampler.

`src` decodes as rank = src // src_stride, local row = src % src_stride.
The device map is allowed to change on each graph replay. Every local row
referenced by any rank must exist in that source rank's inputs.
All operations are integer-exact; no atomic add touches an output.
"""
import torch
C=8
MAX_REQUESTS=256

def reference(local, src, src_stride):
    """`local`: four tuples (tokens, sampled, rejected, nans or None, cu)."""
    packed=[]
    has_nans=local[0][3] is not None
    for tokens,ns,nr,nans,cu in local:
        a=torch.zeros(tokens.shape[0],11,dtype=torch.int64,device=tokens.device)
        a[:,:tokens.shape[1]]=tokens
        a[:,8]=ns;a[:,9]=nr
        if has_nans:
            for r in range(tokens.shape[0]):
                a[r,10]=nans[cu[r]:cu[r+1]].to(torch.int64).sum()
        packed.append(a)
    rows=[packed[int(s)//src_stride][int(s)%src_stride] for s in src]
    out=torch.stack(rows) if rows else packed[0].new_empty((0,11))
    return (out[:,:8].contiguous(),out[:,8].int(),out[:,9].int(),out[:,10].int() if has_nans else None)
