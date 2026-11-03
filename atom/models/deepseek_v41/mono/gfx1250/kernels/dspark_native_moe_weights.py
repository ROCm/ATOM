# SPDX-License-Identifier: MIT
"""Lossless one-time decode binding from mono scale order to native n32k4."""
def native_scale_binding(scale,experts,rows,cols):
    import torch
    assert rows%32==0 and cols%8==0 and scale.is_contiguous()
    raw=scale.view(torch.uint8)
    assert raw.numel()==experts*rows*cols
    # Legacy shuffle_scale: [E,N32,K8,k4,n16,k2,n2].
    plain=raw.reshape(experts,rows//32,cols//8,4,16,2,2).permute(0,1,6,4,2,5,3).contiguous()
    # Native TDM SB: each N32 superrow, one dword per row per K128.
    return plain.reshape(experts,rows//32,32,cols//4,4).permute(0,1,3,2,4).contiguous().reshape_as(raw).view(scale.dtype)
