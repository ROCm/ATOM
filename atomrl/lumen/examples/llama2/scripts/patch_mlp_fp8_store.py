# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""OBSOLETE — use ``atomrl.lumen.models.megatron_patches.install_mlp_fp8_store()`` instead.

Set ``LUMEN_MLP_FP8_STORE=1`` and import ``atomrl.lumen.models.megatron`` (or call
``install_all()``). This file-level disk patch is kept only for Megatron-only
workflows outside the Lumen import path.

Patch Megatron MLP to use FP8 activation storage for SwiGLU intermediates.

TransformerEngine achieves lower activation memory via fused SwiGLU with FP8
storage (USE_TE_SWIGLU=1 + FP8_ACTIVATION=1). In Lumen's Megatron path the
SwiGLU is decomposed (silu + mul), so PyTorch autograd saves large BF16
intermediates (~0.9 GB/layer for Llama2-70B).

This patch replaces the decomposed SwiGLU with a custom autograd function that:
- Runs the same computation in forward
- Saves only the fc1 input in FP8 (1 byte) instead of BF16 (2 bytes)
- Recomputes the SwiGLU during backward from the FP8-stored input

Memory savings per non-recomputed layer (Llama2-70B, seq=8192, MBS=1):
  Before: ~1.3 GB for SwiGLU intermediates in BF16
  After:  ~0.06 GB for fc1 input in FP8 + ~0.22 GB for fc2 input in FP8
  Saving: ~1.0 GB/layer → ~59 GB for 59 non-recomputed layers (ACL=21)
"""

import pathlib
import sys

megatron_root = sys.argv[1]
MARKER = "# patched-mlp-fp8-store"

mlp_path = pathlib.Path(megatron_root) / "megatron" / "core" / "transformer" / "mlp.py"
if not mlp_path.exists():
    print(f"[Patch] mlp.py not found at {mlp_path}")
    sys.exit(1)

src = mlp_path.read_text()

if MARKER in src:
    print("[Patch] mlp.py: already patched with FP8 activation store")
    sys.exit(0)

SWIGLU_FUNCTION = '''

class _SwiGLU_FP8Store(torch.autograd.Function):
    """SwiGLU with FP8 activation storage to reduce memory.

    Forward: silu(gate) * up  (same as decomposed)
    Saved: only the pre-split input in FP8 (1 byte/element vs 2 bytes BF16).
    Backward: recomputes gate/up from FP8-stored input.
    """

    @staticmethod
    def forward(ctx, fc1_output, activation_func, clamp_value, glu_linear_offset):
        x_gate, x_linear = torch.chunk(fc1_output, 2, dim=-1)
        if clamp_value is not None:
            x_gate = x_gate.clamp(min=None, max=clamp_value)
            x_linear = x_linear.clamp(min=-clamp_value, max=clamp_value)
        activated = activation_func(x_gate) * (x_linear + glu_linear_offset)

        fp8_dtype = torch.float8_e4m3fnuz
        fc1_flat = fc1_output.reshape(-1, fc1_output.shape[-1]).contiguous()
        amax = fc1_flat.abs().amax()
        scale = torch.finfo(fp8_dtype).max / amax.clamp(min=1e-12)
        fc1_fp8 = (fc1_flat * scale).to(fp8_dtype).view(torch.uint8)
        inv_scale = 1.0 / scale

        ctx.save_for_backward(fc1_fp8, inv_scale)
        ctx._orig_shape = fc1_output.shape
        ctx._fp8_dtype = fp8_dtype
        ctx._clamp_value = clamp_value
        ctx._glu_linear_offset = glu_linear_offset
        ctx._activation_func = activation_func
        return activated

    @staticmethod
    def backward(ctx, grad_output):
        fc1_fp8, inv_scale = ctx.saved_tensors
        fp8_dtype = ctx._fp8_dtype
        clamp_value = ctx._clamp_value
        activation_func = ctx._activation_func
        glu_linear_offset = ctx._glu_linear_offset
        orig_shape = ctx._orig_shape

        fc1_recon = (fc1_fp8.view(fp8_dtype).to(torch.float32) * inv_scale).to(grad_output.dtype)
        fc1_recon = fc1_recon.reshape(orig_shape)
        x_gate, x_linear = torch.chunk(fc1_recon, 2, dim=-1)
        if clamp_value is not None:
            x_gate = x_gate.clamp(min=None, max=clamp_value)
            x_linear = x_linear.clamp(min=-clamp_value, max=clamp_value)

        with torch.enable_grad():
            x_gate_g = x_gate.detach().requires_grad_(True)
            act_val = activation_func(x_gate_g)
            act_grad = torch.autograd.grad(
                act_val, x_gate_g, torch.ones_like(act_val), retain_graph=False
            )[0]

        gate_activated = activation_func(x_gate)
        up_val = x_linear + glu_linear_offset

        grad_fc1 = torch.empty_like(fc1_recon)
        N = grad_fc1.shape[-1] // 2
        torch.mul(grad_output, up_val * act_grad, out=grad_fc1[..., :N])
        torch.mul(grad_output, gate_activated, out=grad_fc1[..., N:])
        return grad_fc1, None, None, None


'''

PATCHED_GLU = f"""
                {MARKER}
                def glu(x):
                    return _SwiGLU_FP8Store.apply(
                        x,
                        self.config.activation_func,
                        self.config.activation_func_clamp_value,
                        self.config.glu_linear_offset,
                    )
"""

ORIGINAL_GLU_BLOCK = """                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )"""

if ORIGINAL_GLU_BLOCK not in src:
    print("[Patch] mlp.py: original GLU block not found, cannot patch")
    sys.exit(1)

import_marker = "import torch"
if import_marker in src:
    idx = src.index(import_marker) + len(import_marker)
    next_newline = src.index("\n", idx)
    src = src[: next_newline + 1] + SWIGLU_FUNCTION + src[next_newline + 1 :]

src = src.replace(ORIGINAL_GLU_BLOCK, PATCHED_GLU)

mlp_path.write_text(src)
print("[Patch] mlp.py: patched GLU with _SwiGLU_FP8Store for FP8 activation storage")
