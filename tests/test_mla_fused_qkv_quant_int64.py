# SPDX-License-Identifier: MIT
"""The fused QKV quantizer forms its block offsets in 64 bits. Runs on the CPU.

A TP1 prefill over a long cached context quantizes more than 2**31 elements
(K alone is context x 128 heads x 192), so ``program_id * BLOCK`` formed in
32 bits wraps negative, passes the ``offsets < N`` mask and faults. Running
that takes a multi-GiB input and over 262K programs, and the Triton
interpreter cannot run ``_quant_tensor`` (inline asm), so this compiles both
kernels for an AMD target, which needs no GPU, and reads their Triton IR.
"""

import re

import pytest

triton = pytest.importorskip("triton")

from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from atom.model_ops import triton_fused_qkv_quant as m

# The launcher's configuration for 4M elements and more, on MLA prefill shapes.
BLOCK, PARTS = 8192, 128
HEADS, QK_DIM, V_DIM, ROPE_DIM = 128, 192, 128, 64


def _ttir(kernel, n_type, rope):
    signature = {name: "constexpr" for name in kernel.arg_names}
    signature.update(Q="*bf16", K="*bf16", V="*bf16", Partial="*fp32")
    signature.update(NQ=n_type, NK=n_type, NV=n_type)
    constexprs = {
        "SHAPES": ((HEADS, QK_DIM), (HEADS, QK_DIM), (HEADS, V_DIM)),
        "STRIDES": (
            (HEADS * QK_DIM, QK_DIM, 1),
            (HEADS * QK_DIM, QK_DIM, 1),
            (HEADS * V_DIM, V_DIM, 1),
        ),
        "PARTS": PARTS,
        "BLOCK": BLOCK,
        "ROPE_DIM": ROPE_DIM if rope else 0,
        # One RoPE head broadcast over K's heads, as the MLA prefill passes it.
        "ROPE_STRIDE": (ROPE_DIM, 0, 1) if rope else (0, 0, 0),
    }
    if rope:
        signature["K_ROPE"] = "*bf16"
    else:
        constexprs["K_ROPE"] = None
    if kernel is m._fused_qkv_quant:
        signature.update(Q8="*fp8e4nv", K8="*fp8e4nv", V8="*fp8e4nv")
        signature["Scales"] = "*fp32"
        constexprs["SINGLE_PASS"] = False
    source = ASTSource(kernel, signature, constexprs)
    return triton.compile(source, target=GPUTarget("hip", "gfx942", 64)).asm["ttir"]


@pytest.mark.parametrize("rope", [False, True], ids=["k", "split_k_rope"])
# Triton passes N as i64 from 2**31 elements, as in the faulting prefill.
@pytest.mark.parametrize("n_type", ["i32", "i64"])
@pytest.mark.parametrize("kernel", ["_fused_qkv_amax", "_fused_qkv_quant"])
def test_block_offsets_are_64_bit(kernel, n_type, rope):
    ttir = _ttir(getattr(m, kernel), n_type, rope)
    # program_id * BLOCK, and the amax loop's step * PARTS * BLOCK, ...
    assert re.findall(r"arith\.muli [^\n]*: i32\b", ttir) == []
    # ... and the offset of every BLOCK-wide load and store.
    assert re.findall(rf"tt\.addptr [^\n]*tensor<{BLOCK}xi32>", ttir) == []
