# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Quick test: Lumen AITER attention patch on HF SDPA."""

import logging

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Lumen attention requires CUDA/HIP", allow_module_level=True)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from atomrl.lumen.ops.attention.hf_patch import patch_hf_sdpa

patch_hf_sdpa()

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

lumen_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]

B, S, H_Q, H_KV, D = 2, 256, 32, 8, 128


class FakeModule:
    is_causal = True
    num_key_value_groups = H_Q // H_KV


mod = FakeModule()

q = torch.randn(B, H_Q, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(B, H_KV, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(B, H_KV, S, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
print(f"Q={q.shape}, K={k.shape}, V={v.shape}")

out, _ = lumen_fn(mod, q, k, v, attention_mask=None, dropout=0.0, scaling=D**-0.5)
print(f"Out={out.shape} dtype={out.dtype}")

loss = out.sum()
loss.backward()
print(f"dQ={q.grad.shape} dK={k.grad.shape} dV={v.grad.shape}")

assert out.shape == (B, S, H_Q, D), f"Expected (B,S,H_Q,D) got {out.shape}"
assert q.grad.shape == q.shape
assert k.grad.shape == k.shape
assert v.grad.shape == v.shape
print("PASS: Lumen AITER GQA attention forward+backward")
