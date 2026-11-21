"""E2E test: Load Qwen3-8B with Lumen AITER attention, run forward+backward."""

import logging
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Lumen attention requires CUDA/HIP", allow_module_level=True)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger("test_e2e")

# Patch attention BEFORE loading the model
from atomrl.lumen.ops.attention.hf_patch import patch_hf_sdpa

# The module runs the model at import, so a missing prerequisite has to skip it
# here rather than fail the whole collection.
ok = patch_hf_sdpa()
if not ok:
    pytest.skip("Lumen could not patch HF SDPA attention", allow_module_level=True)

from transformers import AutoModelForCausalLM

model_path = "/dev/shm/atomrl_weight_sync"
if not os.path.exists(os.path.join(model_path, "config.json")):
    model_path = os.environ.get("MODEL_PATH", model_path)
if not os.path.exists(os.path.join(model_path, "config.json")):
    pytest.skip(f"no model at {model_path}; set MODEL_PATH", allow_module_level=True)

logger.info("Loading model from %s", model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
)
model.to("cuda:0")
model.train()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

logger.info("Model loaded: %d params", sum(p.numel() for p in model.parameters()))

# Prepare dummy input
seq_len = 512
input_ids = torch.randint(0, 32000, (1, seq_len), device="cuda:0")
labels = input_ids.clone()

logger.info("Running forward pass (seq_len=%d)...", seq_len)
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logger.info("Loss: %.4f", loss.item())

logger.info("Running backward pass...")
loss.backward()

total_grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_grad_norm += p.grad.data.norm(2).item() ** 2
total_grad_norm = total_grad_norm**0.5
logger.info("Grad norm: %.4f", total_grad_norm)

assert total_grad_norm > 0, "Grad norm is zero — backward didn't propagate"
logger.info("PASS: Qwen3 + Lumen AITER attention forward+backward E2E")
