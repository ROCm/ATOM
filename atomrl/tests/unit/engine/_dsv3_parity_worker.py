# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Real-checkpoint parity worker for DSv3. Not collected by pytest.

Loads a real DeepSeek-V3 checkpoint twice -- once into the HF reference
implementation, once into Megatron through the bridge -- and compares logits.

Runs in fp32 on purpose. In bf16 the two stacks differ by ~0.8% mean relative
error from kernel-ordering noise alone, which is too coarse to distinguish a
wrong softmax scale or a mis-ordered head from rounding. In fp32 a correct
bridge lands at ~1e-6, so any real defect is unmissable.

Prints ``DSV3_PARITY_OK`` on success; any assertion failure exits non-zero.
"""

import glob
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ["ATOMRL_ROOT"])

from atomrl.engine.training.bridges import dsv3
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from safetensors.torch import load_file
from transformers import DeepseekV3ForCausalLM

CKPT = os.environ["DSV3_CKPT"]
SEQ = 64
DTYPE = torch.float32
# Measured: a correct bridge gives 1.4e-6 mean relative error and 100% top-1
# agreement. Megatron's own rope/group defaults give 0.11 and 71.9%.
MAX_REL_ERROR = 1e-3

# Slots that are derived at construction time, not carried in a checkpoint.
_NOT_IN_CHECKPOINT = ("rotary_pos_emb", "local_tokens_per_expert", "_extra_state")


def _load_hf_state() -> dict:
    state: dict = {}
    for shard in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
        state.update(load_file(shard))
    return state


def _reference_logits(ids: torch.Tensor) -> torch.Tensor:
    """Logits from the HF implementation, freed before Megatron allocates."""
    ref = DeepseekV3ForCausalLM.from_pretrained(CKPT, dtype=DTYPE).cuda().eval()
    try:
        with torch.no_grad():
            return ref(ids.cuda()).logits.float().cpu()
    finally:
        del ref
        torch.cuda.empty_cache()


def _megatron_logits(hf_cfg: dict, state: dict, ids: torch.Tensor) -> torch.Tensor:
    cfg = dsv3.build_dsv3_config(hf_cfg, {"moe_grouped_gemm": True})
    cfg.bf16 = False
    cfg.params_dtype = DTYPE
    cfg.pipeline_dtype = DTYPE
    cfg.use_cpu_initialization = False
    model = (
        GPTModel(
            config=cfg,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                cfg, use_transformer_engine=True
            ),
            vocab_size=hf_cfg["vocab_size"],
            max_sequence_length=SEQ,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            parallel_output=False,
        )
        .cuda()
        .eval()
    )

    meg = {
        k: v.to(DTYPE)
        for k, v in dsv3.hf_to_megatron(
            state, dsv3.build_dsv3_dims(hf_cfg), True
        ).items()
    }
    missing, unexpected = model.load_state_dict(meg, strict=False)
    real_missing = [k for k in missing if not any(s in k for s in _NOT_IN_CHECKPOINT)]
    assert (
        not real_missing
    ), f"checkpoint left {len(real_missing)} slots unfilled: {real_missing[:5]}"
    assert (
        not unexpected
    ), f"bridge produced {len(unexpected)} unusable names: {list(unexpected)[:5]}"

    pos = torch.arange(SEQ, device="cuda").unsqueeze(0)
    mask = torch.triu(
        torch.ones(1, 1, SEQ, SEQ, dtype=torch.bool, device="cuda"), diagonal=1
    )
    with torch.no_grad():
        out = (
            model(input_ids=ids.cuda(), position_ids=pos, attention_mask=mask)
            .float()
            .cpu()
        )
    # Megatron may return [s, b, v]; the reference is [b, s, v].
    return out.transpose(0, 1) if out.shape[0] == SEQ else out


def main() -> None:
    with open(os.path.join(CKPT, "config.json")) as config_file:
        hf_cfg = json.load(config_file)
    torch.manual_seed(0)
    ids = torch.randint(0, hf_cfg["vocab_size"], (1, SEQ))

    ref_logits = _reference_logits(ids)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29596")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    try:
        mpu.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(0)
        meg_logits = _megatron_logits(hf_cfg, _load_hf_state(), ids)

        assert (
            meg_logits.shape == ref_logits.shape
        ), f"shape {tuple(meg_logits.shape)} != {tuple(ref_logits.shape)}"
        diff = (meg_logits - ref_logits).abs()
        rel = diff.mean().item() / ref_logits.abs().mean().item()
        agree = (meg_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        print(
            f"rel_mean_error={rel:.3e} max_abs={diff.max().item():.3e} top1={agree:.4f}"
        )

        assert (
            rel < MAX_REL_ERROR
        ), f"mean relative logit error {rel:.3e} >= {MAX_REL_ERROR}"
        assert agree == 1.0, f"top-1 agreement {agree:.4f}, expected exact in fp32"
        print("DSV3_PARITY_OK", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
