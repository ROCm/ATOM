# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""torchrun worker for the DSv3 parallelism tests. Not collected by pytest.

Builds the tiny DSv3 model at the topology given by the ``DSV3_TP``/``DSV3_PP``/
``DSV3_EP``/``DSV3_LAYERS`` env vars and asserts that the model describes itself
the way the engine's weight gather assumes.

Two properties, by topology:

* TP/EP -- ``sharded_state_dict()`` must report the right global shapes and shard
  axes for MLA. The gather is driven entirely by that metadata, so if it is right
  the generic gather reconstructs MLA with no MLA-specific code, and if it is
  wrong the rollout receives silently mis-assembled attention weights.
* PP -- each stage must publish its layers under GLOBAL layer numbers, and the
  dense/MoE split must land on the global layers ``first_k_dense_replace``
  names. DSv3 always has a ``moe_layer_freq`` list, which is exactly the
  heterogeneous case where the old metadata-derived offset returned 0 with no
  error (see ``_pp_layer_offset``).

Prints ``DSV3_SHARD_OK`` on success; any assertion failure exits non-zero.
"""

import os
import re
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ["ATOMRL_ROOT"])

from atomrl.engine.training.bridges.dsv3 import build_dsv3_config
from atomrl.engine.training.megatron_native_engine import (
    _pp_layer_offset,
    _to_global_key,
)
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)

TP = int(os.environ.get("DSV3_TP", "2"))
PP = int(os.environ.get("DSV3_PP", "1"))
EP = int(os.environ.get("DSV3_EP", "2"))
LAYERS = int(os.environ.get("DSV3_LAYERS", "3"))
FIRST_K_DENSE = 1

TINY = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_hidden_layers": LAYERS,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 256,
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "first_k_dense_replace": FIRST_K_DENSE,
    "routed_scaling_factor": 2.5,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
}

_LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.")


def _expected_mla_metadata(tp: int) -> dict:
    """Global shape and per-axis fragmentation the engine's gather relies on.

    hidden=128  heads=4  qk_nope=16  qk_rope=8  v=16  q_lora=32  kv_lora=16
    """
    return {
        # LoRA down-projections compress to the latent and are replicated.
        "self_attention.linear_q_down_proj.weight": ((32, 128), (1, 1)),
        "self_attention.linear_kv_down_proj.weight": ((24, 128), (1, 1)),
        # Up-projections fan out to heads -> column-parallel on dim 0.
        "self_attention.linear_q_up_proj.weight": ((96, 32), (tp, 1)),
        "self_attention.linear_kv_up_proj.weight": ((128, 16), (tp, 1)),
        # Output projection consumes the head dim -> row-parallel on dim 1.
        "self_attention.linear_proj.weight": ((128, 64), (1, tp)),
        # Fused LoRA norms live on the latent and are replicated.
        "self_attention.linear_q_up_proj.layer_norm_weight": ((32,), (1,)),
        "self_attention.linear_kv_up_proj.layer_norm_weight": ((16,), (1,)),
    }


def _build():
    cfg = build_dsv3_config(
        TINY, {"moe_grouped_gemm": True}, tp=TP, pp=PP, ep=EP, etp=1, sp=TP > 1
    )
    spec = get_gpt_decoder_block_spec(cfg, use_transformer_engine=True)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    return GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=TINY["vocab_size"],
        max_sequence_length=64,
        pre_process=(pp_rank == 0),
        post_process=(pp_rank == PP - 1),
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        parallel_output=False,
    )


def _check_mla_metadata(model, rank: int) -> None:
    """MLA global shapes and shard axes must match what the gather assumes."""
    ssd = model.sharded_state_dict()
    # Pick a local MoE layer: local 0 is global FIRST_K_DENSE on stage 0 only.
    local = 1 if _pp_layer_offset(model) == 0 else 0
    for suffix, (want_global, want_frag) in _expected_mla_metadata(TP).items():
        key = f"decoder.layers.{local}.{suffix}"
        assert key in ssd, f"[rank{rank}] {key} absent from sharded_state_dict"
        st = ssd[key]
        assert isinstance(
            st, ShardedTensor
        ), f"[rank{rank}] {key} is {type(st).__name__}"
        assert (
            tuple(st.global_shape) == want_global
        ), f"[rank{rank}] {key} global {tuple(st.global_shape)} != {want_global}"
        assert (
            tuple(st.axis_fragmentations) == want_frag
        ), f"[rank{rank}] {key} fragmentation {tuple(st.axis_fragmentations)} != {want_frag}"
        # The local shard must be the global shape divided along the sharded axis.
        expect_local = tuple(g // f for g, f in zip(want_global, want_frag))
        assert (
            tuple(st.local_shape) == expect_local
        ), f"[rank{rank}] {key} local {tuple(st.local_shape)} != {expect_local}"


def _check_experts_and_router(model, rank: int) -> None:
    local = 1 if _pp_layer_offset(model) == 0 else 0
    # EP: experts are split across the expert-parallel group, not replicated.
    local_experts = sum(
        1
        for n, _ in model.named_parameters()
        if f"layers.{local}.mlp.experts.linear_fc1.weight" in n
    )
    assert local_experts == TINY["n_routed_experts"] // EP, (
        f"[rank{rank}] {local_experts} local experts, expected "
        f"{TINY['n_routed_experts'] // EP} at EP={EP}"
    )
    # The router and its bias are replicated: every rank must be able to emit them.
    assert f"decoder.layers.{local}.mlp.router.weight" in dict(model.named_parameters())
    assert any(
        n.endswith(f"layers.{local}.mlp.router.expert_bias")
        for n, _ in model.named_buffers()
    ), f"[rank{rank}] router bias buffer missing"


def _layer_kinds(model) -> dict:
    """{global_layer: 'dense'|'moe'} from this stage's relabelled parameter names.

    Order matters: a MoE layer also has ``mlp.shared_experts.linear_fc*``, which
    would read as dense if ``linear_fc`` were tested first.
    """
    offset = _pp_layer_offset(model)
    kinds: dict[int, str] = {}
    for name, _ in model.named_parameters():
        m = _LAYER_RE.search(_to_global_key(name, offset))
        if not m:
            continue
        gl = int(m.group(1))
        if ".mlp.experts." in name or ".mlp.shared_experts." in name:
            kinds[gl] = "moe"
        elif ".mlp.linear_fc" in name:
            kinds.setdefault(gl, "dense")
    return kinds


def _check_pipeline_layout(model, rank: int) -> None:
    """Stages must publish GLOBAL layer numbers, and the dense layer must be global.

    A stale offset republishes later stages under the first stage's numbers: the
    union check then sees a gap and the overlap check sees a collision, which is
    what the rollout weight sync would otherwise ship silently.
    """
    offset = _pp_layer_offset(model)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    per_stage = LAYERS // PP
    assert (
        offset == pp_rank * per_stage
    ), f"[rank{rank}] pp_rank={pp_rank} offset={offset}, expected {pp_rank * per_stage}"
    assert (
        len(model.decoder.layers) == per_stage
    ), f"[rank{rank}] {len(model.decoder.layers)} local layers, expected {per_stage}"

    kinds = _layer_kinds(model)
    assert sorted(kinds) == list(range(offset, offset + per_stage)), (
        f"[rank{rank}] publishes layers {sorted(kinds)}, expected "
        f"{list(range(offset, offset + per_stage))}"
    )

    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, kinds)
    merged: dict[int, str] = {}
    for g in gathered:
        for gl, kind in g.items():
            # Same global layer from two stages means the offset collapsed.
            assert merged.get(gl, kind) == kind, f"layer {gl} disagrees across stages"
            merged[gl] = kind
    assert sorted(merged) == list(range(LAYERS)), (
        f"stages cover {sorted(merged)}, expected 0..{LAYERS - 1} -- "
        "a gap or overlap means the pipeline offset is wrong"
    )
    for gl in range(LAYERS):
        want = "dense" if gl < FIRST_K_DENSE else "moe"
        assert merged[gl] == want, (
            f"global layer {gl} is {merged[gl]}, expected {want} "
            f"(first_k_dense_replace={FIRST_K_DENSE})"
        )


def main() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)

    model = _build()

    if PP > 1:
        _check_pipeline_layout(model, rank)
    else:
        _check_mla_metadata(model, rank)
        _check_experts_and_router(model, rank)

    dist.barrier()
    if rank == 0:
        print("DSV3_SHARD_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
