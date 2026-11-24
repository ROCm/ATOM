# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""torchrun worker: streaming Megatron-native export vs the original materializing copy.

Not collected by pytest. Env:

* ``NATIVE_TP`` / ``NATIVE_PP`` / ``NATIVE_EP`` / ``NATIVE_MOE`` (0/1)
* ``ATOMRL_SYNC_PREFETCH_MB`` (0, 1024, or ``mixed``: 1024 on even ranks, 0 on odd)

Prints ``NATIVE_EXPORT_OK`` on success.
"""

from __future__ import annotations

import os
import re
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.environ["ATOMRL_ROOT"])

from atomrl.engine.training.bridges.core import (
    expert_local_index,
    relabel_expert_index,
)
from atomrl.engine.training.bridges.gpt import GPTDims, GPTMoEDims
from atomrl.engine.training.megatron_native_engine import (
    MegatronNativeEngine,
    _pp_layer_offset,
    _to_global_key,
)
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.mapping import (
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.transformer_config import TransformerConfig

TP = int(os.environ.get("NATIVE_TP", "1"))
PP = int(os.environ.get("NATIVE_PP", "1"))
EP = int(os.environ.get("NATIVE_EP", "1"))
MOE = os.environ.get("NATIVE_MOE", "0") == "1"
LAYERS = 4
HIDDEN = 128
HEADS = 4
HEAD_DIM = 32
FFN = 256
VOCAB = 256
N_EXPERTS = 8
MOE_FFN = 64


# ---------------------------------------------------------------------------
# Verbatim old export (the reference). Do not "improve" this copy.
# ---------------------------------------------------------------------------


def _old_tp_gather_named_params(engine) -> dict:
    tp = mpu.get_tensor_model_parallel_world_size()
    ssd = engine.module.sharded_state_dict()
    offset = _pp_layer_offset(engine.module)
    ffn = engine._dims.ffn
    out: dict = {}
    group = mpu.get_tensor_model_parallel_group() if tp > 1 else None
    for name, p in engine.module.named_parameters():
        p = p.detach().contiguous()
        gkey = _to_global_key(name, offset)
        if tp == 1:
            out[gkey] = p
            continue
        gathered = [torch.empty_like(p) for _ in range(tp)]
        dist.all_gather(gathered, p, group=group)
        st = ssd.get(name)
        if isinstance(st, ShardedTensorFactory):
            shard = ffn // tp
            gate = torch.cat([g[:shard] for g in gathered], dim=0)
            up = torch.cat([g[shard:] for g in gathered], dim=0)
            full = torch.cat([gate, up], dim=0)
        elif isinstance(st, ShardedTensor):
            gshape = tuple(st.global_shape[st.prepend_axis_num :])
            lshape = tuple(st.local_shape)
            split_dim = next(
                (d for d in range(len(lshape)) if lshape[d] != gshape[d]), None
            )
            full = (
                gathered[0] if split_dim is None else torch.cat(gathered, dim=split_dim)
            )
        else:
            full = gathered[0]
        out[gkey] = full
    return out


def _old_full_megatron_named_params(engine):
    stage_params = _old_tp_gather_named_params(engine)
    pp = mpu.get_pipeline_model_parallel_world_size()
    if pp == 1:
        return list(stage_params.items())
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    meta_local = [(k, tuple(v.shape), v.dtype) for k, v in stage_params.items()]
    gathered_meta: list = [None] * pp
    dist.all_gather_object(gathered_meta, meta_local, group=pp_group)
    out: dict = {}
    for src in range(pp):
        src_global = dist.get_global_rank(pp_group, src)
        for k, shape, dtype in gathered_meta[src]:
            if src == pp_rank:
                t = stage_params[k].contiguous()
            else:
                t = torch.empty(shape, dtype=dtype, device="cuda")
            dist.broadcast(t, src=src_global, group=pp_group)
            out[k] = t
    return list(out.items())


def _old_moe_stage_named_params(engine):
    ep = mpu.get_expert_model_parallel_world_size()
    etp = mpu.get_expert_tensor_parallel_world_size()
    tp = mpu.get_tensor_model_parallel_world_size()
    num_local = engine._num_experts // ep
    ssd = engine.module.sharded_state_dict()
    offset = _pp_layer_offset(engine.module)
    tp_group = mpu.get_tensor_model_parallel_group() if tp > 1 else None
    etp_group = mpu.get_expert_tensor_parallel_group() if etp > 1 else None
    ep_group = mpu.get_expert_model_parallel_group() if ep > 1 else None
    for name, param in engine.module.named_parameters():
        p = param.detach().contiguous()
        exp = expert_local_index(name)
        if exp is None:
            gkey = _to_global_key(name, offset)
            if tp == 1:
                yield gkey, p
                continue
            gathered = [torch.empty_like(p) for _ in range(tp)]
            dist.all_gather(gathered, p, group=tp_group)
            st = ssd.get(name)
            if isinstance(st, ShardedTensorFactory):
                sh = p.shape[0] // 2
                gate = torch.cat([g[:sh] for g in gathered], dim=0)
                up = torch.cat([g[sh:] for g in gathered], dim=0)
                yield gkey, torch.cat([gate, up], dim=0)
            elif isinstance(st, ShardedTensor):
                gshape = tuple(st.global_shape[st.prepend_axis_num :])
                lshape = tuple(st.local_shape)
                split_dim = next(
                    (dd for dd in range(len(lshape)) if lshape[dd] != gshape[dd]), None
                )
                full = (
                    gathered[0]
                    if split_dim is None
                    else torch.cat(gathered, dim=split_dim)
                )
                yield gkey, full
            else:
                yield gkey, gathered[0]
            continue
        local_e, which_fc = exp
        if etp > 1:
            g = [torch.empty_like(p) for _ in range(etp)]
            dist.all_gather(g, p, group=etp_group)
            if which_fc == "1":
                sh = p.shape[0] // 2
                gate = torch.cat([x[:sh] for x in g], dim=0)
                up = torch.cat([x[sh:] for x in g], dim=0)
                p = torch.cat([gate, up], dim=0)
            else:
                p = torch.cat(g, dim=1)
            del g
        if ep == 1:
            yield _to_global_key(relabel_expert_index(name, local_e), offset), p
            continue
        g = [torch.empty_like(p) for _ in range(ep)]
        dist.all_gather(g, p, group=ep_group)
        for j in range(ep):
            global_e = j * num_local + local_e
            gname = _to_global_key(relabel_expert_index(name, global_e), offset)
            yield gname, g[j]
            g[j] = None


def _old_full_megatron_named_params_moe(engine):
    pp = mpu.get_pipeline_model_parallel_world_size()
    if pp == 1:
        yield from _old_moe_stage_named_params(engine)
        return
    stage: dict = dict(_old_moe_stage_named_params(engine))
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    meta_local = [(k, tuple(v.shape), v.dtype) for k, v in stage.items()]
    gathered_meta: list = [None] * pp
    dist.all_gather_object(gathered_meta, meta_local, group=pp_group)
    for src in range(pp):
        src_global = dist.get_global_rank(pp_group, src)
        for k, shape, dtype in gathered_meta[src]:
            if src == pp_rank:
                t = stage.pop(k).contiguous()
            else:
                t = torch.empty(shape, dtype=dtype, device="cuda")
            dist.broadcast(t, src=src_global, group=pp_group)
            yield k, t


# ---------------------------------------------------------------------------
# Tiny TE GPTModel + dummy native engine
# ---------------------------------------------------------------------------


def _build_model() -> GPTModel:
    sp = MOE and TP > 1
    moe_kwargs: dict = {}
    if MOE:
        moe_kwargs = {
            "num_moe_experts": N_EXPERTS,
            "moe_ffn_hidden_size": MOE_FFN,
            "moe_router_topk": 2,
            "moe_grouped_gemm": True,
            "moe_router_load_balancing_type": "aux_loss",
            "moe_aux_loss_coeff": 0.0,
            "expert_model_parallel_size": EP,
            "expert_tensor_parallel_size": 1,
            "moe_router_pre_softmax": False,
        }
    if MOE or PP > 1:
        moe_kwargs["moe_token_dispatcher_type"] = "alltoall"
    tfcfg = TransformerConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=HEADS,
        kv_channels=HEAD_DIM,
        ffn_hidden_size=FFN,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        sequence_parallel=sp,
        use_cpu_initialization=True,
        variable_seq_lengths=(PP > 1),
        **moe_kwargs,
    )
    if MOE:
        spec = get_gpt_decoder_block_spec(tfcfg, use_transformer_engine=True)
    else:
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    model = GPTModel(
        config=tfcfg,
        transformer_layer_spec=spec,
        vocab_size=VOCAB,
        max_sequence_length=32,
        pre_process=(pp_rank == 0),
        post_process=(pp_rank == PP - 1),
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        parallel_output=False,
    )
    return model.cuda()


def _make_engine(model) -> MegatronNativeEngine:
    engine = MegatronNativeEngine.__new__(MegatronNativeEngine)
    engine.module = model
    engine._num_experts = N_EXPERTS if MOE else 0
    if MOE:
        engine._dims = GPTMoEDims(
            num_layers=LAYERS,
            hidden=HIDDEN,
            num_heads=HEADS,
            num_kv_groups=HEADS,
            head_dim=HEAD_DIM,
            ffn=FFN,
            vocab=VOCAB,
            num_experts=N_EXPERTS,
            moe_ffn=MOE_FFN,
        )
    else:
        engine._dims = GPTDims(
            num_layers=LAYERS,
            hidden=HIDDEN,
            num_heads=HEADS,
            num_kv_groups=HEADS,
            head_dim=HEAD_DIM,
            ffn=FFN,
            vocab=VOCAB,
        )
    return engine


def _to_cpu(pairs):
    return [(n, t.detach().contiguous().cpu()) for n, t in pairs]


def _assert_owners(engine, rank: int) -> None:
    local = [
        (o.name, o.pp_stage, o.ep_owner)
        for rec in engine._stage_plan()
        for o in rec.outputs
    ]
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    owners: dict[str, tuple[int, int | None]] = {}
    for recs in gathered:
        for name, pp_stage, ep_owner in recs:
            key = (pp_stage, ep_owner)
            prev = owners.get(name)
            assert (
                prev is None or prev == key
            ), f"[rank{rank}] {name} has owners {prev} and {key}"
            owners[name] = key
    meta = engine._stage_metadata()
    got = [(n, tuple(t.shape), t.dtype) for n, t in engine._stage_named_params()]
    assert got == [
        (n, tuple(s), d) for n, s, d in meta
    ], f"[rank{rank}] _stage_metadata does not match _stage_named_params"


_STACK_TOKEN = re.compile(r"(\d+)::(\d+)")


def _expand_stacks(pairs):
    """``stack_experts`` output -> the per-expert names and tensors it stands for."""
    out = []
    for name, t in pairs:
        m = _STACK_TOKEN.search(name)
        if m is None:
            out.append((name, t))
            continue
        start, step = int(m.group(1)), int(m.group(2))
        for k in range(t.shape[0]):
            out.append((f"{name[:m.start()]}{start + k * step}{name[m.end():]}", t[k]))
    return out


def _assert_stacked(engine, per_expert, rank: int) -> None:
    recipes = engine._stage_plan(stack_experts=True)
    meta = engine._stage_metadata(recipes)
    got = [(n, tuple(t.shape), t.dtype) for n, t in engine._stage_named_params(recipes)]
    assert got == [
        (n, tuple(s), d) for n, s, d in meta
    ], f"[rank{rank}] stacked _stage_metadata does not match _stage_named_params"
    stacked = _to_cpu(engine._full_megatron_named_params_moe(stack_experts=True))
    assert any("::" in n for n, _ in stacked), f"[rank{rank}] no expert stacks emitted"
    expanded = _expand_stacks(stacked)
    assert [n for n, _ in expanded] == [
        n for n, _ in per_expert
    ], f"[rank{rank}] stacked name/order mismatch"
    for (n1, t1), (_, t2) in zip(expanded, per_expert, strict=True):
        assert torch.equal(t1, t2), f"[rank{rank}] stacked {n1} differs"


def main() -> None:
    rank = int(os.environ["RANK"])
    if os.environ.get("ATOMRL_SYNC_PREFETCH_MB") == "mixed":
        os.environ["ATOMRL_SYNC_PREFETCH_MB"] = "1024" if rank % 2 == 0 else "0"
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)
    engine = _make_engine(_build_model())

    _assert_owners(engine, rank)
    dist.barrier()

    torch.cuda.reset_peak_memory_stats()
    if MOE:
        new = _to_cpu(engine._full_megatron_named_params_moe())
    else:
        new = _to_cpu(engine._full_megatron_named_params())
    peak = torch.cuda.max_memory_allocated()
    dist.barrier()

    if MOE:
        old = _to_cpu(_old_full_megatron_named_params_moe(engine))
    else:
        old = _to_cpu(_old_full_megatron_named_params(engine))

    assert [n for n, _ in new] == [n for n, _ in old], (
        f"[rank{rank}] name/order mismatch:\n new={[n for n, _ in new][:8]}\n"
        f" old={[n for n, _ in old][:8]}"
    )
    for (n1, t1), (n2, t2) in zip(new, old, strict=True):
        assert n1 == n2
        assert torch.equal(t1, t2), f"[rank{rank}] {n1} tensors differ"

    if MOE and EP > 1:
        dist.barrier()
        _assert_stacked(engine, new, rank)

    dist.barrier()
    if rank == 0:
        print(
            f"NATIVE_EXPORT_PEAK_ALLOC_BYTES={peak} "
            f"tensors={len(new)} prefetch={os.environ.get('ATOMRL_SYNC_PREFETCH_MB', '?')}",
            flush=True,
        )
        print("NATIVE_EXPORT_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
