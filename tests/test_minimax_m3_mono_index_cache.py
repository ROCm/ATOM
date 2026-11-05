# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU regressions for mono with index storage only on selecting layers.

Execute production binding, pointer packing and probing without importing GPU
compilers. Meta weights retain the real geometry without allocating a model.
"""

import ast
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from atom.models.minimax_m3.mono import config
from atom.mono.runtime.consensus import MonoUnsupported
from tests.test_minimax_m3_index_cache_rows import ROOT, definitions

MONO = "atom/models/minimax_m3/mono/"
FP8 = torch.float8_e4m3fn


@pytest.fixture
def sparse_layer_type():
    namespace = dict(vars(config))
    namespace.update(
        __name__=__name__,
        torch=torch,
        dataclass=dataclass,
        MonoUnsupported=MonoUnsupported,
        dtypes=NS(fp8=FP8),
        QuantType=NS(per_Token=NS(value=1)),
        weight_is_stored_preshuffled=lambda *_: True,
    )
    definitions(
        MONO + "weights.py",
        {"_need", "_ptpc_fp8", "_bf16_vec", "_cos_sin", "SparseMoeLayer"},
        namespace,
    )
    return namespace["SparseMoeLayer"]


def _layer(skip, cache, kv_dtype="fp8"):
    def tensor(*shape, dtype=torch.bfloat16):
        return torch.empty(*shape, dtype=dtype, device="meta")

    def linear(rows, cols):
        return NS(
            weight=tensor(rows, cols, dtype=FP8),
            weight_scale=tensor(rows, dtype=torch.float32),
            quant_type=NS(value=1),
            params_dtype=FP8,
        )

    hidden, inter, experts = config.HIDDEN, config.INTER, config.N_ROUTED + 1
    w13 = tensor(experts * 2 * inter * hidden // 2, dtype=torch.uint8)
    w2 = tensor(experts * hidden * inter // 2, dtype=torch.uint8)
    w13.is_shuffled = w2.is_shuffled = True
    norm = NS(weight=tensor(config.HEAD_DIM))
    return NS(
        is_moe_layer=True,
        self_attn=NS(
            is_indexed_sparse_attention=True,
            num_heads=config.LOCAL_Q_HEADS,
            num_kv_heads=1,
            num_idx_heads=1,
            head_dim=config.HEAD_DIM,
            rotary_emb=NS(
                rotary_dim=config.ROTARY_DIM,
                _minimax_m3_cos_sin_cache=tensor(1, config.ROTARY_DIM),
            ),
            topk_blocks=config.TOPK_BLOCKS,
            init_blocks=0,
            local_blocks=1,
            attn=NS(impl=NS(kv_cache_dtype=kv_dtype, index_cache=cache)),
            skip_index_topk=skip,
            qkv_proj=linear(config.IndexHeads().rows, hidden),
            o_proj=linear(hidden, config.O_K),
            q_norm=norm,
            k_norm=norm,
            index_q_norm=norm,
            index_k_norm=norm,
        ),
        input_layernorm=NS(weight=tensor(hidden)),
        post_attention_layernorm=NS(weight=tensor(hidden)),
        block_sparse_moe=NS(
            fuse_shared_experts=True,
            experts=NS(
                num_fused_shared_experts=1,
                top_k=config.TOP_K,
                quant_method=NS(
                    is_guinterleave=False, intermediate_pad=0, hidden_pad=0
                ),
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=tensor(experts * 2 * inter * hidden // 32),
                w2_weight_scale=tensor(experts * hidden * inter // 32),
            ),
            gate=NS(weight=tensor(config.N_ROUTED, hidden)),
            e_score_correction_bias=tensor(config.N_ROUTED, dtype=torch.float32),
        ),
    )


@pytest.mark.parametrize("skip,dtype", [(True, None), (True, FP8), (False, FP8)])
def test_binding_accepts_compact_and_full_index_storage(sparse_layer_type, skip, dtype):
    cache = None if dtype is None else torch.empty(1, dtype=dtype)
    bound = sparse_layer_type.from_layer(_layer(skip, cache), 4, config.IndexHeads())
    assert bound.index_topk is (not skip)
    assert bound.attn_impl.index_cache is cache


@pytest.mark.parametrize(
    "skip,dtype,kv_dtype",
    [(False, None, "fp8"), (False, torch.bfloat16, "fp8"), (True, None, "bf16")],
)
def test_binding_still_rejects_required_cache_mismatches(
    sparse_layer_type, skip, dtype, kv_dtype
):
    cache = None if dtype is None else torch.empty(1, dtype=dtype)
    with pytest.raises(MonoUnsupported):
        sparse_layer_type.from_layer(
            _layer(skip, cache, kv_dtype), 4, config.IndexHeads()
        )


@pytest.mark.parametrize("reuse_has_storage", [False, True])
def test_allocate_packs_null_index_pointer_only_for_reuse(reuse_has_storage):
    namespace = dict(vars(config))
    namespace.update(
        torch=torch,
        K1_SCRATCH_BYTES=16,
        K4_SCRATCH_BYTES=16,
        owned_peer_buffer=lambda *_: (None, None),
        sym_layout=lambda _: {"_bytes": 0},
        StepMailboxes=lambda *_, **__: None,
    )
    source = ast.parse((ROOT / MONO / "kernels/pre_attn.py").read_text())
    namespace["K1_ARGS"] = next(
        ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "K1_ARGS" for t in node.targets)
    )
    definitions(MONO + "runner.py", {"_ptr"}, namespace)
    definitions(MONO + "runner.py", {"_allocate"}, namespace, "MonoDecodeRunner")
    cache = torch.empty(1, dtype=FP8)
    weights = {
        name: torch.empty(1)
        for name in namespace["K1_ARGS"]
        if name not in {"ar", "index_cache", "iq_out", "scratch"}
    }
    runner = NS(
        dev="cpu",
        npes=1,
        rank=0,
        debug=False,
        sparse=[
            NS(**weights, index_topk=True, attn_impl=NS(index_cache=cache)),
            NS(
                **weights,
                index_topk=False,
                attn_impl=NS(index_cache=cache if reuse_has_storage else None),
            ),
        ],
    )
    namespace["_allocate"](runner, None)
    packed = [dict(zip(namespace["K1_ARGS"], row.tolist())) for row in runner.k1_args]
    assert packed[0]["index_cache"] == cache.data_ptr()
    assert packed[1]["index_cache"] == 0
    for i, ptrs in enumerate(packed):
        assert ptrs["ar"] == runner.ars[i].data_ptr()
        assert all(ptrs[name] == value.data_ptr() for name, value in weights.items())


@pytest.mark.parametrize("reuse_has_storage", [False, True])
def test_probe_fingerprints_only_caches_read_by_each_layer(reuse_has_storage):
    cache = torch.ones(1, 128, 2, dtype=torch.bfloat16)
    kv = NS(k_cache=cache, v_cache=cache * 2)
    fwd = NS(kv_cache_data={"layer_4": kv, "layer_5": kv})
    namespace = {
        "torch": torch,
        "N_DENSE": 1,
        "HIDDEN": 2,
        "get_forward_context": lambda: fwd,
        "tensor_model_parallel_all_reduce": lambda t: t,
        "token_rows": lambda *_: (torch.tensor([[0]]), torch.tensor([129])),
    }
    definitions(MONO + "check.py", {"_fingerprint", "forward_probed"}, namespace)
    hidden = torch.ones(1, 2, dtype=torch.bfloat16)
    runner = NS(
        model=NS(
            get_input_embeddings=lambda _: hidden,
            layers=[lambda pos, h, res: (h, h.clone())],
        ),
        mailboxes=NS(begin_step=lambda: None),
        ars=[hidden.clone(), hidden.clone()],
        sparse=[
            NS(layer_id=4, index_topk=True, attn_impl=NS(index_cache=cache)),
            NS(
                layer_id=5,
                index_topk=False,
                attn_impl=NS(index_cache=cache if reuse_has_storage else None),
            ),
        ],
        run_sparse_layer=lambda i, lw, fwd, pos, res, rows: res,
        finish_step=lambda res: res,
    )
    sums = {}
    out = namespace["forward_probed"](
        runner, torch.tensor([1]), torch.tensor([128]), sums
    )
    fingerprint = namespace["_fingerprint"]
    kv_sum = fingerprint(kv.k_cache) + fingerprint(kv.v_cache)
    assert sums["cache4"] == kv_sum + fingerprint(cache)
    assert sums["cache5"] == kv_sum
    torch.testing.assert_close(out, hidden)
