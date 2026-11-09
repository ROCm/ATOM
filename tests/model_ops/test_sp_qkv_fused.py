# SPDX-License-Identifier: MIT
"""CPU tests for the mode-B `sp_qkv_fused` call site (no aiter, no GPU).

Pins the static coverage matrix, the argument mapping into the neutral op, the
attention-tail hand-off through the impl seam (no packed-qkv leak), and the
flag-off path staying byte-identical.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

import atom.model_ops.sp_qkv_fused as SQF


# ---------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(SQF, "_WARNED", set())
    monkeypatch.setattr(SQF, "_COVERS_CACHE", {})
    monkeypatch.setattr(SQF, "_OUTPUTS", {})
    monkeypatch.setattr(SQF, "_get_gfx", lambda: "gfx950")
    monkeypatch.setattr(SQF, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(SQF, "get_sp_group", lambda: None)
    monkeypatch.setattr(
        SQF, "_cos_sin_cache", lambda *a: torch.zeros(8, 64, dtype=torch.bfloat16)
    )
    monkeypatch.setattr(SQF, "_linear_consumes_per_token_fp8", lambda _o: True)
    monkeypatch.setattr(SQF, "_cache_layout_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        SQF, "_finish_attention", lambda out, padded, real: (out, "scale")
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)


def _aiter(monkeypatch, *, routed=True, covers=True, covers_fn=None):
    op = []

    def fake_op(*args, **kwargs):
        op.append((args, kwargs))

    fake_op.calls = op
    module = types.ModuleType("aiter")
    module.sp_qkv_fused = fake_op
    if covers_fn is None:
        covers_fn = lambda *_a, **_kw: covers
    module.sp_qkv_covers = covers_fn
    routed_ops = {"sp_qkv_fused": {"version": 1}} if routed else {}
    module.plugin = SimpleNamespace(status=lambda: {"routed_ops": routed_ops})
    monkeypatch.setitem(sys.modules, "aiter", module)
    return module, op


def _drop_covers(monkeypatch):
    del sys.modules["aiter"].sp_qkv_covers


def _impl_class(*, mode, index_cache=True):
    class FakeImpl:
        num_heads = 16
        head_dim = 128
        num_kv_heads = 1
        kv_cache_dtype = "fp8"
        index_head_dim = 128 if mode != "dense" else 0
        num_idx_heads = 1 if mode != "dense" else 0

        def __init__(self):
            self.calls = []
            self.index_cache = torch.zeros(2, dtype=torch.float8_e4m3fn)

        def gemm_fusion_prologue_state(self, fwd_ctx):
            k = torch.zeros((2, 2, 1, 4, 16), dtype=torch.float8_e4m3fn)
            v = torch.zeros((2, 2, 1, 4, 16), dtype=torch.float8_e4m3fn)
            scale = torch.zeros((2, 2, 1, 128), dtype=torch.float32)
            state = {
                "k_cache": k,
                "v_cache": v,
                "k_scale": scale,
                "v_scale": scale,
                "slot_mapping": torch.arange(8, dtype=torch.int64),
                "k_scale_view": scale,
                "v_scale_view": scale,
            }
            return state

        def forward_after_prologue(
            self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx, index_q=None
        ):
            self.calls.append(
                SimpleNamespace(
                    q=q,
                    k=k,
                    v=v,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    index_q=index_q,
                    qkv_passed=False,
                )
            )
            return torch.zeros(
                (q.shape[0], self.num_heads * self.head_dim), dtype=torch.bfloat16
            )

    return FakeImpl


def _module(*, mode="dense", shuffled=True, weight_scale=True, o_proj_width=8192):
    impl = _impl_class(mode=mode)()
    weight = torch.zeros((9216, 6144 // 2), dtype=torch.uint8)
    weight.is_shuffled = shuffled
    qkv_proj = SimpleNamespace(
        weight=weight,
        weight_scale=torch.zeros(9216) if weight_scale else None,
    )
    module = SimpleNamespace(
        layer_num=3,
        is_indexed_sparse_attention=(mode != "dense"),
        skip_index_topk=(mode == "reuse"),
        qkv_proj=qkv_proj,
        o_proj=SimpleNamespace(input_size=o_proj_width),
        q_norm=SimpleNamespace(weight=torch.zeros(128), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.zeros(128)),
        rotary_emb=SimpleNamespace(rotary_dim=64),
    )
    module.index_q_norm = (
        SimpleNamespace(weight=torch.zeros(128)) if mode != "dense" else None
    )
    module.index_k_norm = module.index_q_norm
    attn = SimpleNamespace(
        impl=impl,
        layer_name=f"layer{module.layer_num}.attn",
        _m3_sp_fp8_output_supported=True,
        _m3_tp_replicated_o_proj=False,
        _sp_qkv_fused=None,
    )
    module.attn = attn
    return module


def _config(**overrides):
    values = {
        "enable_sp_qkv_fused": True,
        "sequence_parallel_size": 4,
        "tensor_parallel_size": 1,
        "kv_cache_dtype": "fp8",
        "kv_cache_block_size": 128,
        "max_num_batched_tokens": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------------------------------------------------------------- bind matrix
def test_flag_off_binds_nothing_and_install_is_inert(monkeypatch):
    _aiter(monkeypatch)
    monkeypatch.setattr(SQF, "_is_m3_attention", lambda _m: True)
    module = _module()
    original = module.forward = lambda *a, **k: "native"
    assert SQF.bind(module, _config(enable_sp_qkv_fused=False), 3) is None
    assert (
        SQF.install_m3(
            SimpleNamespace(modules=lambda: [module]),
            _config(enable_sp_qkv_fused=False),
        )
        == 0
    )
    assert module.forward is original
    assert module.attn._sp_qkv_fused is None


def test_no_op_in_aiter_refuses(monkeypatch):
    _aiter(monkeypatch)
    module = _module()
    monkeypatch.setattr(
        SQF, "_aiter_op", lambda: (None, "this aiter has no sp_qkv_fused op")
    )
    assert SQF.bind(module, _config(), 3) is None


def test_not_routed_refuses(monkeypatch):
    _aiter(monkeypatch, routed=False)
    module = _module()
    assert SQF.bind(module, _config(), 3) is None


def test_coverage_predicate_receives_mode_and_cell_set(monkeypatch):
    seen = []

    def covers(*args, **kwargs):
        x = args[1]
        cos, sin = args[7], args[8]
        seen.append(
            {
                "mode": kwargs["mode"],
                "m": int(x.shape[0]),
                "cos_stride": cos.stride(0),
                "sin_stride": sin.stride(0),
                "row_stride_ok": cos.stride(1) == sin.stride(1) == 1,
            }
        )
        return int(x.shape[0]) <= 256

    _aiter(monkeypatch, covers_fn=covers)
    assert SQF.bind(_module(), _config(), 3) is not None
    assert seen
    # The prologue band is queried per bucket cell, not just at the max.
    assert [s["m"] for s in seen] == [1, 2]
    assert seen[0]["mode"] == "dense"
    assert seen[0]["cos_stride"] == seen[0]["sin_stride"]
    assert seen[0]["cos_stride"] % 8 == 0
    assert seen[0]["row_stride_ok"]


def test_uncovered_cell_refuses_and_keeps_existing_path(monkeypatch):
    _aiter(monkeypatch, covers=False)
    assert SQF.bind(_module(), _config(), 3) is None


def test_missing_coverage_predicate_refuses(monkeypatch):
    _aiter(monkeypatch)
    _drop_covers(monkeypatch)
    assert SQF.bind(_module(), _config(), 3) is None


def test_coverage_predicate_failure_refuses(monkeypatch):
    def covers(*args, **kwargs):
        raise RuntimeError("predicate exploded")

    _aiter(monkeypatch, covers_fn=covers)
    assert SQF.bind(_module(), _config(), 3) is None


def test_large_band_config_is_not_bound_by_the_prologue_predicate(monkeypatch):
    # The 16K recipe (max_num_batched_tokens 65536, SP4 -> local m 16384) mixes
    # prologue-size cells with the one-stage large band; because every cell the
    # deployment can present is queried, the uncovered 16384 cell refuses the
    # binding and the existing ATOM path keeps 16K in this PR.
    seen = []

    def covers(*args, **kwargs):
        x = args[1]
        seen.append(int(x.shape[0]))
        return int(x.shape[0]) <= 256

    _aiter(monkeypatch, covers_fn=covers)
    cfg = _config(max_num_batched_tokens=65536)
    assert SQF.bind(_module(), cfg, 3) is None
    assert 16384 in seen
    assert set(range(1, 257)) & set(seen)


def test_wrong_arch_refuses(monkeypatch):
    _aiter(monkeypatch)
    monkeypatch.setattr(SQF, "_get_gfx", lambda: "gfx942")
    assert SQF.bind(_module(), _config(), 3) is None


def test_world_sp_tp_refuse(monkeypatch):
    _aiter(monkeypatch)
    assert SQF.bind(_module(), _config(sequence_parallel_size=2), 3) is None
    assert SQF.bind(_module(), _config(tensor_parallel_size=2), 3) is None
    monkeypatch.setattr(SQF, "get_sp_world_size", lambda: 2)
    assert SQF.bind(_module(), _config(), 3) is None


def test_kv_matrix_refuses(monkeypatch):
    _aiter(monkeypatch)
    assert SQF.bind(_module(), _config(kv_cache_dtype="bf16"), 3) is None
    assert SQF.bind(_module(), _config(kv_cache_block_size=16), 3) is None
    module = _module()
    module.attn.impl.kv_cache_dtype = "bf16"
    assert SQF.bind(module, _config(), 3) is None


def test_weight_matrix_refuses(monkeypatch):
    _aiter(monkeypatch)
    assert SQF.bind(_module(shuffled=False), _config(), 3) is None
    assert SQF.bind(_module(weight_scale=False), _config(), 3) is None
    module = _module()
    module.qkv_proj.weight = torch.zeros((9216, 6144), dtype=torch.bfloat16)
    assert SQF.bind(module, _config(), 3) is None


def test_tail_seam_refuses(monkeypatch):
    _aiter(monkeypatch)
    module = _module()
    module.attn._m3_sp_fp8_output_supported = False
    assert SQF.bind(module, _config(), 3) is None
    module = _module(o_proj_width=4096)
    assert SQF.bind(module, _config(), 3) is None
    module = _module()
    module.attn.impl = SimpleNamespace(kv_cache_dtype="fp8")
    assert SQF.bind(module, _config(), 3) is None


# ---------------------------------------------------------------- supported path
def _bound(monkeypatch, *, mode="dense"):
    _aiter(monkeypatch)
    module = _module(mode=mode)
    bound = SQF.bind(module, _config(), 3)
    assert bound is not None
    return module, bound


def test_cos_sin_are_separate_zero_copy_views_with_driver_constraints(monkeypatch):
    _, bound = _bound(monkeypatch)
    cos, sin = bound._cos_cache, bound._sin_cache
    assert cos.shape == (8, 32) and sin.shape == (8, 32)
    # Halves of M3's contiguous [max_pos, rotary_dim] cache are valid as-is:
    # equal stride(0), stride(0) % 8 == 0, contiguous rows, aligned bases.
    assert cos.stride(1) == sin.stride(1) == 1
    assert cos.stride(0) == sin.stride(0)
    assert cos.stride(0) % 8 == 0
    assert cos.data_ptr() != sin.data_ptr()


def test_output_buffers_are_shared_bucket_sized_and_address_stable(monkeypatch):
    # The runtime stages bucket-sized outputs in its binding and copies the real
    # rows into the caller's buffer, so caller output pointers are not bound:
    # one bucket-sized set can be shared by every site and must stay put.
    _, first = _bound(monkeypatch)
    _, second = _bound(monkeypatch)
    assert first._outs.q.data_ptr() == second._outs.q.data_ptr()
    q_before = first._outs.q.data_ptr()
    first._outputs(4)
    first._outputs(8)
    assert first._outs.q.data_ptr() == q_before
    assert first._max_tokens == 8  # sp_pad_len(8, 4)


def test_supported_dense_calls_op_with_contract_args(monkeypatch):
    module, bound = _bound(monkeypatch)
    x = torch.zeros((2, 6144), dtype=torch.uint8)
    x_scale = torch.zeros(2, dtype=torch.float32)
    positions = torch.arange(6, dtype=torch.int64)
    fwd_ctx = SimpleNamespace()

    out, scale = bound(x, x_scale, positions, fwd_ctx)

    calls = sys.modules["aiter"].sp_qkv_fused.calls
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 15  # the 15 positional contract tensors
    assert args[1] is x and args[2] is x_scale
    assert args[9] is positions
    assert kwargs["mode"] == "dense"
    assert kwargs["k"] is not None and kwargs["v"] is not None
    assert kwargs["index_q"] is None
    assert kwargs["eps"] == 1e-6
    assert kwargs["index_cache"] is None
    assert out.shape[0] == positions.numel() and scale == "scale"

    impl_call = module.attn.impl.calls[0]
    assert impl_call.q.shape[0] == positions.numel()
    assert impl_call.qkv_passed is False


def test_supported_indexer_requires_index_tensors(monkeypatch):
    module, bound = _bound(monkeypatch, mode="indexer")
    out, _ = bound(
        torch.zeros((2, 6144), dtype=torch.uint8),
        torch.zeros(2, dtype=torch.float32),
        torch.arange(6, dtype=torch.int64),
        SimpleNamespace(),
    )
    _, kwargs = sys.modules["aiter"].sp_qkv_fused.calls[0]
    assert kwargs["mode"] == "indexer"
    assert kwargs["index_q"] is not None
    assert kwargs["k"] is None and kwargs["v"] is None
    impl_call = module.attn.impl.calls[0]
    assert impl_call.index_q is not None
    assert out.shape[0] == 6


def test_supported_reuse_keeps_kv_but_no_index(monkeypatch):
    _module, bound = _bound(monkeypatch, mode="reuse")
    bound(
        torch.zeros((2, 6144), dtype=torch.uint8),
        torch.zeros(2, dtype=torch.float32),
        torch.arange(6, dtype=torch.int64),
        SimpleNamespace(),
    )
    _, kwargs = sys.modules["aiter"].sp_qkv_fused.calls[0]
    assert kwargs["mode"] == "reuse"
    assert kwargs["k"] is not None and kwargs["v"] is not None
    assert kwargs["index_q"] is None
    assert kwargs["index_cache"] is None


def test_install_swaps_only_eligible_forwards(monkeypatch):
    _aiter(monkeypatch)
    module = _module()
    original = module.forward = lambda *a, **k: "native"
    monkeypatch.setattr(SQF, "_is_m3_attention", lambda m: m is module)
    n = SQF.install_m3(
        SimpleNamespace(modules=lambda: [module, SimpleNamespace()]), _config()
    )
    assert n == 1
    assert module.attn._sp_qkv_fused is not None
    assert module.forward is not original
    # Idempotent per module.
    assert SQF.install_m3(SimpleNamespace(modules=lambda: [module]), _config()) == 0
