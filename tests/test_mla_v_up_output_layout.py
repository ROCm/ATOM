# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise V-up's output layout on CPU with only the GPU BMMs stubbed."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def projection():
    # Module-level AITER/Triton imports prevent a normal import on the CPU gate.
    # Execute the production method bodies, removing only the trace decorator.
    path = Path(__file__).resolve().parents[1] / "atom/model_ops/attention_mla.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLAAttention"
    )
    names = {
        "_v_up_proj",
        "_v_up_proj_and_o_proj",
        "_dcp_project_merge_out",
        "_dcp_project_merge_out_prefill",
        "_dcp_project_merge_out_head_major",
        "_configure_dcp_prefill_routes",
    }
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    for method in methods:
        method.decorator_list = []
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_DCP_BF16_PACKED_TAIL_MAX_ROWS"
            for target in node.targets
        )
    ]
    assert len(constants) == 1
    namespace = {"torch": torch}
    exec(  # noqa: S102 - execute only the checked-in production method bodies
        compile(
            ast.Module(body=constants + methods, type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    attn = type("Projection", (), {name: namespace[name] for name in names})()
    attn.qrep_enabled = False
    attn.dcp_prefill_merge_bf16_ok = True
    attn.num_heads = 3
    attn.kv_lora_rank = 8
    attn.v_head_dim = 4
    return attn, namespace


def _transport(monkeypatch, function):
    module = ModuleType("atom.model_ops.dcp_ops")
    module.cp_lse_a2a_head_major = function
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _stub_bmm(namespace, branch, weights, scale):
    seen = {}

    def write_output(x, weight, output):
        assert weight is weights
        seen.update(x=x, output=output)
        result = torch.bmm(x.float(), weight.float().transpose(1, 2))
        output.copy_(result.transpose(0, 1))
        return output

    def bf16(x, weight, *, YQ):
        seen["YQ"] = YQ
        write_output(x, weight, YQ.transpose(0, 1))
        return YQ

    def fp8(x, weight, weight_scale, *, group_size, YQ, transpose_bm):
        assert weight_scale is scale
        assert group_size == 128 and transpose_bm
        return write_output(x, weight, YQ)

    def fp4(x, weight, weight_scale, *, y, transpose_bm, prequant, y_scale):
        assert weight_scale is scale
        assert transpose_bm and prequant and y_scale is None
        return write_output(x, weight, y)

    def unexpected(*args, **kwargs):
        pytest.fail("V-up dispatched to the wrong BMM branch")

    kernels = {
        "bf16": "_aiter_triton_bf16_bmm",
        "fp8": "_aiter_triton_fp8_bmm",
        "fp4": "batched_gemm_a16wfp4",
    }
    namespace.update({name: unexpected for name in kernels.values()})
    namespace[kernels[branch]] = {"bf16": bf16, "fp8": fp8, "fp4": fp4}[branch]
    namespace["is_rocm_aiter_fp4bmm_enabled"] = lambda: branch == "fp4"
    return seen


def _inputs(tokens, heads, rank, v_dim, noncontiguous=False):
    width = 2 * rank if noncontiguous else rank
    x = (torch.arange(tokens * heads * width) % 7).to(torch.bfloat16)
    x = x.reshape(tokens, heads, width)
    if noncontiguous:
        x = x[..., 1::2]
        assert not x.is_contiguous() and x.storage_offset() == 1
    weights = (torch.arange(heads * v_dim * rank) % 3).to(torch.bfloat16)
    return x, weights.reshape(heads, v_dim, rank)


@pytest.mark.parametrize("branch", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_v_up_output_is_contiguous_and_o_proj_receives_a_view(
    projection, branch, noncontiguous
):
    attn, namespace = projection
    tokens, heads, rank, v_dim = 5, attn.num_heads, attn.kv_lora_rank, attn.v_head_dim
    x, weights = _inputs(tokens, heads, rank, v_dim, noncontiguous)
    scale = None if branch == "bf16" else torch.ones(heads)
    attn.W_V, attn.W_V_scale = weights, scale
    seen = _stub_bmm(namespace, branch, weights, scale)

    def o_proj(flat):
        output = seen["output"]
        assert output.shape == (tokens, heads, v_dim)
        assert output.dtype == torch.bfloat16
        assert output.stride() == (heads * v_dim, v_dim, 1)
        assert output.is_contiguous()
        assert flat.shape == (tokens, heads * v_dim)
        assert flat.is_contiguous()
        assert flat.data_ptr() == output.data_ptr()
        if branch == "bf16":
            yq = seen["YQ"]
            assert yq.shape == (heads, tokens, v_dim)
            assert yq.stride() == (v_dim, heads * v_dim, 1)
            assert yq.data_ptr() == output.data_ptr()
        return flat

    attn.o_proj = o_proj
    result = attn._v_up_proj_and_o_proj(x)
    expected = torch.einsum("tnl,nvl->tnv", x.float(), weights.float())
    assert torch.equal(result, expected.to(torch.bfloat16).reshape(tokens, -1))


@pytest.mark.parametrize("branch", ["bf16", "fp8", "fp4"])
def test_v_up_output_uses_the_dcp_head_and_weight_overrides(projection, branch):
    attn, namespace = projection
    tokens, heads = 5, 2 * attn.num_heads
    x, weights = _inputs(tokens, heads, attn.kv_lora_rank, attn.v_head_dim, True)
    scale = None if branch == "bf16" else torch.ones(heads)
    attn.W_V = torch.empty(attn.num_heads, attn.v_head_dim, attn.kv_lora_rank)
    attn.W_V_scale = None if branch == "bf16" else torch.zeros(attn.num_heads)
    seen = _stub_bmm(namespace, branch, weights, scale)

    result = attn._v_up_proj(x, W_V=weights, W_V_scale=scale, num_heads=heads)

    assert result.shape == (tokens, heads, attn.v_head_dim)
    assert result.is_contiguous()
    assert result.data_ptr() == seen["output"].data_ptr()
    expected = torch.einsum("tnl,nvl->tnv", x.float(), weights.float())
    assert torch.equal(result, expected.to(torch.bfloat16))


def test_head_major_projection_stores_heads_for_exchange(projection, monkeypatch):
    attn, namespace = projection
    tokens, heads = 5, attn.num_heads
    x, weights = _inputs(tokens, heads, attn.kv_lora_rank, attn.v_head_dim, True)
    attn.W_V_dcp, attn.dcp_group = weights, object()
    attn.o_proj = lambda flat: flat
    seen = _stub_bmm(namespace, "bf16", weights, None)

    def exchange(output, lse, group, **kwargs):
        assert group is attn.dcp_group
        assert output.shape == (heads, tokens, attn.v_head_dim)
        assert output.is_contiguous() and output.data_ptr() == seen["YQ"].data_ptr()
        return output.transpose(0, 1).contiguous()

    _transport(monkeypatch, exchange)
    result = attn._dcp_project_merge_out_head_major(x, None)
    expected = torch.einsum("tnl,nvl->tnv", x.float(), weights.float())
    assert torch.equal(result, expected.to(torch.bfloat16).reshape(tokens, -1))


@pytest.mark.parametrize(
    "backend,decode,fused,fp32,heads_layout",
    [
        ("a2a", False, False, False, True),
        ("mailbox", False, False, False, True),
        ("mailbox", True, False, False, False),
        ("a2a", False, True, False, False),
        ("ag_rs", False, False, False, False),
        ("a2a", False, False, True, True),
    ],
)
@pytest.mark.parametrize("tokens", [1, 32, 33])
def test_projection_layout_follows_the_merge_that_can_use_it(
    projection, monkeypatch, backend, decode, fused, fp32, heads_layout, tokens
):
    attn, namespace = projection
    heads = attn.num_heads
    heads_layout = heads_layout and tokens > 32
    x, weights = _inputs(tokens, heads, attn.kv_lora_rank, attn.v_head_dim)
    attn.W_V, attn.W_V_dcp = weights, weights
    attn.W_V_scale = attn.W_V_dcp_scale = None
    attn.pbm_enabled, attn.dcp_comm_backend = True, backend
    dtype = torch.float8_e4m3fn
    attn._dcp_fused_quant_dtype = lambda: dtype if fused else None
    attn.o_proj = lambda flat: flat
    seen = _stub_bmm(namespace, "bf16", weights, None)
    counts = torch.ones(tokens, dtype=torch.int32)

    def merge(out, lse, **kwargs):
        assert "head_major" not in kwargs and not heads_layout
        assert kwargs.get("decode", False) == decode
        assert kwargs["owned_counts"] is counts
        assert out.dtype == (torch.float32 if fp32 else torch.bfloat16)
        assert seen["output"].is_contiguous()
        return (out, None) if kwargs.get("quant_dtype") is not None else out

    def head_merge(out, lse, group, **kwargs):
        assert heads_layout and out.is_contiguous()
        assert kwargs["owned_counts"] is counts
        assert out.dtype == (torch.float32 if fp32 else torch.bfloat16)
        return out.transpose(0, 1).contiguous()

    attn.dcp_group = object()
    attn.dcp_prefill_merge_bf16_ok = not fp32
    attn._dcp_merge = merge
    _transport(monkeypatch, head_merge)
    attn._configure_dcp_prefill_routes()
    if decode:
        result = attn._dcp_project_merge_out(
            x, None, decode=True, merge_in_fp32=fp32, owned_counts=counts
        )
    else:
        result = attn._dcp_prefill_merge_out(
            x, None, merge_in_fp32=fp32, owned_counts=counts
        )
    expected = torch.einsum("tnl,nvl->tnv", x.float(), weights.float())
    assert torch.equal(result, expected.to(torch.bfloat16).reshape(tokens, -1))


@pytest.mark.parametrize("name", ["dcp_lse_merge", "cp_lse_a2a"])
def test_ordinary_merge_interface_has_no_head_major_switch(name):
    path = Path(__file__).resolve().parents[1] / "atom/model_ops/dcp_ops.py"
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    assert "head_major" not in {a.arg for a in function.args.args}


@pytest.mark.parametrize("branch", ["fp8", "fp4", "bf16"])
@pytest.mark.parametrize("tokens", [1, 32, 33])
def test_prefill_row_count_selects_projection_and_transport_together(
    projection, monkeypatch, branch, tokens
):
    attn, namespace = projection
    attn.num_heads = 2
    heads = 2 * attn.num_heads
    x, weights = _inputs(tokens, heads, attn.kv_lora_rank, attn.v_head_dim)
    scale = None if branch == "bf16" else torch.ones(heads)
    attn.W_V = attn.W_V_dcp = weights
    attn.W_V_scale = attn.W_V_dcp_scale = scale
    attn.pbm_enabled, attn.dcp_comm_backend = True, "a2a"
    attn._dcp_fused_quant_dtype = lambda: None
    attn.o_proj = lambda value: value
    seen = _stub_bmm(namespace, branch, weights, scale)

    path = Path(__file__).resolve().parents[1] / "atom/model_ops/dcp_ops.py"
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"cp_lse_a2a", "cp_lse_a2a_head_major", "_lse_pack_slots"}
    ]
    transport = {
        "torch": torch,
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    }
    exec(  # noqa: S102 - real transport wrapper, with collectives/kernels stubbed
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        transport,
    )
    calls = []

    def exchange(recv, send, *, group):
        calls.append(send.shape)
        recv.zero_()

    class Kernel:
        def __init__(self, output_index):
            self.output_index = output_index

        def __getitem__(self, grid):
            return lambda *args, **kwargs: args[self.output_index].zero_()

    transport.update(
        _dcp_a2a_pack_kernel=Kernel(3),
        _dcp_a2a_unpack_combine_kernel=Kernel(1),
        _dcp_a2a_combine_kernel=Kernel(2),
    )
    monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange)
    cp_group = SimpleNamespace(world_size=2, device_group=object())

    def merge(out, lse, **kwargs):
        return transport["cp_lse_a2a"](out, lse, cp_group)

    attn._dcp_merge = merge
    attn.dcp_group = cp_group
    _transport(monkeypatch, transport["cp_lse_a2a_head_major"])
    attn._configure_dcp_prefill_routes()
    result = attn._dcp_prefill_merge_out(x, torch.zeros(tokens, heads))
    assert result.shape == (tokens, attn.num_heads * attn.v_head_dim)
    head_major = branch == "bf16" and tokens > 32
    if head_major:
        assert calls == [(heads, tokens, attn.v_head_dim), (heads, tokens)]
        assert seen["YQ"].is_contiguous()
    else:
        assert calls == [(2, tokens, attn.num_heads, attn.v_head_dim + 2)]
        assert seen["output"].is_contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask_empty", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_head_major_exchange_preserves_partials_and_fp32_lse(
    monkeypatch, dtype, mask_empty, return_lse
):
    """Exercise the real exchange wrapper; only the collective/kernel are CPU stand-ins."""
    path = Path(__file__).resolve().parents[1] / "atom/model_ops/dcp_ops.py"
    names = {"cp_lse_a2a_head_major", "_lse_pack_slots", "mask_empty_dcp_lse"}
    functions = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    }
    exec(  # noqa: S102 - execute only the checked-in CPU wrapper bodies
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    ranks, tokens, heads, width = 2, 3, 4, 8
    local_heads = heads // ranks
    g = torch.Generator().manual_seed(3311)
    partials = torch.randn(ranks, tokens, heads, width, generator=g).to(dtype)
    lse = torch.randn(ranks, tokens, heads, generator=g)
    counts = torch.ones(ranks, tokens, dtype=torch.int32)
    if mask_empty:
        counts[0, 0] = 0
        counts[:, 1] = 0
        partials[counts == 0] = float("nan")
    expected_lse = lse.masked_fill(counts[:, :, None] == 0, float("-inf"))
    calls = []
    group = object()

    def exchange(recv, send, *, group):
        assert group is cp_group.device_group
        calls.append(send)
        if len(calls) == 1:
            assert send.dtype == dtype
            expected = partials[0].transpose(0, 1).contiguous()
            assert torch.equal(send.view(torch.uint8), expected.view(torch.uint8))
            peers = partials[:, :, :local_heads].transpose(1, 2).contiguous()
            recv.copy_(peers.reshape(heads, tokens, width))
        else:
            assert len(calls) == 2
            assert send.dtype == torch.float32 and send.shape == (heads, tokens)
            assert torch.equal(send, expected_lse[0].T)
            peers = expected_lse[:, :, :local_heads].transpose(1, 2).contiguous()
            recv.copy_(peers.reshape(heads, tokens))

    class Combine:
        def __getitem__(self, grid):
            assert grid == (tokens, local_heads)
            return self.run

        def run(
            self,
            recv,
            recv_lse_ptr,
            out,
            out_lse,
            recv_stride_n,
            recv_stride_b,
            recv_stride_h,
            recv_lse_stride_n,
            recv_lse_stride_b,
            recv_lse_stride_h,
            out_stride_b,
            out_stride_h,
            lse_stride_b,
            lse_stride_h,
            n_ranks,
            *,
            WRITE_LSE,
            **kwargs,
        ):
            assert recv.dtype == dtype
            assert n_ranks == ranks and WRITE_LSE == return_lse
            received = torch.as_strided(
                recv,
                (ranks, tokens, local_heads, width),
                (recv_stride_n, recv_stride_b, recv_stride_h, 1),
            )
            assert torch.equal(
                received.view(torch.uint8),
                partials[:, :, :local_heads].view(torch.uint8),
            )
            received_lse = torch.as_strided(
                recv_lse_ptr,
                (ranks, tokens, local_heads),
                (recv_lse_stride_n, recv_lse_stride_b, recv_lse_stride_h),
            )
            assert torch.equal(received_lse, expected_lse[:, :, :local_heads])
            weights = torch.softmax(received_lse.double(), dim=0).nan_to_num()
            torch.as_strided(
                out,
                (tokens, local_heads, width),
                (out_stride_b, out_stride_h, 1),
            ).copy_((received.double().nan_to_num() * weights[..., None]).sum(0))
            if WRITE_LSE:
                torch.as_strided(
                    out_lse,
                    (tokens, local_heads),
                    (lse_stride_b, lse_stride_h),
                ).copy_(torch.logsumexp(received_lse, dim=0))
            else:
                assert out_lse is None

    cp_group = SimpleNamespace(world_size=ranks, device_group=group)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange)
    namespace["_dcp_a2a_combine_kernel"] = Combine()
    result = namespace["cp_lse_a2a_head_major"](
        partials[0].transpose(0, 1).contiguous(),
        lse[0],
        cp_group,
        owned_counts=counts[0] if mask_empty else None,
        return_lse=return_lse,
    )
    out, out_lse = result if return_lse else (result, None)
    assert len(calls) == 2
    expected_weights = torch.softmax(expected_lse.double(), dim=0).nan_to_num()
    expected = (partials.double().nan_to_num() * expected_weights[..., None]).sum(0)
    assert torch.equal(out, expected[:, :local_heads].to(dtype))
    if return_lse:
        assert torch.equal(
            out_lse, torch.logsumexp(expected_lse, dim=0)[:, :local_heads]
        )
