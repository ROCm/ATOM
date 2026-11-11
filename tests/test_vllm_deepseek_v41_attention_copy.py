# SPDX-License-Identifier: MIT
"""The plugin's attention forward is a copy, and copies go stale.

`atom.plugin.vllm.v41_core_overrides` carries its own `Attention.forward` --
the compressor fork, the rotations, the decode/prefill dispatch and the ring
write -- so that the attention half can sit inside a capture break writing
into a caller-owned buffer. Core returns its output directly to ATOM's own
engine.

The hazard that buys is specific: if `atom/models/deepseek_v41/attention.py`
changes how it attends and this copy does not, the copy keeps running the
old arithmetic. That is not a crash. The PR that added `queries` for bounded
replay would have been exactly this -- a new branch in core, absent here,
answering with the wrong rows on a path nobody re-measured.

So this pins core's body by digest. When it goes red the message is "core's
forward moved; port the change", not "the copy is wrong" -- the test cannot
know which. Updating the digest without reading the diff is the one way to
defeat it, which is why the failure says so.

Full-line comments and blank lines are excluded: a re-worded comment is not a change
in what the forward computes, and a digest that fired on those would be
retrained into noise within a month.
"""

import ast
import hashlib
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

NEWLINE = chr(10)
ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "atom/models/deepseek_v41/attention.py"
PLUGIN = ROOT / "atom/plugin/vllm/v41_core_overrides.py"

# sha256 of core's `Attention.forward` and `_attend`, comments and blank
# lines stripped.
CORE_PREPARE_DIGEST = "50a174931df035a4"
CORE_FORWARD_DIGEST = "cf3b6f63a9e378be"


@pytest.fixture
def plugin_attention():
    """Exercise the copied functions in CPU CI without importing vLLM/AITER."""
    names = {"_attn_out_buffer", "_attend_block", "_forward"}
    functions = [
        node
        for node in ast.parse(PLUGIN.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    scope = {"torch": torch, "FP8_DTYPE": torch.float8_e4m3fn}
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=functions, type_ignores=[]), str(PLUGIN), "exec"),
        scope,
    )
    return SimpleNamespace(**{name: scope[name] for name in names})


@pytest.mark.parametrize("fused_quant", [False, True])
@pytest.mark.parametrize("bounded_queries", [False, True])
def test_plugin_forward_passes_fused_quant_into_capture_break(
    plugin_attention, fused_quant, bounded_queries
):
    calls = []

    class Attention:
        _use_fused_attention = staticmethod(lambda *args: False)
        heads = 2
        head_dim = 16

        def _atom_v41_attend_block(
            self, attn_out, hidden, scale, cache, step, rope, queries, hidden_fp8=None
        ):
            calls.append(
                (self, attn_out, hidden, scale, cache, step, rope, queries, hidden_fp8)
            )

    attention = Attention()
    attention._project_out = Mock(return_value=object())
    hidden = torch.empty(1, 2, 32, dtype=torch.bfloat16)
    hidden_fp8 = (
        torch.empty_like(hidden, dtype=torch.float8_e4m3fn) if fused_quant else None
    )
    scale = torch.empty(1, 2, 1, dtype=torch.uint8) if fused_quant else None
    positions = object()
    cache = SimpleNamespace(rope_positions=Mock(return_value=positions))
    queries = SimpleNamespace(positions=object()) if bounded_queries else None
    step, rope = object(), object()
    kwargs = {"hidden_fp8": hidden_fp8} if fused_quant else {}

    output = plugin_attention._forward(
        attention, hidden, scale, cache, step, rope, queries, **kwargs
    )

    assert len(calls) == 1
    call = calls[0]
    assert call[0] is attention
    attn_out = call[1]
    assert attn_out.shape == (1, 2, 2, 16)
    assert attn_out.dtype == torch.bfloat16
    assert call[2] is hidden and call[3] is scale
    assert call[4] is cache and call[5] is step and call[6] is rope
    assert call[7] is queries and call[8] is hidden_fp8
    attention._project_out.assert_called_once_with(
        attn_out, rope, queries.positions if bounded_queries else positions
    )
    assert output is attention._project_out.return_value


def test_plugin_forward_carries_mx_scale_across_the_break_as_bytes(
    plugin_attention,
):
    """vLLM's weak-ref op refuses float8_e8m0fnu (ScalarType 44), so the MX
    scale must reach the break as uint8 over the same storage."""
    calls = []

    class Attention:
        _use_fused_attention = staticmethod(lambda *args: False)
        heads = 2
        head_dim = 16

        def _atom_v41_attend_block(self, *args, **kwargs):
            calls.append((args, kwargs))

    attention = Attention()
    attention._project_out = Mock(return_value=object())
    hidden = torch.empty(1, 2, 32, dtype=torch.bfloat16)
    hidden_fp8 = torch.empty_like(hidden, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, 2, 1, dtype=torch.float8_e8m0fnu)
    cache = SimpleNamespace(rope_positions=Mock(return_value=object()))

    plugin_attention._forward(
        attention, hidden, scale, cache, object(), object(), None, hidden_fp8=hidden_fp8
    )

    ((args, kwargs),) = calls
    crossed = args[2]  # self is bound, so args starts at attn_out
    assert crossed.dtype == torch.uint8
    assert crossed.data_ptr() == scale.data_ptr() and crossed.shape == scale.shape
    assert args[7] is hidden_fp8
    assert kwargs == {"hidden_scale_dtype": torch.float8_e8m0fnu}


def test_plugin_attend_block_restores_the_mx_scale(plugin_attention):
    class ReachedSelection(Exception):
        pass

    hidden = torch.empty(1, 2, 32, dtype=torch.bfloat16)
    hidden_fp8 = torch.empty_like(hidden, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, 2, 1, dtype=torch.float8_e8m0fnu)
    attention = SimpleNamespace(
        _fork_compress=Mock(return_value=True),
        project_qkv=Mock(return_value=(object(), object())),
        qk_norm=Mock(return_value=(object(), object(), object())),
        _fork_select=Mock(side_effect=ReachedSelection),
    )
    cache = SimpleNamespace(current_step=object())
    with pytest.raises(ReachedSelection):
        plugin_attention._attend_block(
            attention,
            None,
            hidden,
            scale.view(torch.uint8),
            cache,
            object(),
            object(),
            None,
            hidden_fp8,
            hidden_scale_dtype=torch.float8_e8m0fnu,
        )
    (qkv_input, restored), _ = attention.project_qkv.call_args
    assert qkv_input is hidden_fp8
    assert restored.dtype == torch.float8_e8m0fnu
    assert restored.data_ptr() == scale.data_ptr()


@pytest.mark.parametrize("fused_quant", [False, True])
def test_plugin_quantized_qkv_preserves_bf16_auxiliary_readers(
    plugin_attention, fused_quant
):
    class ReachedSelection(Exception):
        pass

    hidden = torch.empty(1, 2, 32, dtype=torch.bfloat16)
    hidden_fp8 = (
        torch.empty_like(hidden, dtype=torch.float8_e4m3fn) if fused_quant else None
    )
    scale = torch.empty(1, 2, 1, dtype=torch.uint8) if fused_quant else None
    qr, qr_scale, kv = object(), object(), object()
    attention = SimpleNamespace(
        _fork_compress=Mock(return_value=True),
        project_qkv=Mock(return_value=(object(), object())),
        qk_norm=Mock(return_value=(qr, qr_scale, kv)),
        _fork_select=Mock(side_effect=ReachedSelection),
    )
    frozen_step, live_step, rope = object(), object(), object()
    cache = SimpleNamespace(current_step=live_step)
    kwargs = {"hidden_fp8": hidden_fp8} if fused_quant else {}
    with pytest.raises(ReachedSelection):
        plugin_attention._attend_block(
            attention,
            None,
            hidden,
            scale,
            cache,
            frozen_step,
            rope,
            None,
            **kwargs,
        )
    attention._fork_compress.assert_called_once_with(hidden, cache, live_step, rope)
    attention.project_qkv.assert_called_once_with(
        hidden_fp8 if fused_quant else hidden, scale
    )
    attention._fork_select.assert_called_once_with(
        hidden, qr, qr_scale, cache, live_step, rope, compressed=True
    )


@pytest.fixture
def core_attention():
    """Exercise native eager-boundary arguments without loading GPU kernels."""
    cls = next(
        node
        for node in ast.parse(CORE.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Attention"
    )
    functions = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"forward", "_attend"}
    ]
    assert len(functions) == 2
    for node in functions:
        node.decorator_list = []
    scope = {
        "torch": torch,
        "FP8_DTYPE": torch.float8_e4m3fn,
        "join_stream": lambda *args: None,
        "fused_sparse_attn_v41_paged_2src": Mock(),
    }
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=functions, type_ignores=[]), str(CORE), "exec"),
        scope,
    )
    return SimpleNamespace(**scope)


@pytest.mark.parametrize("stage", ["idle", "decode", "prefill"])
def test_core_fused_output_scale_crosses_break_without_copy(core_attention, stage):
    """A byte-only weak-ref boundary must preserve the kernel's E8M0 storage."""
    hidden = torch.zeros(1, 2, 32, dtype=torch.bfloat16)
    kv = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    cache = SimpleNamespace(
        norm_rope_quant_window=Mock(return_value=(hidden, None, kv, None)),
        attention_indices=Mock(return_value=(None, None, None, None)),
        rope_positions=Mock(return_value=torch.zeros(2, dtype=torch.int64)),
        ring_view=Mock(return_value=object()),
        write_window=Mock(),
        pool=object(),
        packed=False,
    )
    step = SimpleNamespace(
        requests=() if stage == "idle" else (object(),), decode=stage == "decode"
    )
    attention = SimpleNamespace(
        spec=SimpleNamespace(layer_id=0),
        heads=8,
        head_dim=4,
        groups=1,
        attn_sink=None,
        softmax_scale=1.0,
        index_stream=None,
        compress_stream=None,
        qk_norm=object(),
        _fork_compress=Mock(),
        project_qkv=Mock(return_value=(hidden, kv)),
        _use_fused_attention=Mock(return_value=True),
        _fork_select=Mock(),
        wq_b=Mock(return_value=hidden),
        _project_quantized_out=Mock(),
    )
    addresses = []

    def byte_only_boundary(*args, **kwargs):
        scale = kwargs["out_scale"]
        assert scale.dtype == torch.uint8
        addresses.append(scale.data_ptr())
        return core_attention._attend(attention, *args, **kwargs)

    def kernel(*args, out, out_scale, **kwargs):
        assert out_scale.dtype == torch.float8_e8m0fnu
        assert out_scale.data_ptr() == addresses[-1]
        out.view(torch.uint8).zero_()
        out_scale.view(torch.uint8).fill_(127)

    attention._attend = byte_only_boundary
    core_attention.fused_sparse_attn_v41_paged_2src.side_effect = kernel
    core_attention.forward(attention, hidden, None, cache, step, object())

    assert len(addresses) == 1
    output, scale, _, _ = attention._project_quantized_out.call_args.args
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.data_ptr() == addresses[0]
    assert torch.all(scale.view(torch.uint8) == 127)
    assert torch.count_nonzero(output.view(torch.uint8)) == 0
    assert core_attention.fused_sparse_attn_v41_paged_2src.call_count == (
        stage != "idle"
    )


def _core_forward() -> str:
    """The source of the attention half the plugin copies: `Attention.forward`
    and `_attend`, the region it attends in, by AST rather than by text
    markers.

    A "next def" marker would have nothing to find after the class's last
    method -- and a marker that silently matches to end-of-file would fold
    every later edit into the digest.
    """
    src = CORE.read_text()
    cls = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.ClassDef) and n.name == "Attention"
    )
    lines = src.split("\n")
    return "\n".join(
        line.rstrip()
        for name in ("forward", "_attend")
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
        for line in lines[fn.lineno - 1 : fn.end_lineno]
        if line.strip() and not line.strip().startswith("#")
    )


def test_the_markers_still_match_something():
    """A digest over an empty string would pass forever."""
    body = _core_forward()
    assert len(body.split("\n")) > 40, body[:200]
    assert "sparse_attn_v4_paged_2src" in body


def test_core_forward_has_not_moved_under_the_copy():
    digest = hashlib.sha256(_core_forward().encode()).hexdigest()[:16]
    assert digest == CORE_FORWARD_DIGEST, (
        "`Attention.forward` in atom/models/deepseek_v41/attention.py has "
        "changed. The vLLM plugin keeps a copy of it in "
        "atom/plugin/vllm/v41_core_overrides.py (`_attend_block` / "
        "`_forward`) so the attention half can run inside a capture break. "
        "Port the change there, then update CORE_FORWARD_DIGEST to "
        f"{digest!r}. Updating the digest alone leaves the plugin running "
        "the old arithmetic, which produces wrong rows rather than an error."
    )


def _core_prepare() -> str:
    src = (ROOT / "atom/model_ops/attentions/deepseek_v41/backend.py").read_text()
    cls = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41MetadataBuilder"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare"
    )
    return NEWLINE.join(
        line.rstrip()
        for line in src.split(NEWLINE)[fn.lineno - 1 : fn.end_lineno]
        if line.strip() and not line.strip().startswith("#")
    )


def test_core_prepare_has_not_moved_under_its_copy():
    """The second copy, and the one with the worse history.

    `_prepare` reads a batch protocol that main extends: the V4.1 report
    added `is_final_chunk` and `replaying` to it, and the plugin's fabricated
    batches did not carry them -- which cost a four-minute model load and an
    AttributeError in a GPU worker before a single token. A copy of the
    reader makes that class of change silent instead, so it is pinned.
    """
    body = _core_prepare()
    assert len(body.split(NEWLINE)) > 60, "the _prepare marker matched too little"
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    assert digest == CORE_PREPARE_DIGEST, (
        "`DeepseekV41MetadataBuilder._prepare` has changed. The vLLM plugin "
        "keeps a copy in atom/plugin/vllm/v41_core_overrides.py so a captured "
        "dummy batch can reuse one scratch cache. Port the change, then set "
        f"CORE_PREPARE_DIGEST to {digest!r}."
    )


def test_the_copy_still_covers_both_dispatch_arms():
    """Cheap structural check beside the digest: the copy must still carry
    the four kernel paths, so a merge that drops one is caught even if the
    digest was refreshed."""
    copy = PLUGIN.read_text()
    for name in (
        "packed_decode",
        "packed_prefill",
        "sparse_attn_v4_paged_2src",
        "write_window",
        "rope_quant_window",
        "norm_rope_quant_window",
        "fused_sparse_attn_v41_paged_2src",
        "_project_quantized_out",
    ):
        assert name in copy, f"the attention copy lost {name}"


@pytest.mark.parametrize(
    "path,class_name,methods,expected",
    [
        (
            "atom/models/deepseek_v41/model.py",
            "DeepseekV41ForCausalLM",
            ("forward_hidden",),
            "e5d8ebd181754862",
        ),
        # Core's `start` / `join` skip the lookup on a step the model's decode
        # round serves (`EngramStep.served_by_round`); not ported: the plugin's
        # runner has no decode round, so its steps are never served by one.
        (
            "atom/model_ops/engram/device/staging.py",
            "EngramStaging",
            ("start", "consume", "join"),
            "74eade6acaa9b896",
        ),
    ],
)
def test_copied_model_and_staging_sources_have_not_drifted(
    path, class_name, methods, expected
):
    source = (ROOT / path).read_text()
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    selected = [
        next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for name in methods
    ]
    body = "\n".join(
        "\n".join(
            line.rstrip()
            for line in source.split("\n")[node.lineno - 1 : node.end_lineno]
            if line.strip() and not line.strip().startswith("#")
        )
        for node in selected
    )
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    assert digest == expected, (
        f"{path} {methods} changed. Audit and port the relevant behavior into "
        "the vLLM plugin copy before updating this source digest. "
        f"Observed digest: {digest}"
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_plugin_fused_forward_owns_both_output_planes(plugin_attention, bounded):
    calls = []

    class Attention:
        heads, head_dim, groups = 16, 64, 2
        _use_fused_attention = staticmethod(lambda *args: True)

        def _atom_v41_attend_block(self, *args, **kwargs):
            calls.append((args, kwargs))
            args[0].view(torch.uint8).fill_(5)
            kwargs["attn_out_scale"].fill_(127)

    model = Attention()
    model._project_out = Mock()
    model._project_quantized_out = Mock(return_value=object())
    hidden = torch.empty(1, 4, 32, dtype=torch.bfloat16)
    hidden_scale = torch.empty(1, 4, 1, dtype=torch.float8_e8m0fnu)
    queries = (
        SimpleNamespace(rows=torch.tensor([1, 3]), positions=object())
        if bounded
        else None
    )
    result = plugin_attention._forward(
        model, hidden, hidden_scale, object(), object(), object(), queries
    )
    ((args, kwargs),) = calls
    values, scales, leading, dtype = model._project_quantized_out.call_args.args
    rows = 2 if bounded else 4
    assert values is args[0]
    assert values.shape == (rows, 2, 8 * model.head_dim)
    assert values.dtype == torch.float8_e4m3fn
    assert scales.shape == (rows, 2, 8 * model.head_dim // 32)
    assert scales.dtype == torch.float8_e8m0fnu
    assert kwargs["attn_out_scale"].dtype == torch.uint8
    assert kwargs["attn_out_scale"].data_ptr() == scales.data_ptr()
    assert args[2].dtype == torch.uint8
    assert args[2].data_ptr() == hidden_scale.data_ptr()
    assert kwargs["hidden_scale_dtype"] == hidden_scale.dtype
    assert tuple(leading) == (1, rows) and dtype == hidden.dtype
    assert torch.all(values.view(torch.uint8) == 5)
    assert torch.all(scales.view(torch.uint8) == 127)
    model._project_out.assert_not_called()
    assert result is model._project_quantized_out.return_value


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_plugin_fused_region_writes_capture_buffers_and_orders_ring(
    plugin_attention, monkeypatch, decode, bounded
):
    events = []
    hidden = torch.empty(1, 4, 32, dtype=torch.bfloat16)
    q = torch.empty_like(hidden)
    kv = torch.empty(1, 4, 4, dtype=torch.bfloat16)
    q_scale = torch.empty(1, 4, 1, dtype=torch.uint8)
    window = None if decode else object()
    live = SimpleNamespace(decode=decode, requests=(object(),), width=4)
    frozen = SimpleNamespace(decode=not decode, requests=(), width=8)
    positions, rope = object(), object()

    def norm(*args):
        assert args[-1] is live
        events.append("norm/ring" if decode else "norm")
        return q, q_scale, None if decode else kv, window

    def write(layer, rows, step):
        assert step is live and rows is window
        if rows is not None:
            events.append("write")

    cache = SimpleNamespace(
        current_step=live,
        norm_rope_quant_window=norm,
        query_fp8=Mock(side_effect=AssertionError("fused Q must remain BF16")),
        rope_quant_window=Mock(side_effect=AssertionError("KV is already prepared")),
        attention_indices=Mock(return_value=(0, 1, 2, 3)),
        rope_positions=Mock(return_value=positions),
        write_window=write,
        ring_view=Mock(return_value=object()),
        pool=object(),
        packed=False,
        kv_fp8=True,
    )
    model = SimpleNamespace(
        heads=8,
        head_dim=4,
        spec=SimpleNamespace(layer_id=0),
        attn_sink=object(),
        softmax_scale=0.5,
        _fork_compress=Mock(return_value=True),
        project_qkv=Mock(return_value=(q, kv)),
        qk_norm=Mock(side_effect=AssertionError("norm should be fused")),
        _fork_select=lambda *a, **k: events.append("select"),
        wq_b=lambda q, **k: q,
    )
    queries = None
    rows = 4
    if bounded:
        rows = 2
        queries = SimpleNamespace(
            positions=object(),
            take=lambda *xs: tuple(x[:, [1, 3]] for x in xs),
            indices=Mock(return_value=(4, 5, 6, 7)),
        )
    values = torch.empty(rows, 1, 32, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows, 1, 1, dtype=torch.uint8)

    def attend(query, pool, *args, **kwargs):
        events.append("attend")
        assert query.shape == (rows, 8, 4) and query.dtype == torch.bfloat16
        assert pool is cache.pool
        assert kwargs["out"] is values
        assert kwargs["out_scale"].data_ptr() == scales.data_ptr()
        assert kwargs["out_scale"].dtype == torch.float8_e8m0fnu
        assert kwargs["positions"] is (queries.positions if bounded else positions)
        kwargs["out"].view(torch.uint8).fill_(7)
        kwargs["out_scale"].view(torch.uint8).fill_(127)
        return kwargs["out"], kwargs["out_scale"]

    scope = plugin_attention._attend_block.__globals__
    monkeypatch.setitem(scope, "fused_sparse_attn_v41_paged_2src", attend)
    monkeypatch.setitem(scope, "join_stream", lambda *args: events.append("join"))
    model.index_stream, model.compress_stream = object(), object()
    result = plugin_attention._attend_block(
        model,
        values,
        hidden,
        None,
        cache,
        frozen,
        rope,
        queries,
        attn_out_scale=scales,
    )
    assert result is None
    assert torch.all(values.view(torch.uint8) == 7) and torch.all(scales == 127)
    model._fork_compress.assert_called_once_with(hidden, cache, live, rope)
    assert events == (
        ["norm/ring", "select", "join", "attend"]
        if decode
        else ["norm", "select", "join", "attend", "write"]
    )
    cache.rope_quant_window.assert_not_called()


def test_plugin_fused_idle_replay_initializes_both_planes(plugin_attention):
    values = torch.full((2, 1, 32), 255, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.full((2, 1, 1), 255, dtype=torch.uint8)
    cache = SimpleNamespace(current_step=SimpleNamespace(requests=()))
    plugin_attention._attend_block(
        object(),
        values,
        None,
        None,
        cache,
        object(),
        None,
        None,
        attn_out_scale=scales,
    )
    assert values.view(torch.uint8).count_nonzero() == 0
    assert torch.all(scales == 127)
