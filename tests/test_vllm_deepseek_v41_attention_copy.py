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

# sha256 of core's `Attention.forward`, comments and blank lines stripped.
CORE_PREPARE_DIGEST = "4f502d41815e8e82"
CORE_FORWARD_DIGEST = "ae2d04bba437a30e"


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
    scope = {"torch": torch}
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


def _core_forward() -> str:
    """`Attention.forward`'s source, by AST rather than by text markers.

    `forward` is the last method in its class, so a "next def" marker has
    nothing to find -- and a marker that silently matches to end-of-file
    would fold every later edit into the digest.
    """
    src = CORE.read_text()
    cls = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.ClassDef) and n.name == "Attention"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    return "\n".join(
        line.rstrip()
        for line in src.split("\n")[fn.lineno - 1 : fn.end_lineno]
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
    assert len(body.split(NEWLINE)) > 80, "the _prepare marker matched too little"
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
        (
            "atom/model_ops/engram/device/staging.py",
            "EngramStaging",
            ("start", "consume", "join"),
            "7d1dcc8553e48b9d",
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
