# SPDX-License-Identifier: MIT
"""Exercise the runtime validator's capture checks without loading model weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tests/attentions/deepseek_v41/validate_runtime.py"


def capture_checks(runner, capture_sizes, script=VALIDATOR):
    """Run the actual entrypoint's two graph checks, excluding model startup."""
    main = next(
        node
        for node in ast.parse(script.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    checks = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.graph"
    ]
    assert len(checks) == 2
    shapes = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "shapes" for t in node.targets)
    )
    bucket = next(
        node
        for node in ast.parse((ROOT / "atom/utils/forward_context.py").read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "request_bucket"
    )
    scope = {
        "args": SimpleNamespace(graph=True),
        "runner": runner,
        "capture_sizes": capture_sizes,
        "torch": torch,
    }
    module = ast.Module(body=[bucket, checks[0], shapes, checks[1]], type_ignores=[])
    # Execute repository-owned AST without importing optional GPU/model dependencies.
    exec(compile(module, str(script), "exec"), scope)  # noqa: S102
    return scope["shapes"]


def make_runner(dtype, *, draft=0, corrupt=None, graph_keys=None):
    pages = 8
    if dtype == "bf16":
        page_bytes = torch.full((pages, 32), 17, dtype=torch.uint8)
        regions = [("page", page_bytes)]
    else:
        data_width, scale_width = (256, 32) if dtype == "fp4" else (512, 16)
        source = torch.full(
            (pages * (data_width + scale_width),), 17, dtype=torch.uint8
        )
        cut = pages * data_width
        regions = [
            ("main_0.data", source[:cut].view(pages, data_width)),
            ("main_0.scale", source[cut:].view(pages, scale_width)),
            ("index_0.data", torch.full((pages, 16), 17, dtype=torch.uint8)),
            ("index_0.scale", torch.full((pages, 4), 17, dtype=torch.uint8)),
            ("page_padding", torch.full((pages, 4), 17, dtype=torch.uint8)),
        ]
        page_bytes = None
    cache = SimpleNamespace(
        page_bytes=page_bytes,
        page_regions=regions,
        geometry=SimpleNamespace(window_size=4),
    )
    scratch = -(-(4 + draft + 1) // 4)
    queries = [1] if draft == 0 else [1, draft + 1]
    if graph_keys is None:
        graph_keys = [(bs, q - 1) for bs in (1, 2, 4) for q in queries]
    runner = SimpleNamespace(
        attn_metadata_builder=SimpleNamespace(cache=cache, block_size=4),
        _dspark_capture_q_buckets=lambda _: queries,
        req_graph=SimpleNamespace(keys=lambda: graph_keys),
    )
    if draft:
        runner.drafter = SimpleNamespace(mtp_k=draft)

    def capture():
        for name, region in regions:
            region[:scratch].fill_(23)
            if name == corrupt:
                region[scratch, 0] = 99

    runner.capture_cudagraph = capture
    return runner, scratch


@pytest.mark.parametrize("dtype", ["bf16", "fp4", "fp8"])
@pytest.mark.parametrize("draft", [0, 5])
def test_capture_accepts_scratch_writes_and_current_request_graph_keys(dtype, draft):
    runner, scratch = make_runner(dtype, draft=draft)
    shapes = capture_checks(runner, [1, 2, 4])
    drafts = [0] if draft == 0 else [0, 5]
    assert shapes == [(bs, k) for bs in (1, 2, 4) for k in drafts]
    for _, region in runner.attn_metadata_builder.cache.page_regions:
        assert (region[:scratch] == 23).all()
        assert (region[scratch:] == 17).all()


@pytest.mark.parametrize("dtype", ["fp4", "fp8"])
@pytest.mark.parametrize(
    "region",
    ["main_0.data", "main_0.scale", "index_0.data", "index_0.scale", "page_padding"],
)
def test_capture_rejects_non_scratch_writes_in_every_quantized_region(dtype, region):
    runner, _ = make_runner(dtype, corrupt=region)
    with pytest.raises(AssertionError, match="non-scratch pages in " + region):
        capture_checks(runner, [1, 2, 4])


def test_capture_rejects_non_scratch_bf16_writes():
    runner, _ = make_runner("bf16", corrupt="page")
    with pytest.raises(AssertionError, match="non-scratch pages in page"):
        capture_checks(runner, [1, 2, 4])


@pytest.mark.parametrize("keys", [[(1, 0), (2, 0)], [(1, 1), (2, 1), (4, 1)]])
def test_capture_rejects_missing_or_old_format_graph_keys(keys):
    runner, _ = make_runner("fp4", graph_keys=keys)
    with pytest.raises(AssertionError, match="captured"):
        capture_checks(runner, [1, 2, 4])
