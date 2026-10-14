# SPDX-License-Identifier: MIT
"""Delayed MHC integration: model math, ownership and changing graph inputs."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)
if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
    pytest.skip("HIP Mega-MHC integration requires gfx950", allow_module_level=True)

pytest.importorskip("aiter.ops.mega_mhc")

from atom.model_ops.deepseek_v41 import mhc_pre_delayed as mhc


def inputs(rows=8, hidden=5120, post=True, fn=None, packed=None):
    g = torch.Generator(device="cuda").manual_seed(413 + rows + hidden)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g)

    residual = randn(1, rows, 4, hidden).bfloat16()
    pre = randn(1, rows, 4).sigmoid()
    fn = randn(24, 4 * hidden) * 0.01 if fn is None else fn
    scale = torch.tensor([0.7, 1.3, 0.9], device="cuda")
    base = randn(24) * 0.2
    weight = (1 + 0.1 * randn(hidden)).bfloat16()
    packed = mhc.pack_mhc_fn(fn) if packed is None else packed
    options = {
        "rms_eps": 1e-20,
        "hc_eps": 1e-6,
        "sinkhorn_iters": 20,
        "post_mult": 2.0,
        "norm_eps": 1e-20,
        "w_preshuffle_bf16": packed is not None,
    }
    if post:
        options.update(
            sublayer_output=randn(1, rows, hidden).bfloat16(),
            post_mix=randn(1, rows, 4).sigmoid() * 2,
            combination=randn(1, rows, 4, 4).softmax(-1),
        )
    # Keep the FP32 reference weight outside the operator arguments.
    args = (residual, pre, packed if packed is not None else fn, scale, base, weight)
    return args, options, fn


def reference(args, options, fn):
    residual, pre, _, scale, base, weight = args
    if "sublayer_output" in options:
        residual = (
            torch.einsum("...ij,...ih->...jh", options["combination"], residual.float())
            + options["post_mix"].unsqueeze(-1)
            * options["sublayer_output"].float().unsqueeze(-2)
        ).bfloat16()
    flat = residual.flatten(-2).float()
    gates = (flat @ fn.T) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + options["rms_eps"]
    )
    eps = options["hc_eps"]
    next_pre = (gates[..., :4] * scale[0] + base[:4]).sigmoid() + eps
    post = (gates[..., 4:8] * scale[1] + base[4:8]).sigmoid() * options["post_mult"]
    comb = (gates[..., 8:] * scale[2] + base[8:]).unflatten(-1, (4, 4))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(options["sinkhorn_iters"] - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    # Delayed pre uses the incoming mix, not next_pre. Collapse rounds to BF16
    # before RMSNorm, matching the model's two operation boundaries.
    normed = (residual.float() * pre.unsqueeze(-1)).sum(-2).bfloat16().float()
    normed = (
        normed
        * torch.rsqrt(normed.square().mean(-1, keepdim=True) + options["norm_eps"])
        * weight.float()
    ).bfloat16()
    return residual, normed, next_pre, post, comb


def check(actual, expected):
    # Same tolerances as AITER's Mega-MHC reference checks, in ATOM return order.
    torch.testing.assert_close(actual[0], expected[0], rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=2e-2)
    for a, e in zip(actual[2:], expected[2:]):
        torch.testing.assert_close(a, e, rtol=1e-3, atol=5e-4)


@pytest.mark.parametrize("rows", [1, 8, 16, 128, 256, 1024, 2048, 4096])
@pytest.mark.parametrize("post", [False, True])
def test_hip_delayed_seam_matches_model_math(monkeypatch, rows, post):
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    args, options, fn = inputs(rows, post=post)
    with torch.inference_mode():
        before = args[0].clone()
        actual = mhc.pre_delayed(*args, **options)
        check(actual, reference(args, options, fn))
        assert (actual[0] is args[0]) == (not post)
        torch.testing.assert_close(args[0], before, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", [4096, 6144, 7168])
def test_other_supported_widths(monkeypatch, hidden):
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    args, options, fn = inputs(16, hidden)
    with torch.inference_mode():
        check(mhc.pre_delayed(*args, **options), reference(args, options, fn))


@pytest.mark.parametrize("post", [False, True])
def test_backend_switch_and_operator_schema(monkeypatch, post):
    with torch.inference_mode():
        monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "triton")
        args, options, _ = inputs(post=post)
        assert not options["w_preshuffle_bf16"]
        baseline = mhc.pre_delayed(*args, **options)
        monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "auto")
        args, options, _ = inputs(post=post)
        assert options["w_preshuffle_bf16"]
        check(mhc.pre_delayed(*args, **options), baseline)
        torch.library.opcheck(
            torch.ops.aiter.v41_mhc_pre_delayed.default,
            args,
            options,
            test_utils=("test_schema", "test_faketensor"),
        )


def test_forward_reuses_loaded_pack_across_shapes_and_streams(monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    from aiter.ops import mega_mhc as hip

    args, options, fn = inputs()
    packed = args[2]
    assert packed.nbytes == fn.nbytes

    def unexpected_pack(*args):
        raise AssertionError("packed forward must not prepare or version weights")

    monkeypatch.setattr(hip, "mega_mhc_shuffle_fn", unexpected_pack)
    monkeypatch.setattr(hip, "prepare_mega_mhc", unexpected_pack)
    streams = [torch.cuda.current_stream(), torch.cuda.Stream()]
    streams[1].wait_stream(streams[0])
    with torch.inference_mode():
        for stream in streams:
            with torch.cuda.stream(stream):
                for rows in (1, 8, 16, 32, 64, 128, 256, 4096):
                    args, options, fn = inputs(rows, fn=fn, packed=packed)
                    check(
                        mhc.pre_delayed(*args, **options), reference(args, options, fn)
                    )
    streams[0].wait_stream(streams[1])


@pytest.mark.parametrize("post", [False, True])
def test_graph_replay_after_other_shapes_and_on_another_stream(monkeypatch, post):
    from aiter.ops import mega_mhc as hip

    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    # Start small so the eager calls below grow scratch used by the old graph.
    monkeypatch.setattr(hip, "_SCRATCH", hip._Scratch())
    args, options, fn = inputs(post=post)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.stream(stream):
        mhc.pre_delayed(*args, **options)
        with torch.cuda.graph(graph, stream=stream):
            actual = mhc.pre_delayed(*args, **options)
        for _ in range(3):
            args[0].add_(0.125)
            args[1].mul_(0.9)
            if post:
                options["sublayer_output"].mul_(0.75)
            graph.replay()
            check(actual, reference(args, options, fn))
        for rows in (1, 16, 32, 64, 128, 256):
            other, kw, _ = inputs(rows, fn=fn, packed=args[2])
            mhc.pre_delayed(*other, **kw)
        graph.replay()
        check(actual, reference(args, options, fn))
    torch.cuda.current_stream().wait_stream(stream)
    # Replay and eager execution share immutable weights, with independent scratch.
    with torch.inference_mode():
        check(mhc.pre_delayed(*args, **options), reference(args, options, fn))
        graph.replay()
        check(actual, reference(args, options, fn))


def test_scratch_reuse_across_weights_shapes_and_protocols(monkeypatch):
    from aiter.ops import mega_mhc as hip

    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    monkeypatch.setattr(hip, "_SCRATCH", hip._Scratch())
    with torch.inference_mode():
        args, options, fn = inputs(4096)
        mhc.pre_delayed(*args, **options)

        def unexpected_zero(*args, **kwargs):
            raise AssertionError("warmed scratch must not be reinitialized")

        for rows, hidden, repeats in (
            (256, 5120, 20),
            (8, 6144, 7),
            (128, 5120, 20),
            (384, 7168, 1),
            (16, 4096, 20),
            (4096, 5120, 7),
            (256, 5120, 20),
        ):
            args, options, fn = inputs(rows, hidden)
            options["sinkhorn_iters"] = repeats
            with monkeypatch.context() as patch:
                patch.setattr(torch, "zeros", unexpected_zero)
                actual = mhc.pre_delayed(*args, **options)
            check(actual, reference(args, options, fn))


def make_block(args, options, fn):
    from atom.models.deepseek_v41.model import Block

    block = Block.__new__(Block)
    torch.nn.Module.__init__(block)
    block.engram = None
    block.hc_options = {
        k: options[k] for k in ("rms_eps", "hc_eps", "sinkhorn_iters", "post_mult")
    }
    for name in ("attn", "ffn"):
        block.register_parameter(
            f"hc_{name}_fn", torch.nn.Parameter(fn.clone(), requires_grad=False)
        )
        block.register_buffer(f"hc_{name}_fn_packed", None, persistent=False)
        setattr(block, f"hc_{name}_scale", args[3])
        setattr(block, f"hc_{name}_base", args[4])
        norm = torch.nn.Module()
        norm.weight, norm.eps = args[5], options["norm_eps"]
        setattr(block, f"{name}_norm", norm)
    block.process_weights_after_loading()
    return block


def test_model_reload_repacks_inference_weights(monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    with torch.inference_mode():
        args, options, fn = inputs(post=False)
        block = make_block(args, options, fn)
        before = block.hc_attn_fn_packed
        block.hc_attn_fn.mul_(0.5)
        block.process_weights_after_loading()
        assert block.hc_attn_fn_packed is not before
        assert block.hc_attn_fn.dtype == torch.float32  # Mono still needs this.
        assert "hc_attn_fn_packed" not in block.state_dict()
        fn = block.hc_attn_fn
        args = (*args[:2], block.hc_attn_fn_packed, *args[3:])
        check(mhc.pre_delayed(*args, **options), reference(args, options, fn))
        monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "triton")
        block.process_weights_after_loading()
        assert block.hc_attn_fn_packed is None
        assert block.hc_ffn_fn_packed is None


@pytest.mark.parametrize("backend", ["hip", "triton"])
def test_compiled_delayed_seam_with_changing_token_counts(monkeypatch, backend):
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", backend)
    compiled = torch.compile(mhc.pre_delayed, fullgraph=True, dynamic=True)
    args, options, fn = inputs()
    packed = args[2] if options["w_preshuffle_bf16"] else None
    with torch.inference_mode():
        for rows in (8, 32, 8):
            args, options, fn = inputs(rows, fn=fn, packed=packed)
            check(compiled(*args, **options), reference(args, options, fn))


@pytest.mark.parametrize("backend", ["hip", "triton"])
def test_block_attention_and_ffn_seams(monkeypatch, backend):
    from atom.model_ops.deepseek_v41.mhc import SinglePassHCState

    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", backend)
    args, options, fn = inputs()
    block = make_block(args, options, fn)
    state = SinglePassHCState(
        args[0],
        args[1],
        options["sublayer_output"],
        options["post_mix"],
        options["combination"],
    )
    with torch.inference_mode():
        normed, residual, pre, post, comb = block.prepare_attention(state, None, None)
        check((residual, normed, pre, post, comb), reference(args, options, fn))
        output = normed * 0.5
        ffn_args = (residual, pre, block.hc_ffn_fn, args[3], args[4], args[5])
        ffn_options = dict(
            options, sublayer_output=output, post_mix=post, combination=comb
        )
        normed, residual, pre, post, comb = block.prepare_ffn(
            output, residual, pre, post, comb
        )
        check(
            (residual, normed, pre, post, comb),
            reference(ffn_args, ffn_options, block.hc_ffn_fn),
        )


@pytest.mark.parametrize("sinkhorn_iters", [1, 7, 20])
@pytest.mark.parametrize("out_dtype", ["bf16", "mxfp8"])
def test_packed_api_matches_explicit_workspace(sinkhorn_iters, out_dtype):
    from aiter.ops.mega_mhc import mega_mhc, mega_mhc_shuffle_fn, prepare_mega_mhc

    args, _, fn = inputs()
    residual, pre, packed, scale, base, norm = args
    positional = (
        residual.view(-1, 4, 5120),
        packed,
        scale,
        base,
        1e-20,
        1e-6,
        1e-6,
        2.0,
        sinkhorn_iters,
    )
    kw = {
        "pre_mix": pre.view(-1, 4),
        "norm_weight": norm,
        "norm_eps": 1e-20,
        "out_dtype": out_dtype,
        "w_preshuffle_bf16": True,
    }
    workspace = prepare_mega_mhc(residual.device, 8, 5120)
    expected = mega_mhc(*positional, **kw, workspace=workspace)
    actual = mega_mhc(*positional, **kw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A different weight must be consumed directly while reusing the same scratch.
    other_packed = mega_mhc_shuffle_fn(fn * 0.5)
    other_positional = (positional[0], other_packed, *positional[2:])
    actual = mega_mhc(*other_positional, **kw, workspace=workspace)
    torch.testing.assert_close(
        actual, mega_mhc(*other_positional, **kw), rtol=0, atol=0
    )
    assert not torch.equal(actual[4], expected[4])
    # Reuse the packed tensor but never the mutable scratch on a second stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        other = prepare_mega_mhc(residual.device, 8, 5120)
        mega_mhc(*positional, **kw, workspace=other)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            first = mega_mhc(*positional, **kw, workspace=other)
            second = mega_mhc(*other_positional, **kw, workspace=other)
        for _ in range(3):
            graph.replay()
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(second, actual, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("explicit_workspace", [False, True])
def test_hip_rejects_unpacked_weights(explicit_workspace):
    from aiter.ops.mega_mhc import mega_mhc, prepare_mega_mhc

    args, _, fn = inputs()
    residual, pre, _, scale, base, norm = args
    workspace = (
        prepare_mega_mhc(residual.device, 8, 5120) if explicit_workspace else None
    )
    with pytest.raises(AssertionError, match="w_preshuffle_bf16=True"):
        mega_mhc(
            residual.view(-1, 4, 5120),
            fn,
            scale,
            base,
            1e-20,
            1e-6,
            1e-6,
            2.0,
            20,
            pre.view(-1, 4),
            norm_weight=norm,
            workspace=workspace,
            w_preshuffle_bf16=False,
        )


def test_workspace_rejects_wrong_shape_device_or_stream():
    from dataclasses import replace

    from aiter.ops.mega_mhc import mega_mhc, prepare_mega_mhc

    args, _, _ = inputs()
    residual, pre, packed, scale, base, norm = args
    positional = (
        residual.view(-1, 4, 5120),
        packed,
        scale,
        base,
        1e-20,
        1e-6,
        1e-6,
        2.0,
        20,
        pre.view(-1, 4),
    )
    kw = {"norm_weight": norm, "w_preshuffle_bf16": True}
    wrong_shape = prepare_mega_mhc(residual.device, 16, 5120)
    with pytest.raises(ValueError, match=r"device and \(T, H\)"):
        mega_mhc(*positional, **kw, workspace=wrong_shape)
    workspace = prepare_mega_mhc(residual.device, 8, 5120)
    wrong_device = replace(workspace, partials=torch.empty(0))
    with pytest.raises(ValueError, match=r"device and \(T, H\)"):
        mega_mhc(*positional, **kw, workspace=wrong_device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.cuda.stream(stream),
        pytest.raises(ValueError, match="current stream"),
    ):
        mega_mhc(*positional, **kw, workspace=workspace)


def test_concurrent_graphs_share_one_pack(monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    args, options, fn = inputs()
    packed = args[2]
    captures = []
    with torch.inference_mode():
        for rows in (8, 256):
            args, options, fn = inputs(rows, fn=fn, packed=packed)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                mhc.pre_delayed(*args, **options)
                with torch.cuda.graph(graph, stream=stream):
                    actual = mhc.pre_delayed(*args, **options)
            captures.append((stream, graph, args, options, actual))
        for _ in range(10):
            for stream, graph, args, options, _ in captures:
                with torch.cuda.stream(stream):
                    args[0].add_(0.01)
                    graph.replay()
        for stream, _, args, options, actual in captures:
            torch.cuda.current_stream().wait_stream(stream)
            check(actual, reference(args, options, fn))


def test_empty_rows_and_invalid_backend(monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    args, options, fn = inputs(0, post=False)
    actual = mhc.pre_delayed(*args, **options)
    assert actual[0] is args[0]
    assert all(t.numel() == 0 for t in actual)
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "typo")
    with pytest.raises(ValueError, match="auto, hip or triton"):
        mhc.pack_mhc_fn(fn)


def test_auto_falls_back_with_an_older_aiter(monkeypatch):
    monkeypatch.setattr(mhc, "_hip_ops", lambda: None)
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "auto")
    args, options, fn = inputs()
    assert not options["w_preshuffle_bf16"]
    with torch.inference_mode():
        check(mhc.pre_delayed(*args, **options), reference(args, options, fn))
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "hip")
    with pytest.raises(RuntimeError, match="mega_mhc_shuffle_fn"):
        mhc.pack_mhc_fn(fn)
