# SPDX-License-Identifier: MIT
"""Exercise the decorated runtime, live metadata and decoder input hooks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    # Runtime imports reach Triton; skip before importing GPU-only modules.
    pytest.skip("ROCm GPU required", allow_module_level=True)

from atom.config import CompilationConfig, CUDAGraphMode
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41.multimodal import DeepseekV41MultimodalModel
from atom.models.deepseek_v41.runtime import (
    DeepseekV41RuntimeModel,
    RuntimeBlock,
)
from atom.spec_decode.drafter import Drafter
from atom.utils import forward_context
from atom.utils.backends import VllmBackend


class Step:
    def __init__(self, rows, shift, empty=False):
        self.width = rows
        self.shift = shift
        self.requests = [] if empty else [object()]
        self.selected = {"stale": True}

    def begin_forward(self):
        self.selected.clear()


class Rows(dict):
    def __init__(self):
        super().__init__()
        self.events = []

    def stage(self):
        self.events.append("stage")

    def join(self):
        self.events.append("join")


class Attention(nn.Module):
    def forward(self, hidden, hidden_scale, cache, step, rope):
        assert not step.selected
        return hidden + step.shift


class FFN(nn.Module):
    def forward(self, hidden):
        image_mask = forward_context.get_forward_context().attn_metadata.image_mask
        result = hidden * 2
        if image_mask is not None:
            result = result + image_mask.unsqueeze(-1) * 10
        return result


class Engram(nn.Module):
    layer_id = 0

    def forward(self, residual, embeddings, token_mask):
        update = embeddings.unsqueeze(-2)
        if token_mask is not None:
            update = update * token_mask[..., None, None]
        return residual + update


class TinyBlock(RuntimeBlock):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer_name = "v41.layers.0"
        # The block norms its own input now; this stub's is already normed.
        self.attn_norm = nn.Identity()
        self.attn = Attention()
        self.ffn = FFN()
        self.engram = None

    def prepare_attention(self, state, embeddings, image_mask):
        residual = state.settle().residual
        if self.engram is not None:
            residual = self.engram_forward(residual, embeddings, image_mask)
        return residual.mean(-2), residual, state.pre_mix, state.pre_mix, state.pre_mix

    def prepare_ffn(self, output, residual, pre, post, comb):
        return output, residual, pre, post, comb

    def finish_ffn(self, output, residual, pre, post, comb):
        # Settled, not owed: this stub's post is a broadcast, not the mHC
        # expansion `settle` applies.
        return SinglePassHCState(
            output.unsqueeze(-2).expand_as(residual).contiguous(), pre
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm compiler")
@pytest.mark.parametrize("level, images", [(0, False), (0, True), (3, False)])
def test_runtime_guard_with_live_steps_and_aux_hooks(
    monkeypatch, tmp_path, level, images
):
    def tiny_init(self, config, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.embed = nn.Embedding(32, 8, device="cuda")
        self.layers = nn.ModuleList([TinyBlock()])
        if images:
            self.layers[0].engram = Engram()
        self.topology = [SimpleNamespace(layer_id=0, ratio=0)]
        self.global_rope = self.window_rope = None

    monkeypatch.setattr(DeepseekV41MultimodalModel, "__init__", tiny_init)
    calls = []
    original_backend = VllmBackend.__call__

    def record_backend(self, graph, inputs):
        calls.append(graph)
        return original_backend(self, graph, inputs)

    monkeypatch.setattr(VllmBackend, "__call__", record_backend)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(hc_mult=4, hidden_size=8),
        max_model_len=32,
        online_quant_config=None,
        compilation_config=CompilationConfig(
            level=level,
            cudagraph_mode=CUDAGraphMode.FULL,
            cache_dir=str(tmp_path),
            splitting_ops=[],
            compile_sizes=[],
        ),
    )
    model = DeepseekV41RuntimeModel(config)
    names = tuple(model.state_dict())
    aux = torch.full((32, 8), float("nan"), device="cuda")
    drafter = SimpleNamespace(_aux_buffers=[aux])

    def extract(inputs, block):
        return inputs[0].residual.mean(-2).reshape(-1, 8)

    hook = Drafter._make_aux_hook(drafter, 0, extract)

    def pre_hook(module, inputs):
        hook(module, (), inputs)

    model.layers[0].register_forward_pre_hook(pre_hook)
    with torch.inference_mode():
        for rows, shift, empty in [
            (6, 1.0, False),
            (17, 3.0, False),
            (4, -2.0, False),
            (6, 0.0, True),
        ]:
            step = Step(rows, shift, empty)
            embeddings = Rows()
            mask = None
            if images:
                embeddings[0] = torch.full((1, rows, 8), 5.0, device="cuda")
                # Mixed images, then text, then an image-only chunk. The mask
                # must be read anew on each serving invocation.
                if not empty and rows != 17:
                    mask = torch.arange(rows, device="cuda")[None] % 2 == 0
                    if rows == 4:
                        mask.fill_(True)
            metadata = SimpleNamespace(
                step=step, cache=None, engram_embeddings=embeddings, image_mask=mask
            )
            monkeypatch.setattr(
                forward_context,
                "_forward_context",
                forward_context.ForwardContext(
                    attn_metadata=metadata,
                    no_compile_layers=config.compilation_config.static_forward_context,
                    context=SimpleNamespace(is_draft=False, ubatch_token_offset=0),
                ),
            )
            tokens = torch.arange(rows, device="cuda", dtype=torch.int64)
            expected_embed = model.embed(tokens)
            inputs_embeds = expected_embed + 0.25 if images else None
            if inputs_embeds is not None:
                expected_embed = inputs_embeds
            actual = model(tokens, tokens, inputs_embeds=inputs_embeds)
            engram_update = 5.0 if mask is None else (~mask).T * 5.0
            expected = (
                torch.zeros_like(expected_embed)
                if empty
                else (expected_embed + (engram_update if images else 0) + shift) * 2
            )
            if mask is not None:
                expected = expected + mask.T * 10
            torch.testing.assert_close(actual, expected)
            if not empty:
                torch.testing.assert_close(aux[:rows], expected_embed)
            assert not step.selected
            assert embeddings.events == ([] if empty else ["stage", "join"])
    if level == 3:
        assert len(calls) == len(model.compiled_codes) == 1
    else:
        assert not calls
    assert tuple(model.state_dict()) == names == ("embed.weight",)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm mHC")
@pytest.mark.parametrize("mode", ["pre", "fused_post_pre", "unfused_post_pre"])
def test_mhc_guard_outputs_and_input_alias(monkeypatch, mode):
    from atom.model_ops.deepseek_v41 import mhc_pre_delayed as mhc

    monkeypatch.setattr(mhc, "prefers_unfused", lambda rows: mode == "unfused_post_pre")
    device = "cuda"
    rows, hc, hidden = 6, 4, 5120
    residual = torch.randn(1, rows, hc, hidden, device=device, dtype=torch.bfloat16)
    pre = torch.full((1, rows, hc), 1 / hc, device=device)
    fn = torch.randn(hc * (hc + 2), hc * hidden, device=device) * 0.01
    scale = torch.ones(3, device=device)
    base = torch.zeros(hc * (hc + 2), device=device)
    kwargs = {"rms_eps": 1e-6, "hc_eps": 1e-6, "sinkhorn_iters": 20, "post_mult": 2.0}
    if mode != "pre":
        kwargs.update(
            sublayer_output=torch.randn(
                1, rows, hidden, device=device, dtype=torch.bfloat16
            ),
            post_mix=torch.ones_like(pre),
            combination=torch.eye(hc, device=device)
            .expand(1, rows, hc, hc)
            .contiguous(),
        )
    args = (residual, pre, fn, scale, base)
    with torch.inference_mode():
        before = residual.clone()
        outputs = mhc.pre_delayed(*args, **kwargs)
        assert (outputs[0] is residual) == (mode == "pre")
        torch.testing.assert_close(residual, before, rtol=0, atol=0)
        assert all(torch.isfinite(out).all() for out in outputs)
        torch.library.opcheck(
            torch.ops.aiter.v41_mhc_pre_delayed.default,
            args,
            kwargs,
            test_utils=("test_schema", "test_faketensor"),
        )


@torch.inference_mode()
def test_idle_prefill_runs_padding_ffn_without_attention_or_cache_writes(monkeypatch):
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.utils.forward_context import AttnState
    from tests.attentions.deepseek_v41.helpers import metadata_buffers

    geo = V41PoolGeometry(
        1, ((0, 2),), 32, 4, 512, 32, layer_ratios=(2,), index_block_rows=8
    )
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.geometry, builder.device, builder.block_size = geo, "cuda", 32
    builder.cache = PagedAttentionCache(geo, 4, 2, "cuda", max_tokens=8)
    builder.model_runner = SimpleNamespace(
        forward_vars=metadata_buffers(2, 8, 4, "cuda", geo)
    )
    batch = SimpleNamespace(
        is_dummy_run=True,
        state_slots_committed=[],
        req_ids=(-1,),
        num_scheduled_tokens=(8,),
        context_lens=(8,),
        total_seqs_num=1,
        total_tokens_num=8,
    )
    metadata, positions = builder.prepare_prefill(batch, 2)
    metadata.engram_embeddings = Rows()
    step = metadata.step
    assert step.requests == () and step.is_prefill and not step.decode
    assert metadata.state == AttnState.PREFILL_PREFIX
    assert step.width == 8 and torch.all(step.batch_ids == -1)
    builder.cache.backing.fill_(57)
    before = builder.cache.backing.clone()

    model = DeepseekV41RuntimeModel.__new__(DeepseekV41RuntimeModel)
    nn.Module.__init__(model)
    model.do_not_compile = True
    model.config = SimpleNamespace(hidden_size=8, hc_mult=4)
    model.layers = nn.ModuleList([TinyBlock()])
    model.layers[0].engram = Engram()
    model.topology = [SimpleNamespace(layer_id=0, ratio=2)]
    model.global_rope = model.window_rope = None
    model.embed = nn.Embedding(32, 8, device="cuda")
    calls = []

    def unexpected_attention_or_engram(*args):
        pytest.fail("An idle rank must not enter attention or Engram kernels")

    model.layers[0].attn.register_forward_pre_hook(unexpected_attention_or_engram)
    model.layers[0].engram.register_forward_pre_hook(unexpected_attention_or_engram)
    model.layers[0].ffn.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[-2])
    )
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        forward_context.ForwardContext(
            attn_metadata=metadata,
            no_compile_layers={"v41.layers.0": (model.layers[0], None)},
            context=SimpleNamespace(is_draft=False, ubatch_token_offset=0),
        ),
    )
    output = model(torch.zeros(8, dtype=torch.int32, device="cuda"), positions)
    assert calls == [8]
    assert output.shape == (8, 8) and output.count_nonzero() == 0
    assert metadata.engram_embeddings.events == []
    torch.testing.assert_close(builder.cache.backing, before, rtol=0, atol=0)

    # The serving guard skips attention, but a direct metadata consumer must
    # also accept this empty prefill and retain the full table capacity.
    tiles = builder.cache.unit_tiles(step, 2)
    assert tiles.shape == (8, 4 * (geo.rows_per_page(2) // geo.index_block_rows))
    assert tiles.count_nonzero() == 0
    assert builder.cache.unit_tiles(step, 2) is tiles
