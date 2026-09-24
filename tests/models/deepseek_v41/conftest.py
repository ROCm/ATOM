# SPDX-License-Identifier: MIT
"""Pinned model math, isolated per differential test."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from .reference import load_reference


@pytest.fixture
def reference():
    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for pinned model methods")
    with load_reference(directory) as module:
        yield module


@pytest.fixture
def single_rank(monkeypatch):
    # Every module patched here reaches AITER, so a test that asks for this
    # fixture cannot run without it; say so as a skip rather than letting the
    # import fail during setup.
    pytest.importorskip("aiter", reason="the patched layers are AITER-backed")
    from atom.model_ops import embed_head, layernorm, linear
    from atom.model_ops import moe as fused_moe
    from atom.models.deepseek_v41 import attention, model

    group = SimpleNamespace(rank_in_group=0, world_size=1)
    # `fused_moe`, not the V4.1 `moe` module: the routed experts are V4's
    # `FusedMoE`, which reads the group where it lives. `layers` is absent
    # because the output projection reduces through `RowParallelLinear` now,
    # which reads the group in `linear`.
    for module in (linear, attention, model, fused_moe, embed_head):
        monkeypatch.setattr(module, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        layernorm, "get_tensor_model_parallel_world_size", lambda: group.world_size
    )
    # A biased `LinearBase` reads the engine-wide config for the dtype to
    # allocate its bias in, which is the one thing a layer built outside an
    # engine cannot supply. Same role as the group stub above: make ATOM's
    # layers constructible on their own.
    monkeypatch.setattr(
        linear,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    return group


@pytest.fixture
def unallocated_moe(monkeypatch):
    """Build the real model, Block, DraftBlock and V4.1 MoE constructors.

    Stubbed out is only what needs a GPU or a checkpoint: weight allocation,
    attention, and V4's `MoE.__init__` -- the last replaced by a recorder, so
    what the V4.1 layer passes down to V4 is readable as plain attributes.
    A test that wants a constructor exercised for real must not find it here.
    """
    from torch import nn

    from atom.config import get_hf_config
    from atom.models.deepseek_v4 import MoE as V4MoE
    from atom.models.deepseek_v41 import dspark, model, multimodal

    from .reference import FIXTURES

    class UnallocatedModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    for module, names in (
        (model, ("VocabParallelEmbedding", "ParallelHead", "RMSNorm")),
        (dspark, ("ReplicatedLinear", "DSparkMarkovHead", "DSparkConfidenceHead")),
        (multimodal, ("ViT", "Aligner")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, UnallocatedModule)
    monkeypatch.setattr(model.Block, "attention_cls", UnallocatedModule)
    monkeypatch.setattr(dspark.DraftBlock, "attention_cls", UnallocatedModule)

    def capture_v4(self, layer_id, args, prefix="", alt_stream=None):
        nn.Module.__init__(self)
        self.gate = nn.Module()
        self.experts = SimpleNamespace(custom_routing_function=None)
        self.quant_config = args.quant_config
        self.prefix = prefix
        self.alt_stream = alt_stream
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts

    monkeypatch.setattr(V4MoE, "__init__", capture_v4)

    hf = get_hf_config(str(FIXTURES))
    hf.engram_layer_ids = ()
    return hf


@pytest.fixture
def build_v41(unallocated_moe):
    """Build one V4.1 entry point by name, on the meta device.

    All four are here because the model is reachable four ways and each
    reaches its layers by its own path: what one hands them is not evidence
    about what another does.
    """
    import torch

    from atom.models.deepseek_v41 import dspark, model, runtime

    hf = unallocated_moe

    def build(entrypoint, online=None, alt_stream=None):
        engine = SimpleNamespace(
            hf_config=hf,
            max_model_len=32,
            enforce_eager=True,
            compilation_config=SimpleNamespace(level=0, static_forward_context={}),
            online_quant_config=online,
        )
        with torch.device("meta"):
            if entrypoint == "runtime":
                return runtime.DeepseekV41RuntimeModel(engine)
            if entrypoint == "offline":
                return model.DeepseekV41ForCausalLM(
                    hf, max_length=32, online_quant_config=online
                )
            # The draft takes the backbone's stream; serving hands it down
            # from `DSparkProposer`, which is where the backbone is.
            if entrypoint == "draft":
                return dspark.DeepseekV41DSpark(engine, alt_stream=alt_stream)
            if entrypoint == "draft_offline":
                return dspark.DeepseekV41DSpark(
                    hf, max_length=32, alt_stream=alt_stream
                )
            raise ValueError(f"No V4.1 entry point named {entrypoint!r}")

    return build


@pytest.fixture
def small_config():
    """Small model graph retaining D=512 and eight local attention heads."""
    from atom.models.deepseek_v41.config import DeepseekV41TextConfig

    return DeepseekV41TextConfig(
        hidden_size=64,
        head_dim=512,
        num_attention_heads=8,
        q_lora_rank=32,
        o_groups=1,
        o_lora_rank=32,
        rms_norm_eps=1e-20,
        sliding_window=4,
        index_n_heads=32,
        index_head_dim=32,
        index_topk=4,
        candidate_block_size=2,
        candidate_topk_blocks=4,
        max_position_embeddings=32,
        num_hidden_layers=5,
        num_nextn_predict_layers=0,
        compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layer_ids=(1, 3),
        index_source_layer_ids=(1, 3, 4),
        candidate_source_layer_id=3,
    )


@pytest.fixture
def attention_contract(monkeypatch, reference):
    """Check the real backend before controlling its rounding for graph tests.

    FP8 projection quantization is discontinuous at bin boundaries. Validate
    each actual BF16 attention output against upstream independently, then feed
    the upstream output to the remaining graph so its assertions isolate model
    composition. This is not an end-to-end numerical acceptance test.
    """
    import torch

    from atom.models.deepseek_v41 import attention

    captured = []
    original = reference.sparse_attn

    def capture(*args):
        output = original(*args)
        captured.append(output.detach().clone())
        return output

    monkeypatch.setattr(reference, "sparse_attn", capture)

    @contextmanager
    def check(outputs):
        remaining = iter(outputs)

        def checked(function):
            def call(*args, **kwargs):
                actual = function(*args, **kwargs)
                expected = next(remaining).flatten(0, 1).to(actual.device)
                assert torch.isfinite(actual).all()
                error = (actual.float() - expected.float()).norm()
                assert error <= 3e-3 * expected.float().norm().clamp_min(1e-30)
                return expected.clone()

            return call

        with pytest.MonkeyPatch.context() as patch:
            for name in ("sparse_attn_v4_paged_decode", "sparse_attn_v4_paged_prefill"):
                patch.setattr(attention, name, checked(getattr(attention, name)))
            yield
        assert next(remaining, None) is None

    return captured, check
