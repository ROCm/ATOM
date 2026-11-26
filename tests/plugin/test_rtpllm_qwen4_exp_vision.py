"""RTP Qwen4Exp image-position and multimodal registration contracts."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("rtp_llm")

from rtp_llm.multimodal.multimodal_mixin_register import (
    get_multimodal_mixin_cls,
)

from atom.plugin.rtpllm.models.qwen4_exp import (
    _ATOMQwen4ExpMTPRuntime,
    _ATOMQwen4ExpRuntime,
)
from atom.plugin.rtpllm.models.qwen4_exp_vision import Qwen4ExpVisionMixin
from atom.plugin.rtpllm.models.qwen4_exp_vision_impl import Qwen4ExpImageEmbedding


def test_qwen4_exp_vision_mixin_registered():
    assert get_multimodal_mixin_cls("atom_qwen4_exp") is Qwen4ExpVisionMixin
    assert get_multimodal_mixin_cls("atom_qwen4_exp_mtp") is Qwen4ExpVisionMixin


def test_batched_vision_attention_is_isolated_per_image():
    class MixingVision:
        device = torch.device("cpu")
        dtype = torch.float32
        spatial_merge_size = 2

        def __call__(self, pixels, grid):
            assert grid.shape == (1, 3)
            return pixels.mean().reshape(1, 1)

    encoder = object.__new__(Qwen4ExpImageEmbedding)
    encoder.visual = MixingVision()
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int32)
    outputs = encoder.batched_embedding(
        [(torch.ones(4, 1), grid), (torch.full((4, 1), 9.0), grid)],
        [1, 1],
    )
    assert [embedding.item() for embedding, _ in outputs] == [1.0, 9.0]

    # A single request may also carry multiple image grids.
    outputs = encoder.batched_embedding(
        [
            (
                torch.cat([torch.ones(4, 1), torch.full((4, 1), 9.0)]),
                torch.cat([grid, grid]),
            )
        ],
        [1],
    )
    assert [embedding.item() for embedding, _ in outputs] == [1.0, 9.0]


@pytest.mark.parametrize("tp_size", [1, 2])
def test_vision_embedding_masks_image_placeholders_on_single_rank(monkeypatch, tp_size):
    from atom.model_ops import embed_head

    input_ids = torch.tensor([1, 999], dtype=torch.int32)
    weight = torch.ones(4, 2)
    calls = []
    expected = torch.zeros(2, 2)

    def replicated(ids, table):
        calls.append("masked")
        assert ids is input_ids and table is weight
        return expected

    def partitioned(ids):
        calls.append("partitioned")
        assert ids is input_ids
        return expected

    monkeypatch.setattr(embed_head, "replicated_embedding", replicated)
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(tp_size=tp_size, weight=weight),
            embed_input_ids=partitioned,
        )
    )
    actual = _ATOMQwen4ExpRuntime._embed_vision_input_ids(runtime, input_ids)
    assert actual is expected
    assert calls == (["masked"] if tp_size == 1 else ["partitioned"])


@pytest.mark.parametrize(
    ("is_multimodal", "tp_size", "expect_masked"),
    [(True, 1, True), (True, 2, False), (False, 1, False)],
)
def test_vision_draft_masks_negative_padding_before_embedding(
    monkeypatch, is_multimodal, tp_size, expect_masked
):
    from atom.model_ops import embed_head

    input_ids = torch.tensor([7, -1, 9], dtype=torch.int32)
    weight = torch.ones(16, 2)
    expected = torch.zeros(3, 2)
    calls = []

    def masked(ids, table):
        calls.append(True)
        assert ids is input_ids and table is weight
        return expected

    monkeypatch.setattr(embed_head, "replicated_embedding", masked)
    embed_tokens = SimpleNamespace(tp_size=tp_size, weight=weight)
    actual = _ATOMQwen4ExpMTPRuntime._draft_vision_embeddings(
        input_ids, embed_tokens, is_multimodal
    )
    assert actual is (expected if expect_masked else None)
    assert calls == ([True] if expect_masked else [])


def test_vision_placeholder_ids_are_canonicalized_for_ple_history():
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._image_token_id = 248056
    ids = torch.tensor([12, 999999, 999999, 15, 999999, 16])
    features = [torch.zeros(2, 3), torch.zeros(1, 3)]
    locations = torch.tensor([1, 4], dtype=torch.int32)
    canonical = runtime._canonical_vision_input_ids(ids, features, locations)
    assert canonical.tolist() == [12, 248056, 248056, 15, 248056, 16]
    assert ids.tolist() == [12, 999999, 999999, 15, 999999, 16]


def test_vision_image_token_id_comes_from_checkpoint_root(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"image_token_id": 248056, "text_config": {}})
    )
    assert _ATOMQwen4ExpRuntime._read_image_token_id(str(tmp_path)) == 248056


def test_vision_weight_budget_reads_checkpoint_metadata(tmp_path):
    save_file(
        {"model.visual.weight": torch.zeros(2, 3)}, tmp_path / "visual.safetensors"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.visual.weight": "visual.safetensors"}})
    )
    params = SimpleNamespace(config={"ckpt_path": str(tmp_path)})
    assert Qwen4ExpVisionMixin.eval_mm_model_param_count(params, "", "") == 6
    assert Qwen4ExpVisionMixin.eval_mm_model_size(params, "", "") == 12


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([0, 0, 0, 1, 2, 3], dtype=torch.int32),
        torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.int32),
    ],
)
def test_rtp_interleaved_mrope_uses_token_major_order(positions):
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(combo_position_ids=positions),
        combo_position_ids=None,
    )
    actual = _ATOMQwen4ExpRuntime._rope_positions(inputs, torch.device("cpu"), 2)
    assert torch.equal(actual, torch.tensor([[0, 1], [0, 2], [0, 3]]))


def test_rtp_text_and_rope_positions_share_token_major_parser(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._rtp_model_config = SimpleNamespace(
        mm_model_config=SimpleNamespace(is_multimodal=False), is_mtp=False
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # N=3 is deliberately ambiguous if the two-dimensional shape is used
    # to guess whether rows or tokens come first.
    triples = torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32]])
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=True, combo_position_ids=triples.reshape(-1)
        ),
        combo_position_ids=None,
    )
    assert runtime._extract_positions(inputs, torch.device("cpu"), 3).tolist() == [
        10,
        11,
        12,
    ]
    assert runtime._rope_positions(inputs, torch.device("cpu"), 3).tolist() == [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
    ]
    assert runtime._parse_rtp_mrope_positions(
        triples, torch.device("cpu"), 3
    ).tolist() == [[10, 11, 12], [20, 21, 22], [30, 31, 32]]


def test_logical_cache_position_is_not_image_rope_position():
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        input_lengths=torch.tensor([4], dtype=torch.int32),
        prefix_lengths=torch.tensor([0], dtype=torch.int32),
    )
    actual = runtime._logical_positions(attn_inputs, torch.device("cpu"))
    assert torch.equal(actual, torch.arange(4, dtype=torch.int32))


def test_decode_cache_position_uses_sequence_length_not_mrope():
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    attn_inputs = SimpleNamespace(
        is_prefill=False,
        sequence_lengths_plus_1_device=torch.tensor([129, 257], dtype=torch.int32),
    )
    actual = runtime._logical_positions(attn_inputs, torch.device("cpu"))
    assert torch.equal(actual, torch.tensor([128, 256], dtype=torch.int32))


@pytest.mark.parametrize("is_draft", [False, True])
@pytest.mark.parametrize("is_graph", [False, True])
def test_vision_mtp_uses_candidate_cache_positions(monkeypatch, is_draft, is_graph):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._rtp_model_config = SimpleNamespace(
        mm_model_config=SimpleNamespace(is_multimodal=True),
        is_mtp=is_draft,
        gen_num_per_cycle=2,
    )
    runtime._cg_mtp_meta_bufs = {
        "qwen4_positions_i32": torch.empty(6, dtype=torch.int32)
    }
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=True,
            is_target_verify=not is_draft,
            is_cuda_graph=is_graph,
            prefix_lengths_device=torch.tensor([16, 129], dtype=torch.int32),
            input_lengths=torch.tensor([3, 3], dtype=torch.int32),
            sequence_lengths_plus_1_device=torch.tensor([17, 130], dtype=torch.int32),
            combo_position_ids=torch.tensor(
                [16, 17, 18, 129, 130, 131], dtype=torch.int32
            ),
        )
    )
    actual = runtime._extract_positions(inputs, torch.device("cpu"), 6)
    assert actual.tolist() == [16, 17, 18, 129, 130, 131]


@pytest.mark.parametrize("is_graph", [False, True])
@pytest.mark.parametrize("live_length", [2, 3])
def test_vision_mtp_draft_padding_keeps_fixed_window_positions(
    monkeypatch, is_graph, live_length
):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._rtp_model_config = SimpleNamespace(
        mm_model_config=SimpleNamespace(is_multimodal=True),
        is_mtp=True,
        gen_num_per_cycle=2,
    )
    runtime._cg_mtp_meta_bufs = {
        "qwen4_positions_i32": torch.empty(6, dtype=torch.int32)
    }
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=is_graph,
            prefix_lengths_device=torch.tensor([16, 0], dtype=torch.int32),
            input_lengths=torch.tensor([live_length, 0], dtype=torch.int32),
        )
    )
    actual = runtime._extract_positions(inputs, torch.device("cpu"), 6)
    assert actual.tolist() == [16, 17, 18, 0, 1, 2]


def test_vision_graph_decode_uses_prewarmed_position_buffer(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._rtp_model_config = SimpleNamespace(
        mm_model_config=SimpleNamespace(is_multimodal=True), is_mtp=False
    )
    runtime._cg_meta_bufs = {"qwen4_positions_i32": torch.empty(2, dtype=torch.int32)}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=False,
            is_cuda_graph=True,
            sequence_lengths_plus_1_device=torch.tensor([129, 257], dtype=torch.int32),
        )
    )
    actual = runtime._extract_positions(inputs, torch.device("cpu"), 2)
    assert actual.data_ptr() == runtime._cg_meta_bufs["qwen4_positions_i32"].data_ptr()
    assert actual.tolist() == [128, 256]


def test_vision_draft_eager_uses_logical_cache_positions(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._rtp_model_config = SimpleNamespace(
        mm_model_config=SimpleNamespace(is_multimodal=True), is_mtp=True
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(
            is_prefill=False,
            is_cuda_graph=False,
            combo_position_ids=torch.tensor([[36, 70], [38, 72], [40, 74]]),
            sequence_lengths_plus_1_device=torch.tensor([33, 65], dtype=torch.int32),
        )
    )
    actual = runtime._extract_positions(inputs, torch.device("cpu"), 2)
    assert actual.tolist() == [32, 64]


def test_vision_draft_initial_prefill_is_ragged_not_mtp_window(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._rtp_model_config = SimpleNamespace(
        mm_model_config=SimpleNamespace(is_multimodal=True),
        is_mtp=True,
        gen_num_per_cycle=2,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = SimpleNamespace(
        is_prefill=True,
        is_cuda_graph=False,
        is_target_verify=False,
        input_lengths=torch.tensor([2, 3], dtype=torch.int32),
        prefix_lengths=torch.tensor([8, 16], dtype=torch.int32),
        prefix_lengths_device=torch.tensor([8, 16], dtype=torch.int32),
    )
    actual = runtime._extract_positions(
        SimpleNamespace(attention_inputs=attn), torch.device("cpu"), 5
    )
    assert actual.tolist() == [8, 9, 16, 17, 18]


def test_vision_draft_ragged_prefill_with_fixed_total_uses_actual_lengths(monkeypatch):
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    runtime._rtp_model_config = SimpleNamespace(
        mm_model_config=SimpleNamespace(is_multimodal=True),
        is_mtp=True,
        gen_num_per_cycle=2,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = SimpleNamespace(
        is_prefill=True,
        is_cuda_graph=False,
        is_target_verify=False,
        input_lengths=torch.tensor([1, 5], dtype=torch.int32),
        prefix_lengths=torch.tensor([8, 16], dtype=torch.int32),
        prefix_lengths_device=torch.tensor([8, 16], dtype=torch.int32),
    )
    actual = runtime._extract_positions(
        SimpleNamespace(attention_inputs=attn), torch.device("cpu"), 6
    )
    assert actual.tolist() == [8, 16, 17, 18, 19, 20]
