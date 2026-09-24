# SPDX-License-Identifier: MIT
"""Image pipeline differential tests against the pinned released reference."""

import io
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from atom.config import get_hf_config
from atom.entrypoints.openai.chat_encoders import load_custom_message_encoder
from atom.model_engine.multimodal_runtime import (
    embedding_indices,
    multimodal_cache_seed,
)
from atom.models.deepseek_v41.image_processing import (
    DeepseekV41ImageProcessor,
    image_token_types,
    preprocess_image,
)
from atom.models.deepseek_v41.weights import (
    CheckpointReader,
    build_weight_manifest,
    checkpoint_schema,
)

from .reference import FIXTURES


def configs():
    config = get_hf_config(str(FIXTURES))
    vision = config._multimodal_config.vision_config
    names = {
        "vision_patch_size": "patch_size",
        "vision_downsample_ratio": "downsample_ratio",
        "vision_max_wh_ratio": "max_wh_ratio",
        "vision_min_pixels": "min_pixels",
        "vision_max_n_token": "max_image_tokens",
        "vision_n_layers": "num_hidden_layers",
        "vision_dim": "hidden_size",
        "vision_n_heads": "num_attention_heads",
        "vision_inter_dim": "intermediate_size",
        "vision_rope_theta": "rope_theta",
    }
    args = SimpleNamespace(
        **{k: getattr(vision, v) for k, v in names.items()}, dim=config.hidden_size
    )
    return config, vision, args


@pytest.mark.parametrize(
    "size", [(1, 1), (31, 517), (4000, 40), (400, 900), (999, 701), (2500, 2500)]
)
def test_preprocessing_matches_reference(reference, size):
    _, vision, args = configs()
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    official = sys.modules[reference.__package__ + ".image_processor"]
    expected = official.load_image({"data": buf.getvalue()}, args)
    actual = preprocess_image(image, vision)
    assert actual[1:] == expected[1:]
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    types = image_token_types(*actual[-2:])
    assert len(types) <= 1024
    torch.testing.assert_close(
        types, official.image_token_types(*actual[-2:]), rtol=0, atol=0
    )


def test_explicit_span_intersections_do_not_infer_from_token_ids():
    spans = ((3, 5), (11, 4))
    assert embedding_indices(spans, 5, 8) == ([0, 1, 2, 6, 7], [2, 3, 4, 5, 6])
    assert embedding_indices(spans, 15, 2) == ([], [])


def test_image_identity_survives_equal_placeholder_tokens(reference):
    from atom.model_engine.block_manager import BlockManager
    from atom.model_engine.sequence import Sequence

    directory = os.environ["ATOM_DSV41_REFERENCE"]
    config, _, _ = configs()
    processor = DeepseekV41ImageProcessor(
        SimpleNamespace(hf_config=config),
        AutoTokenizer.from_pretrained(directory),
        load_custom_message_encoder(directory),
    )

    def prepare(color):
        image = Image.new("RGB", (100, 100), color)
        return processor.prepare(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Before"},
                        {"type": "image", "image": image},
                        {"type": "text", "text": "After"},
                    ],
                }
            ],
            [image],
            {"thinking_mode": "chat"},
        )

    ids, red = prepare("red")
    blue_ids, blue = prepare("blue")
    assert ids == blue_ids
    assert red["cache_seed"] != blue["cache_seed"]
    assert red["cache_seed"] == multimodal_cache_seed(red)
    seq = Sequence(ids, 64, multimodal_data=red)
    seed = seq.cache_seed
    seq.multimodal_data = None
    assert seq.cache_seed == seed
    assert BlockManager.compute_hash(ids[:64], seed) != BlockManager.compute_hash(
        ids[:64], blue["cache_seed"]
    )
    assert len(red["token_types"]) == len(ids)
    for start, length in red["embedding_spans"]:
        assert np.all(red["token_types"][start : start + length] >= 0)
        assert red["token_types"][start - 1] == -1
        assert red["token_types"][start + length] == -1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm device required")
def test_real_vision_checkpoint_matches_reference(reference, single_rank):
    # `single_rank` because the tower's layers are ATOM's now, and those read
    # the process group at construction where `nn.Linear` read nothing.
    # The tower's layers are ATOM's, which reach AITER; everything above this
    # test is host-side preprocessing and span arithmetic that a CPU-only
    # runner can still check, so the imports live here.
    from atom.models.deepseek_v41.multimodal import DeepseekV41MultimodalModel
    from atom.models.deepseek_v41.vision import Aligner, ViT

    config, vision_config, args = configs()
    official = sys.modules[reference.__package__ + ".vision"]
    # Only the 1.1 GB vision scope: no text model or Engram allocation.
    with torch.device("cuda"):
        target = torch.nn.Module()
        target.vision = ViT(vision_config)
        target.aligner = Aligner(vision_config, config.hidden_size)
        for name in ("image_start", "image_end", "image_newline"):
            target.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(config.hidden_size, dtype=torch.bfloat16)
                ),
            )
        expected_vision = official.ViT(args).to(torch.bfloat16)
        expected_aligner = official.Aligner(args).to(torch.bfloat16)
    # Copied here rather than through `load_model`, which would want the whole
    # checkpoint and a model to put it in. The vision scope is unquantized,
    # untiled BF16 whose schema names are the module names, so reading the
    # slice and copying it is the entire operation; the set comparison is what
    # keeps that from silently covering less than the stub declares.
    schema = checkpoint_schema(config)
    manifest = build_weight_manifest(schema, scopes=("vision",))
    parameters = dict(target.named_parameters())
    assert parameters.keys() == {entry.target for entry in manifest}
    with CheckpointReader(os.environ["ATOM_DSV41_REFERENCE"], schema) as reader:
        for entry in manifest:
            parameters[entry.target].data.copy_(reader.read(entry))
    expected_vision.load_state_dict(target.vision.state_dict())
    expected_aligner.load_state_dict(target.aligner.state_dict())
    images = [Image.new("RGB", (315, 224), "red"), Image.new("RGB", (111, 333), "blue")]
    values = [preprocess_image(im, vision_config) for im in images]
    patches = torch.cat([v[0] for v in values]).cuda()
    grids = torch.tensor([(1, v[1], v[2]) for v in values])
    expected = []
    with torch.inference_mode(), torch.device("cuda"):
        for data in values:
            patch, h, w, lh, lw = data
            hidden = expected_aligner(expected_vision(patch.cuda(), h, w), h, w)
            types = image_token_types(lh, lw).cuda()
            rows = torch.empty(len(types), config.hidden_size, dtype=torch.bfloat16)
            for kind, parameter in [
                (0, target.image_start),
                (2, target.image_newline),
                (3, target.image_end),
            ]:
                rows[types == kind] = parameter
            rows[types == 1] = hidden
            expected.append(rows)
        actual = DeepseekV41MultimodalModel.get_vision_embeddings(
            target, patches, grids
        )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
