# SPDX-License-Identifier: MIT
"""Kimi-K3 video inputs: probing and decoding, sampling and grid sizing, media
binding, the one Kimi-K3 builder path, and bounded vision passes.

CPU only. The encoded-video cases need PyAV. The checkpoint-processor cases
read ``ATOM_KIMI_K3_REFERENCE``, a Kimi-K3 checkpoint directory (only the
processor and tokenizer files are read).
"""

import io
import itertools
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import stubbed_aiter
from PIL import Image

import atom.models.kimi_k3.image_processing as k3
from atom.model_engine.multimodal_runtime import (
    encode_grids_in_passes,
    vision_pass_groups,
)
from atom.multimodal import EncodedMedia, MediaLoader, VideoData, processing
from atom.multimodal.media import video as video_media

PAD = 42
CFG = {
    "in_patch_limit": 65536,
    "patch_size": 14,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_kernel_size": 2,
    "fixed_output_tokens": None,
    "patch_limit_on_one_side": 512,
    "in_patch_limit_each_frame": 16384,
    "in_patch_limit_video": 655360,
    "sample_fps": 8.0,
    "max_num_frames_each_video": None,
    "temporal_merge_kernel_size": 4,
    "timestamp_mode": "hh:mm:ss.fff",
}


def test_sampler_arithmetic():
    sample = video_media.sample_frame_indices
    assert sample(48, 8.0, 8.0, None) == list(range(48))
    assert len(sample(300, 30.0, 8.0, None)) == 80
    assert sample(300, 30.0, 8.0, 5) == [0, 75, 150, 224, 299]
    assert sample(1, 30.0, 8.0, None) == [0]
    with pytest.raises(ValueError):
        sample(0, 30.0, 8.0, None)
    with pytest.raises(ValueError):
        sample(10, 0.0, 8.0, None)


def test_timestamps():
    assert k3._format_timestamp(0.5, "hh:mm:ss.fff") == "00:00:00.500"
    assert k3._format_timestamp(3725.25, "hh:mm:ss.fff") == "01:02:05.250"
    assert k3._format_timestamp(65.0, "mm:ss") == "01:05"
    with pytest.raises(ValueError):
        k3._format_timestamp(1.0, "seconds")


def _padded(size, patch=14):
    return ((size["new_width"] + size["pad_width"]) // patch) * (
        (size["new_height"] + size["pad_height"]) // patch
    )


def test_frames_share_the_video_patch_budget():
    # A short clip keeps in_patch_limit_each_frame per frame; a long one
    # shares in_patch_limit_video among its sampled frames.
    one = k3.video_frame_size(3840, 2160, CFG, 1)
    assert 16000 < (one["new_width"] // 14) * (one["new_height"] // 14) <= 16384
    many = k3.video_frame_size(3840, 2160, CFG, 320)
    assert (many["new_width"] // 14) * (many["new_height"] // 14) <= 2048
    # Padding never takes the clip past in_patch_limit_video.
    for width, height in [(3840, 2160), (1280, 720), (553, 7000), (100, 7000)]:
        for frames in (37, 160, 320, 1000, 5000):
            size = k3.video_frame_size(width, height, CFG, frames)
            assert _padded(size) * frames <= CFG["in_patch_limit_video"]
    # Small frames are never scaled up; padding reaches the merge kernel.
    assert k3.video_frame_size(64, 48, CFG, 4) == {
        "new_width": 64,
        "new_height": 48,
        "pad_width": 20,
        "pad_height": 8,
    }


def _mp4(values, fps, size=(64, 48)):
    av = pytest.importorskip("av")
    out = io.BytesIO()
    container = av.open(out, mode="w", format="mp4")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = *size, "yuv420p"
    for value in values:
        frame = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out.getvalue()


def test_probe_counts_frames_without_decoding(monkeypatch):
    data = _mp4([10 * i for i in range(16)], 4)
    info = video_media.probe_video(data)
    assert (info.num_frames, info.fps, info.width, info.height) == (16, 4.0, 64, 48)
    monkeypatch.setattr(video_media, "MAX_SOURCE_FRAMES", 15)
    with pytest.raises(ValueError, match="more than 15 frames"):
        video_media.probe_video(data)
    with pytest.raises(ValueError, match="Invalid video data"):
        video_media.probe_video(b"not a video")


def test_decode_keeps_only_transformed_sampled_frames_in_order():
    data = _mp4([10 * i for i in range(16)], 4)
    seen = []

    def shrink(position, image):
        seen.append((position, image.size))
        return np.asarray(image.resize((8, 6)))

    frames, decoded = video_media.decode_video_frames(data, [0, 5, 10, 15], shrink)
    assert decoded == 16
    assert seen == [(0, (64, 48)), (1, (64, 48)), (2, (64, 48)), (3, (64, 48))]
    assert [frame.shape for frame in frames] == [(6, 8, 3)] * 4
    means = [frame.mean() for frame in frames]
    assert means == sorted(means) and means[0] < means[-1]


def test_media_binds_in_encounter_order():
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "and"},
                {"type": "image"},
                {"type": "video"},
            ],
        }
    ]
    bound = processing._bind_media_to_messages(conversation, ["i0"], ["v0", "v1"])
    assert [
        p.get("video", p.get("image", p.get("text"))) for p in bound[0]["content"]
    ] == ["v0", "and", "i0", "v1"]
    with pytest.raises(ValueError):
        processing._bind_media_to_messages(conversation, ["i0"], ["v0"])


def test_image_only_requests_keep_their_binding_errors():
    conversation = [{"role": "user", "content": [{"type": "video"}]}]
    with pytest.raises(ValueError, match="image/text parts"):
        processing._bind_media_to_messages(conversation, [])
    with pytest.raises(ValueError, match="0 image markers but 1 images"):
        processing._bind_media_to_messages(
            [{"role": "user", "content": [{"type": "text", "text": "q"}]}], ["i0"]
        )


def test_loader_reads_encoded_video_and_passes_decoded_video_through(tmp_path):
    loader = MediaLoader()
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"encoded")
    assert loader.load_video(str(path)) == EncodedMedia(b"encoded")
    assert loader.load_video(EncodedMedia(b"raw")) == EncodedMedia(b"raw")
    decoded = VideoData(frames=np.zeros((1, 2, 2, 3), dtype=np.uint8), fps=1.0)
    assert loader.load_video(decoded) is decoded


def _architecture(name):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[name]),
        multimodal_config=SimpleNamespace(
            media_placeholder_token_id=PAD,
            vision_config=SimpleNamespace(merge_kernel_size=[2, 2]),
        ),
        max_model_len=None,
    )


def _video_request():
    return [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "q"}]}
    ]


def test_video_modality_and_loading_follow_the_registry(monkeypatch):
    calls = []

    class Loader(MediaLoader):
        def load_video(self, data):
            calls.append(("load_video", data))
            return EncodedMedia(b"from the loader")

    def builder(config, proc, messages, images, kwargs, tools=None):
        calls.append(("builder", messages[0]["content"][0], images))
        return [1], {}

    monkeypatch.setattr(
        processing, "get_multimodal_input_builder", lambda *a, **k: builder
    )
    processing.prepare_multimodal_inputs(
        _architecture("KimiK3ForConditionalGeneration"),
        None,
        _video_request(),
        {"video": b"encoded"},
        media_loader=Loader(),
    )
    # One load path: a loader's own load_video is used, and the builder gets
    # the video bound to its part.
    assert calls == [
        ("load_video", EncodedMedia(b"encoded")),
        ("builder", {"type": "video", "video": EncodedMedia(b"from the loader")}, []),
    ]

    with pytest.raises(ValueError, match="supported modalities: image$"):
        processing.prepare_multimodal_inputs(
            _architecture("Qwen3_5ForConditionalGeneration"),
            None,
            _video_request(),
            {"video": b"encoded"},
        )


def test_video_in_a_text_prompt_is_refused_before_loading():
    with pytest.raises(ValueError, match="requires a chat conversation"):
        processing.prepare_multimodal_inputs(
            _architecture("KimiK3ForConditionalGeneration"),
            None,
            "<video> prompt",
            {"video": "unused.mp4"},
        )


class _MediaProcessor:
    """The public surface of the checkpoint's ``KimiK3VisionProcessor``."""

    def __init__(self, cfg=None):
        self.media_proc_cfg = {**CFG, "patch_size": 2, **(cfg or {})}

    def preprocess(self, medias, return_tensors=None):
        (media,) = medias
        array = np.asarray(media["image"].convert("RGB").resize((4, 4)))
        values, grid = k3._patchify(array[None], self.media_proc_cfg)
        return {"pixel_values": values, "grid_thws": grid}

    @staticmethod
    def resize_image(image, width, height, pad_width, pad_height):
        array = np.asarray(image.resize((width, height)))
        return np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)))


class _Processor:
    def __init__(self, cfg=None):
        self.media_processor = _MediaProcessor(cfg)
        self.tokenizer = SimpleNamespace(apply_chat_template=self._render)
        self.prompts = None

    @staticmethod
    def preprocess_medias(medias):
        return medias, [
            "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        ]

    def _render(self, rendered, image_prompts, **kwargs):
        self.prompts = image_prompts
        self.rendered = rendered
        pads = sum(prompt.count("<|media_pad|>") for prompt in image_prompts)
        return {"input_ids": torch.tensor([[1] + [9, PAD] * pads + [2]])}


def _build(processor, messages, images=(), max_model_len=None):
    config = _architecture("KimiK3ForConditionalGeneration")
    config.max_model_len = max_model_len
    return k3.build_kimi_k3_inputs(config, processor, messages, list(images), {})


def test_image_and_video_share_one_request_in_source_order():
    image = Image.new("RGB", (6, 6), "red")
    frames = np.arange(8 * 6 * 6 * 3, dtype=np.uint8).reshape(8, 6, 6, 3)
    video = VideoData(frames=frames, fps=8.0)
    messages = processing._bind_media_to_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "and"},
                    {"type": "image"},
                ],
            }
        ],
        [image],
        [video],
    )
    processor = _Processor()
    ids, data = _build(processor, messages, [image])

    # 8 frames at 8 fps -> two grids of 4 labelled by their first frame, then
    # the image; the video renders as one image part.
    assert processor.prompts == [
        (
            "00:00:00.000<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
            "00:00:00.500<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
        ),
        "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>",
    ]
    assert [p["type"] for p in processor.rendered[0]["content"]] == [
        "image",
        "text",
        "image",
    ]
    # 6x6 frames pad to 8x8: 16 patches of 2x2 per frame, 4x4 merged to 4 tokens.
    assert data["image_grid_thw"].tolist() == [[4, 4, 4], [4, 4, 4], [1, 2, 2]]
    assert data["pixel_values"].shape[0] == 64 + 64 + 4
    assert data["embedding_spans"] == ((2, 4), (7, 4), (12, 1))
    assert all(
        ids[start : start + n] == [PAD] * n for start, n in data["embedding_spans"]
    )
    reversed_video = VideoData(frames=frames[::-1].copy(), fps=8.0)
    _, other = _build(
        _Processor(),
        processing._bind_media_to_messages(_video_request(), [], [reversed_video]),
    )
    assert other["cache_seed"] != data["cache_seed"]


def test_video_data_with_timestamps_only_labels_grids_by_them():
    times = (0.0, 0.1, 0.2, 0.3, 2.0, 2.1, 2.2, 2.3)
    video = VideoData(frames=np.zeros((8, 6, 6, 3), dtype=np.uint8), timestamps=times)
    processor = _Processor({"sample_fps": 100.0})
    _build(processor, processing._bind_media_to_messages(_video_request(), [], [video]))
    assert [
        chunk[:12] for chunk in processor.prompts[0].split("<|media_end|>")[:2]
    ] == [
        "00:00:00.000",
        "00:00:02.000",
    ]


def test_unbound_image_parts_take_the_images_argument_and_counts_must_match():
    image = Image.new("RGB", (6, 6))
    request = [{"role": "user", "content": [{"type": "image"}]}]
    _, data = _build(_Processor(), request, [image])
    assert data["embedding_spans"] == ((2, 1),)
    with pytest.raises(ValueError, match="1 image parts but 2 images"):
        _build(_Processor(), request, [image, image])
    with pytest.raises(ValueError, match="video part has no video"):
        _build(_Processor(), _video_request())


def test_a_video_over_the_context_is_refused_before_decoding(monkeypatch):
    def no_decode(*args, **kwargs):
        raise AssertionError("refused videos must not be decoded")

    monkeypatch.setattr(video_media, "decode_video_frames", no_decode)
    data = _mp4([0] * 16, 8)
    messages = processing._bind_media_to_messages(
        _video_request(), [], [EncodedMedia(data)]
    )
    # 16 frames -> 4 grids of 64x48 frames padded to 64x48 (patch 2, merge 2):
    # 32 x 24 patches -> 192 tokens per grid.
    with pytest.raises(ValueError, match="needs 768 tokens .*767-token context"):
        _build(_Processor(), messages, max_model_len=767)


def test_encoded_video_is_resized_while_decoding():
    data = _mp4([40 * i for i in range(6)], 6, size=(128, 96))
    messages = processing._bind_media_to_messages(
        _video_request(), [], [EncodedMedia(data)]
    )
    processor = _Processor({"in_patch_limit_video": 6 * 64})
    _, out = _build(processor, messages, max_model_len=10_000)
    # 6 frames share 384 patches (patch 2): grids of 4 and 2 frames of at most
    # 64 patches each, padding included.
    grids = out["image_grid_thw"].tolist()
    assert [t for t, _, _ in grids] == [4, 2]
    assert all(h * w <= 64 for _, h, w in grids)
    assert grids[0][1:] == grids[1][1:]
    assert processor.prompts[0].count("<|media_pad|>") == 2


@pytest.fixture(scope="module")
def vision():
    with stubbed_aiter("aiter.tuned_gemm"):
        from atom.models import kimi_k3_vl
    return kimi_k3_vl


def _segment_attention(q, k, v, cu_seqlens, max_seqlen):
    """Exact attention within each ``cu_seqlens`` segment (aiter's contract)."""
    out = []
    bounds = cu_seqlens.tolist()
    for start, end in itertools.pairwise(bounds):
        out.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[start:end].transpose(0, 1),
                k[start:end].transpose(0, 1),
                v[start:end].transpose(0, 1),
            ).transpose(0, 1)
        )
    return torch.cat(out).flatten(start_dim=-2)


def _tiny_tower(vision):
    config = SimpleNamespace(
        merge_kernel_size=[2, 2],
        patch_size=2,
        init_pos_emb_height=4,
        init_pos_emb_width=4,
        init_pos_emb_time=4,
        vt_hidden_size=16,
        vt_num_hidden_layers=2,
        vt_num_attention_heads=2,
        vt_intermediate_size=32,
        text_hidden_size=8,
    )
    torch.manual_seed(0)
    tower, projector = vision.build_vision_modules(config)
    for parameter in [*tower.parameters(), *projector.parameters()]:
        torch.nn.init.normal_(parameter, std=0.2)
    return tower.double(), projector.double()


def test_the_vision_tower_gives_the_same_rows_in_bounded_passes(vision, monkeypatch):
    monkeypatch.setattr(vision, "varlen_attention", _segment_attention)
    tower, projector = _tiny_tower(vision)
    grids = torch.tensor([[1, 4, 6], [4, 4, 4], [2, 6, 4], [1, 2, 2], [4, 4, 6]])
    pixels = torch.randn(int(grids.prod(-1).sum()), 3, 2, 2, dtype=torch.float64)
    seen = []

    def encode(values, grid_thws):
        seen.append(int(values.shape[0]))
        return projector(tower(values, grid_thws))

    one = encode_grids_in_passes(encode, pixels, grids, None, "cpu", torch.float64)
    split = encode_grids_in_passes(encode, pixels, grids, 100, "cpu", torch.float64)
    assert seen == [236, 24 + 64, 48 + 4, 96]
    torch.testing.assert_close(split, one, rtol=1e-10, atol=1e-10)


def test_pass_groups_never_split_a_grid():
    assert vision_pass_groups([24, 64, 48, 4, 96], 100) == [2, 2, 1]
    # A grid above the budget is encoded alone.
    assert vision_pass_groups([10, 150, 10, 10], 100) == [1, 1, 2]
    assert vision_pass_groups([5, 5, 5], 1) == [1, 1, 1]
    with pytest.raises(ValueError, match="describe 4"):
        encode_grids_in_passes(
            None, torch.zeros(5, 3), torch.tensor([[1, 2, 2]]), 8, "cpu", None
        )


def test_video_time_embedding_follows_the_embedding_device_and_dtype(vision):
    # time_weight is a from_numpy buffer that stays on the CPU in fp32 while
    # the tower is built on the accelerator; only t > 1 grids read it. The meta
    # device stands in for the accelerator here.
    embed = vision.KimiK3InterpPosEmb(4, 4, 4, 8)
    embed.weight = torch.nn.Parameter(
        torch.empty(4, 4, 8, device="meta", dtype=torch.bfloat16)
    )
    x = torch.empty(2 * 16, 8, device="meta", dtype=torch.bfloat16)
    out = embed(x, torch.tensor([[2, 4, 4]]))
    assert out.device.type == "meta" and out.dtype == torch.bfloat16
    assert embed.time_weight.device.type == "cpu"


def test_pass_budget_comes_from_the_processor_config(tmp_path):
    assert k3.kimi_k3_vision_pass_patches(CFG) == 65536
    assert (
        k3.kimi_k3_vision_pass_patches({**CFG, "in_patch_limit_each_frame": 32768})
        == 131072
    )
    (tmp_path / "preprocessor_config.json").write_text(
        '{"media_proc_cfg": {"in_patch_limit": 1024, '
        '"in_patch_limit_each_frame": 512, "temporal_merge_kernel_size": 4}}'
    )
    assert k3.load_kimi_k3_vision_pass_patches(str(tmp_path)) == 2048
    assert k3.load_kimi_k3_vision_pass_patches(None) is None


@pytest.fixture(scope="module")
def reference():
    directory = os.environ.get("ATOM_KIMI_K3_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_KIMI_K3_REFERENCE for the checkpoint processor")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(directory, trust_remote_code=True)


def test_image_requests_match_the_checkpoint_processor(reference):
    images = [Image.new("RGBA", (300, 200), (0, 0, 0, 0)), Image.new("L", (90, 700))]
    images[0].paste((20, 20, 20, 255), (50, 50, 250, 150))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "text", "text": "and"},
                {"type": "image", "image": images[1]},
            ],
        }
    ]
    expected = reference(
        messages=messages,
        medias=[{"type": "image", "image": image} for image in images],
        return_tensors="pt",
    )
    config = _architecture("KimiK3ForConditionalGeneration")
    pad = k3.DEFAULT_MEDIA_PLACEHOLDER_TOKEN_ID
    config.multimodal_config.media_placeholder_token_id = pad
    ids, data = k3.build_kimi_k3_inputs(config, reference, messages, images, {})
    torch.testing.assert_close(data["pixel_values"], expected["pixel_values"])
    assert torch.equal(data["image_grid_thw"], expected["grid_thws"])
    collapsed = [
        token
        for index, token in enumerate(ids)
        if token != pad or index == 0 or ids[index - 1] != pad
    ]
    assert collapsed == expected["input_ids"][0].tolist()
    assert [n for _, n in data["embedding_spans"]] == k3.kimi_k3_tokens_per_grid(
        expected["grid_thws"], [2, 2]
    )


def test_frame_sizing_and_patches_match_the_checkpoint_processor(reference):
    import sys

    media = reference.media_processor
    utils = sys.modules[type(media).__module__]
    cfg = media.media_proc_cfg
    for width, height in [(64, 48), (1920, 1080), (2079, 7168), (553, 7000), (15, 900)]:
        for limit in (16384, 4096, 65536):
            expected = utils.navit_resize_image(
                width,
                height,
                cfg["patch_size"],
                cfg["merge_kernel_size"],
                limit,
                cfg["patch_limit_on_one_side"],
                cfg["fixed_output_tokens"],
            )
            actual = k3.navit_frame_size(width, height, cfg, limit)
            assert actual == {key: expected[key] for key in actual}
    # One frame through the video path equals the same frame as an image.
    image = Image.fromarray(
        np.random.default_rng(3).integers(0, 256, (90, 130, 3), dtype=np.uint8)
    )
    size = k3.navit_frame_size(130, 90, cfg, cfg["in_patch_limit"])
    frame = media.resize_image(
        image,
        size["new_width"],
        size["new_height"],
        size["pad_width"],
        size["pad_height"],
    )
    values, grid = k3._patchify(frame[None], cfg)
    expected = media.preprocess(
        [{"type": "image", "image": image}], return_tensors="pt"
    )
    assert torch.equal(values, expected["pixel_values"])
    assert torch.equal(grid, expected["grid_thws"])
