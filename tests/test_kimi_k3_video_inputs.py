# SPDX-License-Identifier: MIT
"""Kimi-K3 video inputs: sampling, chunking, prompt binding, spans and bounded
vision passes. CPU only; PyAV is needed only for the encoded-video case."""

import io
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from atom.models.kimi_k3.image_processing import (
    _format_k3_timestamp,
    _k3_video_chunks,
    build_kimi_k3_inputs,
)
from atom.multimodal import VideoData, processing
from atom.multimodal.media.video import decode_video, sample_frame_indices

PAD = 42
CFG = {
    "in_patch_limit": 65536,
    "in_patch_limit_each_frame": 16384,
    "in_patch_limit_video": 655360,
    "sample_fps": 8.0,
    "max_num_frames_each_video": None,
    "temporal_merge_kernel_size": 4,
    "timestamp_mode": "hh:mm:ss.fff",
}


def test_reference_sampler_arithmetic():
    assert sample_frame_indices(48, 8.0, 8.0, None) == list(range(48))
    assert len(sample_frame_indices(300, 30.0, 8.0, None)) == 80
    assert sample_frame_indices(300, 30.0, 8.0, 5) == [0, 75, 150, 224, 299]
    assert sample_frame_indices(1, 30.0, 8.0, None) == [0]
    with pytest.raises(ValueError):
        sample_frame_indices(0, 30.0, 8.0, None)
    with pytest.raises(ValueError):
        sample_frame_indices(10, 0.0, 8.0, None)


def test_timestamps():
    assert _format_k3_timestamp(0.5, "hh:mm:ss.fff") == "00:00:00.500"
    assert _format_k3_timestamp(3725.25, "hh:mm:ss.fff") == "01:02:05.250"
    assert _format_k3_timestamp(65.0, "mm:ss") == "01:05"
    with pytest.raises(ValueError):
        _format_k3_timestamp(1.0, "seconds")


def test_chunks_follow_the_reference_contract():
    frames = np.zeros((10, 48, 64, 3), dtype=np.uint8)
    chunks = _k3_video_chunks(VideoData(frames=frames, fps=8.0), CFG)
    assert [len(c[1]) for c in chunks] == [4, 4, 2]
    assert [c[0] for c in chunks] == ["00:00:00.000", "00:00:00.500", "00:00:01.000"]
    assert all(c[2] == 16384 for c in chunks)
    # The per-frame patch limit shrinks so the whole clip fits in_patch_limit_video.
    many = VideoData(frames=np.zeros((80, 4, 4, 3), dtype=np.uint8), fps=8.0)
    assert _k3_video_chunks(many, CFG)[0][2] == 8192


def _mp4(values, fps):
    av = pytest.importorskip("av")
    out = io.BytesIO()
    container = av.open(out, mode="w", format="mp4")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for value in values:
        frame = np.full((48, 64, 3), value, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out.getvalue()


def test_decode_keeps_order_size_and_time():
    values = [10 * i for i in range(16)]
    video = decode_video(_mp4(values, 4), sample_fps=8.0)
    assert video.frames.shape == (16, 48, 64, 3)
    assert video.timestamps[:3] == (0.0, 0.25, 0.5)
    assert np.all(np.diff(video.frames.reshape(16, -1).mean(axis=1)) > 0)
    assert len(decode_video(_mp4(values, 16), sample_fps=8.0).frames) == 8
    with pytest.raises(ValueError):
        decode_video(b"not a video", sample_fps=8.0)


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
    ] == [
        "v0",
        "and",
        "i0",
        "v1",
    ]
    with pytest.raises(ValueError):
        processing._bind_media_to_messages(conversation, ["i0"], ["v0"])


def _video_request():
    return [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": "q"}],
        }
    ]


def test_video_reaches_only_builders_that_declare_it(monkeypatch):
    calls = []

    def video_builder(config, proc, messages, images, kwargs, tools=None, videos=None):
        calls.append((messages, images, videos))
        return [1], {}

    video_builder.supports_video = True

    def image_builder(*args, **kwargs):
        raise AssertionError("an image-only builder must not receive video")

    monkeypatch.setattr(
        processing, "get_multimodal_input_builder", lambda *a, **k: video_builder
    )
    processing.prepare_multimodal_inputs(
        SimpleNamespace(), None, _video_request(), {"video": b"encoded"}
    )
    messages, images, videos = calls[0]
    assert videos == [b"encoded"] and images == []
    assert messages[0]["content"][0] == {"type": "video", "video": b"encoded"}

    monkeypatch.setattr(
        processing, "get_multimodal_input_builder", lambda *a, **k: image_builder
    )
    with pytest.raises(ValueError, match="supported modalities: image"):
        processing.prepare_multimodal_inputs(
            SimpleNamespace(), None, _video_request(), {"video": b"encoded"}
        )


# A minimal stand-in for the K3 media processor module: 2x2 merged patches per
# frame group and one <|media_pad|> per grid, enough to exercise the builder.
def navit_resize_image(width, height, patch, merge, limit, one_side, fixed):
    return {"new_width": 4, "new_height": 4, "pad_width": 0, "pad_height": 0}


def normalize(stack, mean, std_inv):
    return stack.astype(np.float32)


def navit_patchify(array, patch):
    frames = array.shape[0]
    return {
        "pixel_values": array.reshape(frames * 4, -1),
        "grid_thw": np.array([frames, 2, 2]),
    }


def _to_tensor(value, dtype=None):
    return torch.as_tensor(np.asarray(value), dtype=dtype)


class _MediaProcessor:
    media_proc_cfg: ClassVar[dict] = {
        **CFG,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "patch_size": 2,
        "merge_kernel_size": 2,
        "patch_limit_on_one_side": 512,
        "fixed_output_tokens": None,
    }

    @staticmethod
    def resize_image(image, width, height, pad_width, pad_height):
        return np.asarray(image.resize((width, height)))


class _Tokenizer:
    def __init__(self):
        self.prompts = None

    def apply_chat_template(self, rendered, image_prompts, **kwargs):
        self.prompts = image_prompts
        pads = sum(prompt.count("<|media_pad|>") for prompt in image_prompts)
        return {"input_ids": torch.tensor([[1] + [9, PAD] * pads + [2]])}


def test_video_grids_become_chunkable_spans():
    processor = SimpleNamespace(
        media_processor=_MediaProcessor(), tokenizer=_Tokenizer()
    )
    config = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            media_placeholder_token_id=PAD,
            vision_config=SimpleNamespace(merge_kernel_size=[2, 2]),
        )
    )
    frames = np.arange(8 * 6 * 6 * 3, dtype=np.uint8).reshape(8, 6, 6, 3)
    video = VideoData(frames=frames, fps=8.0)
    messages = processing._bind_media_to_messages(_video_request(), [], [video])

    ids, data = build_kimi_k3_inputs(
        config, processor, messages, [], {}, videos=[video]
    )

    # 8 frames at 8 fps -> two temporal grids of 4, labelled by their first frame.
    assert processor.tokenizer.prompts == [
        (
            "00:00:00.000<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
            "00:00:00.500<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
        )
    ]
    assert data["image_grid_thw"].tolist() == [[4, 2, 2], [4, 2, 2]]
    assert data["pixel_values"].shape[0] == 2 * 4 * 4
    # One span per grid, each covering exactly that grid's expanded placeholder run.
    assert data["embedding_spans"] == ((2, 1), (4, 1))
    assert all(
        ids[start : start + n] == [PAD] * n for start, n in data["embedding_spans"]
    )
    other = VideoData(frames=frames[::-1].copy(), fps=8.0)
    _, other_data = build_kimi_k3_inputs(
        config,
        processor,
        processing._bind_media_to_messages(_video_request(), [], [other]),
        [],
        {},
        videos=[other],
    )
    assert other_data["cache_seed"] != data["cache_seed"]


@pytest.fixture
def vision_passes():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from aiter_stub import stubbed_aiter

    with stubbed_aiter("aiter.tuned_gemm"):
        from atom.models import kimi_k3_vl
    return kimi_k3_vl


def _per_grid_tower(seen):
    # Every real tower op is per grid or per token; model that directly.
    def tower(pixel_values, grid_thws):
        seen.append(int(pixel_values.shape[0]))
        rows, offset = [], 0
        for t, h, w in grid_thws.tolist():
            size = t * h * w
            grid = pixel_values[offset : offset + size].sum()
            rows.append(grid + torch.arange((h // 2) * (w // 2), dtype=torch.float32))
            offset += size
        return torch.cat(rows).unsqueeze(-1)

    return tower


def test_vision_passes_are_bounded_and_match_one_pass(vision_passes):
    grids = torch.tensor([[1, 4, 6], [4, 4, 4], [4, 6, 4], [1, 2, 2], [4, 4, 6]])
    pixels = torch.randn(int(grids.prod(-1).sum()), 3)
    whole, parts = [], []
    one = vision_passes.encode_media_in_passes(
        _per_grid_tower(whole), lambda x: x, pixels, grids, max_patches=10**6
    )
    split = vision_passes.encode_media_in_passes(
        _per_grid_tower(parts), lambda x: x, pixels, grids, max_patches=100
    )
    assert whole == [sum(int(g.prod()) for g in grids)]
    assert parts == [24 + 64, 96 + 4, 96]
    assert torch.equal(split, one)


def test_vision_pass_rejects_patch_count_mismatch(vision_passes):
    with pytest.raises(ValueError):
        vision_passes.encode_media_in_passes(
            None, None, torch.zeros(5, 3), torch.tensor([[1, 2, 2]])
        )
