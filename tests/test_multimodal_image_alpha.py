# SPDX-License-Identifier: MIT
"""Transparent images keep their alpha only for builders that fill it themselves."""

import io
from types import SimpleNamespace

import pytest

from atom.multimodal import processing
from atom.multimodal.media import MediaLoader

Image = pytest.importorskip("PIL.Image")


def _png(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def test_alpha_is_kept_only_on_request():
    loader = MediaLoader()
    rgba = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    assert loader.load_image(_png(rgba)).mode == "RGB"
    assert loader.load_image(_png(rgba), keep_alpha=True).mode == "RGBA"
    assert loader.load_image(rgba, keep_alpha=True).mode == "RGBA"
    # Opaque images stay RGB, and palette transparency counts as alpha.
    assert loader.load_image(_png(rgba.convert("RGB")), keep_alpha=True).mode == "RGB"
    palette = Image.new("P", (8, 8))
    palette.info["transparency"] = 0
    assert loader.load_image(palette, keep_alpha=True).mode == "RGBA"


def _chat():
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]}
    ]


@pytest.mark.parametrize("keeps_alpha, mode", [(True, "RGBA"), (False, "RGB")])
def test_builder_declares_whether_it_takes_alpha(monkeypatch, keeps_alpha, mode):
    seen = []

    def builder(config, proc, messages, images, kwargs, tools=None):
        seen.append(images[0].mode)
        return [1], {}

    if keeps_alpha:
        builder.keeps_alpha = True
    monkeypatch.setattr(
        processing, "get_multimodal_input_builder", lambda *a, **k: builder
    )
    transparent = _png(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    processing.prepare_multimodal_inputs(
        SimpleNamespace(), None, _chat(), {"image": transparent}
    )
    assert seen == [mode]


def test_kimi_k3_takes_alpha():
    from atom.models.kimi_k3.image_processing import build_kimi_k3_inputs

    assert build_kimi_k3_inputs.keeps_alpha is True
