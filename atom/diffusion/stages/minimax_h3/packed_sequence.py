# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Layout rules follow the reference in sgl-project/sglang
# (.../minimax_h3/packed_sequence.py, Apache-2.0).

"""MiniMax-H3 packed-sequence layout for t2va and fl2va.

One request becomes a single packed row block:

    [ text L | imgvid_cond C | audio A (= audio_t x 2ch) | video V | pad P ]

``C`` is empty for t2va. For fl2va it holds one full frame of latent rows per
keyframe anchor, and those rows are part of ``img_pos`` but excluded from
``update_mask`` -- they are conditioning, not generated.

with ``cu_seqlens = [0, used, seq_len]`` -- two segments, the second being pure
padding. That matches the captured ``[0, 37712, 37760]``.

The 3-D position grid is fp64 and deliberately odd:

* the temporal axis *continues the text counter*, so video time starts at
  ``text_len`` rather than 0, and advances by ``5/3 * frame_per_token`` where
  ``frame_per_token`` cycles ``(1, 4, 4, 4, 4)``;
* each spatial axis is centred on its aspect ratio against ``sqrt(H*W)`` and
  spans a half-open interval scaled by 32;
* audio rows are channel-major and pinned to the two extremes of the w grid.

ref2va uses a different block builder and is not covered here.

One numerics trap: the fl2va last-frame anchor sums the temporal spans with
numpy (pairwise summation), while the per-frame t grid accumulates
sequentially. The two orders diverge in the last ulp from n=16 onward, so they
must not be unified -- see ``temporal_position_span`` vs ``video_t_grid``.
"""

from collections.abc import Sequence

import numpy as np
import torch

from atom.diffusion.stages.minimax_h3.geometry import (
    PACKED_SEQUENCE_ALIGNMENT,
)

INTERP = 32
T_GROUP = 5
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
PATCH_H = 2
PATCH_W = 2

# token_tags values consumed by the DiT's AdaLN modality gather.
TAG_PAD = -1
TAG_VIDEO = 0
TAG_TEXT = 1
TAG_AUDIO = 2


def axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """Evenly spaced coordinates for one spatial axis, right endpoint excluded."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * INTERP
    return torch.from_numpy(grid).to(torch.float64)


def video_t_grid(n: int, origin: float) -> torch.Tensor:
    """Temporal coordinates for ``n`` latent frames, continuing from ``origin``."""
    spans = torch.tensor(
        [FRAME_RESCALE * FRAME_PER_TOKEN[k % T_GROUP] for k in range(n)],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


# The only keyframe anchor sets the released checkpoint accepts.
FL2VA_KEYFRAME_SIGNATURES: tuple[tuple[int, ...], ...] = ((0,), (-1,), (0, -1))


def temporal_position_span(temporal_length: int) -> float:
    """Total temporal span of ``n`` latent frames, in fp64.

    Deliberately **not** shared with :func:`video_t_grid`. This one sums via
    numpy (pairwise summation) to match the fl2va anchor computation; the grid
    accumulates sequentially. The two orders differ in the last ulp from n=16
    onward, and the anchor position feeds RoPE, so the distinction is real.
    """
    spans = np.ones(int(temporal_length), dtype=np.float64) * FRAME_RESCALE
    for token_index in range(T_GROUP):
        spans[token_index::T_GROUP] *= FRAME_PER_TOKEN[token_index]
    return float(spans.sum())


def resolve_keyframe_indices(
    frame_indices: Sequence[int], *, frame_count: int
) -> list[int]:
    """Map semantic keyframe indices (0 / -1) onto concrete frame numbers."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    seen: dict[int, int] = {}
    resolved: list[int] = []
    for block_index, semantic in enumerate(frame_indices):
        if semantic == -1:
            index = frame_count - 1
        elif 0 <= semantic < frame_count:
            index = semantic
        else:
            raise ValueError(
                f"keyframe index {semantic} must be -1 or in [0, {frame_count})"
            )
        if index in seen:
            raise ValueError(
                f"keyframe block {block_index} resolves to frame {index}, "
                f"already bound by block {seen[index]}"
            )
        seen[index] = block_index
        resolved.append(index)
    return resolved


def validate_keyframe_signature(frame_indices: Sequence[int] | None) -> tuple[int, ...]:
    """Check the anchor set is one the checkpoint supports."""
    if frame_indices is None:
        raise ValueError("fl2va requires keyframe_frame_indices")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in frame_indices):
        raise ValueError("keyframe_frame_indices must be integers")
    sig = tuple(frame_indices)
    if sig not in FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            f"keyframe_frame_indices must be one of "
            f"{FL2VA_KEYFRAME_SIGNATURES}, got {sig}"
        )
    return sig


def build_packed_sequence(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
    keyframe_frame_indices: Sequence[int] | None = None,
    frame_count: int | None = None,
    text_token_tags: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    """Build the structural fields of a t2va or fl2va packed sequence.

    Pass ``keyframe_frame_indices`` (and ``frame_count``) for fl2va; omit both
    for t2va.

    ``text_token_tags`` overrides the per-token modality tags of the text
    block. fl2va needs it: the keyframe is encoded *into the prompt* by
    Qwen3-VL's vision tower, and those image tokens are tagged VIDEO rather
    than TEXT (observed run structure for a 1344x768 anchor: 6 text, 1010
    image, 13 text). Leave it None for pure-text prompts.
    """
    if text_len < 1:
        raise ValueError(f"text_len must be >= 1, got {text_len}")
    if latent_h % PATCH_H or latent_w % PATCH_W:
        raise ValueError(
            f"latent grid {latent_h}x{latent_w} not divisible by patch "
            f"{PATCH_H}x{PATCH_W}"
        )

    ph, pw = latent_h // PATCH_H, latent_w // PATCH_W
    frame_rows = ph * pw
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel

    if keyframe_frame_indices is None:
        cond_signature: tuple[int, ...] = ()
        resolved_cond: list[int] = []
    else:
        cond_signature = validate_keyframe_signature(keyframe_frame_indices)
        if frame_count is None:
            raise ValueError("frame_count is required with keyframe_frame_indices")
        resolved_cond = resolve_keyframe_indices(
            cond_signature, frame_count=frame_count
        )
    cond_rows = len(cond_signature) * frame_rows

    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = (
        (used + PACKED_SEQUENCE_ALIGNMENT - 1)
        // PACKED_SEQUENCE_ALIGNMENT
        * PACKED_SEQUENCE_ALIGNMENT
    )

    text_sl = slice(0, text_len)
    cond_sl = slice(text_len, text_len + cond_rows)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

    target_img_pos = torch.arange(video_sl.start, video_sl.stop, dtype=torch.long)
    # Conditioning rows are image rows too: they are embedded through
    # video_patch_proj and attended, they are just not written back.
    img_pos = (
        torch.cat(
            [
                torch.arange(cond_sl.start, cond_sl.stop, dtype=torch.long),
                target_img_pos,
            ]
        )
        if cond_rows
        else target_img_pos
    )
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop, dtype=torch.long)
    text_pos = torch.arange(0, text_len, dtype=torch.long)

    # Conditioning rows must not be updated by the sampler; target rows must.
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[cond_rows:] = True

    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    t_grid = video_t_grid(latent_t, float(text_len))
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = axis_from_sqrt_area(latent_h, PATCH_H, sqrt_area)
    w_grid = axis_from_sqrt_area(latent_w, PATCH_W, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    video_g = g[video_sl].view(latent_t, frame_rows, 3)
    video_g[:, :, 0] = t_grid[:, None]
    video_g[:, :, 1:] = frame[None]

    # Keyframe anchors reuse the target spatial grid but sit at the temporal
    # position of the frame they condition: the first frame shares the video
    # origin, the last sits one frame-span before the end of the clip.
    for block_index, pixel_index in enumerate(resolved_cond):
        sl = slice(
            cond_sl.start + block_index * frame_rows,
            cond_sl.start + (block_index + 1) * frame_rows,
        )
        if pixel_index == 0:
            cond_t = float(text_len)
        elif frame_count is not None and pixel_index == frame_count - 1:
            cond_t = float(text_len) + temporal_position_span(latent_t) - FRAME_RESCALE
        else:
            raise ValueError(
                "fl2va packed layout supports only first/last keyframe anchors, "
                f"got resolved frame index {pixel_index}"
            )
        g[sl, 0] = cond_t
        g[sl, 1:] = frame

    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    # Channel-major: first channel pinned to the left edge of w, second to the
    # right edge.
    g[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    g[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    token_tags = torch.full((seq_len,), TAG_PAD, dtype=torch.long)
    if text_token_tags is None:
        token_tags[text_sl] = TAG_TEXT
    else:
        tags = text_token_tags.view(-1).to(torch.long)
        if int(tags.numel()) != text_len:
            raise ValueError(
                f"text_token_tags has {int(tags.numel())} entries but text_len "
                f"is {text_len}"
            )
        token_tags[text_sl] = tags
    token_tags[audio_sl] = TAG_AUDIO
    token_tags[img_pos] = TAG_VIDEO

    return {
        "seq_len": seq_len,
        "used_len": used,
        "cond_rows": cond_rows,
        "frame_rows": frame_rows,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "img_position_ids": g,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


def build_local_embedding_layout(
    *,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    text_pos: torch.Tensor,
    row_start: int,
    row_stop: int,
) -> dict[str, torch.Tensor | int]:
    """Rows of this Ulysses rank's shard, as the DiT's ``_embed`` expects.

    ``*_global_ids`` index the full packed sequence (used to gather latents),
    ``*_row_ids`` index this rank's local rows (used to scatter into them).
    Text is contiguous at the head of the sequence, so it is expressed as a
    source range rather than an index vector.
    """
    if row_stop <= row_start:
        raise ValueError(f"empty row shard [{row_start}, {row_stop})")

    def _slice(pos: torch.Tensor) -> torch.Tensor:
        sel = torch.nonzero((pos >= row_start) & (pos < row_stop), as_tuple=False).view(
            -1
        )
        return pos.index_select(0, sel)

    # Text occupies global rows [0, text_len). Intersect that with this shard
    # and express it in source coordinates. Clamping (rather than searching for
    # selected indices) is what makes a text-free shard report an empty range
    # *at text_len* -- e.g. (2, 2) for rank 1 -- which is what the reference
    # emits. Any empty range behaves identically downstream, but matching the
    # reference exactly keeps golden comparisons clean.
    text_len = int(text_pos.numel())
    text_source_start = min(max(row_start, 0), text_len)
    text_source_stop = min(max(row_stop, 0), text_len)

    img_global = _slice(img_pos)
    audio_global = _slice(audio_pos)
    return {
        "text_source_start": text_source_start,
        "text_source_stop": text_source_stop,
        "img_global_ids": img_global,
        "img_row_ids": img_global - row_start,
        "audio_global_ids": audio_global,
        "audio_row_ids": audio_global - row_start,
    }


def build_packed_sequence_t2va(**kwargs) -> dict[str, torch.Tensor | int]:
    """t2va convenience wrapper: no keyframe conditioning."""
    if kwargs.get("keyframe_frame_indices") is not None:
        raise ValueError("use build_packed_sequence() for fl2va")
    kwargs.pop("keyframe_frame_indices", None)
    kwargs.pop("frame_count", None)
    return build_packed_sequence(**kwargs)


# ---------------------------------------------------------------------------
# ref2va
# ---------------------------------------------------------------------------

REF2VA_BLOCK_KINDS = ("image", "audio", "video", "video_audio")


def _block_int(block: dict, key: str, path: str, *, allow_zero: bool = False) -> int:
    value = block.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path}.{key} must be an int, got {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{path}.{key} must be positive, got {value}")
    return value


def _parse_ref_block(raw: dict, path: str, audio_channel: int) -> dict:
    kind = raw.get("kind", raw.get("type"))
    if kind not in REF2VA_BLOCK_KINDS:
        raise ValueError(
            f"{path}.kind must be one of {list(REF2VA_BLOCK_KINDS)}, got {kind!r}"
        )
    if kind == "image":
        rh = _block_int(raw, "latent_h", path)
        rw = _block_int(raw, "latent_w", path)
        if rh % PATCH_H or rw % PATCH_W:
            raise ValueError(f"{path} latent grid {rh}x{rw} is not patch-aligned")
        return {
            "kind": kind,
            "latent_h": rh,
            "latent_w": rw,
            "visual_rows": (rh // PATCH_H) * (rw // PATCH_W),
            "audio_rows": 0,
        }
    if kind == "audio":
        rt = _block_int(raw, "ref_audio_t", path, allow_zero=True)
        return {
            "kind": kind,
            "ref_audio_t": rt,
            "visual_rows": 0,
            "audio_rows": rt * audio_channel,
        }
    rt = _block_int(raw, "ref_audio_t", path, allow_zero=True)
    vt = _block_int(raw, "latent_t", path)
    vh = _block_int(raw, "latent_h", path)
    vw = _block_int(raw, "latent_w", path)
    if vh % PATCH_H or vw % PATCH_W:
        raise ValueError(f"{path} latent grid {vh}x{vw} is not patch-aligned")
    frame_rows = (vh // PATCH_H) * (vw // PATCH_W)
    return {
        "kind": kind,
        "ref_audio_t": rt,
        "latent_t": vt,
        "latent_h": vh,
        "latent_w": vw,
        "frame_rows": frame_rows,
        "visual_rows": vt * frame_rows,
        "audio_rows": rt * audio_channel,
    }


def build_packed_sequence_ref2va(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    ref_blocks: Sequence[dict],
    audio_channel: int = 2,
    seq_len: int | None = None,
    text_token_tags: torch.Tensor | None = None,
) -> dict:
    """Packed layout for ref2va: reference material, then the target.

        [ text | ref blocks (in request order) | target audio | target video | pad ]

    Reference blocks are consumed in request order and each advances a shared
    temporal cursor that starts at ``text_len``, so references and target sit
    on one continuous timeline rather than overlapping:

    * ``image`` occupies one integer slot;
    * ``audio`` advances by its own latent length;
    * ``video``/``video_audio`` pack their audio rows **immediately before**
      their video rows, share a temporal origin, and advance by whichever of
      the two spans is longer.

    Unlike fl2va there are two update masks. Reference *audio* rows are real
    audio conditioning, so ``audio_update_mask`` is the audio-side counterpart
    of ``update_mask`` -- a single mask cannot express "hold these audio rows
    but step those".
    """
    if text_len < 1:
        raise ValueError(f"text_len must be >= 1, got {text_len}")
    if latent_h % PATCH_H or latent_w % PATCH_W:
        raise ValueError(
            f"latent grid {latent_h}x{latent_w} not divisible by patch "
            f"{PATCH_H}x{PATCH_W}"
        )
    if not isinstance(ref_blocks, Sequence) or isinstance(ref_blocks, (str, bytes)):
        raise TypeError("ref_blocks must be a sequence of block descriptions")

    parsed = [
        _parse_ref_block(raw, f"ref_blocks[{i}]", audio_channel)
        for i, raw in enumerate(ref_blocks)
    ]
    ref_visual_rows = sum(int(b["visual_rows"]) for b in parsed)
    ref_audio_rows = sum(int(b["audio_rows"]) for b in parsed)

    ph, pw = latent_h // PATCH_H, latent_w // PATCH_W
    frame_rows = ph * pw
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel

    used = text_len + ref_visual_rows + ref_audio_rows + audio_rows + video_rows
    if seq_len is None:
        seq_len = (
            (used + PACKED_SEQUENCE_ALIGNMENT - 1)
            // PACKED_SEQUENCE_ALIGNMENT
            * PACKED_SEQUENCE_ALIGNMENT
        )
    if seq_len < used:
        raise ValueError(f"seq_len {seq_len} is smaller than the {used} rows used")

    # --- slice assignment -------------------------------------------------
    cursor = text_len
    for block in parsed:
        if block["kind"] == "image":
            block["visual_sl"] = slice(cursor, cursor + int(block["visual_rows"]))
            cursor = block["visual_sl"].stop
        elif block["kind"] == "audio":
            block["audio_sl"] = slice(cursor, cursor + int(block["audio_rows"]))
            cursor = block["audio_sl"].stop
        else:
            block["audio_sl"] = slice(cursor, cursor + int(block["audio_rows"]))
            block["visual_sl"] = slice(
                block["audio_sl"].stop,
                block["audio_sl"].stop + int(block["visual_rows"]),
            )
            cursor = block["visual_sl"].stop

    text_sl = slice(0, text_len)
    audio_sl = slice(cursor, cursor + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

    # --- position grid ----------------------------------------------------
    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = axis_from_sqrt_area(latent_h, PATCH_H, sqrt_area)
    w_grid = axis_from_sqrt_area(latent_w, PATCH_W, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    target_frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    def _pin_audio_extremes(sl: slice, count: int, grid: torch.Tensor) -> None:
        """Audio rows are channel-major and pinned to the two w extremes."""
        if count:
            g[sl.start : sl.start + count, 2] = float(grid[0])
            g[sl.start + count : sl.stop, 2] = float(grid[-1])

    ref_img_parts: list[torch.Tensor] = []
    ref_audio_parts: list[torch.Tensor] = []
    t_cursor = float(text_len)
    for block in parsed:
        kind = block["kind"]
        if kind == "image":
            sl = block["visual_sl"]
            ref_img_parts.append(torch.arange(sl.start, sl.stop, dtype=torch.long))
            rh, rw = int(block["latent_h"]), int(block["latent_w"])
            area = np.sqrt(rh * rw)
            r_hh, r_ww = torch.meshgrid(
                axis_from_sqrt_area(rh, PATCH_H, area),
                axis_from_sqrt_area(rw, PATCH_W, area),
                indexing="ij",
            )
            g[sl, 0] = t_cursor
            g[sl, 1] = r_hh.reshape(-1)
            g[sl, 2] = r_ww.reshape(-1)
            t_cursor += 1.0
        elif kind == "audio":
            sl = block["audio_sl"]
            ref_t = int(block["ref_audio_t"])
            ref_audio_parts.append(torch.arange(sl.start, sl.stop, dtype=torch.long))
            g[sl, 0] = (t_cursor + torch.arange(ref_t, dtype=torch.float64)).repeat(
                audio_channel
            )
            _pin_audio_extremes(sl, ref_t, w_grid)
            t_cursor += float(ref_t)
        else:
            a_sl, v_sl = block["audio_sl"], block["visual_sl"]
            ref_t = int(block["ref_audio_t"])
            vt = int(block["latent_t"])
            vh, vw = int(block["latent_h"]), int(block["latent_w"])
            ref_audio_parts.append(
                torch.arange(a_sl.start, a_sl.stop, dtype=torch.long)
            )
            ref_img_parts.append(torch.arange(v_sl.start, v_sl.stop, dtype=torch.long))

            area = np.sqrt(vh * vw)
            rv_h_grid = axis_from_sqrt_area(vh, PATCH_H, area)
            rv_w_grid = axis_from_sqrt_area(vw, PATCH_W, area)
            rv_hh, rv_ww = torch.meshgrid(rv_h_grid, rv_w_grid, indexing="ij")

            g[a_sl, 0] = (t_cursor + torch.arange(ref_t, dtype=torch.float64)).repeat(
                audio_channel
            )
            # A video block's audio pins to *its own* w grid, not the target's.
            _pin_audio_extremes(a_sl, ref_t, rv_w_grid)

            rv_frame = torch.stack([rv_hh.reshape(-1), rv_ww.reshape(-1)], dim=-1)
            rv_g = g[v_sl].view(vt, int(block["frame_rows"]), 3)
            rv_g[:, :, 0] = video_t_grid(vt, t_cursor)[:, None]
            rv_g[:, :, 1:] = rv_frame[None]
            t_cursor += max(float(ref_t), temporal_position_span(vt))

    g[audio_sl, 0] = (t_cursor + torch.arange(audio_t, dtype=torch.float64)).repeat(
        audio_channel
    )
    _pin_audio_extremes(audio_sl, audio_t, w_grid)

    video_g = g[video_sl].view(latent_t, frame_rows, 3)
    video_g[:, :, 0] = video_t_grid(latent_t, t_cursor)[:, None]
    video_g[:, :, 1:] = target_frame[None]

    # --- index vectors ----------------------------------------------------
    target_img_pos = torch.arange(video_sl.start, video_sl.stop, dtype=torch.long)
    target_audio_pos = torch.arange(audio_sl.start, audio_sl.stop, dtype=torch.long)
    img_pos = (
        torch.cat(ref_img_parts + [target_img_pos]) if ref_img_parts else target_img_pos
    )
    audio_pos = (
        torch.cat(ref_audio_parts + [target_audio_pos])
        if ref_audio_parts
        else target_audio_pos
    )
    text_pos = torch.arange(0, text_len, dtype=torch.long)

    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[ref_visual_rows:] = True
    audio_update_mask = torch.zeros(audio_pos.shape[0], dtype=torch.bool)
    audio_update_mask[ref_audio_rows:] = True

    token_tags = torch.full((seq_len,), TAG_PAD, dtype=torch.long)
    if text_token_tags is None:
        token_tags[text_sl] = TAG_TEXT
    else:
        tags = text_token_tags.view(-1).to(torch.long)
        if int(tags.numel()) != text_len:
            raise ValueError(
                f"text_token_tags has {int(tags.numel())} entries for a "
                f"{text_len}-token text block"
            )
        token_tags[text_sl] = tags
    token_tags[audio_pos] = TAG_AUDIO
    token_tags[img_pos] = TAG_VIDEO

    return {
        "seq_len": seq_len,
        "used_len": used,
        "cond_rows": ref_visual_rows,
        "cond_audio_rows": ref_audio_rows,
        "frame_rows": frame_rows,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "audio_update_mask": audio_update_mask,
        "img_position_ids": g,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
        "blocks": parsed,
    }
