# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 request geometry and the packed-sequence layout.

Turns a request (canvas, duration, prompt length) into the row layout the
DiT consumes: frame/latent lattice, the packed [text | audio | video | pad]
sequence with its 3-D position grid, patchify/unpatchify, and the seeded
initial latents."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

# The video VAE compresses 16x spatially; the DiT then applies a (1, 2, 2)
# patch, so each latent frame contributes (H/16/2) * (W/16/2) rows.
VAE_SPATIAL_COMPRESSION = 16
AUDIO_LATENT_HZ = 40.0
AUDIO_CHANNELS = 2
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
PACKED_SEQUENCE_ALIGNMENT = 64
DEFAULT_SEED = 42


def align_packed_length(used: int) -> int:
    """Round a used-row count up to the packed-sequence alignment."""
    return -(-int(used) // PACKED_SEQUENCE_ALIGNMENT) * PACKED_SEQUENCE_ALIGNMENT


def align_frame_count(frame_count: int) -> int:
    """Snap up to H3's 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def video_latent_t(frame_count: int) -> int:
    """Frames -> video latent T (1 or 5n+2)."""
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def audio_latent_t(duration_seconds: float) -> int:
    """Duration -> audio latent T, rounded at the 40 Hz boundary."""
    return round(float(duration_seconds) * AUDIO_LATENT_HZ)


def time_shift_sigmas(*, num_steps: int = 50, shift_scale: float = 6.0) -> list[float]:
    """Rectified-flow sigma schedule with a flow shift.

    ``sigma = s*b / (1 + (s-1)*b)`` over ``b`` linearly spaced on [1, 0].
    Returns ``num_steps`` sigmas ending at 0, so a denoise loop runs
    ``num_steps - 1`` iterations (50 sigmas -> 49 steps, which is what the
    reference server reports).
    """
    if shift_scale <= 0:
        raise ValueError(f"shift_scale must be > 0, got {shift_scale}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    base = torch.linspace(1.0, 0.0, int(num_steps), dtype=torch.float32)
    shifted = shift_scale * base / (1 + (shift_scale - 1) * base)
    shifted = torch.unique_consecutive(shifted)
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.zeros(1, dtype=shifted.dtype)])
    return [float(v) for v in shifted.tolist()]


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    """Resolved token layout for one request."""

    height: int
    width: int
    frame_count: int
    duration_seconds: float
    text_len: int

    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int

    video_rows: int
    audio_rows: int
    used_len: int
    seq_len: int

    @classmethod
    def resolve(
        cls,
        *,
        height: int,
        width: int,
        frame_count: int,
        duration_seconds: float,
        text_len: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> "MiniMaxH3Geometry":
        if height % VAE_SPATIAL_COMPRESSION or width % VAE_SPATIAL_COMPRESSION:
            raise ValueError(
                f"height and width must be multiples of "
                f"{VAE_SPATIAL_COMPRESSION}, got {height}x{width}"
            )
        aligned_frames = align_frame_count(frame_count)
        lt = video_latent_t(aligned_frames)
        lh = height // VAE_SPATIAL_COMPRESSION
        lw = width // VAE_SPATIAL_COMPRESSION

        _, ph, pw = patch_size
        if lh % ph or lw % pw:
            raise ValueError(
                f"latent grid {lh}x{lw} is not divisible by patch {ph}x{pw}"
            )
        frame_rows = (lh // ph) * (lw // pw)
        video_rows = lt * frame_rows

        at = audio_latent_t(duration_seconds)
        audio_rows = at * AUDIO_CHANNELS

        used = text_len + audio_rows + video_rows
        return cls(
            height=height,
            width=width,
            frame_count=aligned_frames,
            duration_seconds=duration_seconds,
            text_len=text_len,
            latent_t=lt,
            latent_h=lh,
            latent_w=lw,
            audio_t=at,
            video_rows=video_rows,
            audio_rows=audio_rows,
            used_len=used,
            seq_len=align_packed_length(used),
        )

    def validate_ulysses(self, world_size: int) -> None:
        """The padded sequence must split evenly across the Ulysses group."""
        if self.seq_len % world_size:
            raise ValueError(
                f"packed sequence {self.seq_len} does not divide across "
                f"ulysses world size {world_size}; alignment is "
                f"{PACKED_SEQUENCE_ALIGNMENT}, so degrees above that or "
                f"non-divisors of it cannot work"
            )


def _int_tuple(value: Sequence[int], name: str, length: int) -> tuple[int, ...]:
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {list(value)!r}")
    out = tuple(int(v) for v in value)
    if any(v <= 0 for v in out):
        raise ValueError(f"{name} values must be positive, got {list(value)!r}")
    return out


def patchify_video_latent(
    latent: torch.Tensor, *, patch_size: Sequence[int] = (1, 2, 2)
) -> torch.Tensor:
    """``[B, C, T, H, W]`` -> ``[B*t*h*w, C*pt*ph*pw]``."""
    if latent.ndim != 5:
        raise ValueError(f"video latent must be rank 5, got shape {list(latent.shape)}")
    pt, ph, pw = _int_tuple(patch_size, "patch_size", 3)
    b, c, full_t, full_h, full_w = (int(d) for d in latent.shape)
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError(
            f"latent dims {list(latent.shape)} not divisible by patch "
            f"{[pt, ph, pw]}"
        )
    t, h, w = full_t // pt, full_h // ph, full_w // pw
    packed = latent.reshape(b, c, t, pt, h, ph, w, pw)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(b * t * h * w, c * pt * ph * pw).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    *,
    latent_shape: Sequence[int],
    patch_size: Sequence[int] = (1, 2, 2),
) -> torch.Tensor:
    """``[N, C*pt*ph*pw]`` -> ``[B, C, T, H, W]``. ``latent_shape`` is (t,h,w,C)."""
    if rows.ndim != 2:
        raise ValueError(f"token rows must be rank 2, got {list(rows.shape)}")
    t, h, w, channel = _int_tuple(latent_shape, "latent_shape", 4)
    pt, ph, pw = _int_tuple(patch_size, "patch_size", 3)
    expected = pt * ph * pw * channel
    if int(rows.shape[-1]) != expected:
        raise ValueError(
            f"token width {int(rows.shape[-1])} != patch volume x channel {expected}"
        )
    per_sample = t * h * w
    if int(rows.shape[0]) % per_sample:
        raise ValueError(
            f"row count {int(rows.shape[0])} not divisible by t*h*w {per_sample}"
        )
    packed = rows.reshape(-1, t, h, w, channel, pt, ph, pw)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channel, t * pt, h * ph, w * pw).contiguous()


def unpack_audio_tokens(
    rows: torch.Tensor, *, audio_t: int, audio_channel: int = 2
) -> torch.Tensor:
    """``[audio_t, D]`` -> ``[C, D, audio_t // C]`` for the audio VAE.

    ``audio_t`` here is the *row* count (latent steps x channels); rows are
    channel-major, matching how the packed sequence lays them out.
    """
    if rows.ndim != 2:
        raise ValueError(f"audio token rows must be rank 2, got {list(rows.shape)}")
    audio_t = int(audio_t)
    audio_channel = int(audio_channel)
    if audio_t <= 0 or audio_channel <= 0:
        raise ValueError(
            f"audio_t and audio_channel must be positive, got {audio_t}, "
            f"{audio_channel}"
        )
    if int(rows.shape[0]) != audio_t:
        raise ValueError(f"audio rows {int(rows.shape[0])} != audio_t {audio_t}")
    if audio_t % audio_channel:
        raise ValueError(
            f"audio_t {audio_t} not divisible by audio_channel {audio_channel}"
        )
    native = rows.reshape(audio_channel, audio_t // audio_channel, int(rows.shape[-1]))
    return native.permute(0, 2, 1).contiguous()


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


def spatial_grid(latent_h: int, latent_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, w) coordinates for one latent frame: ``([rows, 2], w_axis)``."""
    area = np.sqrt(latent_h * latent_w)
    h_axis = axis_from_sqrt_area(latent_h, PATCH_H, area)
    w_axis = axis_from_sqrt_area(latent_w, PATCH_W, area)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_axis


def _pin_audio_w(g: torch.Tensor, sl: slice, count: int, w_axis: torch.Tensor) -> None:
    """Audio rows are channel-major, pinned to the two extremes of the w axis."""
    if count:
        g[sl.start : sl.start + count, 2] = float(w_axis[0])
        g[sl.start + count : sl.stop, 2] = float(w_axis[-1])


def _text_tags(
    token_tags: torch.Tensor, text_len: int, override: torch.Tensor | None
) -> None:
    """Tag the text block, honouring per-token tags from the encoder."""
    if override is None:
        token_tags[:text_len] = TAG_TEXT
        return
    tags = override.view(-1).to(torch.long)
    if int(tags.numel()) != text_len:
        raise ValueError(
            f"text_token_tags has {int(tags.numel())} entries but text_len "
            f"is {text_len}"
        )
    token_tags[:text_len] = tags


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

    Deliberately not shared with :func:`video_t_grid`: this sums via numpy
    (pairwise) to match the fl2va anchor, that one accumulates sequentially.
    They differ in the last ulp from n=16, and the anchor feeds RoPE.
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
    seq_len = align_packed_length(used)

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
    frame, w_grid = spatial_grid(latent_h, latent_w)

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
    _pin_audio_w(g, audio_sl, audio_t, w_grid)

    token_tags = torch.full((seq_len,), TAG_PAD, dtype=torch.long)
    _text_tags(token_tags, text_len, text_token_tags)
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

    # Text is global rows [0, text_len); intersect with this shard. Clamping
    # makes a text-free shard report its empty range *at* text_len -- (2, 2),
    # not (0, 0) -- matching the reference so golden diffs stay clean.
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

        [ text | ref blocks (request order) | target audio | target video | pad ]

    Blocks share one temporal cursor starting at ``text_len``: ``image`` takes a
    single integer slot, ``audio`` advances by its own latent length, and
    ``video``/``video_audio`` pack their audio immediately before their video,
    share a temporal origin, and advance by the longer of the two spans.

    Two update masks, because a single one cannot express "hold these audio rows
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
        seq_len = align_packed_length(used)
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

    target_frame, w_grid = spatial_grid(latent_h, latent_w)

    ref_img_parts: list[torch.Tensor] = []
    ref_audio_parts: list[torch.Tensor] = []
    t_cursor = float(text_len)
    for block in parsed:
        kind = block["kind"]
        if kind == "image":
            sl = block["visual_sl"]
            ref_img_parts.append(torch.arange(sl.start, sl.stop, dtype=torch.long))
            frame, _ = spatial_grid(int(block["latent_h"]), int(block["latent_w"]))
            g[sl, 0] = t_cursor
            g[sl, 1:] = frame
            t_cursor += 1.0
        elif kind == "audio":
            sl = block["audio_sl"]
            ref_t = int(block["ref_audio_t"])
            ref_audio_parts.append(torch.arange(sl.start, sl.stop, dtype=torch.long))
            g[sl, 0] = (t_cursor + torch.arange(ref_t, dtype=torch.float64)).repeat(
                audio_channel
            )
            _pin_audio_w(g, sl, ref_t, w_grid)
            t_cursor += float(ref_t)
        else:
            a_sl, v_sl = block["audio_sl"], block["visual_sl"]
            ref_t = int(block["ref_audio_t"])
            vt = int(block["latent_t"])
            ref_audio_parts.append(
                torch.arange(a_sl.start, a_sl.stop, dtype=torch.long)
            )
            ref_img_parts.append(torch.arange(v_sl.start, v_sl.stop, dtype=torch.long))
            rv_frame, rv_w_grid = spatial_grid(
                int(block["latent_h"]), int(block["latent_w"])
            )

            g[a_sl, 0] = (t_cursor + torch.arange(ref_t, dtype=torch.float64)).repeat(
                audio_channel
            )
            # A video block's audio pins to *its own* w grid, not the target's.
            _pin_audio_w(g, a_sl, ref_t, rv_w_grid)

            rv_g = g[v_sl].view(vt, int(block["frame_rows"]), 3)
            rv_g[:, :, 0] = video_t_grid(vt, t_cursor)[:, None]
            rv_g[:, :, 1:] = rv_frame[None]
            t_cursor += max(float(ref_t), temporal_position_span(vt))

    g[audio_sl, 0] = (t_cursor + torch.arange(audio_t, dtype=torch.float64)).repeat(
        audio_channel
    )
    _pin_audio_w(g, audio_sl, audio_t, w_grid)

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
    _text_tags(token_tags, text_len, text_token_tags)
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


def build_initial_latents(
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    seed: int | None = None,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(video_rows [Nv, 96], audio_rows [Na, 32])`` fp32 on CPU."""
    if seed is None:
        seed = DEFAULT_SEED
    seed = int(seed)

    pt, ph, pw = patch_size
    if latent_t % pt or latent_h % ph or latent_w % pw:
        raise ValueError(
            f"latent grid {latent_t}x{latent_h}x{latent_w} not divisible by "
            f"patch {patch_size}"
        )

    gen_v = torch.Generator().manual_seed(seed)
    video_tensor = torch.randn(
        1,
        VIDEO_LATENT_CHANNELS,
        latent_t,
        latent_h,
        latent_w,
        generator=gen_v,
        dtype=torch.float32,
    )
    video_rows = patchify_video_latent(video_tensor, patch_size=patch_size).to(
        torch.float32
    )

    # Independent generator, same seed -- each modality re-seeds its own.
    gen_a = torch.Generator().manual_seed(seed)
    audio_rows = torch.randn(
        audio_t * AUDIO_CHANNELS,
        AUDIO_LATENT_CHANNELS,
        generator=gen_a,
        dtype=torch.float32,
    )

    expected_video = (
        (latent_t // pt) * (latent_h // ph) * (latent_w // pw),
        VIDEO_LATENT_CHANNELS * pt * ph * pw,
    )
    if tuple(video_rows.shape) != expected_video:
        raise ValueError(
            f"video noise shape {tuple(video_rows.shape)} != {expected_video}"
        )
    return video_rows, audio_rows


def scatter_rows_into_packed(
    *,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-form latents -> the full-length ``x`` / ``audio_x`` the DiT takes.

    The DiT reads ``[1, S, 96]`` and ``[1, S, 32]`` buffers indexed by global
    row id, even though only the media rows carry data; padding and text rows
    stay zero.
    """
    device = video_rows.device
    x = torch.zeros(
        1, seq_len, video_rows.shape[-1], dtype=video_rows.dtype, device=device
    )
    audio_x = torch.zeros(
        1, seq_len, audio_rows.shape[-1], dtype=audio_rows.dtype, device=device
    )
    x[0].index_copy_(0, img_pos.to(device), video_rows)
    audio_x[0].index_copy_(0, audio_pos.to(device), audio_rows)
    return x, audio_x
