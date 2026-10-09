# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 networks other than the DiT: both VAEs, the text encoder,
and DiT weight loading."""

import contextlib
import json
import logging
import os
from typing import Any

import torch

from atom.diffusion.models.minimax_h3.layout import (
    unpack_audio_tokens,
    unpatchify_video_tokens,
)

logger = logging.getLogger(__name__)


# The checkpoint ships fp32 weights, but H3's video VAE is transformer-based
# (39.7% of decode is addmm, 0.0% convolution) so fp32 GEMMs dominate it.
# Measured at 1344x768x124f: fp32 88.4 s, bf16 24.4 s, and the two agree to
# 51.4 dB -- an order of magnitude inside the 41 dB bar the whole pipeline is
# validated at. Encode still runs fp32; see encode_keyframe_cond_rows.
VIDEO_VAE_DECODE_DTYPE = torch.bfloat16


def load_checkpoint_vae(
    path: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> Any:
    """Instantiate a VAE from the code bundled in the checkpoint directory."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_path = os.path.join(path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    auto_map = config.get("auto_map", {})
    ref = auto_map.get("AutoModel")
    if not ref:
        raise ValueError(
            f"{config_path} has no auto_map.AutoModel entry; cannot locate the "
            f"bundled VAE class"
        )

    cls = get_class_from_dynamic_module(ref, path)
    model = cls.from_pretrained(path)
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    model = model.eval()
    logger.info(
        "loaded %s from %s on %s (%s)",
        type(model).__name__,
        path,
        device,
        dtype or next(model.parameters()).dtype,
    )
    return model


def enable_parallel_tiled_decode(vae, *, group, rank, world_size) -> bool:
    """Point the checkpoint's bundled VAE at our sequence-parallel group.

    It already implements tiled decode and its rank sharding, but cannot know
    our process group, so it seeds ``sp_size = 1`` and runs every tile on one
    rank. Overwriting that state is the whole change. False when world size 1.
    """
    if world_size <= 1:
        return False
    import sys

    module = sys.modules.get(type(vae).__module__)
    get_state = getattr(module, "get_parallel_state", None)
    if get_state is None:
        logger.warning("bundled VAE exposes no parallel state; decode stays serial")
        return False

    # Mutated in place: klvae holds a reference to the same dict.
    get_state().update(
        {
            "group_size": world_size,
            "group_rank": rank,
            "local_process_group": group,
            "sp_size": world_size,
            "sp_rank": rank,
            "sp_enabled": True,
            "sp_process_group": group,
        }
    )
    logger.info("parallel tiled decode enabled: rank %d of %d", rank, world_size)
    return True


def latent_stats(path: str) -> tuple[list[float], list[float]] | None:
    """Read ``latents_mean``/``latents_std`` from a VAE config.

    Raises if the config is missing -- that is a wrong checkpoint path and
    should say so. Returns ``None`` only when the config exists but declares no
    stats, which is the audio VAE's case on some partitions.
    """
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    mean = config.get("latents_mean")
    std = config.get("latents_std")
    if mean is None or std is None:
        return None
    return list(mean), list(std)


def denormalize_latents(
    latents: torch.Tensor,
    *,
    mean: list[float] | torch.Tensor,
    std: list[float] | torch.Tensor,
    name: str = "vae",
) -> torch.Tensor:
    """``z * std + mean`` over the channel axis (dim 1). Returns a new tensor."""
    mean_t = torch.as_tensor(mean, device=latents.device, dtype=latents.dtype)
    std_t = torch.as_tensor(std, device=latents.device, dtype=latents.dtype)
    if mean_t.ndim != 1 or std_t.ndim != 1:
        raise ValueError(f"{name} latents_mean/std must be 1-D")
    if mean_t.shape != std_t.shape:
        raise ValueError(
            f"{name} mean/std shape mismatch: {tuple(mean_t.shape)} vs "
            f"{tuple(std_t.shape)}"
        )
    if latents.ndim < 2:
        raise ValueError(f"{name} latents need a channel dimension")
    if int(latents.shape[1]) != int(mean_t.shape[0]):
        raise ValueError(
            f"{name} channel mismatch: latents.shape[1]="
            f"{int(latents.shape[1])} vs {int(mean_t.shape[0])} stats"
        )
    view = [1] * latents.ndim
    view[1] = int(mean_t.shape[0])
    return latents * std_t.view(*view) + mean_t.view(*view)


def crop_to_canvas(frames: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Crop ``[B, C, T, H, W]`` down to the requested canvas.

    The VAE pads the latent grid up to tile multiples and the padding lands at
    the bottom/right, so cropping from the origin is correct.
    """
    if frames.ndim != 5:
        raise ValueError(f"frames must be rank 5, got {list(frames.shape)}")
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h < height or w < width:
        raise ValueError(
            f"decoded frames {h}x{w} are smaller than the target canvas "
            f"{height}x{width}"
        )
    if h == height and w == width:
        return frames
    return frames[..., :height, :width].contiguous()


def denormalize_pixels(frames: torch.Tensor, vae: Any) -> torch.Tensor:
    """ImageNet-normalized decoder output -> pixels in [0, 1].

    The decoder emits normalized pixel space, not displayable pixels; the
    finish is ``transform_rev(x).clamp(0, 1)``. Treating the output as [-1, 1]
    instead still looks plausible -- the error is per-channel, so it survives
    every structural check and shows up only in a pixel comparison.

    ``transform_rev`` is 4-D (N, C, H, W), so T folds into the batch first.
    """
    transform_rev = getattr(vae, "transform_rev", None)
    if transform_rev is None:
        raise AttributeError(
            "video VAE has no transform_rev; cannot map decoder output out of "
            "normalized pixel space"
        )
    if frames.ndim != 5:
        raise ValueError(f"frames must be rank 5, got {list(frames.shape)}")

    b, c, t, h, w = frames.shape
    flat = frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    flat = transform_rev(flat).clamp_(0.0, 1.0)
    return flat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def decode_video_rows(
    vae: Any,
    rows: torch.Tensor,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    height: int,
    width: int,
    mean: list[float],
    std: list[float],
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """DiT video rows -> decoded frames ``[B, C, T, H, W]``."""
    pt, ph, pw = patch_size
    latent = unpatchify_video_tokens(
        rows.float(),
        latent_shape=(latent_t // pt, latent_h // ph, latent_w // pw, 24),
        patch_size=patch_size,
    )
    latent = denormalize_latents(latent, mean=mean, std=std, name="video_vae")
    z = latent.to(next(vae.parameters()).dtype)

    # Clip-aware path, not the base `decode`: the latter upsamples uniformly by
    # vae_ratio_t (37 latents -> 148 frames) where H3 frames live on the 17n+5
    # lattice (37 -> 124). Nothing downstream catches the difference -- the file
    # is still a valid MP4, just with the wrong frames.
    decode_fn = getattr(vae, "decode_temporal", None) or vae.decode
    frames = decode_fn(z)
    frames = getattr(frames, "sample", frames)
    frames = denormalize_pixels(frames.float(), vae)
    return crop_to_canvas(frames, height=height, width=width)


@torch.no_grad()
def decode_audio_rows(
    vae: Any,
    rows: torch.Tensor,
    *,
    audio_channel: int = 2,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    """DiT audio rows -> decoded waveform."""
    latent = unpack_audio_tokens(
        rows.float(), audio_t=int(rows.shape[0]), audio_channel=audio_channel
    )
    if mean is not None and std is not None:
        latent = denormalize_latents(latent, mean=mean, std=std, name="audio_vae")
    waveform = vae.decode(latent.to(next(vae.parameters()).dtype))
    return getattr(waveform, "sample", waveform)


# Output after layer 49 == hidden_states[50].
MINIMAX_H3_SELECTED_LM_LAYER = 50
MINIMAX_H3_TEXT_DIM = 5120


class MiniMaxH3TextEncoder:
    """Wraps Qwen3-VL and returns H3's ``[T, 5120]`` conditioning rows."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str,
        processor: Any | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device(device)
        # Populated on first resident_on(): the pinned, authoritative host copy.
        self._host_tensors: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        num_layers: int = MINIMAX_H3_SELECTED_LM_LAYER,
        attn_implementation: str | None = None,
    ) -> "MiniMaxH3TextEncoder":
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        # Keep one layer beyond the one we read: the selected state must be an
        # *intermediate* entry of hidden_states, because transformers puts the
        # post-final-norm activation in the last slot.
        keep_layers = num_layers + 1
        if hasattr(config, "text_config"):
            config.text_config.num_hidden_layers = keep_layers
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = keep_layers

        try:
            from transformers import Qwen3VLForConditionalGeneration as _Cls
        except ImportError as exc:  # pragma: no cover - depends on transformers
            raise ImportError(
                "transformers is too old for Qwen3-VL; MiniMax-H3 text "
                "conditioning needs Qwen3VLForConditionalGeneration"
            ) from exc

        kwargs: dict[str, Any] = {}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = _Cls.from_pretrained(
            path, config=config, torch_dtype=dtype, trust_remote_code=True, **kwargs
        )
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 - text-only still works
            logger.warning("no AutoProcessor (%s); fl2va images unavailable", exc)
            processor = None
        logger.info(
            "loaded MiniMax-H3 text encoder from %s (%d layers kept, reading "
            "hidden_states[%d], %s)",
            path,
            keep_layers,
            num_layers,
            dtype,
        )
        return cls(model, tokenizer, device=device, processor=processor)

    @torch.no_grad()
    def encode(self, prompt: str, images: Any | list | None = None) -> torch.Tensor:
        """Prompt (optionally with keyframes) -> ``[T, 5120]`` rows.

        For fl2va the keyframe also goes through Qwen3-VL's vision tower, so it
        occupies real positions in the conditioning sequence and must be tagged
        VIDEO downstream; see :meth:`encode_with_tags`.
        """
        rows, _ = self.encode_with_tags(prompt, images)
        return rows

    def _tensors(self):
        yield from self.model.parameters()
        yield from self.model.buffers()

    def to(self, device: torch.device | str) -> "MiniMaxH3TextEncoder":
        """Move the encoder and remember where it is."""
        self.model = self.model.to(device)
        self.device = torch.device(device)
        return self

    def prime_host_cache(self) -> None:
        """Pin the host weights once, so no request pays for it.

        Pinning 50 GiB takes ~11 s; done at load it is startup cost, done
        lazily it lands on the first request.
        """
        if self._host_tensors is not None:
            return
        self._host_tensors = []
        for tensor in self._tensors():
            host = tensor.data
            if host.device.type == "cpu" and not host.is_pinned():
                try:
                    host = host.pin_memory()
                except RuntimeError:  # pragma: no cover - host-dependent
                    logger.warning("could not pin host weights; uploads will be slower")
            tensor.data = host
            self._host_tensors.append((tensor, host))

    @contextlib.contextmanager
    def resident_on(self, device: torch.device | str):
        """Hold the encoder on ``device`` for one encode, then release it.

        H3 encodes once per request and never touches the encoder again, but at
        50 GiB it is the largest resident tensor on the main rank -- leaving it
        beside the DiT overflows a 192 GB card during denoise (measured: the
        first served request died with 182 GiB allocated).

        Weights are read-only under inference, so the host copy stays
        authoritative and releasing is just dropping the device copy: no
        copy-back. With the host side pinned that turns a 12.7 s round trip
        into a ~2 s upload.
        """
        self.prime_host_cache()

        for tensor, host in self._host_tensors:
            tensor.data = host.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.device = torch.device(device)
        try:
            yield self
        finally:
            for tensor, host in self._host_tensors:
                tensor.data = host
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a prebuilt presentation through Qwen3-VL -> ``[T, 5120]`` rows.

        The caller owns the token stream, which is the whole point: fl2va and
        ref2va differ only in how that stream is built.
        """
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )

        ids = input_ids.to(device=self.device, dtype=torch.long)
        batch: dict[str, Any] = {"input_ids": ids.unsqueeze(0)}
        if pixel_values is not None or pixel_values_videos is not None:
            # Hand-built input_ids means no mm_token_type_ids from the
            # processor, and Qwen3-VL raises rather than guess. Mark exactly the
            # pad positions -- vision_start/end are ordinary text tokens.
            from atom.diffusion.models.minimax_h3.conditioning import (
                IMAGE_PAD,
                VIDEO_PAD,
            )

            mm = torch.zeros_like(ids)
            for token in (IMAGE_PAD, VIDEO_PAD):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    mm |= (ids == token_id).to(torch.long)
            batch["mm_token_type_ids"] = mm.unsqueeze(0)
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            batch["image_grid_thw"] = image_grid_thw.to(self.device, torch.long)
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos.to(
                self.device, torch.bfloat16
            )
            batch["video_grid_thw"] = video_grid_thw.to(self.device, torch.long)

        out = self.model(**batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states
        if len(hidden) <= MINIMAX_H3_SELECTED_LM_LAYER:
            raise ValueError(
                f"encoder returned {len(hidden)} hidden states; need at least "
                f"{MINIMAX_H3_SELECTED_LM_LAYER + 1} to select layer "
                f"{MINIMAX_H3_SELECTED_LM_LAYER}"
            )
        rows = hidden[MINIMAX_H3_SELECTED_LM_LAYER][0]
        if int(rows.shape[-1]) != MINIMAX_H3_TEXT_DIM:
            raise ValueError(
                f"text embeddings are {int(rows.shape[-1])} wide, expected "
                f"{MINIMAX_H3_TEXT_DIM}"
            )
        return rows

    def image_token_counts(self, images: list) -> tuple[dict, list[int]]:
        """Preprocess images and report Qwen's per-image token count."""
        if self.processor is None:
            raise RuntimeError(
                "images supplied but no processor is loaded; conditioning on "
                "images needs AutoProcessor for the vision tower"
            )
        vision = self.processor.image_processor(images=images, return_tensors="pt")
        grid = vision["image_grid_thw"]
        merge = int(self.processor.image_processor.merge_size) ** 2
        counts = [int(grid[i].prod().item()) // merge for i in range(len(images))]
        return vision, counts

    @torch.no_grad()
    def encode_with_tags(
        self, prompt: str, images: Any | list | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rows [T, 5120], token_tags [T])`` for t2va / fl2va.

        Tags mark vision blocks as VIDEO so the DiT's AdaLN gather treats those
        positions as image, not text.
        """
        from atom.diffusion.models.minimax_h3.conditioning import (
            multi_image_presentation,
            text_only_presentation,
        )

        if not prompt:
            raise ValueError("prompt must be non-empty")

        if images is None:
            ids, tags = text_only_presentation(self.tokenizer, prompt=prompt)
            rows = self.encode_ids(ids)
        else:
            if not isinstance(images, list):
                images = [images]
            vision, counts = self.image_token_counts(images)
            ids, tags = multi_image_presentation(
                self.tokenizer, prompt=prompt, image_token_counts=counts
            )
            rows = self.encode_ids(
                ids,
                pixel_values=vision["pixel_values"],
                image_grid_thw=vision["image_grid_thw"],
            )

        if int(tags.numel()) != int(rows.shape[0]):
            raise ValueError(
                f"presentation produced {int(tags.numel())} tags for "
                f"{int(rows.shape[0])} rows"
            )
        return rows, tags


_INDEX_NAME = "model.safetensors.index.json"


def _shard_files(path: str) -> dict[str, str]:
    """Map tensor name -> shard filename, for sharded or single-file dirs."""
    index_path = os.path.join(path, _INDEX_NAME)
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    single = os.path.join(path, "model.safetensors")
    if not os.path.exists(single):
        raise FileNotFoundError(
            f"no {_INDEX_NAME} and no model.safetensors under {path}"
        )
    from safetensors import safe_open

    with safe_open(single, framework="pt") as f:
        return dict.fromkeys(f.keys(), "model.safetensors")


def load_minimax_h3_dit_weights(
    model: torch.nn.Module,
    path: str,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> int:
    """Load a DiT from safetensors shards, applying the QKV reorder.

    The checkpoint stores QKV **interleaved per query group**, not as [Q;K;V]:
    ``num_query_groups`` blocks of ``(heads_per_group + 2) * head_dim`` rows. A
    plain three-way split of the fused tensor is silently wrong.

    Fails loudly on any missing or unexpected tensor -- a partially loaded DiT
    produces plausible noise rather than an error.
    """
    from safetensors import safe_open

    from atom.diffusion.models.minimax_h3.dit import reorder_grouped_qkv_to_qkv

    arch = model.arch
    weight_map = _shard_files(path)
    own = dict(model.state_dict())

    missing = sorted(set(own) - set(weight_map))
    unexpected = sorted(set(weight_map) - set(own))
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"checkpoint/module mismatch: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:3]})"
        )

    # Group by shard so each file opens once.
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name in own:
            by_shard.setdefault(shard, []).append(name)

    loaded = 0
    for shard, names in sorted(by_shard.items()):
        with safe_open(os.path.join(path, shard), framework="pt") as f:
            for name in names:
                tensor = f.get_tensor(name)
                if name.endswith("attn.qkv_proj.weight"):
                    tensor = reorder_grouped_qkv_to_qkv(
                        tensor,
                        num_query_groups=arch.num_attention_heads,
                        heads_per_group=1,
                        head_dim=arch.attention_head_dim,
                    )
                target = own[name]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"{name}: checkpoint shape {tuple(tensor.shape)} != "
                        f"module shape {tuple(target.shape)}"
                    )
                target.data.copy_(
                    tensor.to(device=device, dtype=target.dtype), non_blocking=False
                )
                loaded += 1
        logger.debug("loaded %d tensors from %s", len(names), shard)

    logger.info("loaded %d tensors from %s", loaded, path)
    return loaded
