# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Output encoding for diffusion pipelines."""

from atom.diffusion.postprocess.mux import (
    frames_to_uint8,
    write_video_with_audio,
)

__all__ = ["frames_to_uint8", "write_video_with_audio"]
