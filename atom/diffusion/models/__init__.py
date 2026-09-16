# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Diffusion models, one package per model family.

Model-major rather than component-major: everything MiniMax-H3 needs -- arch,
DiT, VAEs, encoder, scheduler, loader, stages and pipeline -- lives in one
directory, so adding a model is one new package rather than edits scattered
across dits/, vaes/, encoders/ and schedulers/. Components graduate out to the
shared layer when a *second* model actually uses them.
"""
