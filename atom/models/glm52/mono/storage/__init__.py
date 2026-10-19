# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GLM-5.2's host-side weight preparation: the storage
vocabulary (``config``), MX format conversion (``formats``), and the packing
that turns a loaded tensor into what a kernel reads (``packing``, ``layer``).
"""
