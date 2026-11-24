# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public wire constants shared by ATOM's RDMA weight sender and receiver."""

CMD_END = 0
CMD_BUCKET = 1
HEADER_WORDS = 4  # command, metadata bytes, payload bytes, version
