# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi's Triton MLA builder, kept apart from `kimi_mla_gdn_attn`.

`triton_mla` imports aiter's Triton MLA decode, which the default aiter MLA path
does not need, so only `KimiMLAGDNBackend.get_builder_cls` imports this module,
and only when `ATOM_USE_TRITON_MLA` is set.
"""

from .kimi_mla_gdn_attn import _KimiMLAGDNCommon
from .triton_mla import TritonMLAMetadataBuilder


class KimiTritonMLAGDNMetadataBuilder(_KimiMLAGDNCommon, TritonMLAMetadataBuilder):
    pass
