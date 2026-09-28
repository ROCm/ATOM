"""Patch SGLang LoadConfig for ATOM-only model loader extras.

ATOM reuses SGLang's ``--model-loader-extra-config`` to receive plugin-only
options such as ``online_quant_config``. SGLang parses the same JSON into
``LoadConfig.model_loader_extra_config`` and later passes it to its default
model loader, whose validation rejects keys it does not own. Without this
patch, a valid ATOM online-quant config fails before ATOM can consume it.

The ATOM config translation reads and stores ``online_quant_config`` earlier in
the plugin setup path, so this runtime patch only removes the ATOM-private key
from SGLang's loader-facing copy after ``LoadConfig.__post_init__`` has parsed
the JSON string into a dict.
"""

from __future__ import annotations


def apply_load_config_patch() -> None:
    from sglang.srt.configs.load_config import LoadConfig

    if getattr(LoadConfig, "_atom_online_quant_patch", False):
        return

    original_post_init = LoadConfig.__post_init__

    def patched_post_init(self):
        original_post_init(self)
        extra_config = self.model_loader_extra_config or {}
        if isinstance(extra_config, dict):
            extra_config.pop("online_quant_config", None)

    LoadConfig.__post_init__ = patched_post_init
    LoadConfig._atom_online_quant_patch = True
