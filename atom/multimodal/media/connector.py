# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Read encoded media without depending on a modality or a model."""

import base64
import binascii
import urllib.request
from os import PathLike, fspath
from pathlib import Path
from urllib.parse import unquote, urlsplit


class MediaConnector:
    def __init__(self, timeout: float = 30):
        self.timeout = timeout

    def read(self, source: str | PathLike[str]) -> bytes:
        source = fspath(source)
        if source.startswith("data:"):
            try:
                header, encoded = source.split(",", 1)
                if not header.endswith(";base64"):
                    raise ValueError("Expected a base64 data URL")
                return base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Invalid base64 media data URL") from exc
        if source.startswith(("http://", "https://")):
            with urllib.request.urlopen(source, timeout=self.timeout) as response:
                return response.read()
        if source.startswith("file://"):
            parsed = urlsplit(source)
            if parsed.netloc not in ("", "localhost"):
                raise ValueError("file URLs must refer to a local path")
            source = unquote(parsed.path)
        return Path(source).read_bytes()
