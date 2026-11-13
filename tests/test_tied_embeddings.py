# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A tied LM head is the embedding's Parameter, not a second Parameter on the
same storage.

With ``lm_head.weight.data = embed_tokens.weight.data`` there are two
Parameters; anything that counts parameters by identity, as a weight update's
coverage check does, sees both, and a trainer sending HF ``named_parameters()``
sends only ``embed_tokens.weight``.
"""

import pathlib
import re

MODELS = pathlib.Path(__file__).resolve().parents[1] / "atom" / "models"
TIE_BY_DATA = re.compile(r"\.weight\.data\s*=\s*\S*embed\w*\.weight\.data")


def test_no_model_ties_its_head_by_data():
    offenders = [
        f"{path.relative_to(MODELS)}:{number}"
        for path in sorted(MODELS.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if TIE_BY_DATA.search(line)
    ]
    assert offenders == [], f"tie the Parameter instead: {offenders}"
