# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fill-in-the-middle prompts for ``/beta/completions``.

A FIM request gives the text before and after a gap; the model writes the
middle. Models trained for it read the two sides in prefix-suffix-middle
order between three special tokens, and the tokens are the only
model-specific part -- so a format is recognised by its tokens being in the
vocabulary, the same way for every model, rather than by model name.
"""

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class FimFormat:
    """One model family's prefix-suffix-middle markers."""

    begin: str
    hole: str
    end: str
    # Rendered first, as the model saw every training document start.
    bos: str = ""

    def render(self, prefix: str, suffix: str) -> str:
        return f"{self.bos}{self.begin}{prefix}{self.hole}{suffix}{self.end}"


KNOWN_FIM_FORMATS: tuple[FimFormat, ...] = (
    # DeepSeek-Coder, V3 and V4.1.
    FimFormat(begin="<｜fim▁begin｜>", hole="<｜fim▁hole｜>", end="<｜fim▁end｜>"),
)


def resolve_fim_format(tokenizer: Any) -> FimFormat | None:
    """The FIM format this tokenizer's vocabulary carries, or ``None``.

    Each marker must be a single token: a marker the tokenizer splits into
    pieces of ordinary text is not one the model was trained to read.

    The BOS token is written into the prompt only when encoding does not
    already add it, so the model sees it exactly once either way.
    """
    vocab = tokenizer.get_vocab()
    for fim in KNOWN_FIM_FORMATS:
        if all(marker in vocab for marker in (fim.begin, fim.hole, fim.end)):
            bos = getattr(tokenizer, "bos_token", None) or ""
            if bos and tokenizer.encode("")[:1] == [vocab[bos]]:
                bos = ""
            return replace(fim, bos=bos)
    return None
