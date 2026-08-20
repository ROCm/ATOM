# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat-template dispatch for the OpenAI chat endpoint.

Some models (e.g. DeepSeek V4) ship a Python encoder under ``<model>/encoding/``
instead of a Jinja ``chat_template``. This module discovers such encoders at
server startup and provides :func:`apply_chat_template`, a single entry point
that the request handler calls — it transparently routes to the custom encoder
when one was found, or to ``tokenizer.apply_chat_template`` otherwise.
"""

import glob
import importlib.util
import logging
import os
import pathlib
from typing import Any

from huggingface_hub import snapshot_download
from jinja2 import TemplateError

from .chat_encoder_adapters import (
    MessageEncoderAdapter,
    build_message_encoder_adapter,
)

logger = logging.getLogger("atom")


def _resolve_model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    try:
        return snapshot_download(model, local_files_only=True, allow_patterns=[])
    except Exception:
        return model


def _load_encoder_from_dir(model_path: str) -> MessageEncoderAdapter | None:
    """Look for ``<model>/encoding/encoding_*.py`` and load ``encode_messages``.

    Returns ``None`` when the directory or matching file is absent (model uses
    the standard Jinja path). Returns ``None`` and warns on ambiguity (multiple
    matches) or load failures.

    """
    enc_dir = os.path.join(model_path, "encoding")
    if not os.path.isdir(enc_dir):
        return None

    candidates = sorted(glob.glob(os.path.join(enc_dir, "encoding_*.py")))
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            f"Multiple encoding_*.py found in {enc_dir}, refusing to guess: "
            f"{[os.path.basename(p) for p in candidates]}"
        )
        return None

    enc_path = candidates[0]
    module_name = os.path.splitext(os.path.basename(enc_path))[0]
    try:
        spec = importlib.util.spec_from_file_location(module_name, enc_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw = mod.encode_messages
    except Exception:
        # Broad on purpose: this executes a file the operator supplied, so the
        # failure can be anything that module raises, and a bad encoder must
        # not stop the server. `exc_info` rather than `{e}` -- the message
        # alone was never enough to debug someone else's encoder, and the
        # traceback names the line inside it.
        logger.warning(f"Failed to load encoder from {enc_path}", exc_info=True)
        return None

    logger.info(f"Loaded message encoder from {enc_path}")
    # also valid is "chat" (non-thinking short-form). May need to add as an option.
    # Revisit when a second model ships an encode_*.py — the default may need to be per-model.
    #
    # Handed to the adapter rather than applied in a wrapper here. A wrapper
    # runs *after* the adapter has filtered kwargs against the encoder's
    # signature, so the one kwarg it adds is the one the filter cannot remove:
    # an encoder that does not take `thinking_mode` raised `TypeError` on
    # every real request while the startup probe -- which now counts TypeError
    # as a refusal -- reported only "tool calls will be delivered as plain
    # text". Silent at startup, 500 on every chat.
    return build_message_encoder_adapter(
        module_name, raw, enc_path, defaults={"thinking_mode": "thinking"}
    )


def load_custom_message_encoder(model_path: str) -> MessageEncoderAdapter | None:
    """Probe ``model_path`` once at startup for a custom message encoder.

    Returns the encoder, or ``None`` when the model uses the standard Jinja
    ``chat_template`` path. Result should be cached by the caller — this does
    filesystem IO and a Python import.
    """
    return _load_encoder_from_dir(_resolve_model_path(model_path))


# The smallest request that makes a template show its framing. Nothing is
# ever sent to the model; only what the template wraps a turn in matters.
PROBE_MESSAGES: list[dict] = [{"role": "user", "content": "hi"}]
PROBE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def chat_template_source(
    tokenizer: Any, custom_encoder: MessageEncoderAdapter | None = None
) -> str:
    """The model's chat template as text to search, or ``""`` if it ships none.

    Deliberately not a rendered prompt, and the two are not interchangeable:
    a marker that appears only in the template's own logic is absent from a
    fresh prompt. Measured on this box -- Qwen3.5's source carries `<think>`
    and `</think>`, its rendered prompt carries only `<think>`, and
    Qwen3-8B's rendered prompt carries neither. A question about what the
    template *does with a reply* has to read the source; a question about what
    the prompt *tells the model* has to read the render
    (:func:`render_probe_prompt`).

    Two shapes bite anyone reaching for the raw attribute, which is why this
    exists rather than `getattr(tokenizer, "chat_template", "")`:
    multi-template tokenizers hold a ``dict``, and `"</think>" in <dict>`
    silently tests the *keys*; and it is ``None`` for every model that ships a
    Python encoder instead of Jinja (Kimi-K3, DeepSeek-V4), whose literals
    live in that module's source instead.
    """
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return template
    if isinstance(template, dict):
        # Every named variant, so a marker in any of them counts. Searching
        # the dict itself would have searched the names.
        return "\n".join(str(v) for v in template.values())
    if custom_encoder is not None and custom_encoder.source_path:
        try:
            return pathlib.Path(custom_encoder.source_path).read_text()
        except OSError as e:
            logger.warning("Could not read %s: %s", custom_encoder.source_path, e)
    return ""


# (kwarg, off-value, on-value) triples, tried in order. Every family on this
# box is covered: Qwen reads `enable_thinking`, Kimi-K3 `thinking`, and
# `thinking_mode` is read by two families with disjoint vocabularies --
# MiniMax-M3 wants "disabled"/"enabled" and DeepSeek-V4's encoder asserts on
# anything outside {"chat", "thinking"}. Hence pairs of the same name, and
# hence the order: "disabled" first, so MiniMax matches before V4's rejection
# of it sends the probe on to "chat".
#
# The on-value is the off-value's counterpart and is not probed for. It cannot
# be: most templates default to reasoning on, so passing their on-value renders
# identically to passing nothing, and "the render changed" -- the evidence the
# off-probe runs on -- is absent by construction. Only MiniMax, whose default
# is off, differs when switched on.
REASONING_TOGGLES: tuple[tuple[str, Any, Any], ...] = (
    ("enable_thinking", False, True),
    ("thinking", False, True),
    ("thinking_mode", "disabled", "enabled"),
    ("thinking_mode", "chat", "thinking"),
)

# A template refusing a probe value, by name. A model-shipped Python encoder
# validates with a bare `assert`; Jinja raises its own; a signature that does
# not take the kwarg raises TypeError. Refusal means "not this pair, try the
# next" -- anything else is a bug and is left to propagate.
_PROBE_REFUSALS = (TemplateError, TypeError, ValueError, AssertionError)


def resolve_reasoning_toggle(
    tokenizer: Any, custom_encoder: MessageEncoderAdapter | None = None
) -> tuple[str, Any, Any] | None:
    """The kwarg that turns this model's reasoning off, and the value, or None.

    Rendered twice and compared, rather than matched against a table of model
    families: a Jinja template silently ignores a kwarg it does not read, so
    "the prompt changed" *is* the evidence that it read this one. That silence
    is also why the question has to be asked at all -- the chat path passed a
    hardcoded ``thinking=`` to every model, which is correct for Kimi-K3 and a
    no-op for the entire Qwen family, whose templates read `enable_thinking`.
    A no-op here is invisible: the model reasons anyway and the client that
    asked for no reasoning simply pays for it.

    ``None`` means the template offers no switch, and for a model that begins
    inside the reasoning channel that means reasoning cannot be turned off at
    all -- worth saying out loud at startup, because the request can then only
    be honoured as far as separating the reasoning and reporting it, never by
    discarding it. On this box only gpt-oss and DeepSeek-R1 answer ``None``,
    and neither opens a channel this way to begin with.

    Verified against every model on this box: Qwen3/Qwen3.5 `enable_thinking`,
    Kimi-K3 `thinking`, MiniMax-M3 `thinking_mode="disabled"`, DeepSeek-V4
    `thinking_mode="chat"` -- and for all six that start inside the channel,
    applying the pair takes the rendered prompt back out of it.
    """
    baseline = render_probe_prompt(tokenizer, custom_encoder, tools=False)
    if baseline is None:
        return None
    for name, off_value, on_value in REASONING_TOGGLES:
        try:
            rendered = apply_chat_template(
                tokenizer, custom_encoder, PROBE_MESSAGES, **{name: off_value}
            )
        except _PROBE_REFUSALS:
            continue
        if rendered != baseline:
            return name, off_value, on_value
    return None


def render_probe_prompt(
    tokenizer: Any,
    custom_encoder: MessageEncoderAdapter | None,
    *,
    tools: bool,
) -> str | None:
    """Render a throwaway turn, to ask the template about itself at startup.

    What a template renders *into the prompt* is the model's own instructions
    -- notably how to call a tool -- so a question about those is asked here
    and not of the template source. The reverse question reads the source
    instead; see :func:`chat_template_source` for why one is not a cheaper
    version of the other.

    ``None`` means the template refused the probe, which is a real answer: a
    template may reject the synthetic tools payload, and a model that does not
    do tool calls should still start. Refusal is `_PROBE_REFUSALS`, the same
    list the reasoning probe uses -- these two ask the same template the same
    kind of question and disagreeing about what counts as a refusal is how a
    model with no chat template at all stopped booting. `transformers` raises
    `ValueError` for that, this caught only `TemplateError` and `TypeError`,
    and both probes run *before* the engine is created, so a base checkpoint
    that used to serve `/v1/completions` perfectly well now died at startup.

    Nothing outside that list is caught -- an unexpected failure here is a
    bug, and a bug that silently turns into "this model has no tool-call
    format" is the class of silence this path exists to end.
    """
    try:
        return apply_chat_template(
            tokenizer,
            custom_encoder,
            PROBE_MESSAGES,
            tools=PROBE_TOOLS if tools else None,
        )
    except _PROBE_REFUSALS as e:
        logger.warning("The model's chat template refused the probe: %s", e)
        return None


def apply_chat_template(
    tokenizer: Any,
    custom_encoder: MessageEncoderAdapter | None,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    **kwargs: Any,
) -> str:
    """Render ``messages`` to a prompt string.

    Dispatches to ``custom_encoder`` if one was discovered for this model,
    otherwise to ``tokenizer.apply_chat_template``. Jinja-only kwargs
    (``tokenize``, ``add_generation_prompt``) are stripped on the custom path.
    Model-scoped adapters prepare tools for custom encoders that support them;
    the generic path does not apply DeepSeek-V4-specific message rewriting.
    """
    if custom_encoder is not None:
        for k in ("tokenize", "add_generation_prompt"):
            kwargs.pop(k, None)
        if tools and not custom_encoder.supports_tools:
            logger.warning(
                "tools= is not supported by custom message encoder %s; ignoring.",
                custom_encoder.name,
            )
        messages = custom_encoder.prepare_messages(messages, tools)
        return custom_encoder(messages, **kwargs)

    # ATOM's OpenAI layer normalizes thinking controls for custom encoders.
    # Mirror them to the names used by Qwen3.8's Hugging Face Jinja template.
    if "thinking" in kwargs:
        kwargs.setdefault("enable_thinking", kwargs["thinking"])
    if "thinking_effort" in kwargs:
        kwargs.setdefault("reasoning_effort", kwargs["thinking_effort"])

    kwargs["tokenize"] = False
    kwargs["add_generation_prompt"] = True
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)
