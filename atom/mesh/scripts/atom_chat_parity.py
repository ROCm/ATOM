# SPDX-License-Identifier: MIT
"""ATOM's prompt tokens for a set of chats, to compare with the router's.

Renders each chat as ATOM's /v1/chat/completions does on a checkpoint, and
prints as JSON the `chat_rendering` its /v1/models card reports with each
chat's request and token ids, or the error ATOM rejects it with. The router's
ignored test `test_checkpoint_chats_tokenize_as_on_atom` reads that, renders
the chats as HTTP routing does for workers reporting that rendering, and
compares the tokens:

    python atom/mesh/scripts/atom_chat_parity.py <checkpoint dir> \\
        [--default-chat-template-kwargs '{"enable_thinking": false}'] > atom.json
    cd atom/mesh && ATOM_CHAT_PARITY=../../atom.json \\
        cargo test --lib checkpoint_chats -- --ignored --nocapture

Needs ATOM's Python dependencies, not a GPU.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    load_custom_message_encoder,
    render_probe_prompt,
    resolve_reasoning_toggle,
    template_needs_tool_call_argument_strings,
)
from atom.entrypoints.openai.images import check_chat_images
from atom.entrypoints.openai.model_capabilities import chat_rendering
from atom.entrypoints.openai.protocol import ChatCompletionRequest
from atom.entrypoints.openai.reasoning_controls import CHAT_REASONING_RULES
from atom.entrypoints.openai.serving_chat import (
    normalize_chat_tools,
    resolve_chat_template_kwargs,
    validate_chat_request,
)
from atom.entrypoints.openai.tool_parser.registry import (
    template_keeps_tool_call_arguments,
)
from atom.model_engine.llm_engine import load_tokenizer

TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Reads a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Read README.md."},
]
CHATS = {
    "plain": {},
    "effort_low": {"reasoning_effort": "low"},
    "effort_medium": {"reasoning_effort": "medium"},
    "effort_high": {"reasoning_effort": "high"},
    "effort_max": {"reasoning_effort": "max"},
    "effort_minimal": {"reasoning_effort": "minimal"},
    "effort_none": {"reasoning_effort": "none"},
    "kwargs_thinking_off": {"chat_template_kwargs": {"enable_thinking": False}},
    "kwargs_thinking_on": {"chat_template_kwargs": {"enable_thinking": True}},
    "thinking_disabled": {"thinking": {"type": "disabled"}},
    "thinking_enabled": {"thinking": {"type": "enabled"}},
    "thinking_effort_30": {"thinking": {"type": "enabled", "effort": 30}},
    "tools_effort_high": {"tools": [TOOL], "reasoning_effort": "high"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="the checkpoint directory")
    parser.add_argument("--default-chat-template-kwargs", default="{}")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    defaults = json.loads(args.default_chat_template_kwargs)

    # As the server starts (api_server's _setup_tokenizer_and_template and
    # _setup_reasoning_and_tools).
    tokenizer = load_tokenizer(args.model, args.trust_remote_code)
    encoder = load_custom_message_encoder(args.model)
    toggle = resolve_reasoning_toggle(tokenizer, encoder)
    keeps = template_keeps_tool_call_arguments(
        render_probe_prompt(tokenizer, encoder, tools=True)
    )
    preserve = keeps or template_needs_tool_call_argument_strings(tokenizer, encoder)
    rules = getattr(encoder, "reasoning_rules", None) or CHAT_REASONING_RULES

    chats = []
    for name, fields in CHATS.items():
        body = {"model": args.model, "messages": MESSAGES, "max_tokens": 16, **fields}
        chat = {"name": name, "request": body}
        # As _chat_completions renders and the engine tokenizes a chat.
        try:
            request = ChatCompletionRequest(**body)
            request.tools = normalize_chat_tools(request.tools)
            validate_chat_request(request)
            messages = check_chat_images(request.get_messages())
            kwargs = resolve_chat_template_kwargs(
                request, defaults=defaults, toggle=toggle, rules=rules
            )
            prompt = apply_chat_template(
                tokenizer,
                encoder,
                messages,
                tools=request.tools,
                preserve_tool_call_arguments=preserve,
                **kwargs,
            )
            chat["prompt"] = prompt
            chat["ids"] = tokenizer.encode(prompt)
        except Exception as e:  # noqa: BLE001 - recorded as ATOM's rejection
            chat["error"] = f"{type(e).__name__}: {e}"
        chats.append(chat)

    rendering = chat_rendering(defaults, toggle, encoder, rules)
    json.dump(
        {"model": args.model, "chat_rendering": rendering, "chats": chats},
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
