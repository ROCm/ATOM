"""Exact CPU prompt tokenization using ATOM's deployed text-chat encoder."""

import copy

from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    load_custom_message_encoder,
    resolve_reasoning_toggle,
)
from atom.entrypoints.openai.images import check_chat_images
from atom.entrypoints.openai.protocol import ChatCompletionRequest
from atom.entrypoints.openai.reasoning_controls import (
    CHAT_REASONING_RULES,
    apply_reasoning_controls,
)
from atom.entrypoints.openai.serving_chat import (
    normalize_chat_tools,
    resolve_thinking,
    validate_chat_request,
)


class ExactAtomChatEncoder:
    def __init__(self, model, tokenizer, *, defaults=None):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = load_custom_message_encoder(model)
        if self.encoder is None:
            raise ValueError(
                "The deployed custom message encoder is required; refusing HF fallback"
            )
        self.defaults = copy.deepcopy(defaults or {})
        self.toggle = resolve_reasoning_toggle(tokenizer, self.encoder)
        self.rules = self.encoder.reasoning_rules or CHAT_REASONING_RULES
        self.preserve_arguments = False  # V4.1 text/tool encoding.

    def encode_body(self, body):
        request = ChatCompletionRequest.model_validate(body)
        request.tools = normalize_chat_tools(request.tools)
        validate_chat_request(request)
        messages = check_chat_images(request.get_messages())
        # This ingress supports text/tool history only. Never fetch images or
        # claim a text tokenizer reproduces a multimodal IOProcessor.
        for message in messages:
            if isinstance(message.content, list) and any(
                (
                    part.get("type")
                    if isinstance(part, dict)
                    else getattr(part, "type", None)
                )
                not in (None, "text")
                for part in message.content
            ):
                raise ValueError("Exact ingress supports text/tool history only")
        supplied = request.get_prompt_token_ids()
        if supplied is not None:
            return list(supplied)
        kwargs = copy.deepcopy(self.defaults)
        if request.chat_template_kwargs:
            kwargs.update(request.chat_template_kwargs)
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        if isinstance(request.tool_choice, str):
            kwargs["tool_choice"] = request.tool_choice
        kwargs = apply_reasoning_controls(
            kwargs,
            self.toggle,
            resolve_thinking(request, rules=self.rules),
            rules=self.rules,
        )
        prompt = apply_chat_template(
            self.tokenizer,
            self.encoder,
            messages,
            tools=request.tools,
            preserve_tool_call_arguments=self.preserve_arguments,
            **kwargs,
        )
        # Same IOProcessor default encode path as actual ATOM P; keep BOS/specials.
        return list(self.tokenizer.encode(prompt))

    def prepare_body(self, body):
        payload = copy.deepcopy(body)
        payload["prompt_token_ids"] = self.encode_body(body)
        return payload
