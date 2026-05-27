"""Python-owned ATOM standalone serving logic.

This adapter keeps OpenAI-compatible request semantics in Python so the Rust
standalone router only needs to bridge requests and responses.
"""

from __future__ import annotations

import numbers
import uuid
from typing import Any

from atom.entrypoints.openai.api_server import _build_sampling_params, _coerce_n
from atom.entrypoints.openai.chat_encoders import (
    apply_chat_template,
    load_custom_message_encoder,
)
from atom.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from atom.entrypoints.openai.serving_chat import (
    build_chat_response,
    build_chat_response_multi,
)
from atom.entrypoints.openai.serving_completion import (
    build_completion_response,
    build_completion_response_multi,
)


class AtomStandaloneService:
    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        model_name: str,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.custom_message_encoder = load_custom_message_encoder(model_name)

    def chat_completions(self, request_data: dict[str, Any]) -> dict[str, Any]:
        request_data = self._normalize_chat_request(request_data)
        request = ChatCompletionRequest.model_validate(request_data)

        if request.stream:
            raise NotImplementedError(
                "Streaming chat completions are not implemented for ATOM standalone yet"
            )

        messages = request.get_messages()
        template_kwargs = dict(self.default_chat_template_kwargs)
        if request.chat_template_kwargs:
            template_kwargs.update(request.chat_template_kwargs)

        prompt = apply_chat_template(
            self.tokenizer,
            self.custom_message_encoder,
            [msg.to_template_dict() for msg in messages],
            tools=request.tools,
            **template_kwargs,
        )

        effective_n = _coerce_n(request.n, request.temperature)
        outputs = self._generate(prompt, request, effective_n)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        if effective_n > 1:
            response = build_chat_response_multi(request_id, self.model_name, outputs)
        else:
            response = build_chat_response(
                request_id, self.model_name, outputs[0]["text"], outputs[0]
            )
        return self._json_safe(response.model_dump(exclude_none=True))

    def completions(self, request_data: dict[str, Any]) -> dict[str, Any]:
        request = CompletionRequest.model_validate(request_data)

        if request.stream:
            raise NotImplementedError(
                "Streaming completions are not implemented for ATOM standalone yet"
            )

        effective_n = _coerce_n(request.n, request.temperature)
        outputs = self._generate(request.prompt, request, effective_n)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        if effective_n > 1:
            response = build_completion_response_multi(
                request_id, self.model_name, outputs
            )
        else:
            response = build_completion_response(
                request_id, self.model_name, outputs[0]
            )
        return self._json_safe(response.model_dump(exclude_none=True))

    def close(self) -> None:
        if hasattr(self.engine, "close"):
            self.engine.close()

    @staticmethod
    def _normalize_chat_request(request_data: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(request_data)
        if (
            normalized.get("max_tokens") is None
            and normalized.get("max_completion_tokens") is not None
        ):
            normalized["max_tokens"] = normalized["max_completion_tokens"]
        return normalized

    def _generate(
        self,
        prompt: str,
        request: ChatCompletionRequest | CompletionRequest,
        effective_n: int,
    ) -> list[dict[str, Any]]:
        sampling_params = _build_sampling_params(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
            top_k=request.top_k,
            top_p=request.top_p,
            n=1,
        )
        outputs = self.engine.generate([prompt] * effective_n, sampling_params)
        if not outputs:
            raise RuntimeError("No output generated")
        return [self._normalize_output(output) for output in outputs]

    @staticmethod
    def _normalize_output(output: Any) -> dict[str, Any]:
        if isinstance(output, str):
            return {
                "text": output,
                "finish_reason": None,
                "num_tokens_input": 0,
                "num_tokens_output": 0,
                "ttft": 0.0,
                "tpot": 0.0,
                "latency": 0.0,
            }
        return AtomStandaloneService._json_safe(dict(output))

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): AtomStandaloneService._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [AtomStandaloneService._json_safe(item) for item in value]
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            return float(value)
        if hasattr(value, "item"):
            return AtomStandaloneService._json_safe(value.item())
        return value
