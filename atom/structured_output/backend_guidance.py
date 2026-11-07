# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Strict JSON Schema compilation: unsupported constraints must never disappear."""

import json

from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper
from llguidance.hf import from_tokenizer
from llguidance.torch import fill_next_token_bitmask

from .dsml_grammar import dsml_grammar
from .schema import check_schema_safety, expansion_budget, normalize_schema


def _prepare_schema(value, budget):
    check_schema_safety(value)
    return normalize_schema(value, budget)


class _VisibleAddedTokenAdapter:
    """Keep HF's visible added tokens as literal bytes in the grammar trie."""

    def __init__(self, tokenizer, source, vocab_size, visible):
        self.source = source
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokens = [source.decode_bytes([i]) for i in range(vocab_size)]
        self.special_token_ids = [
            i
            for i in range(vocab_size)
            if source.is_special_token(i) and i not in visible
        ]

    def __call__(self, value):
        return self.source.tokenize_bytes(value)


def grammar_tokenizer(tokenizer, vocab_size, stop_ids):
    source = from_tokenizer(tokenizer, n_vocab=vocab_size, eos_token=stop_ids)
    # llguidance 1.9.1 classifies HF added tokens as grammar specials even
    # when HF marks them special=False. Its mask can then allow <think> as
    # literal JSON text, but consume_token sees a special-token sentinel and
    # rejects it. Preserve the model IDs/bytes and HF's visible-token contract.
    visible = {
        i
        for i, token in tokenizer.backend_tokenizer.get_added_tokens_decoder().items()
        if i < vocab_size and not token.special and source.is_special_token(i)
    }
    if not visible:
        return source
    adapter = _VisibleAddedTokenAdapter(tokenizer, source, vocab_size, visible)
    return LLTokenizer(
        TokenizerWrapper(adapter), n_vocab=vocab_size, eos_token=stop_ids
    )


class GrammarMatcher:
    def __init__(self, compiled):
        self.matcher = compiled.deep_copy()

    def accept_token(self, token):
        return self.matcher.consume_token(int(token))

    def rollback(self, count):
        self.matcher.rollback(count)
        if self.matcher.is_error():
            raise ValueError(self.matcher.get_error())

    def is_terminated(self):
        return self.matcher.is_stopped()

    def fill_next_token_bitmask(self, mask):
        fill_next_token_bitmask(self.matcher, mask)
        if self.matcher.is_error():
            raise ValueError(self.matcher.get_error())


class GrammarCompiler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compile_json_schema(self, schema, tool_format=None, reasoning_end=""):
        """Compile a JSON Schema answer, DSML tool calls, or either of them.

        ``schema`` is ``None`` when only strict tools constrain the reply.
        The answer and every strict parameter are normalized from one
        expansion budget, as a single schema's siblings are.
        """
        budget = expansion_budget()
        value = None
        if schema is not None:
            value = _prepare_schema(json.loads(schema), budget)
            if isinstance(value, dict):
                # This is an API JSON Schema, not a backend configuration object.
                # In particular, clients must not enable lenient/coerce_one_of or
                # replace JSON punctuation through an x-guidance annotation.
                value.pop("x-guidance", None)
        if tool_format:
            # Only the response text is constrained by response_format; a
            # function call uses the parser's native envelope, its arguments
            # constrained only for strict tools.
            lark = dsml_grammar(
                json.loads(tool_format),
                value,
                lambda parameter: _prepare_schema(parameter, budget),
            )
            grammar = {"grammars": [{"lark_grammar": lark}]}
        else:
            grammar = LLMatcher.grammar_from_json_schema(
                value, overrides={"lenient": False, "coerce_one_of": False}
            )
        if reasoning_end:
            q = json.dumps
            if not tool_format:
                lark = f"start: answer\nanswer: %json {q(value)}\n"
            lark = lark.replace("start: ", "start: reasoning response\nresponse: ", 1)
            lark += f"reasoning[suffix={q(reasoning_end)}]: /(.|\\n)*/\n"
            grammar = {"grammars": [{"lark_grammar": lark}]}
        matcher = LLMatcher(self.tokenizer, grammar, log_level=0)
        if matcher.is_error():
            raise ValueError(matcher.get_error())
        warnings = matcher.get_grammar_warnings()
        if warnings:
            raise ValueError("; ".join(warnings))
        return matcher
