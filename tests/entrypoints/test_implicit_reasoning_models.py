# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Models that begin inside the reasoning channel with no marker anywhere.

DeepSeek-R1 emits `</think>` but neither its prompt nor its output carries
`<think>`. Nothing in a single response says so — its first token is already
reasoning and reads like an answer — so the fact has to be known before the
response starts, and the chat template is what knows it.

vLLM expresses the same fact by registering `DeepSeekR1ReasoningParser`, whose
only job is to override the streaming branch so a stream with no start token
counts as reasoning until the end marker. Registering a class per family and
reading the template are two spellings of one decision; this is the second.
"""

from __future__ import annotations

import pytest

from atom.entrypoints.openai.reasoning import (
    ReasoningFilter,
    separate_reasoning,
    template_opens_reasoning_implicitly,
)

# Shapes taken from the real templates on this box, reduced to what decides.
R1 = "...{{ content.split('</think>')|last }}...<｜Assistant｜>"
QWEN = "...<think>\n{{ reasoning }}\n</think>...<|im_start|>assistant\n<think>\n"
MINIMAX = "...[e~[\\n]~b]ai\\n..."


class TestTheRule:
    def test_a_template_that_closes_what_it_never_opens(self):
        assert template_opens_reasoning_implicitly(R1)

    def test_a_template_that_opens_its_own_does_not_count(self):
        """Qwen mentions both, so its model emits the opener itself.

        Treating it as implicit would put the model's own `<think>` inside the
        reasoning text and start every plain answer in the wrong channel.
        """
        assert not template_opens_reasoning_implicitly(QWEN)

    @pytest.mark.parametrize("template", [MINIMAX, "", "plain assistant template"])
    def test_a_template_with_no_reasoning_channel_does_not_count(self, template):
        assert not template_opens_reasoning_implicitly(template)


class TestWhatItBuys:
    RAW = "Let me work it out.</think>\n\n2 + 2 = 4."

    def test_unseeded_and_unflagged_the_end_marker_is_just_text(self):
        """The default, unchanged: nothing opened a channel, so nothing closed one."""
        reasoning, content = separate_reasoning(self.RAW)
        assert reasoning is None and content == self.RAW

    def test_flagged_the_reasoning_is_recovered(self):
        reasoning, content = separate_reasoning(self.RAW, starts_thinking=True)
        assert reasoning == "Let me work it out."
        assert content == "\n\n2 + 2 = 4."

    def test_streaming_and_non_streaming_agree_once_flagged(self):
        """The flag is one value read by both paths, so they cannot diverge."""
        reasoning, content = separate_reasoning(self.RAW, starts_thinking=True)

        rf = ReasoningFilter(starts_thinking=True)
        segments = []
        for i in range(0, len(self.RAW), 4):
            segments += rf.process(self.RAW[i : i + 4])
        segments += rf.flush()
        streamed_r = "".join(s for f, s in segments if f == "reasoning_content")
        streamed_c = "".join(s for f, s in segments if f == "content")

        # Compared byte-for-byte. This used to `.strip()` the streamed side
        # to make the two agree, which is what a divergence looks like when a
        # test is written around it: `content` was `"2 + 2 = 4."` here and
        # `"\n\n2 + 2 = 4."` on the wire.
        assert (reasoning or "") == streamed_r
        assert content == streamed_c


class TestAskingForNoThinkingOnlyCountsWhereItCanBeHonoured:
    """`thinking: disabled` reaches the model through the chat template's own
    switch. A template with no such switch cannot carry it.

    DeepSeek-R1 is that model: it begins inside the reasoning channel with no
    marker, and `resolve_reasoning_toggle` answers `None` for it. Asking it not
    to think puts nothing in the prompt, so it reasons exactly as always --
    and believing the request anyway stopped the channel being separated, so
    the client got the chain of thought and a literal `</think>` inside
    `content`. Reasoning that was asked not to happen and happened anyway is
    still reasoning; `anthropic_drop_reasoning` exists to withhold it, and it
    can only withhold what was separated.
    """

    ANSWER = "The user wants the capital.</think>Paris."

    SWITCH = ("enable_thinking", False, True)

    @staticmethod
    def _channel(toggle, template_kwargs):
        """The channel this request would get, on a model that begins inside
        one implicitly.

        `template_kwargs` is what actually went into the render -- the server
        defaults merged with the client's own, with the request's `thinking`
        field written in by name on top. Passing the request field instead
        was the defect: an operator's `--default-chat-template-kwargs` never
        reached this decision.
        """
        import atom.entrypoints.openai.api_server as api
        from atom.entrypoints.openai.reasoning_dialects import resolve_dialect

        before = (
            api.reasoning_dialect,
            api.model_starts_in_reasoning,
            api.reasoning_toggle,
        )
        try:
            api.reasoning_dialect, _ = resolve_dialect("<think></think>")
            api.model_starts_in_reasoning = True
            api.reasoning_toggle = toggle
            return api.reasoning_channel(False, template_kwargs=template_kwargs)
        finally:
            (
                api.reasoning_dialect,
                api.model_starts_in_reasoning,
                api.reasoning_toggle,
            ) = before

    SEPARATED = ("The user wants the capital.", "Paris.")

    def test_a_model_with_no_switch_is_separated_anyway(self):
        """No toggle means nothing about thinking reached the prompt, so the
        model reasons as always however the request was written. True by
        construction now: with no toggle there is no kwarg name to look for."""
        channel = self._channel(None, {"enable_thinking": False})
        assert channel.split(self.ANSWER) == self.SEPARATED

    def test_a_model_with_a_switch_takes_the_render_at_its_word(self):
        channel = self._channel(self.SWITCH, {"enable_thinking": False})
        assert channel.split(self.ANSWER) == (None, self.ANSWER)

    @pytest.mark.parametrize(
        "kwargs",
        [None, {}, {"enable_thinking": True}],
        ids=["none", "empty", "switched-on"],
    )
    def test_anything_but_the_off_value_separates(self, kwargs):
        """Absent means unstated, and unstated leaves the model's own default
        alone -- at this layer as at the prompt layer."""
        channel = self._channel(self.SWITCH, kwargs)
        assert channel.split(self.ANSWER) == self.SEPARATED

    def test_a_kwarg_this_template_does_not_read_changes_nothing(self):
        """The name has to be the resolved toggle's. A Qwen-spelled switch
        sent to a Kimi-K3 template is a kwarg the render ignores, so the model
        thinks anyway -- and believing it would stop the channel being
        separated while the chain of thought went on being produced."""
        channel = self._channel(self.SWITCH, {"thinking": False})
        assert channel.split(self.ANSWER) == self.SEPARATED

    def test_an_operator_default_reaches_the_decision(self):
        """The defect. `--default-chat-template-kwargs '{"enable_thinking":
        false}'` renders a prompt that does not open the channel, but
        `thinking_off` was read from the request's own `thinking` field, which
        is absent -- so the model-level fact was OR-ed back in and the whole
        answer came back as `reasoning_content` with `content` empty."""
        merged = {"enable_thinking": False}  # a server default, no request field
        channel = self._channel(self.SWITCH, merged)
        assert channel.split(self.ANSWER) == (None, self.ANSWER)
