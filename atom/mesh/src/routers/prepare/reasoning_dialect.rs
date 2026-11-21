//! What each of mesh's reasoning parsers reads, for what its trait does not
//! expose: its markers, and a whole reply past its `max_buffer_size`.

use super::parser_factory_lookup::create_reasoning_parser;
use crate::reasoning_parser::{
    BaseReasoningParser, ParseError, ParserConfig, ParserFactory, ParserResult, ReasoningParser,
};

/// `(model_type, opener, closer, starts in reasoning)` of every parser
/// reasoning-parser =1.0.0 registers, as it configures them. MiniMax's
/// one-shot parse prepends its opener, which is starting in reasoning.
const DIALECTS: &[(&str, &str, &str, bool)] = &[
    ("base", "<think>", "</think>", false),
    ("deepseek_r1", "<think>", "</think>", true),
    ("qwen3", "<think>", "</think>", false),
    ("qwen_thinking", "<think>", "</think>", true),
    ("kimi", "◁think▷", "◁/think▷", false),
    ("glm45", "<think>", "</think>", false),
    ("step3", "<think>", "</think>", true),
    ("minimax", "<think>", "</think>", true),
    ("passthrough", "", "", false),
];

fn dialect(model_type: &str) -> Option<&'static (&'static str, &'static str, &'static str, bool)> {
    DIALECTS.iter().find(|(name, ..)| *name == model_type)
}

/// The marker a reply read by `parser` opens reasoning with.
pub(crate) fn opener(parser: &dyn ReasoningParser) -> Option<&'static str> {
    dialect(parser.model_type())
        .map(|(_, opener, ..)| *opener)
        .filter(|opener| !opener.is_empty())
}

/// The Python server's `prompt_starts_in_reasoning`: `prompt`, trailing
/// whitespace aside, ends on the opener of the dialect its reply is read in
/// (a template whose generation prompt opens `<think>`, as GLM-5.x and
/// Qwen3.5 do), so the reply starts inside reasoning with no opener of its own.
pub(crate) fn prompt_opens_reasoning(
    prompt: &str,
    factory: Option<&ParserFactory>,
    configured: Option<&str>,
    model: &str,
) -> bool {
    let prompt = prompt.trim_end();
    let ends_on = |opener: &str| !opener.is_empty() && prompt.ends_with(opener);
    if !DIALECTS.iter().any(|(_, opener, ..)| ends_on(opener)) {
        return false;
    }
    factory
        .and_then(|factory| create_reasoning_parser(factory, configured, model))
        .and_then(|parser| opener(parser.as_ref()))
        .is_some_and(ends_on)
}

/// `parser.detect_and_parse_reasoning(text)` for a text of any length: past
/// the parser's `max_buffer_size` the same one-shot parse runs on its
/// dialect without the cap, so a long reply splits exactly as a short one.
pub(crate) fn parse_whole(
    parser: &mut dyn ReasoningParser,
    text: &str,
) -> Result<ParserResult, ParseError> {
    match parser.detect_and_parse_reasoning(text) {
        Err(ParseError::BufferOverflow(len)) => {
            let Some(&(_, opener, closer, initial_in_reasoning)) = dialect(parser.model_type())
            else {
                return Err(ParseError::BufferOverflow(len));
            };
            BaseReasoningParser::new(ParserConfig {
                think_start_token: opener.into(),
                think_end_token: closer.into(),
                stream_reasoning: true,
                max_buffer_size: usize::MAX,
                initial_in_reasoning,
            })
            .detect_and_parse_reasoning(text)
        }
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning_parser::ParserFactory;

    fn parser(name: &str) -> Box<dyn ReasoningParser> {
        let factory = ParserFactory::new();
        // A model no pattern names registers passthrough, its fallback.
        let _ = factory.get_pooled("no-such-model");
        factory.registry().create_parser(name).unwrap()
    }

    const PARSERS: [&str; 9] = [
        "base",
        "deepseek_r1",
        "qwen3",
        "qwen3_thinking",
        "kimi",
        "glm45",
        "step3",
        "minimax",
        "passthrough",
    ];

    /// Every parser the crate registers (its `ParserFactory::new`, which it
    /// cannot list) is in the table, under the model type it reports.
    #[test]
    fn every_registered_parser_has_a_dialect() {
        for name in PARSERS {
            assert!(dialect(parser(name).model_type()).is_some(), "{name}");
        }
    }

    /// Under the cap the uncapped parse is the parser's own, in the parser's
    /// own markers: openers before and after a closer, text before the
    /// opener, a closer only, whitespace.
    #[test]
    fn the_uncapped_parse_is_the_parsers_own() {
        let texts = [
            "plain answer",
            "<think>a b</think> answer",
            "<think>\n  reasoning \n</think>\n\n answer  ",
            "<think>unclosed reasoning  ",
            "preamble <think>r</think>answer",
            "<think>r</think>a <think>b",
            "r</think>a <think>b",
            "reasoning only</think>answer",
            "x < y and z > w",
        ];
        for name in PARSERS {
            let marker = |t: &str| match parser(name).model_type() {
                "kimi" => t
                    .replace("</think>", "◁/think▷")
                    .replace("<think>", "◁think▷"),
                _ => t.to_string(),
            };
            for text in texts.map(marker) {
                let mut capped = parser(name);
                let own = capped.detect_and_parse_reasoning(&text).unwrap();
                let &(_, opener, closer, initial) = dialect(capped.model_type()).unwrap();
                let uncapped = BaseReasoningParser::new(ParserConfig {
                    think_start_token: opener.into(),
                    think_end_token: closer.into(),
                    stream_reasoning: true,
                    max_buffer_size: usize::MAX,
                    initial_in_reasoning: initial,
                })
                .detect_and_parse_reasoning(&text)
                .unwrap();
                assert_eq!(
                    (uncapped.normal_text, uncapped.reasoning_text),
                    (own.normal_text, own.reasoning_text),
                    "{name}: {text:?}"
                );
            }
        }
    }

    /// Past the cap a reply splits where the parser splits it under the cap,
    /// whatever its markers and wherever they fall.
    #[test]
    fn a_reply_past_the_cap_splits_as_one_under_it() {
        let (open, close) = ("◁think▷", "◁/think▷");
        let reasoning = "wörd ".repeat(20_000);
        let text = format!("{open}{reasoning}{close} the answer");
        assert!(text.len() > 65_536);
        let result = parse_whole(parser("kimi").as_mut(), &text).unwrap();
        assert_eq!(result.normal_text, "the answer");
        // The one-shot parse trims what precedes the reasoning, not its end.
        assert!(result.reasoning_text == reasoning);

        let text = format!("preamble <think>{reasoning}</think>a <think>b");
        let result = parse_whole(parser("qwen3").as_mut(), &text).unwrap();
        assert_eq!(result.normal_text, "a b");
        assert!(result.reasoning_text == format!("preamble {reasoning}"));
    }

    #[test]
    fn a_prompt_opens_reasoning_in_its_replys_dialect() {
        let factory = ParserFactory::new();
        let opens = |prompt: &str, parser: &str| {
            prompt_opens_reasoning(prompt, Some(&factory), Some(parser), "m")
        };
        assert!(opens("assistant: <think>\n", "qwen3"));
        assert!(opens("assistant: ◁think▷", "kimi"));
        // Another dialect's opener is text to this one.
        assert!(!opens("assistant: <think>", "kimi"));
        assert!(!opens("assistant: ◁think▷", "qwen3"));
        assert!(!opens("assistant:", "qwen3"));
        assert!(!prompt_opens_reasoning("<think>", None, None, "m"));
    }

    #[test]
    fn each_dialect_opens_with_its_own_marker() {
        assert_eq!(opener(parser("kimi").as_ref()), Some("◁think▷"));
        assert_eq!(opener(parser("qwen3").as_ref()), Some("<think>"));
        assert_eq!(opener(parser("passthrough").as_ref()), None);
    }
}
