use crate::reasoning_parser::ParserFactory as ReasoningParserFactory;
use crate::tool_parser::ParserFactory as ToolParserFactory;

pub(crate) fn reasoning_parser_factory() -> ReasoningParserFactory {
    ReasoningParserFactory::new()
}

pub(crate) fn tool_parser_factory() -> ToolParserFactory {
    ToolParserFactory::new()
}
