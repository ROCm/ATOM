/// Tool parser module for handling function/tool calls in model outputs
///
/// This module provides infrastructure for parsing tool calls from various model formats.
///
/// Source provenance and local adaptations are documented in `UPSTREAM.md`.
// Shared infrastructure
pub mod core;
pub mod partial_json;
pub mod registry;

// Parser implementations
pub mod parsers;

#[cfg(test)]
mod tests;

// Compatibility modules preserve the established internal paths while all
// implementation now lives in core.rs.
pub mod errors {
    pub use super::core::{ParserError, ParserResult};
}

pub mod traits {
    pub use super::core::{PartialJsonParser, TokenToolParser, ToolParser};
}

pub mod types {
    pub use super::core::{
        FunctionCall, PartialToolCall, StreamingParseResult, ToolCall, ToolCallItem,
    };
}

// Re-export types used outside this module.
pub use core::{FunctionCall, PartialToolCall, StreamingParseResult, ToolCall, ToolParser};
pub use parsers::{
    DsmlParser, Glm4MoeParser, JsonParser, KimiK2Parser, KimiK3Parser, MiniMaxParser,
    QwenCoderParser, QwenParser,
};
pub use registry::{ParserFactory, PooledParser};
