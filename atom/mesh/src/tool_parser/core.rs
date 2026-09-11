//! Shared tool-parser types, errors, and parsing traits.

use thiserror::Error;

/// Result type for tool parser operations
pub type ParserResult<T> = Result<T, ParserError>;

/// Errors that can occur during tool parsing
#[derive(Debug, Error)]
pub enum ParserError {
    #[error("Parsing failed: {0}")]
    ParsingFailed(String),

    #[error("Model not supported: {0}")]
    ModelNotSupported(String),

    #[error("Parse depth exceeded: max {0}")]
    DepthExceeded(usize),

    #[error("Invalid JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),

    #[error("Incomplete tool call")]
    Incomplete,

    #[error("Invalid tool name: {0}")]
    InvalidToolName(String),

    #[error("Token not found: {0}")]
    TokenNotFound(String),
}

use serde::{Deserialize, Serialize};

/// Parsed tool call from model output
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Function call details
    pub function: FunctionCall,
}

/// Function call within a tool call
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,
    /// Arguments as JSON string
    pub arguments: String,
}

/// Simple partial tool call for streaming
#[derive(Debug, Clone)]
pub struct PartialToolCall {
    /// Tool name (if parsed)
    pub name: Option<String>,
    /// Buffer for accumulating arguments
    pub arguments_buffer: String,
    /// Start position in the input buffer
    pub start_position: usize,
    /// Whether the name has been sent (for streaming)
    pub name_sent: bool,
    /// Arguments already streamed
    pub streamed_args: String,
}

/// Result of streaming parse operation (matches Python StreamingParseResult)
#[derive(Debug, Clone, Default)]
pub struct StreamingParseResult {
    /// Normal text that's not part of tool calls
    pub normal_text: String,
    /// Tool call items parsed from the chunk
    pub calls: Vec<ToolCallItem>,
}

/// Simple encapsulation of parsed tool call for streaming (matches Python ToolCallItem)
#[derive(Debug, Clone)]
pub struct ToolCallItem {
    /// Tool index in the array
    pub tool_index: usize,
    /// Tool name (only present on first chunk)
    pub name: Option<String>,
    /// Incremental JSON arguments
    pub parameters: String,
}

use async_trait::async_trait;
use openai_protocol::common::Tool;

/// Core trait for all tool parsers
#[async_trait]
pub trait ToolParser: Send + Sync {
    /// Parse complete tool calls from final output
    /// Returns (remaining_normal_text, tool_calls) tuple
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)>;

    /// Parse complete tool calls with access to the request's tool schemas.
    ///
    /// XML-style parsers can override this to preserve declared string types
    /// instead of inferring values from their textual representation.
    async fn parse_complete_with_tools(
        &self,
        output: &str,
        _tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete(output).await
    }

    /// Parse tool calls from model output (streaming)
    /// Parsers now maintain internal state, so self is mutable
    ///
    /// # Arguments
    /// * `chunk` - New text chunk from model output
    /// * `tools` - List of available tools for validation
    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult>;

    /// Check if text contains tool calls in this parser's format
    fn has_tool_markers(&self, text: &str) -> bool;

    /// Optionally expose a token-aware parser implementation.
    /// Default returns `None`, meaning the parser only supports text input.
    fn as_token_parser(&self) -> Option<&dyn TokenToolParser> {
        None
    }

    /// Get unstreamed tool call arguments
    /// Returns tool call items for arguments that have been parsed but not yet streamed
    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        None
    }

    /// Reset the parser state for reuse across requests.
    /// This should clear all buffers and reset state to initial values.
    fn reset(&mut self) {
        // Default no-op implementation
    }
}

/// Trait for partial JSON parsing
pub trait PartialJsonParser: Send + Sync {
    /// Parse potentially incomplete JSON
    fn parse(&self, input: &str) -> ParserResult<(serde_json::Value, usize)>;

    /// Check if JSON is complete
    fn is_complete(&self, input: &str) -> bool;

    /// Get the maximum parsing depth
    fn max_depth(&self) -> usize;
}

#[async_trait]
pub trait TokenToolParser: ToolParser {
    /// Parse complete tool calls when provided with raw token IDs.
    async fn parse_complete_tokens(&self, tokens: &[u32]) -> ParserResult<(String, Vec<ToolCall>)>;

    /// Streaming parser entrypoint for token chunks.
    /// Parsers maintain internal state, so self is mutable
    async fn parse_incremental_tokens(
        &mut self,
        tokens: &[u32],
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult>;
}
