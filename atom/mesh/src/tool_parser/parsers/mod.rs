/// Parser implementations for different model formats
///
/// This module contains concrete parser implementations for various model-specific
/// tool/function call formats.
// Shared utilities
pub mod common;

// Model parser families
pub mod deepseekv4;
pub mod glm;
pub mod json;
pub mod kimi;
pub mod minimax;
pub mod qwen;

// Compatibility module for existing parser internals.
pub(crate) mod helpers {
    pub(crate) use super::common::*;
}

// Re-export parser types for convenience
pub(crate) use common::PassthroughParser;
pub use deepseekv4::DsmlParser;
pub use glm::Glm4MoeParser;
pub use json::JsonParser;
pub use kimi::{KimiK2Parser, KimiK3Parser};
pub use minimax::MiniMaxParser;
pub use qwen::{QwenCoderParser, QwenParser};
