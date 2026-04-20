//! Shared response functionality used by regular implementations

pub(crate) mod context;
pub(crate) mod handlers;
pub(crate) mod streaming;
pub(crate) mod utils;

// Re-export commonly used items
pub(crate) use context::ResponsesContext;
pub(crate) use streaming::build_sse_response;
pub(crate) use utils::persist_response_if_needed;
