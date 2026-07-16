//! Transport-neutral engine boundary types: TokenChunk, TokenHandle, EngineError.

pub mod engine_error;
pub mod test_support;
pub mod token_chunk;
#[allow(clippy::module_inception)]
pub mod token_handle;

#[cfg(test)]
mod tests;
