//! Transport-neutral request preparation: HTTP request → (GenerationPayload, ResponseContext).
//!
//! Module contents are added incrementally by Parts A–C of the gRPC refactor.

pub(crate) mod chat_template;
pub mod generation_payload;
pub(crate) mod parser_factory_lookup;
pub(crate) mod stop_sequence_decoder;
pub(crate) mod tool_constraints;

#[cfg(any())]
mod tests;
