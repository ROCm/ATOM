//! Transport-neutral response rendering: Stream<TokenChunk> + ResponseContext → axum::Response.
//!
//! Module contents are added incrementally by Parts A and E of the gRPC refactor.

#[cfg(test)]
mod tests;
