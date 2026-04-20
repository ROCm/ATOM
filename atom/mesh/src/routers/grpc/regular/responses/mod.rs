//! Regular gRPC Router `/v1/responses` endpoint implementation
//!
//! This module handles all responses-specific logic for the regular pipeline.
//!
//! ## Architecture
//!
//! - `handlers` - Entry points: route_responses (thin dispatcher)
//! - `non_streaming` - Non-streaming execution
//! - `streaming` - Streaming execution
//! - `common` - Shared helpers: tool extraction, conversation history loading
//! - `conversions` - Request/response conversion between Responses and Chat formats

mod common;
mod conversions;
mod handlers;
mod non_streaming;
mod streaming;

// Public exports
pub(crate) use handlers::route_responses;
