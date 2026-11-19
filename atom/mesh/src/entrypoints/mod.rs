//! Entrypoints.
//!
//! This module is the HTTP face of ATOMesh for the token path. It accepts an
//! already-tokenized request and returns token events. Template rendering,
//! encode, and detokenize stay with the caller. Routing, P/D, and the gRPC
//! hop to ATOM live behind [`TokenBackend`].

mod backend;
pub mod http;

pub use backend::{MeshBackend, TokenBackend};
