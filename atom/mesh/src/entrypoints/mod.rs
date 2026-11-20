//! Entrypoints: the HTTP front of ATOMesh.
//!
//! Entrypoints owns the HTTP server, middleware, OpenAI handlers, prepare and
//! render. Downstream code is reached only through [`backend::Backend`]: a
//! token request in, a token stream out. [`mesh_backend::MeshBackend`] wraps
//! Mesh's router token entry. [`standalone::StandaloneBackend`] writes the
//! token ids into an LLMEngine. Mesh no longer owns the HTTP server.

pub mod backend;
pub mod http_error;
pub mod mesh_backend;
pub mod middleware;
pub mod openai;
pub mod prepare;
pub mod render;
pub mod server;
pub mod standalone;
