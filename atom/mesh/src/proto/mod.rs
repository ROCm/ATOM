//! Protobuf types shared with the Python EngineCore processes.
//!
//! Both languages generate these types from `atom/proto/engine/*.proto`; this
//! module must not contain a copied Rust-only schema.

pub mod engine {
    include!(concat!(env!("OUT_DIR"), "/atom.engine.rs"));
}
