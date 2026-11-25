//! Reasoning parsers the `reasoning-parser` crate does not carry, registered
//! into its factory beside the crate's own.

mod kimi_k3;

use crate::reasoning_parser::ParserFactory;

/// Add this crate's parsers to `factory`, by the names `--reasoning-parser`
/// takes.
pub fn register(factory: &ParserFactory) {
    factory
        .registry()
        .register_parser("kimi_k3", || Box::new(kimi_k3::KimiK3Parser::default()));
}
