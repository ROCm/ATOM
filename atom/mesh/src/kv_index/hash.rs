//! Router-side block hashes, bit-compatible with SMG's
//! [`block.rs`](https://github.com/smg-project/smg/blob/2c4ca4141e02747f4d2e048ee72ebd5c4b64739f/crates/kv_index/src/block.rs#L114-L184)
//! and [`salt.rs`](https://github.com/smg-project/smg/blob/2c4ca4141e02747f4d2e048ee72ebd5c4b64739f/crates/kv_index/src/salt.rs#L15-L25),
//! except that only a sequence's first block is seeded with its namespace.

use std::sync::Arc;

use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::PrefixHash;
use crate::kv_events::Namespace;

pub const XXH3_SEED: u64 = 1337;

/// Seed of a sequence's first block. Empty names count as absent.
pub fn namespace_seed(ns: &Namespace) -> u64 {
    let part = |name: &Option<Arc<str>>, seed| {
        name.as_deref()
            .filter(|name| !name.is_empty())
            .map_or(0, |name| xxh3_64_with_seed(name.as_bytes(), seed))
    };
    XXH3_SEED
        .wrapping_add(part(&ns.lora, 0))
        .wrapping_add(part(&ns.salt, 1))
}

/// Hashes one block's tokens with `seed` and chains the result onto the prefix hash of the
/// block before it; a first block's prefix hash is its content hash.
pub fn chain(
    parent: Option<PrefixHash>,
    tokens: &[u32],
    seed: u64,
    buf: &mut Vec<u8>,
) -> PrefixHash {
    let bytes: &[u8] = if cfg!(target_endian = "little") {
        bytemuck::cast_slice(tokens)
    } else {
        buf.clear();
        buf.extend(tokens.iter().flat_map(|token| token.to_le_bytes()));
        buf
    };
    let content = xxh3_64_with_seed(bytes, seed);
    PrefixHash(match parent {
        None => content,
        Some(PrefixHash(parent)) => {
            let mut link = [0; 16];
            link[..8].copy_from_slice(&parent.to_le_bytes());
            link[8..].copy_from_slice(&content.to_le_bytes());
            xxh3_64_with_seed(&link, XXH3_SEED)
        }
    })
}

/// Prefix hashes of the full blocks of a request; empty for an `opaque` namespace, whose
/// blocks are never indexed.
pub fn prefix_hashes(tokens: &[u32], block_size: usize, ns: &Namespace) -> Vec<PrefixHash> {
    if block_size == 0 || ns.opaque {
        return Vec::new();
    }
    let mut buf = Vec::with_capacity(block_size * 4);
    let mut parent = None;
    tokens
        .chunks_exact(block_size)
        .map(|block| {
            let seed = parent.map_or_else(|| namespace_seed(ns), |_| XXH3_SEED);
            let hash = chain(parent, block, seed, &mut buf);
            parent = Some(hash);
            hash
        })
        .collect()
}
