//! Prefix hashes of blocks, bit-compatible with SMG's
//! [`block.rs`](https://github.com/smg-project/smg/blob/2c4ca4141e02747f4d2e048ee72ebd5c4b64739f/crates/kv_index/src/block.rs#L114-L184)
//! for blocks without LoRA or salt, the only ones indexed.

use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::PrefixHash;

const XXH3_SEED: u64 = 1337;

// Tokens are hashed as their little-endian bytes, as SMG hashes them, borrowed in place.
const _: () = assert!(cfg!(target_endian = "little"));

/// Chains a block's content hash onto the prefix hash of the block before it; a first
/// block's prefix hash is its content hash.
pub(super) fn chain(parent: Option<PrefixHash>, tokens: &[u32]) -> PrefixHash {
    let content = xxh3_64_with_seed(bytemuck::cast_slice(tokens), XXH3_SEED);
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

/// Prefix hashes of the full blocks of `tokens`.
pub(super) fn prefix_hashes(tokens: &[u32], block_size: u32) -> Vec<PrefixHash> {
    let mut parent = None;
    tokens
        .chunks_exact(block_size as usize)
        .map(|block| {
            let hash = chain(parent, block);
            parent = Some(hash);
            hash
        })
        .collect()
}
