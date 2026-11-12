//! ATOM's chunk keys: `{namespace}/w{rank}of{world}/{digest}`, where digest `i` chains
//! blake2b over the prompt's full chunks up to chunk `i`, as ATOM's
//! `mooncake_store/keys.py` does.

use blake2b_simd::Params;

use super::{FailReason, KeyRule, KeyState, RankProbe, StorageLookup};

const DIGEST_BYTES: usize = 16;
const CHAIN_PERSON: &[u8] = b"atom-kv-chain-v1";
/// Present in a pool a late write may have corrupted; ATOM loads nothing from it.
pub const STORE_TAINT_KEY: &str = "atomkv/store-tainted";

/// The `AtomChunk` key rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomChunk {
    pub namespace: String,
    pub chunk_tokens: u32,
    pub world: u32,
    pub probe: RankProbe,
}

impl AtomChunk {
    /// Ranks probed per chunk. Mooncake evicts every rank's object of a chunk together.
    pub fn ranks(&self) -> u32 {
        match self.probe {
            RankProbe::First => 1,
            RankProbe::All => self.world,
        }
    }

    /// Keys of the probed ranks for every full chunk, chunk by chunk.
    pub fn keys(&self, tokens: &[u32]) -> Vec<String> {
        let prefixes: Vec<String> = (0..self.ranks())
            .map(|rank| format!("{}/w{rank}of{}/", self.namespace, self.world))
            .collect();
        let chain = chunk_chain(tokens, self.chunk_tokens as usize);
        let mut keys = Vec::with_capacity(chain.len() * prefixes.len());
        for digest in &chain {
            for prefix in &prefixes {
                let mut key = String::with_capacity(prefix.len() + 2 * DIGEST_BYTES);
                key.push_str(prefix);
                push_hex(&mut key, digest);
                keys.push(key);
            }
        }
        keys
    }

    /// Leading hit tokens from chunk 0, given the state of each key. A chunk is in memory
    /// only if every probed rank's object is, and missing if any is. An unknown chunk
    /// before the first missing one could change the result, so the lookup is then
    /// unavailable rather than short.
    pub fn fold(&self, states: &[KeyState]) -> StorageLookup {
        let (mut shared, mut disk) = (None, 0);
        for chunk in states.chunks(self.ranks().max(1) as usize) {
            let state = chunk.iter().copied().min_by_key(|state| state.level());
            match state.unwrap_or(KeyState::Absent) {
                KeyState::Absent => break,
                KeyState::Unknown => return StorageLookup::Unavailable(FailReason::Transport),
                KeyState::SsdOnly => {
                    shared.get_or_insert(disk);
                }
                KeyState::Memory => {}
            }
            disk += 1;
        }
        StorageLookup::Hit {
            shared: shared.unwrap_or(disk) * self.chunk_tokens,
            disk: disk * self.chunk_tokens,
        }
    }

    pub fn rule(&self) -> KeyRule {
        KeyRule::AtomChunk {
            namespace: self.namespace.clone(),
            chunk_tokens: self.chunk_tokens,
            world: self.world,
            probe: self.probe,
        }
    }
}

/// Digests of every full chunk of a text prompt; a trailing partial chunk is left out.
pub fn chunk_chain(tokens: &[u32], chunk_tokens: usize) -> Vec<[u8; DIGEST_BYTES]> {
    if chunk_tokens == 0 {
        return Vec::new();
    }
    let start = Params::new()
        .hash_length(DIGEST_BYTES)
        .personal(CHAIN_PERSON)
        .to_state();
    let mut bytes = vec![0; chunk_tokens * 4];
    let mut link = [0; DIGEST_BYTES];
    tokens
        .chunks_exact(chunk_tokens)
        .map(|chunk| {
            for (to, token) in bytes.chunks_exact_mut(4).zip(chunk) {
                to.copy_from_slice(&token.to_le_bytes());
            }
            let digest = start.clone().update(&link).update(&bytes).finalize();
            link.copy_from_slice(digest.as_bytes());
            link
        })
        .collect()
}

/// The `chain_check` ATOM reports: the digest of tokens `0..chunk_tokens`.
pub fn chain_check(chunk_tokens: u32) -> String {
    let tokens: Vec<u32> = (0..chunk_tokens).collect();
    let mut hex = String::with_capacity(2 * DIGEST_BYTES);
    for digest in chunk_chain(&tokens, chunk_tokens as usize) {
        push_hex(&mut hex, &digest);
    }
    hex
}

fn push_hex(out: &mut String, bytes: &[u8]) {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 15) as usize] as char);
    }
}
