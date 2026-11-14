//! ATOM's Mooncake Store keys, `{namespace}/w{rank}of{world}/{digest}`, where the
//! digest of chunk `i` chains blake2b over the prompt's full chunks up to `i`, as
//! ATOM's `mooncake_store/keys.py` builds them.

use blake2b_simd::Params;

const DIGEST_BYTES: usize = 16;
const CHAIN_PERSON: &[u8] = b"atom-kv-chain-v1";
/// Present in a pool a late write may have corrupted; ATOM loads nothing from it.
pub const TAINT_KEY: &str = "atomkv/store-tainted";

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

/// The `chain_check` ATOM reports: the hex digest of tokens `0..chunk_tokens`.
pub fn chain_check(chunk_tokens: u32) -> String {
    let tokens: Vec<u32> = (0..chunk_tokens).collect();
    let mut check = String::new();
    for digest in chunk_chain(&tokens, chunk_tokens as usize) {
        push_hex(&mut check, &digest);
    }
    check
}

/// The key of each chunk of `chain` at rank 0 of `world`. The master evicts a chunk's
/// objects of every rank together, so rank 0's tells whether the chunk is stored.
pub fn rank0_keys(namespace: &str, world: u32, chain: &[[u8; DIGEST_BYTES]]) -> Vec<String> {
    let prefix = format!("{namespace}/w0of{world}/");
    let key = |digest: &[u8; DIGEST_BYTES]| {
        let mut key = String::with_capacity(prefix.len() + 2 * DIGEST_BYTES);
        key.push_str(&prefix);
        push_hex(&mut key, digest);
        key
    };
    chain.iter().map(key).collect()
}

fn push_hex(to: &mut String, bytes: &[u8]) {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        to.push(DIGITS[usize::from(byte >> 4)].into());
        to.push(DIGITS[usize::from(byte & 15)].into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // From ATOM's keys.py: chunk_hash_chain(range(n), n).hex(), and the chain of
    // tokens (i * 7919) % 150000 for i in 0..53 in chunks of 16, and its chunk_keys at
    // rank 0 of 8.
    #[test]
    fn digests_are_atoms() {
        for (chunk_tokens, want) in [
            (16, "b4f719a070effaaae20d233b3341c77d"),
            (64, "a1aef9c4be290e35c90734297b3e6c8a"),
            (256, "368bbb8bdb74a66c980c2a9762fef7fa"),
        ] {
            assert_eq!(chain_check(chunk_tokens), want, "{chunk_tokens}");
        }
        assert_eq!(chain_check(0), "");
        let tokens: Vec<u32> = (0..53).map(|i| (i * 7919) % 150_000).collect();
        let namespace = "atomkv1-0123456789abcdef01234567";
        assert_eq!(
            rank0_keys(namespace, 8, &chunk_chain(&tokens, 16)),
            [
                "48a73da894d8d0ef643123150b31991e",
                "a53c74ba1da5cce48218d8a3729dc7a1",
                "bdff5c9bb8a712526c9ad1e28caa2a8e",
            ]
            .map(|digest| format!("{namespace}/w0of8/{digest}"))
        );
    }

    /// `cargo test --release --lib kv_storage::keys::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_keys_of_100k_tokens() {
        let tokens: Vec<u32> = (0..100_000).map(|i| (i * 7919) % 150_000).collect();
        let samples = (0..2000).map(|_| {
            let started = std::time::Instant::now();
            let keys = rank0_keys(
                "atomkv1-0123456789abcdef01234567",
                8,
                &chunk_chain(&tokens, 256),
            );
            assert_eq!(keys.len(), 390);
            started.elapsed()
        });
        let mut samples: Vec<_> = samples.collect();
        samples.sort();
        let at = |q: f64| samples[(samples.len() as f64 * q) as usize];
        println!(
            "keys of 100k tokens in chunks of 256: p50 {:?}, p99 {:?}",
            at(0.5),
            at(0.99)
        );
    }
}
