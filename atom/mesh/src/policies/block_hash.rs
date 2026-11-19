//! Block hashes of a token-id sequence.
//!
//! The token path never falls back to prompt text. Policies that key a prefix
//! tree by text use this string instead, so a shared token prefix still hits
//! the same worker.

/// Tokens per block. Matches the routing key width used when the engine block
/// size is not on the request.
pub const TOKEN_BLOCK_SIZE: usize = 16;

/// Stable, order-preserving key. Equal token prefixes share an equal string prefix.
pub fn token_block_key(tokens: &[u32]) -> String {
    let mut out = String::new();
    for chunk in tokens.chunks(TOKEN_BLOCK_SIZE) {
        let mut bytes = Vec::with_capacity(chunk.len() * 4);
        for token in chunk {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        if !out.is_empty() {
            out.push('/');
        }
        out.push_str(&format!("{:016x}", xxhash_rust::xxh3::xxh3_64(&bytes)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_prefix_blocks_share_a_key_prefix() {
        let mut first = vec![1u32; TOKEN_BLOCK_SIZE];
        first.extend_from_slice(&[9, 9, 9]);
        let mut second = vec![1u32; TOKEN_BLOCK_SIZE];
        second.extend_from_slice(&[4, 4]);
        let a = token_block_key(&first);
        let b = token_block_key(&second);
        let a_block = a.split('/').next().unwrap();
        let b_block = b.split('/').next().unwrap();
        assert_eq!(a_block, b_block);
        assert_ne!(a, b);
    }

    #[test]
    fn empty_input_is_empty_key() {
        assert!(token_block_key(&[]).is_empty());
    }
}
