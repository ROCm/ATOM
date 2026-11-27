//! The tiers a worker's engine takes a prompt's KV from, with `--kv-events`, nearest
//! first: its device cache and the store it reads.
//! What the router knows of a prompt at one worker is one `TieredHit`, what a token of
//! each tier counts for is one `TierWeights`, in the policy's config, and
//! `effective_cached` is the one score of the two that the policies route by. Another
//! tier is a field of each, and its term in the score.

/// What the router knows of a prompt's leading tokens at one worker, by the tier its
/// engine will take them from. Each tier holds the tokens past those of the tiers before
/// it, so together they are what the engine will report cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TieredHit {
    /// The tokens the worker's device cache holds.
    pub device: u32,
    /// The tokens its engine will load from the store it reads; None when that store was
    /// asked and did not answer.
    pub stored: Option<u32>,
}

impl TieredHit {
    /// What the worker's engine will report cached, every tier's tokens; None when the
    /// store it reads did not answer.
    pub(crate) fn cached(&self) -> Option<u32> {
        Some(self.device + self.stored?)
    }
}

/// What a token of each tier below a worker's device cache counts for in a policy's score,
/// against one its device cache holds, which counts 1. A tier that counts 0 is not asked.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TierWeights {
    /// A token the engine will load from the store it reads: `--kv-storage-weight`.
    pub stored: f64,
}

/// What `hit` is worth in tokens on the device: each tier's tokens at what `weights` counts
/// one of them for, a store that did not answer adding none. Over the prompt's tokens, it
/// is the match rate `cache_aware` holds against `cache_threshold`.
pub(crate) fn effective_cached(hit: &TieredHit, weights: &TierWeights) -> f64 {
    f64::from(hit.device) + weights.stored * f64::from(hit.stored.unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The score `cache_aware` wrote out by hand before the tiers had one, from what the
    /// index then answered: `device` tokens on the worker's device, and `cached` once its
    /// store's are loaded, None when the store does not answer.
    fn hand_written((device, cached): (u32, Option<u32>), store_weight: f64) -> f64 {
        let loaded = cached.map_or(0, |cached| cached.saturating_sub(device));
        f64::from(device) + store_weight * f64::from(loaded)
    }

    #[test]
    fn the_score_of_a_hit_is_the_one_cache_aware_wrote_out_by_hand() {
        let hits = [
            // On the device alone, loaded, both, and nothing at all.
            (60, Some(60)),
            (0, Some(90)),
            (20, Some(96)),
            (0, Some(0)),
            // A store that does not answer.
            (60, None),
            // A long prompt, and a sum of weighed tokens that an f64 rounds.
            (3000, Some(16_384)),
            (1, Some(11)),
        ];
        // The default of the flag, then others, 1 included.
        let weights = [0.0, 0.6, 1.0, 0.3, 0.7];
        assert_eq!(TierWeights::default(), TierWeights { stored: weights[0] });
        for stored in weights {
            let tier_weights = TierWeights { stored };
            for (device, cached) in hits {
                let hit = TieredHit {
                    device,
                    stored: cached.map(|cached| cached - device),
                };
                assert_eq!(hit.cached(), cached);
                // The same bits, not only the same within a rounding.
                let want = hand_written((device, cached), stored);
                let got = effective_cached(&hit, &tier_weights);
                assert_eq!(got.to_bits(), want.to_bits(), "{hit:?} at {tier_weights:?}");
            }
        }
        // 80 on the device, and 16 loaded at half each.
        let hit = TieredHit {
            device: 80,
            stored: Some(16),
        };
        assert_eq!(effective_cached(&hit, &TierWeights { stored: 0.5 }), 88.0);
    }
}
