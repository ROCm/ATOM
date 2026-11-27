//! The tiers a worker's engine takes a prompt's KV from, with `--kv-events`, nearest
//! first: its device cache, the device pools of its share group, and the store it reads.
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
    /// The tokens its engine will copy in from the device pools of its share group.
    pub shared: u32,
    /// The tokens its engine will load from the store it reads; None when that store was
    /// asked and did not answer.
    pub stored: Option<u32>,
}

impl TieredHit {
    /// What the worker's device cache will hold once its engine has copied in what its
    /// share group holds: what the store it reads adds to.
    pub(crate) fn on_device(&self) -> u32 {
        self.device + self.shared
    }

    /// What the worker's engine will report cached, every tier's tokens; None when the
    /// store it reads did not answer.
    pub(crate) fn cached(&self) -> Option<u32> {
        Some(self.on_device() + self.stored?)
    }
}

/// What a token of each tier below a worker's device cache counts for in a policy's score,
/// against one its device cache holds, which counts 1. A tier that counts 0 is not asked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierWeights {
    /// A token the engine will copy in from the device pools of its share group:
    /// `--kv-share-weight`.
    pub shared: f64,
    /// A token the engine will load from the store it reads: `--kv-storage-weight`.
    pub stored: f64,
}

impl Default for TierWeights {
    fn default() -> Self {
        Self {
            shared: 0.9,
            stored: 0.0,
        }
    }
}

/// What `hit` is worth in tokens on the device: each tier's tokens at what `weights` counts
/// one of them for, a store that did not answer adding none. Over the prompt's tokens, it
/// is the match rate `cache_aware` holds against `cache_threshold`.
pub(crate) fn effective_cached(hit: &TieredHit, weights: &TierWeights) -> f64 {
    f64::from(hit.device)
        + weights.shared * f64::from(hit.shared)
        + weights.stored * f64::from(hit.stored.unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The score `cache_aware` wrote out by hand, a term a tier, before the tiers had one,
    /// from what the index then answered a tier at a time: `device` tokens on the worker's
    /// device, `held` there once its share group's are copied in, and `cached` once its
    /// store's are loaded, None when the store does not answer.
    fn hand_written(
        (device, held, cached): (u32, u32, Option<u32>),
        (share_weight, store_weight): (f64, f64),
    ) -> f64 {
        let copied = held.saturating_sub(device);
        let loaded = cached.map_or(0, |cached| cached.saturating_sub(device + copied));
        f64::from(device) + share_weight * f64::from(copied) + store_weight * f64::from(loaded)
    }

    #[test]
    fn the_score_of_a_hit_is_the_one_cache_aware_wrote_out_by_hand() {
        let hits = [
            // On the device alone, copied in, loaded, both, and nothing at all.
            (60, 60, Some(60)),
            (20, 80, Some(80)),
            (0, 0, Some(90)),
            (20, 80, Some(96)),
            (0, 0, Some(0)),
            // A store that does not answer.
            (60, 60, None),
            (20, 80, None),
            // A long prompt, and a sum of weighed tokens that an f64 rounds.
            (3000, 11_008, Some(16_384)),
            (1, 4, Some(11)),
        ];
        // The defaults of the two flags, then others, 0 and 1 included.
        let weights = [
            (0.9, 0.0),
            (0.9, 0.6),
            (0.5, 1.0),
            (0.0, 0.0),
            (0.0, 0.3),
            (1.0, 1.0),
            (0.1, 0.7),
        ];
        assert_eq!(
            TierWeights::default(),
            TierWeights {
                shared: weights[0].0,
                stored: weights[0].1
            }
        );
        for (shared, stored) in weights {
            let tier_weights = TierWeights { shared, stored };
            for (device, held, cached) in hits {
                let hit = TieredHit {
                    device,
                    shared: held - device,
                    stored: cached.map(|cached| cached - held),
                };
                assert_eq!((hit.on_device(), hit.cached()), (held, cached));
                // The same bits, not only the same within a rounding.
                let want = hand_written((device, held, cached), (shared, stored));
                let got = effective_cached(&hit, &tier_weights);
                assert_eq!(got.to_bits(), want.to_bits(), "{hit:?} at {tier_weights:?}");
            }
        }
        // 20 on the device, 60 copied in at 0.9 each, and 16 loaded at nothing.
        let hit = TieredHit {
            device: 20,
            shared: 60,
            stored: Some(16),
        };
        assert_eq!(effective_cached(&hit, &TierWeights::default()), 74.0);
    }
}
