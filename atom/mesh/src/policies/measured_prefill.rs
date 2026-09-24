//! Decision kernel for measured, synchronized-DP prefill routing.
//!
//! A placement pays for its effect on every unfinished prefill, not just the
//! incoming request. Inputs must describe remaining token work, resident cache,
//! scheduler geometry, and a whole-DP cycle-time model. Historical prefix text
//! and original dispatched prompt sizes do not satisfy this contract.
//!
//! This kernel is not enabled by a routing-policy name. An integration must
//! acquire coherent observations and atomically reserve the chosen placement.
//! The empirical residual guard is not a counterfactual confidence guarantee.

use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub id: u64,
    pub prompt_tokens: usize,
    /// Accepted prefix plus successfully completed chunks, never dispatched work.
    pub completed_tokens: usize,
    pub checkpoint_demand: usize,
    /// None derives the anchor; Some(0) explicitly disables the prompt-end anchor.
    pub checkpoint_end: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Geometry {
    pub token_budget: usize,
    pub alignment: usize,
    pub hash_block: usize,
    /// Zero disables all checkpoints; negative disables only the periodic grid.
    pub checkpoint_interval: i64,
    pub successor_room: usize,
    pub readable_midstep: bool,
    pub max_sequences: usize,
}

#[derive(Clone, Debug)]
pub struct CostModel {
    /// Intercept, sum(query), max(query), sum(query*context), max(query*context).
    pub coefficients: [f64; 5],
    /// Absolute cycle-time residual calibrated separately from the test data.
    pub step_error_s: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Evidence {
    /// Full pool snapshot with fresh, ordered successful chunk progress.
    pub coherent_progress: bool,
    /// Cache for backlog and candidate placements was probed, not inferred
    /// from a historical prefix tree. A cold probe may legitimately report zero.
    pub resident_cache_probed: bool,
    /// No unmodeled slot/connector/allocation stall or truncated queue.
    pub capacity_accounted_for: bool,
    /// Calibration belongs to this runtime and covers the candidate features.
    pub model_in_domain: bool,
}

impl Evidence {
    fn ready(&self) -> bool {
        self.coherent_progress
            && self.resident_cache_probed
            && self.capacity_accounted_for
            && self.model_in_domain
    }
}

#[derive(Clone, Debug)]
pub struct Completion {
    pub point_s: f64,
    pub lower_s: f64,
    pub upper_s: f64,
    pub first_step: usize,
    pub last_step: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Forecast {
    pub completions: BTreeMap<u64, Completion>,
    pub sum_completion_s: f64,
    pub sum_lower_s: f64,
    pub sum_upper_s: f64,
    pub steps: usize,
}

#[derive(Clone, Debug)]
pub struct Placement {
    pub rank: usize,
    pub request: Job,
}

#[derive(Clone, Debug)]
pub struct Score {
    pub total_s: f64,
    pub total_lower_s: f64,
    pub total_upper_s: f64,
    pub request: Completion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    Beneficial,
    KeepAffinity,
    InsufficientEvidence,
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub rank: usize,
    pub reason: Reason,
    pub scores: BTreeMap<usize, Score>,
}

fn valid_job(job: &Job) -> bool {
    job.prompt_tokens > job.completed_tokens
        && job.checkpoint_demand <= job.prompt_tokens
        && job.checkpoint_end.is_none_or(|end| end < job.prompt_tokens)
}

fn validate(
    queues: &[Vec<Job>],
    geometry: &Geometry,
    model: &CostModel,
) -> Result<HashSet<u64>, &'static str> {
    if queues.is_empty()
        || geometry.token_budget == 0
        || geometry.alignment == 0
        || geometry.hash_block == 0
        || geometry.max_sequences == 0
        || (geometry.checkpoint_interval > 0
            && geometry.checkpoint_interval as usize % geometry.hash_block != 0)
    {
        return Err("invalid pool or scheduler geometry");
    }
    if model.coefficients.iter().all(|&c| c == 0.0)
        || model
            .coefficients
            .iter()
            .any(|c| !c.is_finite() || *c < 0.0)
        || !model.step_error_s.is_finite()
        || model.step_error_s < 0.0
    {
        return Err("invalid cycle-time model");
    }
    let mut keys = HashSet::new();
    for job in queues.iter().flatten() {
        if !valid_job(job) || !keys.insert(job.id) {
            return Err("invalid or duplicate unfinished request");
        }
    }
    Ok(keys)
}

fn chunk(job: &Job, used: usize, g: &Geometry) -> usize {
    let remaining = job.prompt_tokens - job.completed_tokens;
    let mut amount = remaining.min(g.token_budget.saturating_sub(used));
    if amount < remaining {
        let aligned = amount - amount % g.alignment;
        if aligned > 0 || used > 0 {
            amount = aligned;
        }
    }
    if amount == 0 || g.readable_midstep {
        return amount;
    }
    let available_end = job.prompt_tokens.saturating_sub(g.successor_room);
    let rung = if g.checkpoint_interval > 0 {
        let interval = g.checkpoint_interval as usize;
        let limit = available_end / interval * interval;
        (job.completed_tokens + amount).min(limit) / interval * interval
    } else {
        0
    };
    let anchor = job.checkpoint_end.unwrap_or_else(|| {
        if g.checkpoint_interval == 0 {
            0
        } else {
            (available_end / g.hash_block * g.hash_block)
                .min((job.prompt_tokens - 1) / g.hash_block * g.hash_block)
        }
    });
    [rung, anchor, job.checkpoint_demand]
        .into_iter()
        .filter(|&p| job.completed_tokens < p && p <= job.completed_tokens + amount)
        .min()
        .map_or(amount, |p| p - job.completed_tokens)
}

pub fn forecast(
    queues: &[Vec<Job>],
    geometry: &Geometry,
    model: &CostModel,
    max_steps: usize,
) -> Result<Forecast, &'static str> {
    validate(queues, geometry, model)?;
    let mut queues = queues.to_vec();
    let mut result = Forecast::default();
    let mut first = BTreeMap::new();
    let (mut elapsed, mut lower, mut upper) = (0.0, 0.0, 0.0);
    while queues.iter().any(|q| !q.is_empty()) {
        if result.steps >= max_steps {
            return Err("forecast horizon exceeded; keep affinity");
        }
        let count = queues.iter().map(Vec::len).sum::<usize>();
        let mut features = [1.0_f64, 0.0, 0.0, 0.0, 0.0];
        let mut batches = Vec::with_capacity(queues.len());
        for queue in &queues {
            let (mut used, mut context) = (0, 0.0);
            let mut batch = Vec::new();
            for (index, job) in queue.iter().take(geometry.max_sequences).enumerate() {
                let amount = chunk(job, used, geometry);
                if amount == 0 {
                    break;
                }
                used += amount;
                context += amount as f64 * (job.completed_tokens + amount) as f64;
                batch.push((index, amount));
            }
            features[1] += used as f64;
            features[2] = features[2].max(used as f64);
            features[3] += context;
            features[4] = features[4].max(context);
            batches.push(batch);
        }
        if batches.iter().all(Vec::is_empty) {
            return Err("nonempty snapshot cannot make progress");
        }
        let duration: f64 = features
            .iter()
            .zip(model.coefficients)
            .map(|(f, c)| f * c)
            .sum();
        if !duration.is_finite() {
            return Err("non-finite cycle cost");
        }
        elapsed += duration;
        let lo = (duration - model.step_error_s).max(0.0);
        let hi = duration + model.step_error_s;
        lower += lo;
        upper += hi;
        result.sum_completion_s += count as f64 * duration;
        result.sum_lower_s += count as f64 * lo;
        result.sum_upper_s += count as f64 * hi;
        for (queue, batch) in queues.iter_mut().zip(batches) {
            for (index, amount) in batch {
                let job = &mut queue[index];
                let start = *first.entry(job.id).or_insert(result.steps);
                job.completed_tokens += amount;
                if job.completed_tokens == job.prompt_tokens {
                    result.completions.insert(
                        job.id,
                        Completion {
                            point_s: elapsed,
                            lower_s: lower,
                            upper_s: upper,
                            first_step: start,
                            last_step: result.steps,
                        },
                    );
                }
            }
            queue.retain(|job| job.completed_tokens < job.prompt_tokens);
        }
        result.steps += 1;
    }
    if !result.sum_upper_s.is_finite() {
        return Err("non-finite total cost");
    }
    Ok(result)
}

/// Evaluate placements in one synchronized DP pool. The caller owns the atomic
/// snapshot -> decision -> reservation transaction, including retries/cancels.
pub fn choose(
    queues: &[Vec<Job>],
    placements: &[Placement],
    source: usize,
    geometry: &Geometry,
    model: &CostModel,
    evidence: &Evidence,
    max_steps: usize,
) -> Result<Decision, &'static str> {
    let keys = validate(queues, geometry, model)?;
    let Some(affinity) = placements.iter().find(|p| p.rank == source) else {
        return Err("missing affinity placement");
    };
    let mut ranks = HashSet::new();
    for p in placements {
        if p.rank >= queues.len()
            || !ranks.insert(p.rank)
            || !valid_job(&p.request)
            || keys.contains(&p.request.id)
            || p.request.id != affinity.request.id
            || p.request.prompt_tokens != affinity.request.prompt_tokens
        {
            return Err("invalid, duplicate, or inconsistent placement");
        }
    }
    let mut decision = Decision {
        rank: source,
        reason: Reason::InsufficientEvidence,
        scores: BTreeMap::new(),
    };
    if !evidence.ready() {
        return Ok(decision);
    }
    for p in placements {
        let mut candidate = queues.to_vec();
        candidate[p.rank].push(p.request.clone());
        let f = forecast(&candidate, geometry, model, max_steps)?;
        decision.scores.insert(
            p.rank,
            Score {
                total_s: f.sum_completion_s,
                total_lower_s: f.sum_lower_s,
                total_upper_s: f.sum_upper_s,
                request: f.completions[&p.request.id].clone(),
            },
        );
    }
    let baseline = &decision.scores[&source];
    let mut best_upper = f64::INFINITY;
    decision.reason = Reason::KeepAffinity;
    for (&rank, score) in &decision.scores {
        if rank != source
            && score.total_upper_s < baseline.total_lower_s
            && score.request.upper_s < baseline.request.lower_s
            && score.total_upper_s < best_upper
        {
            decision.rank = rank;
            decision.reason = Reason::Beneficial;
            best_upper = score.total_upper_s;
        }
    }
    Ok(decision)
}

#[cfg(test)]
#[path = "measured_prefill/tests.rs"]
mod tests;
