use super::*;

fn geometry() -> Geometry {
    Geometry {
        token_budget: 16384,
        alignment: 256,
        hash_block: 256,
        checkpoint_interval: 8192,
        successor_room: 0,
        readable_midstep: false,
        max_sequences: 256,
    }
}

fn job(id: u64, prompt: usize, done: usize) -> Job {
    Job {
        id,
        prompt_tokens: prompt,
        completed_tokens: done,
        checkpoint_demand: 0,
        checkpoint_end: None,
    }
}

fn evidence() -> Evidence {
    Evidence {
        coherent_progress: true,
        resident_cache_probed: true,
        capacity_accounted_for: true,
        model_in_domain: true,
    }
}

fn token_model() -> CostModel {
    CostModel {
        coefficients: [0.0, 1.0 / 16384.0, 0.0, 0.0, 0.0],
        step_error_s: 0.0,
    }
}

fn placements(prompt: usize) -> Vec<Placement> {
    vec![
        Placement {
            rank: 0,
            request: job(2, prompt, prompt - 1),
        },
        Placement {
            rank: 1,
            request: job(2, prompt, 0),
        },
    ]
}

#[test]
fn independent_completion_area() {
    let g = Geometry {
        token_budget: 1,
        alignment: 1,
        hash_block: 1,
        checkpoint_interval: 0,
        ..geometry()
    };
    let m = CostModel {
        coefficients: [1.0, 0.0, 0.0, 0.0, 0.0],
        step_error_s: 0.0,
    };
    let f = forecast(
        &[vec![job(1, 2, 0), job(2, 1, 0)], vec![job(3, 1, 0)]],
        &g,
        &m,
        100,
    )
    .unwrap();
    assert_eq!(f.completions[&1].point_s, 2.0);
    assert_eq!(f.completions[&2].point_s, 3.0);
    assert_eq!(f.completions[&3].point_s, 1.0);
    assert_eq!(f.sum_completion_s, 6.0);
}

#[test]
fn rejects_speedup_that_hurts_total_completion() {
    let d = choose(
        &[vec![job(1, 1048576, 0)], vec![]],
        &placements(393216),
        0,
        &geometry(),
        &token_model(),
        &evidence(),
        8192,
    )
    .unwrap();
    assert!(d.scores[&1].request.point_s < d.scores[&0].request.point_s);
    assert!(d.scores[&1].total_s > d.scores[&0].total_s);
    assert_eq!(d.rank, 0);
}

#[test]
fn accepts_cold_work_when_both_improve() {
    let d = choose(
        &[vec![job(1, 1048576, 0)], vec![]],
        &placements(262144),
        0,
        &geometry(),
        &token_model(),
        &evidence(),
        8192,
    )
    .unwrap();
    assert_eq!(d.rank, 1);
    assert_eq!(d.reason, Reason::Beneficial);
}

#[test]
fn measured_residual_rejects_small_gain() {
    let mut m = token_model();
    m.step_error_s = 0.6681326836048957;
    let d = choose(
        &[vec![job(1, 1048576, 0)], vec![]],
        &placements(262144),
        0,
        &geometry(),
        &m,
        &evidence(),
        8192,
    )
    .unwrap();
    assert_eq!(d.rank, 0);
}

#[test]
fn uncertainty_is_not_a_blanket_ban() {
    let mut m = token_model();
    m.step_error_s = 0.01;
    let d = choose(
        &[vec![job(1, 1048576, 0)], vec![]],
        &placements(16384),
        0,
        &geometry(),
        &m,
        &evidence(),
        8192,
    )
    .unwrap();
    assert_eq!(d.rank, 1);
}

#[test]
fn missing_evidence_does_not_switch() {
    let fields: [fn(&mut Evidence); 4] = [
        |e| e.coherent_progress = false,
        |e| e.resident_cache_probed = false,
        |e| e.capacity_accounted_for = false,
        |e| e.model_in_domain = false,
    ];
    for set_missing in fields {
        let mut e = evidence();
        set_missing(&mut e);
        let d = choose(
            &[vec![job(1, 1048576, 0)], vec![]],
            &placements(16384),
            0,
            &geometry(),
            &token_model(),
            &e,
            8192,
        )
        .unwrap();
        assert_eq!(d.rank, 0);
        assert_eq!(d.reason, Reason::InsufficientEvidence);
        assert!(d.scores.is_empty());
    }
}

#[test]
fn no_quota_or_input_mutation() {
    let queues = vec![vec![job(1, 1048576, 0)], vec![]];
    let before = queues.clone();
    for _ in 0..20 {
        // Independent complete snapshots, not unreserved concurrent dispatches.
        let d = choose(
            &queues,
            &placements(262144),
            0,
            &geometry(),
            &token_model(),
            &evidence(),
            8192,
        )
        .unwrap();
        assert_eq!(d.rank, 1);
    }
    assert_eq!(queues, before);
}

#[test]
fn progress_decrements_remaining_work() {
    let a = forecast(&[vec![job(1, 65536, 0)]], &geometry(), &token_model(), 100).unwrap();
    let b = forecast(
        &[vec![job(1, 65536, 32768)]],
        &geometry(),
        &token_model(),
        100,
    )
    .unwrap();
    assert_eq!(a.sum_completion_s, 4.0);
    assert_eq!(b.sum_completion_s, 2.0);
}

#[test]
fn validates_duplicates_model_geometry_and_horizon() {
    assert!(forecast(
        &[vec![job(1, 100, 0), job(1, 100, 0)]],
        &geometry(),
        &token_model(),
        100
    )
    .is_err());
    assert!(forecast(&[vec![job(1, 100, 100)]], &geometry(), &token_model(), 100).is_err());
    assert!(forecast(&[vec![job(1, 100000, 0)]], &geometry(), &token_model(), 1).is_err());
    let mut m = token_model();
    m.coefficients[1] = f64::NAN;
    assert!(forecast(&[vec![]], &geometry(), &m, 10).is_err());
    let mut g = geometry();
    g.token_budget = 0;
    assert!(forecast(&[vec![]], &g, &token_model(), 10).is_err());
    let mut p = placements(100);
    p[1].request.id = 99;
    assert!(choose(
        &[vec![], vec![]],
        &p,
        0,
        &geometry(),
        &token_model(),
        &evidence(),
        100
    )
    .is_err());
    assert!(choose(
        &[vec![], vec![]],
        &placements(100),
        2,
        &geometry(),
        &token_model(),
        &evidence(),
        100
    )
    .is_err());
}

#[test]
fn captured_scheduler_chunk_contracts() {
    let mut cases = 0;
    for line in include_str!("source-chunks.tsv").lines() {
        let v: Vec<i64> = line.split('\t').map(|s| s.parse().unwrap()).collect();
        let g = Geometry {
            checkpoint_interval: v[3],
            successor_room: v[4] as usize,
            readable_midstep: v[5] != 0,
            ..geometry()
        };
        let j = Job {
            checkpoint_demand: v[6] as usize,
            ..job(1, v[0] as usize, v[1] as usize)
        };
        assert_eq!(chunk(&j, v[2] as usize, &g), v[7] as usize, "{line}");
        cases += 1;
    }
    assert_eq!(cases, 5000);
}

#[test]
fn real_trace_python_rust_forecast_parity() {
    let model = CostModel {
        coefficients: [
            0.19053810792083709,
            1.1447054789687817e-05,
            4.2003272577018524e-05,
            0.0,
            1.1779312439719456e-10,
        ],
        step_error_s: 0.0,
    };
    let mut cases = 0;
    for line in include_str!("factual-forecasts.tsv").lines() {
        let parts: Vec<_> = line.split('|').collect();
        let mut queues = vec![vec![]; 8];
        for spec in parts[1].split(';').filter(|s| !s.is_empty()) {
            let v: Vec<usize> = spec.split(',').map(|s| s.parse().unwrap()).collect();
            queues[v[0]].push(Job {
                checkpoint_demand: v[4],
                ..job(v[1] as u64, v[2], v[3])
            });
        }
        let f = forecast(&queues, &geometry(), &model, 8192).unwrap();
        let expected_sum: f64 = parts[2].parse().unwrap();
        assert!(
            (f.sum_completion_s - expected_sum).abs() < 1e-7,
            "step {}",
            parts[0]
        );
        let mut expected_count = 0;
        for spec in parts[3].split(';').filter(|s| !s.is_empty()) {
            let v: Vec<_> = spec.split(',').collect();
            let key: u64 = v[0].parse().unwrap();
            let c = &f.completions[&key];
            assert_eq!(c.first_step, v[1].parse::<usize>().unwrap());
            assert_eq!(c.last_step, v[2].parse::<usize>().unwrap());
            assert!((c.point_s - v[3].parse::<f64>().unwrap()).abs() < 1e-8);
            expected_count += 1;
        }
        assert_eq!(f.completions.len(), expected_count);
        cases += 1;
    }
    assert_eq!(cases, 130);
}
