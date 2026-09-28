use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const METRIC_SCOPE_COUNT: usize = 5;
const ROLLING_WINDOW_SECS: usize = 5 * 60;
const RESETTING_EPOCH_SECOND: u64 = u64::MAX;
const FINE_LATENCY_BUCKET_US: u64 = 10;
const FINE_LATENCY_MAX_US: u64 = 10_000;
const FINE_LATENCY_BUCKET_COUNT: usize =
    (FINE_LATENCY_MAX_US / FINE_LATENCY_BUCKET_US) as usize;
const COARSE_LATENCY_BUCKETS_US: [u64; 11] = [
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    30_000_000,
    u64::MAX,
];
const LATENCY_BUCKET_COUNT: usize =
    FINE_LATENCY_BUCKET_COUNT + COARSE_LATENCY_BUCKETS_US.len();
const REQUEST_ENDPOINT_SCOPES: [(u16, &str); 4] = [
    (1, "/generate"),
    (2, "/v1/chat/completions"),
    (3, "/v1/completions"),
    (4, "/v1/responses"),
];

#[derive(Clone, Copy, Debug, Default)]
struct MetricStats {
    count: usize,
    success_count: usize,
    failed_count: usize,
    avg_ms: f64,
    p99_ms: f64,
    p999_ms: f64,
    qps: f64,
}

pub(crate) struct VirtualRequestMetrics {
    scopes: Vec<ScopeMetrics>,
}

impl VirtualRequestMetrics {
    pub(crate) fn new() -> Self {
        Self {
            scopes: (0..METRIC_SCOPE_COUNT)
                .map(|_| ScopeMetrics::new())
                .collect(),
        }
    }

    pub(crate) fn record(&self, endpoint: u16, duration: Duration, success: bool) {
        let latency_us = duration.as_micros().min(u128::from(u64::MAX)) as u64;
        let now_second = current_epoch_second();
        self.scopes[0].record(now_second, latency_us, success);

        let endpoint_index = usize::from(endpoint);
        if endpoint_index < self.scopes.len() {
            self.scopes[endpoint_index].record(now_second, latency_us, success);
        }
    }

    pub(crate) fn print(&self) {
        print_metrics(self);
    }
}

struct ScopeMetrics {
    total_count: AtomicU64,
    total_success: AtomicU64,
    total_fail: AtomicU64,
    total_latency_us: AtomicU64,
    total_histogram: Vec<AtomicU64>,
    rolling: Vec<RollingMetricsBucket>,
}

impl ScopeMetrics {
    fn new() -> Self {
        Self {
            total_count: AtomicU64::new(0),
            total_success: AtomicU64::new(0),
            total_fail: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            total_histogram: new_atomic_histogram(),
            rolling: (0..ROLLING_WINDOW_SECS)
                .map(|_| RollingMetricsBucket::new())
                .collect(),
        }
    }

    fn record(&self, now_second: u64, latency_us: u64, success: bool) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        if success {
            self.total_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.total_fail.fetch_add(1, Ordering::Relaxed);
        }
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);
        self.total_histogram[latency_bucket_index(latency_us)].fetch_add(1, Ordering::Relaxed);

        let bucket = &self.rolling[now_second as usize % ROLLING_WINDOW_SECS];
        bucket.record(now_second, latency_us, success);
    }

    fn total_stats(&self) -> MetricStats {
        metric_stats_from_parts(
            self.total_count.load(Ordering::Relaxed),
            self.total_success.load(Ordering::Relaxed),
            self.total_fail.load(Ordering::Relaxed),
            self.total_latency_us.load(Ordering::Relaxed),
            load_histogram(&self.total_histogram),
            None,
        )
    }

    fn window_stats(&self, now_second: u64, window_secs: u64) -> MetricStats {
        let mut count = 0;
        let mut success = 0;
        let mut fail = 0;
        let mut latency_us = 0;
        let mut histogram = vec![0; LATENCY_BUCKET_COUNT];

        for bucket in &self.rolling {
            if !bucket.is_in_window(now_second, window_secs) {
                continue;
            }

            let snapshot = bucket.snapshot();
            count += snapshot.count;
            success += snapshot.success;
            fail += snapshot.fail;
            latency_us += snapshot.latency_us;
            for (target, source) in histogram.iter_mut().zip(snapshot.histogram) {
                *target += source;
            }
        }

        metric_stats_from_parts(count, success, fail, latency_us, histogram, Some(window_secs))
    }
}

struct RollingMetricsBucket {
    epoch_second: AtomicU64,
    active_writers: AtomicU64,
    count: AtomicU64,
    success: AtomicU64,
    fail: AtomicU64,
    latency_us: AtomicU64,
    histogram: Vec<AtomicU64>,
}

impl RollingMetricsBucket {
    fn new() -> Self {
        Self {
            epoch_second: AtomicU64::new(0),
            active_writers: AtomicU64::new(0),
            count: AtomicU64::new(0),
            success: AtomicU64::new(0),
            fail: AtomicU64::new(0),
            latency_us: AtomicU64::new(0),
            histogram: new_atomic_histogram(),
        }
    }

    fn record(&self, epoch_second: u64, latency_us: u64, success: bool) {
        loop {
            self.prepare_for_epoch(epoch_second);

            self.active_writers.fetch_add(1, Ordering::Acquire);
            if self.epoch_second.load(Ordering::Acquire) == epoch_second {
                self.count.fetch_add(1, Ordering::Relaxed);
                if success {
                    self.success.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.fail.fetch_add(1, Ordering::Relaxed);
                }
                self.latency_us.fetch_add(latency_us, Ordering::Relaxed);
                self.histogram[latency_bucket_index(latency_us)].fetch_add(1, Ordering::Relaxed);
                self.active_writers.fetch_sub(1, Ordering::Release);
                break;
            }

            self.active_writers.fetch_sub(1, Ordering::Release);
        }
    }

    fn prepare_for_epoch(&self, epoch_second: u64) {
        let current = self.epoch_second.load(Ordering::Acquire);
        if current == epoch_second {
            return;
        }
        if current == RESETTING_EPOCH_SECOND {
            while self.epoch_second.load(Ordering::Acquire) == RESETTING_EPOCH_SECOND {
                std::hint::spin_loop();
            }
            return;
        }

        if self
            .epoch_second
            .compare_exchange(
                current,
                RESETTING_EPOCH_SECOND,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            while self.active_writers.load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
            self.count.store(0, Ordering::Relaxed);
            self.success.store(0, Ordering::Relaxed);
            self.fail.store(0, Ordering::Relaxed);
            self.latency_us.store(0, Ordering::Relaxed);
            for bucket in &self.histogram {
                bucket.store(0, Ordering::Relaxed);
            }
            self.epoch_second.store(epoch_second, Ordering::Release);
        }
    }

    fn is_in_window(&self, now_second: u64, window_secs: u64) -> bool {
        let epoch_second = self.epoch_second.load(Ordering::Acquire);
        epoch_second != 0
            && epoch_second != RESETTING_EPOCH_SECOND
            && epoch_second <= now_second
            && now_second.saturating_sub(epoch_second) < window_secs
    }

    fn snapshot(&self) -> RollingMetricsSnapshot {
        RollingMetricsSnapshot {
            count: self.count.load(Ordering::Relaxed),
            success: self.success.load(Ordering::Relaxed),
            fail: self.fail.load(Ordering::Relaxed),
            latency_us: self.latency_us.load(Ordering::Relaxed),
            histogram: load_histogram(&self.histogram),
        }
    }
}

struct RollingMetricsSnapshot {
    count: u64,
    success: u64,
    fail: u64,
    latency_us: u64,
    histogram: Vec<u64>,
}

pub(crate) async fn run_metrics_printer(
    metrics: Arc<VirtualRequestMetrics>,
    running: Arc<AtomicBool>,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(5));
    ticker.tick().await;

    while running.load(Ordering::Relaxed) {
        ticker.tick().await;
        metrics.print();
    }
}

fn print_metrics(metrics: &VirtualRequestMetrics) {
    println!(
        "\n{:<24} {:>10} {:>14} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10} {:>12} {:>10} {:>12} {:>10}",
        "scope",
        "total",
        "total_success",
        "total_fail",
        "avg_ms",
        "p99_ms",
        "p999_ms",
        "1s_avg_ms",
        "1s_qps",
        "1m_avg_ms",
        "1m_qps",
        "5m_avg_ms",
        "5m_qps",
    );
    print_metrics_scope("all", &metrics.scopes[0]);

    for (endpoint_code, endpoint_path) in REQUEST_ENDPOINT_SCOPES {
        let endpoint_scope = &metrics.scopes[usize::from(endpoint_code)];
        if endpoint_scope.total_count.load(Ordering::Relaxed) == 0 {
            continue;
        }

        print_metrics_scope(endpoint_path, endpoint_scope);
    }
}

fn print_metrics_scope(scope: &str, metrics: &ScopeMetrics) {
    let complete_second = current_epoch_second().saturating_sub(1);
    let total = metrics.total_stats();
    let one_second = metrics.window_stats(complete_second, 1);
    let one_minute = metrics.window_stats(complete_second, 60);
    let five_minutes = metrics.window_stats(complete_second, 5 * 60);

    println!(
        "{:<24} {:>10} {:>14} {:>12} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>10.3} {:>12.3} {:>10.3} {:>12.3} {:>10.3}",
        scope,
        total.count,
        total.success_count,
        total.failed_count,
        total.avg_ms,
        total.p99_ms,
        total.p999_ms,
        one_second.avg_ms,
        one_second.qps,
        one_minute.avg_ms,
        one_minute.qps,
        five_minutes.avg_ms,
        five_minutes.qps,
    );
}

fn metric_stats_from_parts(
    count: u64,
    success_count: u64,
    failed_count: u64,
    latency_us: u64,
    histogram: Vec<u64>,
    qps_window_secs: Option<u64>,
) -> MetricStats {
    if count == 0 {
        return MetricStats::default();
    }

    let qps = qps_window_secs
        .map(|window_secs| count as f64 / window_secs as f64)
        .unwrap_or_default();

    MetricStats {
        count: count as usize,
        success_count: success_count as usize,
        failed_count: failed_count as usize,
        avg_ms: latency_us as f64 / 1000.0 / count as f64,
        p99_ms: percentile_ms(&histogram, count, 0.99),
        p999_ms: percentile_ms(&histogram, count, 0.999),
        qps,
    }
}

fn percentile_ms(histogram: &[u64], count: u64, percentile: f64) -> f64 {
    if count == 0 {
        return 0.0;
    }

    let target = ((count as f64 * percentile).ceil() as u64).max(1);
    let mut cumulative = 0;
    for (index, bucket_count) in histogram.iter().enumerate() {
        cumulative += *bucket_count;
        if cumulative >= target {
            return latency_bucket_upper_bound_us(index) as f64 / 1000.0;
        }
    }
    latency_bucket_upper_bound_us(LATENCY_BUCKET_COUNT - 1) as f64 / 1000.0
}

fn latency_bucket_index(latency_us: u64) -> usize {
    if latency_us <= FINE_LATENCY_MAX_US {
        return latency_us
            .saturating_sub(1)
            .checked_div(FINE_LATENCY_BUCKET_US)
            .unwrap_or_default() as usize;
    }

    FINE_LATENCY_BUCKET_COUNT
        + COARSE_LATENCY_BUCKETS_US
            .iter()
            .position(|bucket| latency_us <= *bucket)
            .unwrap_or(COARSE_LATENCY_BUCKETS_US.len() - 1)
}

fn latency_bucket_upper_bound_us(index: usize) -> u64 {
    if index < FINE_LATENCY_BUCKET_COUNT {
        return (index as u64 + 1) * FINE_LATENCY_BUCKET_US;
    }

    COARSE_LATENCY_BUCKETS_US
        .get(index - FINE_LATENCY_BUCKET_COUNT)
        .copied()
        .unwrap_or(u64::MAX)
}

fn new_atomic_histogram() -> Vec<AtomicU64> {
    (0..LATENCY_BUCKET_COUNT)
        .map(|_| AtomicU64::new(0))
        .collect()
}

fn load_histogram(histogram: &[AtomicU64]) -> Vec<u64> {
    histogram
        .iter()
        .map(|bucket| bucket.load(Ordering::Relaxed))
        .collect()
}

fn current_epoch_second() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
