//! Share the default CPU thread budget between request preparation and Tokio.

use std::{io, thread};

use tokio::runtime::{Builder, Runtime};

/// Return (preparation workers, Tokio workers), keeping both pools usable on one CPU.
fn split_workers(available: usize) -> (usize, usize) {
    let available = available.max(1);
    let prepare = (available / 2).max(1);
    let tokio = (available - prepare).max(1);
    (prepare, tokio)
}

fn default_worker_counts() -> (usize, usize) {
    split_workers(thread::available_parallelism().map_or(1, |count| count.get()))
}

/// Default preparation threads: half the available parallelism, with a minimum of one.
pub fn default_prepare_workers() -> usize {
    default_worker_counts().0
}

/// Build the server runtime with the remaining default CPU thread budget.
///
/// Explicit `TOKIO_WORKER_THREADS` settings retain Tokio's native behavior. An
/// explicit preparation-pool size does not change the Tokio default.
pub fn build_runtime() -> io::Result<Runtime> {
    let mut builder = Builder::new_multi_thread();
    if std::env::var_os("TOKIO_WORKER_THREADS").is_none() {
        builder.worker_threads(default_worker_counts().1);
    }
    builder.enable_all().build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_budget_handles_even_odd_and_single_cpu() {
        assert_eq!(split_workers(224), (112, 112));
        assert_eq!(split_workers(8), (4, 4));
        assert_eq!(split_workers(7), (3, 4));
        assert_eq!(split_workers(2), (1, 1));
        assert_eq!(split_workers(1), (1, 1));
    }
}
