//! Tokenizer copies shared by small groups of preparation workers.
//!
//! A HuggingFace BPE tokenizer keeps its word cache behind a lock shared by every
//! thread using the instance, and the contention grows with the number of
//! preparation threads (one instance across 16 threads loses about a third of
//! its throughput). Each group of `WORKERS_PER_COPY` workers therefore encodes
//! with its own copy, loaded once from the entry's source with the same factory
//! used at registration. Groups of four keep the throughput of one copy per
//! thread at a quarter of the memory (a 150k-entry vocabulary is about 60 MiB
//! per copy). Template rendering keeps the shared instance; encoding without
//! special tokens does not depend on it.

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, LazyLock, OnceLock, Weak},
};

use parking_lot::Mutex;
use tracing::warn;

use crate::{
    core::prepare_pool::current_worker_index,
    tokenizer::{
        factory, huggingface::HuggingFaceTokenizer, registry::TokenizerEntry, traits::Tokenizer,
    },
};

const WORKERS_PER_COPY: usize = 4;

/// A group's copy of one registered instance. The first worker to need it
/// loads it while the others in the group wait, so a group loads one copy.
/// `GROUP_COPIES` holds the only long-lived strong reference to the copy, so
/// removing the slot releases it once in-flight requests finish.
struct GroupSlot {
    shared: Weak<dyn Tokenizer>,
    local: OnceLock<Option<Arc<dyn Tokenizer>>>,
}

impl GroupSlot {
    fn made_for(&self, entry: &TokenizerEntry) -> bool {
        std::ptr::addr_eq(self.shared.as_ptr(), Arc::as_ptr(&entry.tokenizer))
    }
}

/// A group's loaded copy, as handed to a worker.
struct Copy {
    /// The registered instance this copy was made for; a re-registration under
    /// the same id replaces it, and a removal drops it.
    shared: Weak<dyn Tokenizer>,
    /// `None` after a failed load, so the group keeps the shared instance
    /// without retrying or warning on every request.
    local: Option<Arc<dyn Tokenizer>>,
}

impl Copy {
    fn encoder(&self, entry: &TokenizerEntry) -> Arc<dyn Tokenizer> {
        self.local
            .clone()
            .unwrap_or_else(|| entry.tokenizer.clone())
    }
}

/// A thread's view of its group's copy. It holds the copy weakly so idle
/// workers cannot keep a released copy alive.
struct ThreadCopy {
    shared: Weak<dyn Tokenizer>,
    local: Option<Weak<dyn Tokenizer>>,
}

impl ThreadCopy {
    fn of(copy: &Copy) -> Self {
        Self {
            shared: copy.shared.clone(),
            local: copy.local.as_ref().map(Arc::downgrade),
        }
    }

    /// The tokenizer to encode `entry` with, or `None` when this view is not
    /// for `entry` or its copy has been released.
    fn encoder(&self, entry: &TokenizerEntry) -> Option<Arc<dyn Tokenizer>> {
        if !std::ptr::addr_eq(self.shared.as_ptr(), Arc::as_ptr(&entry.tokenizer)) {
            return None;
        }
        match &self.local {
            Some(local) => local.upgrade(),
            None => Some(entry.tokenizer.clone()),
        }
    }

    fn is_live(&self) -> bool {
        self.shared.strong_count() > 0
            && self
                .local
                .as_ref()
                .is_none_or(|local| local.strong_count() > 0)
    }
}

/// Group slots keyed by (tokenizer id, worker group).
type GroupSlots = HashMap<(String, usize), Arc<GroupSlot>>;

static GROUP_COPIES: LazyLock<Mutex<GroupSlots>> = LazyLock::new(Default::default);

thread_local! {
    /// This thread's view of its group's copies, so encoding takes no lock.
    static THREAD_COPIES: RefCell<HashMap<String, ThreadCopy>> = RefCell::new(HashMap::new());
}

/// Return the tokenizer this thread should encode `entry`'s text with.
///
/// Preparation workers share a copy within their group. Other threads, cached
/// wrappers and non-HuggingFace tokenizers use the shared instance.
pub(crate) fn encoder(entry: &TokenizerEntry) -> Arc<dyn Tokenizer> {
    match current_worker_index() {
        Some(index) => group_encoder(entry, index / WORKERS_PER_COPY),
        None => entry.tokenizer.clone(),
    }
}

/// Release every group's copy of a removed tokenizer. Requests still encoding
/// with a copy keep it until they finish.
pub(crate) fn release_copies(tokenizer_id: &str) {
    GROUP_COPIES
        .lock()
        .retain(|(id, _), slot| id != tokenizer_id && slot.shared.strong_count() > 0);
}

fn group_encoder(entry: &TokenizerEntry, group: usize) -> Arc<dyn Tokenizer> {
    if !entry.tokenizer.as_any().is::<HuggingFaceTokenizer>() {
        return entry.tokenizer.clone();
    }
    THREAD_COPIES.with(|copies| {
        let mut copies = copies.borrow_mut();
        if let Some(encoder) = copies.get(&entry.id).and_then(|copy| copy.encoder(entry)) {
            return encoder;
        }
        copies.retain(|_, copy| copy.is_live());
        let copy = group_copy(entry, group, || factory::create_tokenizer(&entry.source));
        copies.insert(entry.id.clone(), ThreadCopy::of(&copy));
        copy.encoder(entry)
    })
}

fn group_copy<E: std::fmt::Display>(
    entry: &TokenizerEntry,
    group: usize,
    load: impl FnOnce() -> Result<Arc<dyn Tokenizer>, E>,
) -> Copy {
    let key = (entry.id.clone(), group);
    let slot = {
        let mut slots = GROUP_COPIES.lock();
        match slots.get(&key).filter(|slot| slot.made_for(entry)) {
            Some(slot) => slot.clone(),
            None => {
                slots.retain(|_, slot| slot.shared.strong_count() > 0);
                let slot = Arc::new(GroupSlot {
                    shared: Arc::downgrade(&entry.tokenizer),
                    local: OnceLock::new(),
                });
                slots.insert(key, slot.clone());
                slot
            }
        }
    };
    // Load outside the map lock; only this group's other workers wait on it.
    let local = slot.local.get_or_init(|| {
        load()
            .inspect_err(|error| {
                warn!(
                    tokenizer = %entry.name,
                    %error,
                    "Failed to load a tokenizer copy for preparation workers; encoding with the shared instance"
                );
            })
            .ok()
    });
    Copy {
        shared: slot.shared.clone(),
        local: local.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routers::test_mocks::tokenizer::{hf_tokenizer, hf_tokenizer_path};

    #[test]
    fn workers_in_a_group_share_one_copy_and_other_groups_get_their_own() {
        let entry = TokenizerEntry {
            id: "group-copy".into(),
            name: "group-copy".into(),
            source: hf_tokenizer_path().into(),
            tokenizer: hf_tokenizer(),
        };
        let first = group_encoder(&entry, 0);
        assert!(!Arc::ptr_eq(&first, &entry.tokenizer));
        assert!(Arc::ptr_eq(&first, &group_encoder(&entry, 0)));
        // Another worker thread of the same group reuses the copy.
        let same_group = entry.clone();
        let peer = std::thread::spawn(move || group_encoder(&same_group, 0))
            .join()
            .unwrap();
        assert!(Arc::ptr_eq(&first, &peer));
        let other_group = entry.clone();
        let other = std::thread::spawn(move || group_encoder(&other_group, 1))
            .join()
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &other));
        let text = "hello world";
        assert_eq!(
            first.encode(text, false).unwrap().token_ids(),
            entry.tokenizer.encode(text, false).unwrap().token_ids()
        );

        // Threads outside the pool, other tokenizer kinds and copies that cannot
        // load use the shared instance.
        assert!(Arc::ptr_eq(&encoder(&entry), &entry.tokenizer));
        let mock = TokenizerEntry {
            id: "mock".into(),
            tokenizer: Arc::new(crate::tokenizer::MockTokenizer::new()),
            ..entry.clone()
        };
        assert!(Arc::ptr_eq(&group_encoder(&mock, 0), &mock.tokenizer));
        let missing = TokenizerEntry {
            id: "missing".into(),
            source: "/nonexistent/tokenizer.json".into(),
            ..entry
        };
        assert!(Arc::ptr_eq(&group_encoder(&missing, 0), &missing.tokenizer));
    }

    #[test]
    fn removing_a_tokenizer_releases_copies_that_idle_workers_have_seen() {
        let entry = TokenizerEntry {
            id: "released-copy".into(),
            name: "released-copy".into(),
            source: hf_tokenizer_path().into(),
            tokenizer: hf_tokenizer(),
        };
        let in_flight = group_encoder(&entry, 0);
        let copy = Arc::downgrade(&in_flight);
        assert!(!Arc::ptr_eq(&in_flight, &entry.tokenizer));

        release_copies(&entry.id);
        // A request still encoding keeps its copy until it finishes, but this
        // thread's cached view of the copy does not.
        assert!(copy.upgrade().is_some());
        drop(in_flight);
        assert!(copy.upgrade().is_none());

        // The group loads a fresh copy if the entry is used again.
        let reloaded = group_encoder(&entry, 0);
        assert!(!Arc::ptr_eq(&reloaded, &entry.tokenizer));
        assert!(GROUP_COPIES.lock().contains_key(&(entry.id.clone(), 0)));
    }

    #[test]
    fn concurrent_misses_in_a_group_load_one_copy() {
        let entry = TokenizerEntry {
            id: "single-flight".into(),
            name: "single-flight".into(),
            source: hf_tokenizer_path().into(),
            tokenizer: hf_tokenizer(),
        };
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let start = Arc::new(std::sync::Barrier::new(WORKERS_PER_COPY));
        let copies: Vec<_> = (0..WORKERS_PER_COPY)
            .map(|_| {
                let (entry, loads, start) = (entry.clone(), loads.clone(), start.clone());
                std::thread::spawn(move || {
                    start.wait();
                    let copy = group_copy(&entry, 0, || {
                        loads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // Keep the load slow enough for every worker to miss.
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        factory::create_tokenizer(&entry.source)
                    });
                    copy.encoder(&entry)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert_eq!(loads.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(!Arc::ptr_eq(&copies[0], &entry.tokenizer));
        assert!(copies.iter().all(|copy| Arc::ptr_eq(copy, &copies[0])));
    }
}
