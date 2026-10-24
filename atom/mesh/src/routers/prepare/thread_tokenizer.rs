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
    sync::{Arc, LazyLock, Weak},
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

#[derive(Clone)]
struct Copy {
    /// The registered instance this copy was made for; a re-registration under
    /// the same id replaces it, and a removal drops it.
    shared: Weak<dyn Tokenizer>,
    /// `None` after a failed load, so the group keeps the shared instance
    /// without retrying or warning on every request.
    local: Option<Arc<dyn Tokenizer>>,
}

impl Copy {
    fn made_for(&self, entry: &TokenizerEntry) -> bool {
        std::ptr::addr_eq(self.shared.as_ptr(), Arc::as_ptr(&entry.tokenizer))
    }

    fn encoder(&self, entry: &TokenizerEntry) -> Arc<dyn Tokenizer> {
        self.local
            .clone()
            .unwrap_or_else(|| entry.tokenizer.clone())
    }
}

/// Copies keyed by (tokenizer id, worker group).
static GROUP_COPIES: LazyLock<Mutex<HashMap<(String, usize), Copy>>> =
    LazyLock::new(Default::default);

thread_local! {
    /// This thread's view of its group's copies, so encoding takes no lock.
    static THREAD_COPIES: RefCell<HashMap<String, Copy>> = RefCell::new(HashMap::new());
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

fn group_encoder(entry: &TokenizerEntry, group: usize) -> Arc<dyn Tokenizer> {
    if !entry.tokenizer.as_any().is::<HuggingFaceTokenizer>() {
        return entry.tokenizer.clone();
    }
    THREAD_COPIES.with(|copies| {
        let mut copies = copies.borrow_mut();
        if let Some(copy) = copies.get(&entry.id).filter(|copy| copy.made_for(entry)) {
            return copy.encoder(entry);
        }
        copies.retain(|_, copy| copy.shared.strong_count() > 0);
        let copy = group_copy(entry, group);
        copies.insert(entry.id.clone(), copy.clone());
        copy.encoder(entry)
    })
}

fn group_copy(entry: &TokenizerEntry, group: usize) -> Copy {
    let key = (entry.id.clone(), group);
    if let Some(copy) = GROUP_COPIES
        .lock()
        .get(&key)
        .filter(|copy| copy.made_for(entry))
    {
        return copy.clone();
    }
    // Load outside the lock; if another worker of the group raced us, keep its copy.
    let loaded = Copy {
        shared: Arc::downgrade(&entry.tokenizer),
        local: factory::create_tokenizer(&entry.source)
            .inspect_err(|error| {
                warn!(
                    tokenizer = %entry.name,
                    %error,
                    "Failed to load a tokenizer copy for preparation workers; encoding with the shared instance"
                );
            })
            .ok(),
    };
    let mut copies = GROUP_COPIES.lock();
    copies.retain(|_, copy| copy.shared.strong_count() > 0);
    match copies.get(&key).filter(|copy| copy.made_for(entry)) {
        Some(existing) => existing.clone(),
        None => {
            copies.insert(key, loaded.clone());
            loaded
        }
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
}
