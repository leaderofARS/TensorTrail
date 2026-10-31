//! Cached Rayon thread pools.
//!
//! Applications often compare many layers with the same worker count. Building a
//! Rayon pool per comparison would allocate threads and scheduling structures
//! repeatedly, so pools are cached by requested thread count and reused. The
//! global Rayon pool is used when no explicit thread count is supplied.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::{Error, Result};

fn pool_cache() -> &'static Mutex<HashMap<usize, Arc<ThreadPool>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_cache() -> std::sync::MutexGuard<'static, HashMap<usize, Arc<ThreadPool>>> {
    pool_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Return a cached pool with exactly `threads` workers, creating it if needed.
pub fn get_or_create_pool(threads: usize) -> Result<Arc<ThreadPool>> {
    if threads == 0 {
        return Err(Error::InvalidConfiguration(
            "thread count must be at least 1".to_owned(),
        ));
    }

    if let Some(pool) = lock_cache().get(&threads).cloned() {
        return Ok(pool);
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |index| format!("tensortrail-{threads}-{index}"))
        .build()
        .map_err(|error| {
            Error::InvalidConfiguration(format!(
                "could not create a {threads}-thread Rayon pool: {error}"
            ))
        })?;
    let pool = Arc::new(pool);

    let mut cache = lock_cache();
    // Another caller may have inserted an equivalent pool while we were
    // building; keep the first one so callers share a single pool.
    Ok(cache
        .entry(threads)
        .or_insert_with(|| Arc::clone(&pool))
        .clone())
}

/// Run `operation` on the ambient Rayon pool or on a cached pool with the
/// requested worker count.
///
/// The closure must be `Send`; it must not borrow Python objects across the
/// boundary. The bindings only pass plain Rust slices and owned options.
pub fn with_pool<R, F>(threads: Option<usize>, operation: F) -> Result<R>
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    match threads {
        None => Ok(operation()),
        Some(threads) => {
            let pool = get_or_create_pool(threads)?;
            Ok(pool.install(operation))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_pool_is_reused() {
        let first = get_or_create_pool(2).unwrap();
        let second = get_or_create_pool(2).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn zero_threads_is_rejected() {
        assert!(matches!(
            get_or_create_pool(0),
            Err(Error::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn explicit_pool_runs_closure() {
        let value = with_pool(Some(2), || 20 + 22).unwrap();
        assert_eq!(value, 42);
    }
}
