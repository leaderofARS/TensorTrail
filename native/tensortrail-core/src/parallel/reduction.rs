//! Deterministic chunked reductions.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::parallel::execution::with_pool;
use crate::parallel::{DEFAULT_CHUNK_SIZE, DEFAULT_PARALLEL_THRESHOLD};

/// Execution controls for a native reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelConfig {
    /// Enables the parallel path; `false` always runs sequentially.
    pub enabled: bool,
    /// Inputs smaller than this element count run sequentially.
    pub threshold: usize,
    /// Fixed number of elements per chunk. Fixed chunks make reductions
    /// independent of the worker count and of scheduling order.
    pub chunk_size: usize,
    /// Explicit Rayon worker count. `None` uses the global Rayon pool.
    pub threads: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: DEFAULT_PARALLEL_THRESHOLD,
            chunk_size: DEFAULT_CHUNK_SIZE,
            threads: None,
        }
    }
}

impl ParallelConfig {
    /// Validate the configuration and reject values that cannot produce a
    /// meaningful reduction.
    pub fn validate(&self) -> Result<()> {
        if self.chunk_size == 0 {
            return Err(Error::InvalidConfiguration(
                "chunk_size must be at least 1".to_owned(),
            ));
        }
        if let Some(threads) = self.threads {
            if threads == 0 {
                return Err(Error::InvalidConfiguration(
                    "num_threads must be at least 1".to_owned(),
                ));
            }
        }
        Ok(())
    }

    /// Worker count that will be reported if the parallel path is used.
    pub fn effective_threads(&self) -> usize {
        self.threads.unwrap_or_else(rayon::current_num_threads)
    }
}

/// Description of how a reduction actually executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionInfo {
    pub parallel: bool,
    pub threads: usize,
    pub chunks: usize,
    pub chunk_size: usize,
}

impl ExecutionInfo {
    const fn sequential(chunk_size: usize) -> Self {
        Self {
            parallel: false,
            threads: 1,
            chunks: if chunk_size == 0 { 0 } else { 1 },
            chunk_size,
        }
    }
}

/// Apply `operation(start, end)` to deterministic fixed-size chunks and return
/// partial results in input order.
///
/// The sequence of chunk boundaries depends only on `len` and
/// `config.chunk_size`, never on the worker count. That property is what makes
/// the final, ordered merge reproducible across executions and thread counts.
pub fn map_chunks<R, F>(
    len: usize,
    config: &ParallelConfig,
    operation: F,
) -> Result<(Vec<R>, ExecutionInfo)>
where
    R: Send,
    F: Fn(usize, usize) -> R + Sync + Send,
{
    config.validate()?;

    if len == 0 {
        return Ok((Vec::new(), ExecutionInfo::sequential(0)));
    }

    let chunk_size = config.chunk_size.max(1);
    let chunk_count = len.div_ceil(chunk_size);
    let threads = config.effective_threads();
    let should_parallelize =
        config.enabled && len >= config.threshold && threads > 1 && chunk_count > 1;

    if !should_parallelize {
        let partial = operation(0, len);
        return Ok((vec![partial], ExecutionInfo::sequential(len)));
    }

    let partials = with_pool(config.threads, || {
        (0..chunk_count)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(len);
                operation(start, end)
            })
            .collect::<Vec<R>>()
    })?;

    Ok((
        partials,
        ExecutionInfo {
            parallel: true,
            threads,
            chunks: chunk_count,
            chunk_size,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_chunk_size() {
        let config = ParallelConfig {
            chunk_size: 0,
            ..ParallelConfig::default()
        };
        assert!(matches!(
            map_chunks(10, &config, |_, _| 0),
            Err(Error::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn parallel_chunks_cover_input_exactly_once() {
        let config = ParallelConfig {
            enabled: true,
            threshold: 1,
            chunk_size: 3,
            threads: Some(4),
        };
        let (partials, info) = map_chunks(10, &config, |start, end| (start, end)).unwrap();
        assert_eq!(
            partials,
            vec![(0, 3), (3, 6), (6, 9), (9, 10)],
            "chunks must be ordered and cover the tail"
        );
        assert!(info.parallel);
        assert_eq!(info.chunks, 4);
    }

    #[test]
    fn small_inputs_use_sequential_path() {
        let config = ParallelConfig {
            enabled: true,
            threshold: 1_000,
            chunk_size: 16,
            threads: Some(4),
        };
        let (partials, info) = map_chunks(100, &config, |start, end| (start, end)).unwrap();
        assert_eq!(partials, vec![(0, 100)]);
        assert!(!info.parallel);
        assert_eq!(info.threads, 1);
    }

    #[test]
    fn empty_input_produces_no_chunks() {
        let (partials, info) = map_chunks(0, &ParallelConfig::default(), |_, _| 1).unwrap();
        assert!(partials.is_empty());
        assert_eq!(info.chunks, 0);
    }
}
