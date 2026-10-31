//! Parallel execution primitives.
//!
//! TensorTrail uses Rayon for data-parallel CPU work. The reduction strategy is
//! deliberately *chunk-stable*: inputs are split at fixed boundaries, partial
//! results are collected in chunk order, and then merged sequentially. As a
//! result, floating-point results do not depend on scheduling, OS thread count,
//! or the configured worker count. They may still differ from a sequential run
//! or from a different chunk size because floating-point addition is not
//! associative.

pub mod execution;
pub mod reduction;

pub use reduction::{map_chunks, ExecutionInfo, ParallelConfig};

/// Default element count below which TensorTrail stays on one thread.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 32_768;

/// Default number of elements per reduction chunk.
pub const DEFAULT_CHUNK_SIZE: usize = 32_768;
