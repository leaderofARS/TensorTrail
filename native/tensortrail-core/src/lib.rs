//! TensorTrail native numerical engine.
//!
//! This crate is usable as a normal Rust library as well as the PyO3 extension
//! backing the `tensortrail` Python package. The numerical core has no Python
//! dependency; bindings are gated behind the `python` feature.

pub mod error;
pub mod metrics;
pub mod parallel;
pub mod tensor;
pub mod validation;

#[cfg(feature = "python")]
mod bindings;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python extension module backing `tensortrail._core`.
#[cfg(feature = "python")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    bindings::register(module)
}
