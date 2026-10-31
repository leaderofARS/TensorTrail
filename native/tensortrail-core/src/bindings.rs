//! PyO3 bindings for the native engine.
//!
//! The binding layer is intentionally thin: it validates shapes and buffer
//! layout, copies option values, releases the GIL, and translates results into
//! Python objects. Numerical policy lives in the Rust core.
//!
//! Inputs must be C-contiguous, aligned NumPy arrays. The Python layer converts
//! non-contiguous inputs explicitly and records that conversion in report
//! metadata; the binding rejects them rather than reinterpreting memory.

use std::time::Instant;

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::Error;
use crate::metrics::fused::{self, ComparisonResult, FusedOptions};
use crate::metrics::statistics::{self, Statistics};
use crate::metrics::{absolute, cosine, relative, squared};
use crate::parallel::ParallelConfig;

/// Native comparison result exposed as an internal Python object.
#[pyclass(module = "tensortrail._core", frozen)]
#[derive(Clone)]
pub struct NativeComparison {
    #[pyo3(get)]
    pub elements: u64,
    #[pyo3(get)]
    pub finite_elements: u64,
    #[pyo3(get)]
    pub mae: Option<f64>,
    #[pyo3(get)]
    pub mse: Option<f64>,
    #[pyo3(get)]
    pub rmse: Option<f64>,
    #[pyo3(get)]
    pub relative_error_mean: Option<f64>,
    #[pyo3(get)]
    pub relative_error_max: Option<f64>,
    #[pyo3(get)]
    pub cosine_similarity: Option<f64>,
    #[pyo3(get)]
    pub max_abs_error: Option<f64>,
    #[pyo3(get)]
    pub finite_mismatch_count: u64,
    #[pyo3(get)]
    pub nonfinite_mismatch_count: u64,
    #[pyo3(get)]
    pub mismatch_count: u64,
    #[pyo3(get)]
    pub reference_nan_count: u64,
    #[pyo3(get)]
    pub reference_positive_infinity_count: u64,
    #[pyo3(get)]
    pub reference_negative_infinity_count: u64,
    #[pyo3(get)]
    pub candidate_nan_count: u64,
    #[pyo3(get)]
    pub candidate_positive_infinity_count: u64,
    #[pyo3(get)]
    pub candidate_negative_infinity_count: u64,
    #[pyo3(get)]
    pub statistics_reference: Option<NativeStatistics>,
    #[pyo3(get)]
    pub statistics_candidate: Option<NativeStatistics>,
    #[pyo3(get)]
    pub parallel: bool,
    #[pyo3(get)]
    pub threads: usize,
    #[pyo3(get)]
    pub chunks: usize,
    #[pyo3(get)]
    pub chunk_size: usize,
    #[pyo3(get)]
    pub native_seconds: f64,
}

impl NativeComparison {
    fn from_core(result: ComparisonResult, native_seconds: f64) -> Self {
        Self {
            elements: result.elements,
            finite_elements: result.finite_elements,
            mae: result.mae,
            mse: result.mse,
            rmse: result.rmse,
            relative_error_mean: result.relative_error_mean,
            relative_error_max: result.relative_error_max,
            cosine_similarity: result.cosine_similarity,
            max_abs_error: result.max_abs_error,
            finite_mismatch_count: result.finite_mismatch_count,
            nonfinite_mismatch_count: result.nonfinite_mismatch_count,
            mismatch_count: result.mismatch_count,
            reference_nan_count: result.reference_nonfinite.nan,
            reference_positive_infinity_count: result.reference_nonfinite.positive_infinity,
            reference_negative_infinity_count: result.reference_nonfinite.negative_infinity,
            candidate_nan_count: result.candidate_nonfinite.nan,
            candidate_positive_infinity_count: result.candidate_nonfinite.positive_infinity,
            candidate_negative_infinity_count: result.candidate_nonfinite.negative_infinity,
            statistics_reference: result.statistics_reference.map(NativeStatistics::from),
            statistics_candidate: result.statistics_candidate.map(NativeStatistics::from),
            parallel: result.execution.parallel,
            threads: result.execution.threads,
            chunks: result.execution.chunks,
            chunk_size: result.execution.chunk_size,
            native_seconds,
        }
    }
}

/// Native statistics for one tensor.
#[pyclass(module = "tensortrail._core", frozen)]
#[derive(Clone)]
pub struct NativeStatistics {
    #[pyo3(get)]
    pub elements: u64,
    #[pyo3(get)]
    pub finite_elements: u64,
    #[pyo3(get)]
    pub nan_count: u64,
    #[pyo3(get)]
    pub positive_infinity_count: u64,
    #[pyo3(get)]
    pub negative_infinity_count: u64,
    #[pyo3(get)]
    pub min: Option<f64>,
    #[pyo3(get)]
    pub max: Option<f64>,
    #[pyo3(get)]
    pub mean: Option<f64>,
    #[pyo3(get)]
    pub variance: Option<f64>,
    #[pyo3(get)]
    pub std_dev: Option<f64>,
}

impl From<Statistics> for NativeStatistics {
    fn from(statistics: Statistics) -> Self {
        Self {
            elements: statistics.elements,
            finite_elements: statistics.finite_elements,
            nan_count: statistics.nan_count,
            positive_infinity_count: statistics.positive_infinity_count,
            negative_infinity_count: statistics.negative_infinity_count,
            min: statistics.min,
            max: statistics.max,
            mean: statistics.mean,
            variance: statistics.variance,
            std_dev: statistics.std_dev,
        }
    }
}

fn to_py_error(error: Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn ensure_matching_shapes<T: numpy::Element>(
    reference: &PyReadonlyArrayDyn<'_, T>,
    candidate: &PyReadonlyArrayDyn<'_, T>,
) -> PyResult<()> {
    if reference.shape() == candidate.shape() {
        Ok(())
    } else {
        Err(to_py_error(Error::ShapeMismatch {
            reference: reference.shape().to_vec(),
            candidate: candidate.shape().to_vec(),
        }))
    }
}

fn contiguous_slice<'a, T>(array: &'a PyReadonlyArrayDyn<'_, T>, role: &str) -> PyResult<&'a [T]>
where
    T: numpy::Element,
{
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{role} must be C-contiguous and aligned; TensorTrail's Python layer \
             converts non-contiguous inputs explicitly and records the conversion"
        ))
    })
}

fn parallel_config(
    parallel_threshold: usize,
    chunk_size: usize,
    num_threads: Option<usize>,
) -> ParallelConfig {
    ParallelConfig {
        enabled: true,
        threshold: parallel_threshold,
        chunk_size,
        threads: num_threads,
    }
}

#[rustfmt::skip]
macro_rules! define_compare_binding {
    ($name:ident, $ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
                            reference,
                            candidate,
                            *,
                            parallel_threshold = 32768,
                            chunk_size = 32768,
                            num_threads = None,
                            epsilon = 1e-12,
                            atol = 1e-6,
                            rtol = 1e-5,
                            equal_nan = true,
                            compute_statistics = false,
                            release_gil = true
                        ))]
        #[allow(clippy::too_many_arguments)]
        fn $name(
            py: Python<'_>,
            reference: PyReadonlyArrayDyn<'_, $ty>,
            candidate: PyReadonlyArrayDyn<'_, $ty>,
            parallel_threshold: usize,
            chunk_size: usize,
            num_threads: Option<usize>,
            epsilon: f64,
            atol: f64,
            rtol: f64,
            equal_nan: bool,
            compute_statistics: bool,
            release_gil: bool,
        ) -> PyResult<NativeComparison> {
            ensure_matching_shapes(&reference, &candidate)?;
            let reference_slice = contiguous_slice(&reference, "reference")?;
            let candidate_slice = contiguous_slice(&candidate, "candidate")?;
            let options = FusedOptions {
                epsilon,
                atol,
                rtol,
                equal_nan,
                compute_statistics,
                parallel: parallel_config(parallel_threshold, chunk_size, num_threads),
            };

            let started = Instant::now();
            let result = if release_gil {
                py.detach(|| fused::compare(reference_slice, candidate_slice, &options))
            } else {
                fused::compare(reference_slice, candidate_slice, &options)
            };
            let elapsed = started.elapsed().as_secs_f64();
            let result = result.map_err(to_py_error)?;
            Ok(NativeComparison::from_core(result, elapsed))
        }
    };
}

define_compare_binding!(compare_f32, f32);
define_compare_binding!(compare_f64, f64);

#[rustfmt::skip]
macro_rules! define_binary_metric_binding {
    ($name:ident, $ty:ty, $kernel:path) => {
        #[pyfunction]
        #[pyo3(signature = (
                            reference,
                            candidate,
                            *,
                            parallel_threshold = 32768,
                            chunk_size = 32768,
                            num_threads = None,
                            release_gil = true
                        ))]
        fn $name(
            py: Python<'_>,
            reference: PyReadonlyArrayDyn<'_, $ty>,
            candidate: PyReadonlyArrayDyn<'_, $ty>,
            parallel_threshold: usize,
            chunk_size: usize,
            num_threads: Option<usize>,
            release_gil: bool,
        ) -> PyResult<Option<f64>> {
            ensure_matching_shapes(&reference, &candidate)?;
            let reference_slice = contiguous_slice(&reference, "reference")?;
            let candidate_slice = contiguous_slice(&candidate, "candidate")?;
            let config = parallel_config(parallel_threshold, chunk_size, num_threads);

            let result = if release_gil {
                py.detach(|| $kernel(reference_slice, candidate_slice, &config))
            } else {
                $kernel(reference_slice, candidate_slice, &config)
            };
            result.map_err(to_py_error)
        }
    };
}

define_binary_metric_binding!(mae_f32, f32, absolute::mean_absolute_error);
define_binary_metric_binding!(mae_f64, f64, absolute::mean_absolute_error);
define_binary_metric_binding!(mse_f32, f32, squared::mean_squared_error);
define_binary_metric_binding!(mse_f64, f64, squared::mean_squared_error);
define_binary_metric_binding!(rmse_f32, f32, squared::root_mean_squared_error);
define_binary_metric_binding!(rmse_f64, f64, squared::root_mean_squared_error);
define_binary_metric_binding!(cosine_similarity_f32, f32, cosine::cosine_similarity);
define_binary_metric_binding!(cosine_similarity_f64, f64, cosine::cosine_similarity);

#[rustfmt::skip]
macro_rules! define_relative_binding {
    ($name:ident, $ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
                            reference,
                            candidate,
                            *,
                            epsilon = 1e-12,
                            parallel_threshold = 32768,
                            chunk_size = 32768,
                            num_threads = None,
                            release_gil = true
                        ))]
        #[allow(clippy::too_many_arguments)]
        fn $name(
            py: Python<'_>,
            reference: PyReadonlyArrayDyn<'_, $ty>,
            candidate: PyReadonlyArrayDyn<'_, $ty>,
            epsilon: f64,
            parallel_threshold: usize,
            chunk_size: usize,
            num_threads: Option<usize>,
            release_gil: bool,
        ) -> PyResult<(f64, f64)> {
            ensure_matching_shapes(&reference, &candidate)?;
            let reference_slice = contiguous_slice(&reference, "reference")?;
            let candidate_slice = contiguous_slice(&candidate, "candidate")?;
            let config = parallel_config(parallel_threshold, chunk_size, num_threads);

            let result = if release_gil {
                py.detach(|| {
                    relative::relative_error(reference_slice, candidate_slice, epsilon, &config)
                })
            } else {
                relative::relative_error(reference_slice, candidate_slice, epsilon, &config)
            };
            let relative_error = result.map_err(to_py_error)?;
            match (relative_error.mean, relative_error.max) {
                (Some(mean), Some(maximum)) => Ok((mean, maximum)),
                _ => Err(PyValueError::new_err(
                    "mean and maximum relative error are undefined because no finite \
                     reference/candidate pairs were compared",
                )),
            }
        }
    };
}

define_relative_binding!(relative_error_f32, f32);
define_relative_binding!(relative_error_f64, f64);

#[rustfmt::skip]
macro_rules! define_statistics_binding {
    ($name:ident, $ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (
                            values,
                            *,
                            parallel_threshold = 32768,
                            chunk_size = 32768,
                            num_threads = None,
                            release_gil = true
                        ))]
        fn $name(
            py: Python<'_>,
            values: PyReadonlyArrayDyn<'_, $ty>,
            parallel_threshold: usize,
            chunk_size: usize,
            num_threads: Option<usize>,
            release_gil: bool,
        ) -> PyResult<NativeStatistics> {
            let values_slice = contiguous_slice(&values, "values")?;
            let config = parallel_config(parallel_threshold, chunk_size, num_threads);
            let result = if release_gil {
                py.detach(|| statistics::statistics(values_slice, &config))
            } else {
                statistics::statistics(values_slice, &config)
            };
            result.map(NativeStatistics::from).map_err(to_py_error)
        }
    };
}

define_statistics_binding!(statistics_f32, f32);
define_statistics_binding!(statistics_f64, f64);

/// Return build and runtime information for diagnostics.
#[pyfunction]
fn native_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;
    info.set_item("rayon_threads", rayon::current_num_threads())?;
    info.set_item(
        "build_profile",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    info.set_item("simd", "scalar")?;
    Ok(info)
}

/// Register the native API on the extension module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<NativeComparison>()?;
    module.add_class::<NativeStatistics>()?;

    module.add_function(wrap_pyfunction!(compare_f32, module)?)?;
    module.add_function(wrap_pyfunction!(compare_f64, module)?)?;

    module.add_function(wrap_pyfunction!(mae_f32, module)?)?;
    module.add_function(wrap_pyfunction!(mae_f64, module)?)?;
    module.add_function(wrap_pyfunction!(mse_f32, module)?)?;
    module.add_function(wrap_pyfunction!(mse_f64, module)?)?;
    module.add_function(wrap_pyfunction!(rmse_f32, module)?)?;
    module.add_function(wrap_pyfunction!(rmse_f64, module)?)?;
    module.add_function(wrap_pyfunction!(relative_error_f32, module)?)?;
    module.add_function(wrap_pyfunction!(relative_error_f64, module)?)?;
    module.add_function(wrap_pyfunction!(cosine_similarity_f32, module)?)?;
    module.add_function(wrap_pyfunction!(cosine_similarity_f64, module)?)?;
    module.add_function(wrap_pyfunction!(statistics_f32, module)?)?;
    module.add_function(wrap_pyfunction!(statistics_f64, module)?)?;
    module.add_function(wrap_pyfunction!(native_info, module)?)?;
    Ok(())
}
