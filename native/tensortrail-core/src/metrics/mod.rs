//! Numerical metric kernels.
//!
//! # Numerical policy
//!
//! * Hardware-supported inputs are `float32` and `float64`. Both are widened to
//!   `f64` for accumulation, then reported as `f64`.
//! * Non-finite elements are excluded from metric sums and from tensor
//!   statistics. They are counted separately (NaN, +inf, -inf) and non-finite
//!   pattern differences are reflected in the divergence status. This avoids
//!   silently turning an entire report into NaN when one layer overflows.
//! * Empty inputs produce `None` for undefined metrics; they never produce a
//!   fabricated zero.
//! * Finite sums use plain `f64` addition. Results depend on the deterministic
//!   chunk layout, but may differ from a sequential run in the last few bits
//!   because floating-point addition is not associative.

pub mod absolute;
pub mod cosine;
pub mod fused;
pub mod relative;
pub mod squared;
pub mod statistics;

use crate::error::Result;
use crate::tensor::DType;

/// A scalar dtype supported by the native engine.
pub trait Element: Copy + Send + Sync + 'static {
    /// Corresponding dtype tag.
    const DTYPE: DType;

    /// Widen to `f64` for accumulation.
    fn to_f64(self) -> f64;
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Element for f64 {
    const DTYPE: DType = DType::F64;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Comparison semantics shared by the fused and standalone kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparisonOptions {
    /// Positive floor for the relative-error denominator.
    pub epsilon: f64,
    /// Absolute tolerance: `|reference - candidate| <= atol + rtol * |reference|`.
    pub atol: f64,
    /// Relative tolerance, applied to the reference magnitude.
    pub rtol: f64,
    /// Whether two corresponding NaNs are considered a match.
    pub equal_nan: bool,
}

impl Default for ComparisonOptions {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            atol: 1e-6,
            rtol: 1e-5,
            equal_nan: true,
        }
    }
}

impl ComparisonOptions {
    /// Validate user-controlled values.
    pub fn validate(&self) -> Result<()> {
        crate::validation::validate_metric_options(self.epsilon, self.atol, self.rtol)
    }
}

/// Counts of non-finite values on one side of a comparison.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NonFiniteCounts {
    pub nan: u64,
    pub positive_infinity: u64,
    pub negative_infinity: u64,
}

impl NonFiniteCounts {
    /// Total number of non-finite values.
    pub const fn total(&self) -> u64 {
        self.nan + self.positive_infinity + self.negative_infinity
    }

    pub(crate) fn observe(&mut self, value: f64) {
        if value.is_nan() {
            self.nan += 1;
        } else if value == f64::INFINITY {
            self.positive_infinity += 1;
        } else if value == f64::NEG_INFINITY {
            self.negative_infinity += 1;
        }
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.nan += other.nan;
        self.positive_infinity += other.positive_infinity;
        self.negative_infinity += other.negative_infinity;
    }
}

/// Relative-error summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeError {
    pub mean: Option<f64>,
    pub max: Option<f64>,
}

/// Streaming accumulator for all binary comparison metrics.
///
/// One traversal accumulates MAE, MSE, RMSE inputs, relative error, cosine
/// terms, maximum absolute error, and tolerance mismatches. The accumulator is
/// split and merged by the fused kernel; all fields are simple scalars so the
/// memory overhead is constant regardless of tensor size.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PairAccumulator {
    pub elements: u64,
    pub finite_elements: u64,
    pub reference_nonfinite: NonFiniteCounts,
    pub candidate_nonfinite: NonFiniteCounts,
    pub nonfinite_mismatch: u64,
    pub finite_mismatch: u64,
    pub abs_sum: f64,
    pub squared_sum: f64,
    pub dot: f64,
    pub reference_squared_norm: f64,
    pub candidate_squared_norm: f64,
    pub reference_max_abs: f64,
    pub candidate_max_abs: f64,
    pub relative_sum: f64,
    pub relative_max: f64,
    pub max_abs_error: f64,
}

impl PairAccumulator {
    /// Accumulate one reference/candidate pair.
    #[inline]
    pub fn push<T: Element>(&mut self, reference: T, candidate: T, options: &ComparisonOptions) {
        let reference = reference.to_f64();
        let candidate = candidate.to_f64();

        self.elements += 1;

        if reference.is_finite() && candidate.is_finite() {
            self.finite_elements += 1;

            let difference = reference - candidate;
            let absolute = difference.abs();
            let squared = difference * difference;

            self.abs_sum += absolute;
            self.squared_sum += squared;
            self.dot += reference * candidate;
            self.reference_squared_norm += reference * reference;
            self.candidate_squared_norm += candidate * candidate;
            self.reference_max_abs = self.reference_max_abs.max(reference.abs());
            self.candidate_max_abs = self.candidate_max_abs.max(candidate.abs());

            let denominator = reference.abs().max(options.epsilon);
            let relative = absolute / denominator;
            self.relative_sum += relative;
            if relative > self.relative_max {
                self.relative_max = relative;
            }

            if absolute > self.max_abs_error {
                self.max_abs_error = absolute;
            }

            let tolerance = options.atol + options.rtol * reference.abs();
            // `absolute` is a finite magnitude here; overflow to infinity is
            // still a mismatch, and NaN cannot be produced by finite inputs.
            if absolute > tolerance {
                self.finite_mismatch += 1;
            }
        } else {
            self.reference_nonfinite.observe(reference);
            self.candidate_nonfinite.observe(candidate);
            if !nonfinite_values_match(reference, candidate, options.equal_nan) {
                self.nonfinite_mismatch += 1;
            }
        }
    }

    /// Merge a partial accumulator produced from another chunk.
    pub fn merge(&mut self, other: Self) {
        self.elements += other.elements;
        self.finite_elements += other.finite_elements;
        self.reference_nonfinite.merge(other.reference_nonfinite);
        self.candidate_nonfinite.merge(other.candidate_nonfinite);
        self.nonfinite_mismatch += other.nonfinite_mismatch;
        self.finite_mismatch += other.finite_mismatch;
        self.abs_sum += other.abs_sum;
        self.squared_sum += other.squared_sum;
        self.dot += other.dot;
        self.reference_squared_norm += other.reference_squared_norm;
        self.candidate_squared_norm += other.candidate_squared_norm;
        if other.reference_max_abs > self.reference_max_abs {
            self.reference_max_abs = other.reference_max_abs;
        }
        if other.candidate_max_abs > self.candidate_max_abs {
            self.candidate_max_abs = other.candidate_max_abs;
        }
        self.relative_sum += other.relative_sum;
        if other.relative_max > self.relative_max {
            self.relative_max = other.relative_max;
        }
        if other.max_abs_error > self.max_abs_error {
            self.max_abs_error = other.max_abs_error;
        }
    }

    /// Mean absolute error, or `None` when no finite pairs were compared.
    pub fn mae(&self) -> Option<f64> {
        mean(self.abs_sum, self.finite_elements)
    }

    /// Mean squared error, or `None` when no finite pairs were compared.
    pub fn mse(&self) -> Option<f64> {
        mean(self.squared_sum, self.finite_elements)
    }

    /// Root mean squared error, or `None` when no finite pairs were compared.
    pub fn rmse(&self) -> Option<f64> {
        self.mse().map(f64::sqrt)
    }

    /// Mean and maximum relative error.
    pub fn relative_error(&self) -> RelativeError {
        if self.finite_elements == 0 {
            RelativeError {
                mean: None,
                max: None,
            }
        } else {
            RelativeError {
                mean: Some(self.relative_sum / self.finite_elements as f64),
                max: Some(self.relative_max),
            }
        }
    }

    /// Cosine similarity using accumulated dot and squared norms.
    ///
    /// Returns `None` for an empty finite comparison. Identical zero vectors are
    /// defined as similarity 1; exactly one zero vector is defined as 0. If the
    /// accumulated terms overflow `f64`, the returned value is NaN and callers
    /// should use [`cosine::stable_cosine_similarity`] as a fallback.
    pub fn cosine_similarity(&self) -> Option<f64> {
        if self.finite_elements == 0 {
            return None;
        }

        // Use the exact maximum magnitude rather than a squared norm to detect
        // zero vectors: squaring very small values can underflow to zero.
        if self.reference_max_abs == 0.0 && self.candidate_max_abs == 0.0 {
            return Some(1.0);
        }
        if self.reference_max_abs == 0.0 || self.candidate_max_abs == 0.0 {
            return Some(0.0);
        }

        let reference_squared_norm = self.reference_squared_norm;
        let candidate_squared_norm = self.candidate_squared_norm;
        let norm_product = reference_squared_norm * candidate_squared_norm;
        let terms_are_usable = reference_squared_norm.is_finite()
            && candidate_squared_norm.is_finite()
            && reference_squared_norm > 0.0
            && candidate_squared_norm > 0.0
            && norm_product.is_finite()
            && self.dot.is_finite();

        if !terms_are_usable {
            // Sentinel for "requires the scaled two-pass fallback".
            return Some(f64::NAN);
        }

        // sqrt(norm_a * norm_b) is more accurate than sqrt(norm_a) *
        // sqrt(norm_b), and yields exactly 1 for identical vectors in common
        // cases. The product has already been checked for overflow.
        let denominator = norm_product.sqrt();
        let value = self.dot / denominator;
        Some(if value.is_nan() {
            f64::NAN
        } else {
            value.clamp(-1.0, 1.0)
        })
    }

    /// Maximum finite absolute error, or `None` if no finite pairs exist.
    pub fn max_abs_error(&self) -> Option<f64> {
        (self.finite_elements > 0).then_some(self.max_abs_error)
    }

    /// Total tolerance and non-finite-pattern mismatches.
    pub const fn mismatch_count(&self) -> u64 {
        self.finite_mismatch + self.nonfinite_mismatch
    }

    /// Build the public result from the accumulated state.
    pub fn finish(self) -> fused::ComparisonResult {
        let relative_error = self.relative_error();
        fused::ComparisonResult {
            elements: self.elements,
            finite_elements: self.finite_elements,
            mae: self.mae(),
            mse: self.mse(),
            rmse: self.rmse(),
            relative_error_mean: relative_error.mean,
            relative_error_max: relative_error.max,
            cosine_similarity: self.cosine_similarity(),
            max_abs_error: self.max_abs_error(),
            finite_mismatch_count: self.finite_mismatch,
            nonfinite_mismatch_count: self.nonfinite_mismatch,
            mismatch_count: self.mismatch_count(),
            reference_nonfinite: self.reference_nonfinite,
            candidate_nonfinite: self.candidate_nonfinite,
            statistics_reference: None,
            statistics_candidate: None,
            execution: crate::parallel::ExecutionInfo {
                parallel: false,
                threads: 1,
                chunks: 0,
                chunk_size: 0,
            },
        }
    }
}

/// Divide a finite sum by a count without fabricating a value for empty input.
pub(crate) fn mean(sum: f64, count: u64) -> Option<f64> {
    (count > 0).then(|| sum / count as f64)
}

/// Check whether two non-finite values represent the same exception.
#[inline]
fn nonfinite_values_match(reference: f64, candidate: f64, equal_nan: bool) -> bool {
    if reference.is_nan() && candidate.is_nan() {
        return equal_nan;
    }
    if reference.is_nan() || candidate.is_nan() {
        return false;
    }
    reference.is_infinite()
        && candidate.is_infinite()
        && reference.is_sign_positive() == candidate.is_sign_positive()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_handles_identical_values() {
        let options = ComparisonOptions::default();
        let mut accumulator = PairAccumulator::default();
        for (reference, candidate) in [1.0f64, -2.0, 0.0].into_iter().zip([1.0f64, -2.0, 0.0]) {
            accumulator.push(reference, candidate, &options);
        }
        assert_eq!(accumulator.mae(), Some(0.0));
        assert_eq!(accumulator.mse(), Some(0.0));
        assert_eq!(accumulator.max_abs_error(), Some(0.0));
        let cosine = accumulator.cosine_similarity().unwrap();
        assert!((cosine - 1.0).abs() < 1e-15);
        assert_eq!(accumulator.mismatch_count(), 0);
    }

    #[test]
    fn nonfinite_patterns_are_counted() {
        let mut accumulator = PairAccumulator::default();
        let options = ComparisonOptions::default();
        accumulator.push(f64::NAN, f64::NAN, &options);
        accumulator.push(f64::INFINITY, f64::INFINITY, &options);
        accumulator.push(f64::INFINITY, f64::NEG_INFINITY, &options);
        accumulator.push(f64::NAN, 1.0, &options);
        assert_eq!(accumulator.finite_elements, 0);
        assert_eq!(accumulator.reference_nonfinite.nan, 2);
        assert_eq!(accumulator.reference_nonfinite.positive_infinity, 2);
        assert_eq!(accumulator.nonfinite_mismatch, 2);
        assert_eq!(accumulator.nonfinite_mismatch, 2);
        assert_eq!(accumulator.mae(), None);
    }
}
