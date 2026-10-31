//! Fused comparison: all binary metrics in one traversal.
//!
//! The fused kernel touches each input element once and accumulates MAE, MSE,
//! RMSE inputs, relative error, cosine terms, maximum absolute error, and
//! tolerance mismatches. Statistics can optionally be folded into the same
//! traversal. Whether fusion is faster than independent passes depends on the
//! metric mix, input size, and memory bandwidth; benchmarks in this repository
//! measure both rather than assuming an answer.

use crate::error::Result;
use crate::metrics::cosine::stable_cosine_similarity;
use crate::metrics::statistics::{Statistics, StatisticsAccumulator};
use crate::metrics::{Element, NonFiniteCounts, PairAccumulator};
use crate::parallel::{map_chunks, ExecutionInfo, ParallelConfig};
use crate::validation::{validate_metric_options, validate_pair_lengths};

/// Options for the fused comparison kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedOptions {
    pub epsilon: f64,
    pub atol: f64,
    pub rtol: f64,
    pub equal_nan: bool,
    pub compute_statistics: bool,
    pub parallel: ParallelConfig,
}

impl Default for FusedOptions {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            atol: 1e-6,
            rtol: 1e-5,
            equal_nan: true,
            compute_statistics: false,
            parallel: ParallelConfig::default(),
        }
    }
}

/// Result of a fused comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonResult {
    pub elements: u64,
    pub finite_elements: u64,
    pub mae: Option<f64>,
    pub mse: Option<f64>,
    pub rmse: Option<f64>,
    pub relative_error_mean: Option<f64>,
    pub relative_error_max: Option<f64>,
    pub cosine_similarity: Option<f64>,
    pub max_abs_error: Option<f64>,
    pub finite_mismatch_count: u64,
    pub nonfinite_mismatch_count: u64,
    pub mismatch_count: u64,
    pub reference_nonfinite: NonFiniteCounts,
    pub candidate_nonfinite: NonFiniteCounts,
    pub statistics_reference: Option<Statistics>,
    pub statistics_candidate: Option<Statistics>,
    pub execution: ExecutionInfo,
}

impl ComparisonResult {
    /// A comparison passes only when every finite pair is inside the configured
    /// tolerance and every non-finite pattern matches (subject to `equal_nan`).
    pub const fn passed(&self) -> bool {
        self.mismatch_count == 0
    }

    /// Stable status string used by reports.
    pub const fn status(&self) -> &'static str {
        if self.passed() {
            "pass"
        } else {
            "fail"
        }
    }
}

/// Run the fused comparison.
pub fn compare<T: Element>(
    reference: &[T],
    candidate: &[T],
    options: &FusedOptions,
) -> Result<ComparisonResult> {
    validate_metric_options(options.epsilon, options.atol, options.rtol)?;
    options.parallel.validate()?;
    let length = validate_pair_lengths(reference.len(), candidate.len())?;

    let comparison_options = crate::metrics::ComparisonOptions {
        epsilon: options.epsilon,
        atol: options.atol,
        rtol: options.rtol,
        equal_nan: options.equal_nan,
    };

    let (partials, execution) = map_chunks(length, &options.parallel, |start, end| {
        let mut accumulator = PairAccumulator::default();
        let mut reference_statistics = StatisticsAccumulator::default();
        let mut candidate_statistics = StatisticsAccumulator::default();

        if options.compute_statistics {
            for index in start..end {
                accumulator.push(reference[index], candidate[index], &comparison_options);
                reference_statistics.push(reference[index]);
                candidate_statistics.push(candidate[index]);
            }
        } else {
            for index in start..end {
                accumulator.push(reference[index], candidate[index], &comparison_options);
            }
        }

        (accumulator, reference_statistics, candidate_statistics)
    })?;

    let mut accumulator = PairAccumulator::default();
    let mut reference_statistics = StatisticsAccumulator::default();
    let mut candidate_statistics = StatisticsAccumulator::default();

    for (partial, reference_partial, candidate_partial) in partials {
        accumulator.merge(partial);
        reference_statistics.merge(reference_partial);
        candidate_statistics.merge(candidate_partial);
    }

    let mut result = accumulator.finish();
    result.execution = execution;

    if result.cosine_similarity.is_some_and(f64::is_nan) {
        result.cosine_similarity =
            stable_cosine_similarity(reference, candidate, &options.parallel)?;
    }

    if options.compute_statistics {
        result.statistics_reference = Some(reference_statistics.finish());
        result.statistics_candidate = Some(candidate_statistics.finish());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{absolute, squared};

    #[test]
    fn hand_calculated_comparison() {
        let reference = [1.0f64, 2.0, 3.0, 4.0];
        let candidate = [1.0f64, 4.0, 2.0, 8.0];
        let result = compare(
            &reference,
            &candidate,
            &FusedOptions {
                compute_statistics: true,
                ..FusedOptions::default()
            },
        )
        .unwrap();

        assert_eq!(result.elements, 4);
        assert_eq!(result.finite_elements, 4);
        assert_eq!(result.mae, Some(1.75));
        assert!((result.mse.unwrap() - 5.25).abs() < 1e-15);
        assert!((result.rmse.unwrap() - 5.25f64.sqrt()).abs() < 1e-15);
        assert_eq!(result.max_abs_error, Some(4.0));
        assert_eq!(result.finite_mismatch_count, 3);
        assert!(!result.passed());
        assert_eq!(result.status(), "fail");

        let reference_stats = result.statistics_reference.unwrap();
        assert_eq!(reference_stats.mean, Some(2.5));
        assert_eq!(reference_stats.variance, Some(1.25));
    }

    #[test]
    fn fused_matches_standalone_metrics() {
        let reference: Vec<f64> = (0..10_000)
            .map(|index| (index as f64 * 0.3).sin())
            .collect();
        let candidate: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(index, value)| value + (index as f64 * 0.001).cos() * 0.01)
            .collect();
        let config = ParallelConfig {
            enabled: true,
            threshold: 1,
            chunk_size: 257,
            threads: Some(4),
        };
        let result = compare(
            &reference,
            &candidate,
            &FusedOptions {
                parallel: config,
                ..FusedOptions::default()
            },
        )
        .unwrap();

        let mae = absolute::mean_absolute_error(&reference, &candidate, &config)
            .unwrap()
            .unwrap();
        let mse = squared::mean_squared_error(&reference, &candidate, &config)
            .unwrap()
            .unwrap();
        assert!((result.mae.unwrap() - mae).abs() < 1e-12);
        assert!((result.mse.unwrap() - mse).abs() < 1e-12);
        assert!(
            (result.cosine_similarity.unwrap()
                - crate::metrics::cosine::cosine_similarity(&reference, &candidate, &config)
                    .unwrap()
                    .unwrap())
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn parallel_results_are_identical_across_thread_counts() {
        let reference: Vec<f32> = (0..50_003).map(|index| (index as f32).sqrt()).collect();
        let candidate: Vec<f32> = reference.iter().map(|value| value + 0.25).collect();

        let run = |threads: usize| {
            compare(
                &reference,
                &candidate,
                &FusedOptions {
                    parallel: ParallelConfig {
                        enabled: true,
                        threshold: 1,
                        chunk_size: 1000,
                        threads: Some(threads),
                    },
                    ..FusedOptions::default()
                },
            )
            .unwrap()
        };

        let two = run(2);
        let eight = run(8);
        assert_eq!(two.mae, eight.mae);
        assert_eq!(two.mse, eight.mse);
        assert_eq!(two.relative_error_mean, eight.relative_error_mean);
        assert_eq!(two.cosine_similarity, eight.cosine_similarity);
        assert_eq!(two.mismatch_count, eight.mismatch_count);
        assert_eq!(two.finite_elements, eight.finite_elements);
    }

    #[test]
    fn empty_comparison_is_a_pass_with_no_metrics() {
        let empty: [f64; 0] = [];
        let result = compare(
            &empty,
            &empty,
            &FusedOptions {
                compute_statistics: true,
                ..FusedOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.elements, 0);
        assert_eq!(result.mae, None);
        assert_eq!(result.cosine_similarity, None);
        assert_eq!(result.mismatch_count, 0);
        assert!(result.passed());
        assert_eq!(result.statistics_reference.unwrap().elements, 0);
    }

    #[test]
    fn nonfinite_pattern_mismatch_fails_comparison() {
        let reference = [1.0f64, f64::NAN, f64::INFINITY];
        let candidate = [1.0f64, 2.0, f64::INFINITY];
        let result = compare(&reference, &candidate, &FusedOptions::default()).unwrap();
        assert_eq!(result.finite_elements, 1);
        assert_eq!(result.nonfinite_mismatch_count, 1);
        assert_eq!(result.reference_nonfinite.nan, 1);
        assert!(!result.passed());
    }

    #[test]
    fn equal_nan_can_be_disabled() {
        let reference = [f64::NAN];
        let candidate = [f64::NAN];
        let passing = compare(&reference, &candidate, &FusedOptions::default()).unwrap();
        assert!(passing.passed());
        let failing = compare(
            &reference,
            &candidate,
            &FusedOptions {
                equal_nan: false,
                ..FusedOptions::default()
            },
        )
        .unwrap();
        assert!(!failing.passed());
    }

    #[test]
    fn huge_values_trigger_scaled_cosine_fallback() {
        let reference = [1e308f64, 5e307, 2e307];
        let candidate = [1e308f64, 5e307, 2e307];
        let result = compare(&reference, &candidate, &FusedOptions::default()).unwrap();
        assert!((result.cosine_similarity.unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn statistics_are_not_computed_unless_requested() {
        let result = compare(&[1.0f64], &[1.0f64], &FusedOptions::default()).unwrap();
        assert_eq!(result.statistics_reference, None);
        assert_eq!(result.statistics_candidate, None);
    }

    #[test]
    fn tiny_values_trigger_scaled_cosine_fallback() {
        let reference = [1e-300f64, 2e-300, 3e-300];
        let candidate = [2e-300f64, 4e-300, 6e-300];
        let result = compare(&reference, &candidate, &FusedOptions::default()).unwrap();
        assert!((result.cosine_similarity.unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn finite_overflow_produces_infinity_not_nan() {
        let reference = [1e308f64];
        let candidate = [-1e308f64];
        let result = compare(&reference, &candidate, &FusedOptions::default()).unwrap();
        assert!(result.mae.unwrap().is_infinite());
        assert!(result.mse.unwrap().is_infinite());
        assert!(result.rmse.unwrap().is_infinite());
        assert!(result.relative_error_mean.unwrap().is_infinite());
        assert!(result.max_abs_error.unwrap().is_infinite());
        assert_eq!(result.finite_mismatch_count, 1);
        assert!((result.cosine_similarity.unwrap() + 1.0).abs() < 1e-15);
    }

    #[test]
    fn statistics_exclude_nonfinite_values_when_fused() {
        let reference = [1.0f64, f64::NAN, f64::INFINITY, 3.0];
        let candidate = [1.0f64, f64::NAN, f64::INFINITY, 3.0];
        let result = compare(
            &reference,
            &candidate,
            &FusedOptions {
                compute_statistics: true,
                ..FusedOptions::default()
            },
        )
        .unwrap();
        let statistics = result.statistics_reference.unwrap();
        assert_eq!(statistics.elements, 4);
        assert_eq!(statistics.finite_elements, 2);
        assert_eq!(statistics.nan_count, 1);
        assert_eq!(statistics.positive_infinity_count, 1);
        assert_eq!(statistics.mean, Some(2.0));
        assert_eq!(statistics.variance, Some(1.0));
    }

    #[test]
    fn f32_and_f64_agree_for_exactly_representable_values() {
        let reference = [1.0f32, 2.0, 3.0, 4.0];
        let candidate = [1.0f32, 4.0, 2.0, 8.0];
        let single = compare(&reference, &candidate, &FusedOptions::default()).unwrap();
        let double = compare(
            &[1.0f64, 2.0, 3.0, 4.0],
            &[1.0f64, 4.0, 2.0, 8.0],
            &FusedOptions::default(),
        )
        .unwrap();
        assert_eq!(single.mae, double.mae);
        assert_eq!(single.mse, double.mse);
        assert_eq!(single.relative_error_mean, double.relative_error_mean);
        assert_eq!(single.finite_mismatch_count, double.finite_mismatch_count);
    }
}
