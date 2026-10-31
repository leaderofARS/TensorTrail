//! Mean absolute error.

use crate::error::Result;
use crate::metrics::{mean, Element};
use crate::parallel::{map_chunks, ParallelConfig};
use crate::validation::validate_pair_lengths;

/// Compute mean absolute error over finite reference/candidate pairs.
///
/// Returns `None` for an empty input or when no pair contains two finite
/// values. Non-finite pairs are excluded and are reported separately by
/// [`crate::metrics::fused::compare`].
pub fn mean_absolute_error<T: Element>(
    reference: &[T],
    candidate: &[T],
    config: &ParallelConfig,
) -> Result<Option<f64>> {
    let length = validate_pair_lengths(reference.len(), candidate.len())?;
    if length == 0 {
        return Ok(None);
    }

    let (partials, _) = map_chunks(length, config, |start, end| {
        let mut sum = 0.0f64;
        let mut count = 0u64;
        for index in start..end {
            let reference_value = reference[index].to_f64();
            let candidate_value = candidate[index].to_f64();
            if reference_value.is_finite() && candidate_value.is_finite() {
                sum += (reference_value - candidate_value).abs();
                count += 1;
            }
        }
        (sum, count)
    })?;

    let (sum, count) = partials
        .into_iter()
        .fold((0.0f64, 0u64), |(sum, count), (part, part_count)| {
            (sum + part, count + part_count)
        });
    Ok(mean(sum, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_calculated_example() {
        let reference = [1.0f64, 2.0, 3.0, 4.0];
        let candidate = [1.0f64, 4.0, 2.0, 8.0];
        // |0| + |2| + |1| + |4| = 7, divided by 4.
        assert_eq!(
            mean_absolute_error(&reference, &candidate, &ParallelConfig::default()).unwrap(),
            Some(1.75)
        );
    }

    #[test]
    fn negative_values_use_absolute_difference() {
        let reference = [-5.0f32, -1.0];
        let candidate = [-2.0f32, -1.0];
        assert_eq!(
            mean_absolute_error(&reference, &candidate, &ParallelConfig::default()).unwrap(),
            Some(1.5)
        );
    }

    #[test]
    fn nonfinite_pairs_are_excluded() {
        let reference = [1.0f64, f64::NAN, 3.0, f64::INFINITY];
        let candidate = [2.0f64, 1.0, 3.0, 4.0];
        assert_eq!(
            mean_absolute_error(&reference, &candidate, &ParallelConfig::default()).unwrap(),
            Some(0.5)
        );
    }

    #[test]
    fn all_nonfinite_returns_none() {
        let reference = [f64::NAN, f64::INFINITY];
        let candidate = [1.0f64, 2.0];
        assert_eq!(
            mean_absolute_error(&reference, &candidate, &ParallelConfig::default()).unwrap(),
            None
        );
    }

    #[test]
    fn empty_returns_none() {
        let empty: [f64; 0] = [];
        assert_eq!(
            mean_absolute_error(&empty, &empty, &ParallelConfig::default()).unwrap(),
            None
        );
    }

    #[test]
    fn parallel_matches_sequential_within_tolerance() {
        let reference: Vec<f64> = (0..100_003).map(|value| value as f64 * 0.25).collect();
        let candidate: Vec<f64> = reference.iter().map(|value| value + 0.125).collect();
        let sequential = ParallelConfig {
            enabled: false,
            ..ParallelConfig::default()
        };
        let parallel = ParallelConfig {
            enabled: true,
            threshold: 1,
            chunk_size: 997,
            threads: Some(4),
        };
        let expected = mean_absolute_error(&reference, &candidate, &sequential)
            .unwrap()
            .unwrap();
        let actual = mean_absolute_error(&reference, &candidate, &parallel)
            .unwrap()
            .unwrap();
        assert!((actual - expected).abs() < 1e-9);
    }
}
