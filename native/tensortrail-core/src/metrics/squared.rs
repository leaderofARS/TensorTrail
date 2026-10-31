//! Mean squared error and root mean squared error.

use crate::error::Result;
use crate::metrics::{mean, Element};
use crate::parallel::{map_chunks, ParallelConfig};
use crate::validation::validate_pair_lengths;

fn squared_error_terms<T: Element>(
    reference: &[T],
    candidate: &[T],
    config: &ParallelConfig,
) -> Result<(f64, u64)> {
    let length = validate_pair_lengths(reference.len(), candidate.len())?;
    if length == 0 {
        return Ok((0.0, 0));
    }

    let (partials, _) = map_chunks(length, config, |start, end| {
        let mut sum = 0.0f64;
        let mut count = 0u64;
        for index in start..end {
            let reference_value = reference[index].to_f64();
            let candidate_value = candidate[index].to_f64();
            if reference_value.is_finite() && candidate_value.is_finite() {
                let difference = reference_value - candidate_value;
                sum += difference * difference;
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
    Ok((sum, count))
}

/// Compute mean squared error over finite reference/candidate pairs.
///
/// Accumulation is performed in `f64`. If the mathematical result exceeds the
/// range of `f64`, infinity is returned rather than wrapping or saturating.
pub fn mean_squared_error<T: Element>(
    reference: &[T],
    candidate: &[T],
    config: &ParallelConfig,
) -> Result<Option<f64>> {
    let (sum, count) = squared_error_terms(reference, candidate, config)?;
    Ok(mean(sum, count))
}

/// Compute root mean squared error over finite reference/candidate pairs.
///
/// This reuses the same squared-error reduction as [`mean_squared_error`]; MSE
/// is never computed in a separate traversal.
pub fn root_mean_squared_error<T: Element>(
    reference: &[T],
    candidate: &[T],
    config: &ParallelConfig,
) -> Result<Option<f64>> {
    Ok(mean_squared_error(reference, candidate, config)?.map(f64::sqrt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_calculated_example() {
        let reference = [1.0f64, 2.0, 3.0];
        let candidate = [2.0f64, 4.0, 6.0];
        // (1 + 4 + 9) / 3 = 14 / 3
        let mse = mean_squared_error(&reference, &candidate, &ParallelConfig::default())
            .unwrap()
            .unwrap();
        assert!((mse - 14.0 / 3.0).abs() < 1e-15);

        let rmse = root_mean_squared_error(&reference, &candidate, &ParallelConfig::default())
            .unwrap()
            .unwrap();
        assert!((rmse - (14.0f64 / 3.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn rmse_equals_sqrt_mse() {
        let reference = [0.0f64, 1.0, 2.0, 3.0];
        let candidate = [1.0f64, 1.5, 1.0, 4.0];
        let mse = mean_squared_error(&reference, &candidate, &ParallelConfig::default())
            .unwrap()
            .unwrap();
        let rmse = root_mean_squared_error(&reference, &candidate, &ParallelConfig::default())
            .unwrap()
            .unwrap();
        assert_eq!(rmse, mse.sqrt());
    }

    #[test]
    fn empty_returns_none() {
        let empty: [f32; 0] = [];
        assert_eq!(
            mean_squared_error(&empty, &empty, &ParallelConfig::default()).unwrap(),
            None
        );
        assert_eq!(
            root_mean_squared_error(&empty, &empty, &ParallelConfig::default()).unwrap(),
            None
        );
    }

    #[test]
    fn large_values_do_not_use_integer_arithmetic() {
        let reference = [1.0e30f64, 2.0e30];
        let candidate = [0.0f64, 0.0];
        let mse = mean_squared_error(&reference, &candidate, &ParallelConfig::default())
            .unwrap()
            .unwrap();
        assert!(mse.is_infinite() || mse > 1.0e59);
    }

    #[test]
    fn parallel_tail_is_covered() {
        let reference: Vec<f32> = (0..10_001).map(|index| index as f32).collect();
        let candidate: Vec<f32> = reference.iter().map(|value| value + 1.0).collect();
        let config = ParallelConfig {
            enabled: true,
            threshold: 1,
            chunk_size: 64,
            threads: Some(3),
        };
        assert_eq!(
            mean_squared_error(&reference, &candidate, &config).unwrap(),
            Some(1.0)
        );
    }
}
