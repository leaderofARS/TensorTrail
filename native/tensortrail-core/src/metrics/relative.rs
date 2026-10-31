//! Relative error with an explicit epsilon floor.

use crate::error::Result;
use crate::metrics::{Element, RelativeError};
use crate::parallel::{map_chunks, ParallelConfig};
use crate::validation::{validate_metric_options, validate_pair_lengths};

/// Compute mean and maximum relative error.
///
/// The per-element definition is:
///
/// ```text
/// |reference - candidate| / max(|reference|, epsilon)
/// ```
///
/// The epsilon floor prevents division by zero from producing infinity. A pair
/// of zeros therefore contributes zero error, while a zero reference and a
/// non-zero candidate is measured against `epsilon` and can produce a very
/// large (but finite, unless it overflows) value. Non-finite pairs are excluded.
pub fn relative_error<T: Element>(
    reference: &[T],
    candidate: &[T],
    epsilon: f64,
    config: &ParallelConfig,
) -> Result<RelativeError> {
    validate_metric_options(epsilon, 0.0, 0.0)?;
    let length = validate_pair_lengths(reference.len(), candidate.len())?;
    if length == 0 {
        return Ok(RelativeError {
            mean: None,
            max: None,
        });
    }

    let (partials, _) = map_chunks(length, config, |start, end| {
        let mut sum = 0.0f64;
        let mut maximum = 0.0f64;
        let mut count = 0u64;
        for index in start..end {
            let reference_value = reference[index].to_f64();
            let candidate_value = candidate[index].to_f64();
            if reference_value.is_finite() && candidate_value.is_finite() {
                let absolute = (reference_value - candidate_value).abs();
                let value = absolute / reference_value.abs().max(epsilon);
                sum += value;
                if value > maximum {
                    maximum = value;
                }
                count += 1;
            }
        }
        (sum, maximum, count)
    })?;

    let (sum, maximum, count) =
        partials
            .into_iter()
            .fold((0.0f64, 0.0f64, 0u64), |accumulator, part| {
                (
                    accumulator.0 + part.0,
                    accumulator.1.max(part.1),
                    accumulator.2 + part.2,
                )
            });

    if count == 0 {
        return Ok(RelativeError {
            mean: None,
            max: None,
        });
    }

    Ok(RelativeError {
        mean: Some(sum / count as f64),
        max: Some(maximum),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_calculated_example() {
        let reference = [1.0f64, 2.0, 4.0];
        let candidate = [1.5f64, 1.0, 4.0];
        // 0.5 / 1 + 1 / 2 + 0 / 4 = 1.0; mean = 1/3; max = 0.5
        let error =
            relative_error(&reference, &candidate, 1e-12, &ParallelConfig::default()).unwrap();
        assert!((error.mean.unwrap() - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(error.max, Some(0.5));
    }

    #[test]
    fn zero_reference_uses_epsilon_floor() {
        let reference = [0.0f64];
        let candidate = [1e-6f64];
        let error =
            relative_error(&reference, &candidate, 1e-12, &ParallelConfig::default()).unwrap();
        assert!((error.mean.unwrap() - 1e6).abs() < 0.5);
        assert!(error.mean.unwrap().is_finite());
    }

    #[test]
    fn both_zero_has_zero_error() {
        let reference = [0.0f32];
        let candidate = [0.0f32];
        let error =
            relative_error(&reference, &candidate, 1e-12, &ParallelConfig::default()).unwrap();
        assert_eq!(error.mean, Some(0.0));
        assert_eq!(error.max, Some(0.0));
    }

    #[test]
    fn invalid_epsilon_is_rejected() {
        let reference = [1.0f64];
        assert!(relative_error(&reference, &reference, 0.0, &ParallelConfig::default()).is_err());
        assert!(relative_error(&reference, &reference, -1.0, &ParallelConfig::default()).is_err());
        assert!(
            relative_error(&reference, &reference, f64::NAN, &ParallelConfig::default()).is_err()
        );
    }

    #[test]
    fn empty_and_nonfinite_return_none() {
        let empty: [f64; 0] = [];
        assert_eq!(
            relative_error(&empty, &empty, 1e-12, &ParallelConfig::default()).unwrap(),
            RelativeError {
                mean: None,
                max: None
            }
        );
        assert_eq!(
            relative_error(
                &[f64::NAN, f64::INFINITY],
                &[1.0, 2.0],
                1e-12,
                &ParallelConfig::default()
            )
            .unwrap()
            .mean,
            None
        );
    }
}
