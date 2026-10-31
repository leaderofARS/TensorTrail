//! Cosine similarity with explicit zero-vector and magnitude policies.

use crate::error::Result;
use crate::metrics::Element;
use crate::parallel::{map_chunks, ParallelConfig};
use crate::validation::validate_pair_lengths;

struct Terms {
    dot: f64,
    reference_squared_norm: f64,
    candidate_squared_norm: f64,
    reference_max_abs: f64,
    candidate_max_abs: f64,
    count: u64,
}

fn fast_terms<T: Element>(
    reference: &[T],
    candidate: &[T],
    config: &ParallelConfig,
) -> Result<Terms> {
    let length = validate_pair_lengths(reference.len(), candidate.len())?;
    let (partials, _) = map_chunks(length, config, |start, end| {
        let mut terms = Terms {
            dot: 0.0,
            reference_squared_norm: 0.0,
            candidate_squared_norm: 0.0,
            reference_max_abs: 0.0,
            candidate_max_abs: 0.0,
            count: 0,
        };
        for index in start..end {
            let reference_value = reference[index].to_f64();
            let candidate_value = candidate[index].to_f64();
            if reference_value.is_finite() && candidate_value.is_finite() {
                terms.dot += reference_value * candidate_value;
                terms.reference_squared_norm += reference_value * reference_value;
                terms.candidate_squared_norm += candidate_value * candidate_value;
                terms.reference_max_abs = terms.reference_max_abs.max(reference_value.abs());
                terms.candidate_max_abs = terms.candidate_max_abs.max(candidate_value.abs());
                terms.count += 1;
            }
        }
        terms
    })?;

    let mut merged = Terms {
        dot: 0.0,
        reference_squared_norm: 0.0,
        candidate_squared_norm: 0.0,
        reference_max_abs: 0.0,
        candidate_max_abs: 0.0,
        count: 0,
    };
    for part in partials {
        merged.dot += part.dot;
        merged.reference_squared_norm += part.reference_squared_norm;
        merged.candidate_squared_norm += part.candidate_squared_norm;
        merged.reference_max_abs = merged.reference_max_abs.max(part.reference_max_abs);
        merged.candidate_max_abs = merged.candidate_max_abs.max(part.candidate_max_abs);
        merged.count += part.count;
    }
    Ok(merged)
}

fn scaled_maxima<T: Element>(
    reference: &[T],
    candidate: &[T],
    config: &ParallelConfig,
) -> Result<(f64, f64, u64)> {
    let (partials, _) = map_chunks(reference.len(), config, |start, end| {
        let mut reference_maximum = 0.0f64;
        let mut candidate_maximum = 0.0f64;
        let mut count = 0u64;
        for index in start..end {
            let reference_value = reference[index].to_f64();
            let candidate_value = candidate[index].to_f64();
            if reference_value.is_finite() && candidate_value.is_finite() {
                reference_maximum = reference_maximum.max(reference_value.abs());
                candidate_maximum = candidate_maximum.max(candidate_value.abs());
                count += 1;
            }
        }
        (reference_maximum, candidate_maximum, count)
    })?;
    Ok(partials.into_iter().fold(
        (0.0f64, 0.0f64, 0u64),
        |(reference_maximum, candidate_maximum, count), part| {
            (
                reference_maximum.max(part.0),
                candidate_maximum.max(part.1),
                count + part.2,
            )
        },
    ))
}

/// Scaled two-pass cosine similarity used when squared norms overflow or
/// underflow. Normalizing by each vector's maximum absolute finite value keeps
/// intermediate products bounded by one.
///
/// The fallback runs only for extreme-magnitude `float64` data; it costs a
/// second and third traversal and is not used for ordinary inputs.
pub fn stable_cosine_similarity<T: Element>(
    reference: &[T],
    candidate: &[T],
    config: &ParallelConfig,
) -> Result<Option<f64>> {
    let length = validate_pair_lengths(reference.len(), candidate.len())?;
    if length == 0 {
        return Ok(None);
    }

    let (reference_max, candidate_max, finite_count) = scaled_maxima(reference, candidate, config)?;

    if finite_count == 0 {
        return Ok(None);
    }
    if reference_max == 0.0 && candidate_max == 0.0 {
        return Ok(Some(1.0));
    }
    if reference_max == 0.0 || candidate_max == 0.0 {
        return Ok(Some(0.0));
    }

    let (partials, _) = map_chunks(length, config, |start, end| {
        let mut dot = 0.0f64;
        let mut reference_norm = 0.0f64;
        let mut candidate_norm = 0.0f64;
        let mut count = 0u64;
        for index in start..end {
            let reference_value = reference[index].to_f64();
            let candidate_value = candidate[index].to_f64();
            if reference_value.is_finite() && candidate_value.is_finite() {
                let normalized_reference = reference_value / reference_max;
                let normalized_candidate = candidate_value / candidate_max;
                dot += normalized_reference * normalized_candidate;
                reference_norm += normalized_reference * normalized_reference;
                candidate_norm += normalized_candidate * normalized_candidate;
                count += 1;
            }
        }
        (dot, reference_norm, candidate_norm, count)
    })?;

    let (dot, reference_norm, candidate_norm, count) = partials.into_iter().fold(
        (0.0f64, 0.0f64, 0.0f64, 0u64),
        |(dot, reference_norm, candidate_norm, count), part| {
            (
                dot + part.0,
                reference_norm + part.1,
                candidate_norm + part.2,
                count + part.3,
            )
        },
    );

    if count == 0 || reference_norm == 0.0 || candidate_norm == 0.0 {
        return Ok(None);
    }

    let cosine = dot / (reference_norm.sqrt() * candidate_norm.sqrt());
    Ok(Some(if cosine.is_nan() {
        f64::NAN
    } else {
        cosine.clamp(-1.0, 1.0)
    }))
}

/// Compute cosine similarity between finite reference/candidate pairs.
///
/// Policy:
/// * both compared vectors are zero: `1.0`;
/// * exactly one vector is zero: `0.0`;
/// * no finite pairs: `None`;
/// * ordinary inputs: the standard dot-product formula, clamped to `[-1, 1]`
///   to absorb rounding error;
/// * extreme `float64` magnitudes that overflow or underflow squared norms:
///   automatically use the scaled two-pass algorithm.
pub fn cosine_similarity<T: Element>(
    reference: &[T],
    candidate: &[T],
    config: &ParallelConfig,
) -> Result<Option<f64>> {
    let terms = fast_terms(reference, candidate, config)?;
    if terms.count == 0 {
        return Ok(None);
    }
    if terms.reference_max_abs == 0.0 && terms.candidate_max_abs == 0.0 {
        return Ok(Some(1.0));
    }
    if terms.reference_max_abs == 0.0 || terms.candidate_max_abs == 0.0 {
        return Ok(Some(0.0));
    }

    let norm_product = terms.reference_squared_norm * terms.candidate_squared_norm;
    let usable = terms.reference_squared_norm.is_finite()
        && terms.candidate_squared_norm.is_finite()
        && terms.reference_squared_norm > 0.0
        && terms.candidate_squared_norm > 0.0
        && norm_product.is_finite()
        && terms.dot.is_finite();

    if !usable {
        return stable_cosine_similarity(reference, candidate, config);
    }

    let denominator = norm_product.sqrt();
    let cosine = terms.dot / denominator;
    Ok(Some(if cosine.is_nan() {
        f64::NAN
    } else {
        cosine.clamp(-1.0, 1.0)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_vectors() {
        let reference = [1.0f64, 0.0, 0.0];
        let candidate = [0.0f64, 2.0, 0.0];
        assert_eq!(
            cosine_similarity(&reference, &candidate, &ParallelConfig::default()).unwrap(),
            Some(0.0)
        );
    }

    #[test]
    fn opposite_vectors() {
        let reference = [1.0f64, 2.0, 3.0];
        let candidate = [-1.0f64, -2.0, -3.0];
        assert_eq!(
            cosine_similarity(&reference, &candidate, &ParallelConfig::default()).unwrap(),
            Some(-1.0)
        );
    }

    #[test]
    fn hand_calculated_value() {
        let reference = [1.0f64, 2.0, 3.0];
        let candidate = [4.0f64, 5.0, 6.0];
        let expected = 32.0 / (14.0f64.sqrt() * 77.0f64.sqrt());
        let actual = cosine_similarity(&reference, &candidate, &ParallelConfig::default()).unwrap();
        assert!((actual.unwrap() - expected).abs() < 1e-15);
    }

    #[test]
    fn zero_vector_policy() {
        let zeros = [0.0f64, 0.0, 0.0];
        let values = [1.0f64, 2.0, 3.0];
        assert_eq!(
            cosine_similarity(&zeros, &zeros, &ParallelConfig::default()).unwrap(),
            Some(1.0)
        );
        assert_eq!(
            cosine_similarity(&zeros, &values, &ParallelConfig::default()).unwrap(),
            Some(0.0)
        );
        assert_eq!(
            cosine_similarity(&values, &zeros, &ParallelConfig::default()).unwrap(),
            Some(0.0)
        );
    }

    #[test]
    fn nonfinite_pairs_are_excluded() {
        let reference = [1.0f64, f64::NAN, 2.0];
        let candidate = [1.0f64, 3.0, 2.0];
        let cosine = cosine_similarity(&reference, &candidate, &ParallelConfig::default())
            .unwrap()
            .unwrap();
        assert!((cosine - 1.0).abs() < 1e-15);
    }

    #[test]
    fn all_nonfinite_returns_none() {
        assert_eq!(
            cosine_similarity(
                &[f64::NAN, f64::INFINITY],
                &[1.0f64, 2.0],
                &ParallelConfig::default()
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn tiny_values_use_scaled_fallback() {
        let reference = [1e-300f64, 2e-300, 3e-300];
        let candidate = [2e-300f64, 4e-300, 6e-300];
        let cosine = cosine_similarity(&reference, &candidate, &ParallelConfig::default())
            .unwrap()
            .unwrap();
        assert!((cosine - 1.0).abs() < 1e-15);
    }

    #[test]
    fn huge_values_use_scaled_fallback() {
        let reference = [1e308f64, 5e307, 2e307];
        let candidate = [1e308f64, 5e307, 2e307];
        let cosine = cosine_similarity(&reference, &candidate, &ParallelConfig::default())
            .unwrap()
            .unwrap();
        assert!((cosine - 1.0).abs() < 1e-15);
    }

    #[test]
    fn parallel_result_matches_sequential() {
        let reference: Vec<f64> = (0..100_000).map(|value| (value as f64).sin()).collect();
        let candidate: Vec<f64> = reference.iter().map(|value| value * 2.0).collect();
        let sequential = ParallelConfig {
            enabled: false,
            ..ParallelConfig::default()
        };
        let parallel = ParallelConfig {
            enabled: true,
            threshold: 1,
            chunk_size: 1024,
            threads: Some(4),
        };
        let expected = cosine_similarity(&reference, &candidate, &sequential)
            .unwrap()
            .unwrap();
        let actual = cosine_similarity(&reference, &candidate, &parallel)
            .unwrap()
            .unwrap();
        assert!((actual - expected).abs() < 1e-12);
    }
}
