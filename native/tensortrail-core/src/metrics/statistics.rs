//! Numerically stable tensor statistics.
//!
//! Statistics are accumulated with Welford's online algorithm, which avoids the
//! catastrophic cancellation of `sum(x^2) - sum(x)^2 / n`. Partial Welford
//! states from fixed chunks are merged in input order, so results are stable
//! across worker counts and scheduling.

use crate::error::Result;
use crate::metrics::{Element, NonFiniteCounts};
use crate::parallel::{map_chunks, ParallelConfig};

/// Summary statistics for one tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Statistics {
    /// Total number of elements, including non-finite values.
    pub elements: u64,
    /// Number of finite elements included in the statistics.
    pub finite_elements: u64,
    pub nan_count: u64,
    pub positive_infinity_count: u64,
    pub negative_infinity_count: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    /// Population variance (denominator `n`).
    pub variance: Option<f64>,
    pub std_dev: Option<f64>,
}

impl Statistics {
    /// Whether any non-finite value was observed.
    pub const fn has_nonfinite(&self) -> bool {
        self.nan_count + self.positive_infinity_count + self.negative_infinity_count > 0
    }
}

/// Streaming Welford accumulator for one tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StatisticsAccumulator {
    pub elements: u64,
    pub finite_elements: u64,
    pub nonfinite: NonFiniteCounts,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl StatisticsAccumulator {
    /// Add one element.
    #[inline]
    pub fn push<T: Element>(&mut self, value: T) {
        let value = value.to_f64();
        self.elements += 1;

        if !value.is_finite() {
            self.nonfinite.observe(value);
            return;
        }

        self.finite_elements += 1;
        if self.finite_elements == 1 {
            self.mean = value;
            self.m2 = 0.0;
            self.min = value;
            self.max = value;
            return;
        }

        let delta = value - self.mean;
        self.mean += delta / self.finite_elements as f64;
        self.m2 += delta * (value - self.mean);
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Merge another Welford state (Chan et al. parallel update).
    pub fn merge(&mut self, other: Self) {
        self.nonfinite.merge(other.nonfinite);
        self.elements += other.elements;

        if other.finite_elements == 0 {
            return;
        }
        if self.finite_elements == 0 {
            self.finite_elements = other.finite_elements;
            self.mean = other.mean;
            self.m2 = other.m2;
            self.min = other.min;
            self.max = other.max;
            return;
        }

        let total = self.finite_elements + other.finite_elements;
        let total_f64 = total as f64;
        let delta = other.mean - self.mean;
        let weight = other.finite_elements as f64 / total_f64;
        self.mean += delta * weight;
        self.m2 += other.m2 + delta * delta * (self.finite_elements as f64) * weight;
        self.finite_elements = total;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }

    /// Finish the reduction and derive variance and standard deviation.
    pub fn finish(&self) -> Statistics {
        let (min, max, mean, variance, std_dev) = if self.finite_elements == 0 {
            (None, None, None, None, None)
        } else {
            let variance = if self.m2 < 0.0 { 0.0 } else { self.m2 };
            let population_variance = variance / self.finite_elements as f64;
            (
                Some(self.min),
                Some(self.max),
                Some(self.mean),
                Some(population_variance),
                Some(population_variance.sqrt()),
            )
        };

        Statistics {
            elements: self.elements,
            finite_elements: self.finite_elements,
            nan_count: self.nonfinite.nan,
            positive_infinity_count: self.nonfinite.positive_infinity,
            negative_infinity_count: self.nonfinite.negative_infinity,
            min,
            max,
            mean,
            variance,
            std_dev,
        }
    }
}

/// Compute statistics for `values` using deterministic chunked Welford
/// reductions. Non-finite values are counted but excluded from min/max/mean/
/// variance.
pub fn statistics<T: Element>(values: &[T], config: &ParallelConfig) -> Result<Statistics> {
    if values.is_empty() {
        return Ok(StatisticsAccumulator::default().finish());
    }

    let (partials, _) = map_chunks(values.len(), config, |start, end| {
        let mut accumulator = StatisticsAccumulator::default();
        for value in &values[start..end] {
            accumulator.push(*value);
        }
        accumulator
    })?;

    let mut merged = StatisticsAccumulator::default();
    for partial in partials {
        merged.merge(partial);
    }
    Ok(merged.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_calculated_population_statistics() {
        let values = [1.0f64, 2.0, 3.0, 4.0];
        let stats = statistics(&values, &ParallelConfig::default()).unwrap();
        assert_eq!(stats.elements, 4);
        assert_eq!(stats.finite_elements, 4);
        assert_eq!(stats.min, Some(1.0));
        assert_eq!(stats.max, Some(4.0));
        assert_eq!(stats.mean, Some(2.5));
        assert_eq!(stats.variance, Some(1.25));
        assert!((stats.std_dev.unwrap() - 1.25f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn single_element_has_zero_variance() {
        let stats = statistics(&[7.5f32], &ParallelConfig::default()).unwrap();
        assert_eq!(stats.mean, Some(7.5));
        assert_eq!(stats.variance, Some(0.0));
        assert_eq!(stats.std_dev, Some(0.0));
    }

    #[test]
    fn nonfinite_values_are_counted_and_excluded() {
        let values = [1.0f64, f64::NAN, 3.0, f64::INFINITY, f64::NEG_INFINITY];
        let stats = statistics(&values, &ParallelConfig::default()).unwrap();
        assert_eq!(stats.elements, 5);
        assert_eq!(stats.finite_elements, 2);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.positive_infinity_count, 1);
        assert_eq!(stats.negative_infinity_count, 1);
        assert_eq!(stats.mean, Some(2.0));
        assert_eq!(stats.variance, Some(1.0));
        assert!(stats.has_nonfinite());
    }

    #[test]
    fn all_nonfinite_has_no_numeric_statistics() {
        let stats = statistics(&[f64::NAN, f64::INFINITY], &ParallelConfig::default()).unwrap();
        assert_eq!(stats.finite_elements, 0);
        assert_eq!(stats.mean, None);
        assert_eq!(stats.variance, None);
        assert_eq!(stats.std_dev, None);
    }

    #[test]
    fn empty_statistics_are_explicitly_empty() {
        let empty: [f64; 0] = [];
        let stats = statistics(&empty, &ParallelConfig::default()).unwrap();
        assert_eq!(stats.elements, 0);
        assert_eq!(stats.mean, None);
        assert_eq!(stats.min, None);
    }

    #[test]
    fn stable_variance_for_large_offset() {
        // The naive sum-of-squares formula loses all precision here.
        let values = [1.0e9 + 1.0f64, 1.0e9 + 2.0, 1.0e9 + 3.0];
        let stats = statistics(&values, &ParallelConfig::default()).unwrap();
        assert!((stats.variance.unwrap() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn parallel_merge_matches_sequential_within_tolerance() {
        let values: Vec<f64> = (0..100_003)
            .map(|index| (index as f64 * 0.001).sin() + 1.0e6)
            .collect();
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
        let expected = statistics(&values, &sequential).unwrap();
        let actual = statistics(&values, &parallel).unwrap();
        let mean_tolerance = expected.mean.unwrap().abs() * 1e-12 + 1e-9;
        assert!((actual.mean.unwrap() - expected.mean.unwrap()).abs() < mean_tolerance);
        assert!(
            (actual.variance.unwrap() - expected.variance.unwrap()).abs()
                < expected.variance.unwrap().abs() * 1e-10 + 1e-9
        );
    }
}
