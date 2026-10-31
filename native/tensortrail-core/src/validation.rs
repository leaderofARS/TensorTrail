//! Validation shared by numerical kernels.

use crate::error::{Error, Result};

/// Verify that two flat buffers contain the same number of elements.
pub fn validate_pair_lengths(reference: usize, candidate: usize) -> Result<usize> {
    if reference == candidate {
        Ok(reference)
    } else {
        Err(Error::LengthMismatch {
            reference,
            candidate,
        })
    }
}

/// Validate the numerical options used by generic kernels.
pub fn validate_metric_options(epsilon: f64, atol: f64, rtol: f64) -> Result<()> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidConfiguration(format!(
            "relative_error epsilon must be finite and > 0, received {epsilon}"
        )));
    }
    if !atol.is_finite() || atol < 0.0 {
        return Err(Error::InvalidConfiguration(format!(
            "atol must be finite and >= 0, received {atol}"
        )));
    }
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(Error::InvalidConfiguration(format!(
            "rtol must be finite and >= 0, received {rtol}"
        )));
    }
    Ok(())
}
