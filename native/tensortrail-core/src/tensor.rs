//! Tensor metadata and shape validation.
//!
//! Numerical kernels operate on flat, contiguous slices. This module keeps the
//! contract between logical tensor shapes and those slices explicit so a buffer
//! can never be silently reinterpreted. NumPy's C-contiguity conversion is
//! performed (and recorded) by the Python layer; the native layer rejects
//! anything that is not already a contiguous slice.

use crate::error::{Error, Result};

/// Storage dtype supported directly by the native engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    /// Stable string used in reports and binding metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }
}

/// Number of elements described by a shape. A zero-dimensional shape (a scalar)
/// contains exactly one element.
pub fn checked_element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |accumulator, &dimension| {
        accumulator
            .checked_mul(dimension)
            .ok_or(Error::ShapeSizeOverflow)
    })
}

/// A validated borrowed view over contiguous tensor data.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a, T> {
    shape: &'a [usize],
    data: &'a [T],
    dtype: DType,
}

impl<'a, T> TensorView<'a, T> {
    /// Construct a view after checking that `shape` matches `data.len()`.
    pub fn new(shape: &'a [usize], data: &'a [T], dtype: DType) -> Result<Self> {
        let expected = checked_element_count(shape)?;
        if expected != data.len() {
            return Err(Error::InvalidShapeSize {
                shape: shape.to_vec(),
                length: data.len(),
            });
        }
        Ok(Self { shape, data, dtype })
    }

    /// Logical shape.
    pub const fn shape(&self) -> &'a [usize] {
        self.shape
    }

    /// Number of elements.
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the view contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Directly supported dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Borrowed contiguous storage.
    pub const fn data(&self) -> &'a [T] {
        self.data
    }
}

/// Verify that reference and candidate shapes are identical.
pub fn validate_matching_shapes(reference: &[usize], candidate: &[usize]) -> Result<()> {
    if reference == candidate {
        Ok(())
    } else {
        Err(Error::ShapeMismatch {
            reference: reference.to_vec(),
            candidate: candidate.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_shape_has_one_element() {
        assert_eq!(checked_element_count(&[]).unwrap(), 1);
    }

    #[test]
    fn empty_dimension_has_zero_elements() {
        assert_eq!(checked_element_count(&[4, 0, 8]).unwrap(), 0);
    }

    #[test]
    fn overflow_is_reported() {
        assert_eq!(
            checked_element_count(&[usize::MAX, 2]),
            Err(Error::ShapeSizeOverflow)
        );
    }

    #[test]
    fn view_rejects_shape_length_mismatch() {
        let data = [1.0f32, 2.0];
        assert_eq!(
            TensorView::new(&[3], &data, DType::F32).unwrap_err(),
            Error::InvalidShapeSize {
                shape: vec![3],
                length: 2
            }
        );
    }
}
