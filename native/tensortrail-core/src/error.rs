//! Structured errors returned by the native engine.
//!
//! The native layer deliberately avoids panicking for malformed user input.
//! Bindings translate these variants into Python exceptions; the Rust API
//! returns them directly.

use std::fmt;

/// Errors produced by TensorTrail's native numerical engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Reference and candidate flat lengths differ.
    LengthMismatch { reference: usize, candidate: usize },
    /// Reference and candidate shapes differ.
    ShapeMismatch {
        reference: Vec<usize>,
        candidate: Vec<usize>,
    },
    /// A supplied shape describes a number of elements different from the buffer.
    InvalidShapeSize { shape: Vec<usize>, length: usize },
    /// The product of a shape's dimensions exceeds `usize::MAX`.
    ShapeSizeOverflow,
    /// A configuration value is outside its documented domain.
    InvalidConfiguration(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                reference,
                candidate,
            } => write!(
                f,
                "reference and candidate have different numbers of elements \
                 ({reference} vs {candidate}); shapes are not interchangeable"
            ),
            Self::ShapeMismatch {
                reference,
                candidate,
            } => write!(
                f,
                "reference shape {reference:?} does not match candidate shape {candidate:?}"
            ),
            Self::InvalidShapeSize { shape, length } => write!(
                f,
                "shape {shape:?} describes a different number of elements than the {length}-element buffer"
            ),
            Self::ShapeSizeOverflow => {
                write!(f, "shape element count overflows the platform usize type")
            }
            Self::InvalidConfiguration(message) => write!(f, "invalid configuration: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// Convenience result type for the native engine.
pub type Result<T> = std::result::Result<T, Error>;
