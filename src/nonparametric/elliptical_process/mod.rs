pub mod exact;
pub mod ey;
pub mod gaussian_process;
pub mod kernel_matrix;
pub mod kiss_love;
pub mod params_diff_kernel_matrix;
pub mod regressor;
pub mod sparse;

pub use exact::*;
pub use ey::*;
pub use gaussian_process::*;
pub use kernel_matrix::*;
pub use kiss_love::*;
pub use params_diff_kernel_matrix::*;
pub use regressor::*;
pub use sparse::*;

use crate::{DistributionError, EllipticalParams, RandomVariable};
use opensrdk_kernel_method::*;

#[derive(thiserror::Error, Debug)]
pub enum EllipticalProcessError {
    #[error("Data is empty.")]
    Empty,
    #[error("Dimension mismatch.")]
    DimensionMismatch,
    #[error("NaN contaminated.")]
    NaNContamination,
}

/// You can use these methods:
/// - `exact`
/// - `sparse`
/// - `kiss_love`
#[derive(Clone, Debug)]
pub struct BaseEllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    kernel: K,
    x: Vec<T>,
    theta: Vec<f64>,
    sigma: f64,
}

impl<K, T> BaseEllipticalProcessParams<K, T>
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    /// - `kernel`: Kernel function
    /// - `x`: Input value
    /// - `theta`: Params of kernel function
    /// - `sigma`: White noise variance for a regularization likes Ridge regression
    pub fn new(
        kernel: K,
        x: Vec<T>,
        theta: Vec<f64>,
        sigma: f64,
    ) -> Result<Self, DistributionError> {
        if kernel.params_len() != theta.len() {
            return Err(DistributionError::InvalidParameters(
                EllipticalProcessError::DimensionMismatch.into(),
            ));
        }

        Ok(Self {
            kernel,
            x,
            theta,
            sigma,
        })
    }
}

pub trait EllipticalProcessParams<K, T>: EllipticalParams
where
    K: PositiveDefiniteKernel<T>,
    T: RandomVariable,
{
    fn mahalanobis_squared(&self) -> f64;
}
