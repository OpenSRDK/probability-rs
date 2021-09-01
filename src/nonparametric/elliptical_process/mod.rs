pub mod exact;
pub mod ey;
pub mod kernel_matrix;
pub mod kiss_love;
pub mod regressor;
pub mod sparse;

use crate::{DistributionError, EllipticalParams, RandomVariable};
pub use exact::*;
pub use ey::*;
pub use kernel_matrix::*;
pub use kiss_love::*;
use opensrdk_kernel_method::Kernel;
pub use regressor::*;
pub use sparse::*;
use std::fmt::Debug;

#[derive(thiserror::Error, Debug)]
pub enum EllipticalProcessError {
    #[error("Data is empty.")]
    Empty,
    #[error("Dimension mismatch.")]
    DimensionMismatch,
    #[error("NaN contaminated.")]
    NaNContamination,
}

#[derive(Clone, Debug)]
pub struct BaseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    kernel: K,
    x: Vec<T>,
    theta: Vec<f64>,
    sigma: f64,
}

impl<K, T> BaseEllipticalProcessParams<K, T>
where
    K: Kernel<T>,
    T: RandomVariable,
{
    pub fn new(
        kernel: K,
        x: Vec<T>,
        theta: Vec<f64>,
        sigma: f64,
    ) -> Result<Self, DistributionError> {
        if kernel.params_len() != theta.len() {
            return Err(EllipticalProcessError::DimensionMismatch);
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
    K: Kernel<T>,
    T: RandomVariable,
{
    fn mahalanobis_squared(&self) -> f64;
}
