pub mod converted;
pub mod dependent_joint;
pub mod independent_array_joint;
pub mod independent_joint;
pub mod instant;
pub mod variational_inference;

pub use converted::*;
pub use dependent_joint::*;
pub use independent_array_joint::*;
pub use independent_joint::*;
pub use instant::*;
pub use variational_inference::*;

use opensrdk_kernel_method::KernelError;
use opensrdk_linear_algebra::MatrixError;
use rand::prelude::*;
use std::{error::Error, fmt::Debug};

pub trait RandomVariable: Clone + Debug + Send + Sync {}
impl<T> RandomVariable for T where T: Clone + Debug + Send + Sync {}

#[derive(thiserror::Error, Debug)]
pub enum DistributionError {
    #[error("Invalid parameters")]
    InvalidParameters(Box<dyn Error + Send + Sync>),
    #[error("Matrix error")]
    MatrixError(MatrixError),
    #[error("Kernel error")]
    KernelError(KernelError),
    #[error("Others")]
    Others(Box<dyn Error + Send + Sync>),
}

impl From<MatrixError> for DistributionError {
    fn from(e: MatrixError) -> Self {
        Self::MatrixError(e)
    }
}

impl From<KernelError> for DistributionError {
    fn from(e: KernelError) -> Self {
        Self::KernelError(e)
    }
}

impl From<Box<dyn Error + Send + Sync>> for DistributionError {
    fn from(e: Box<dyn Error + Send + Sync>) -> Self {
        Self::Others(e)
    }
}

/// # Distribution
/// ![tex](https://latex.codecogs.com/svg.latex?p%28x%7C\mathbf{\theta}%29)
pub trait Distribution: Clone + Debug + Send + Sync {
    type T: RandomVariable;
    type U: RandomVariable;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError>;
    fn ln_p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        Ok(self.p(x, theta)?.ln())
    }
    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError>;
}
