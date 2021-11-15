pub mod conditioned;
pub mod dependent_joint;
pub mod discrete_posterior;
pub mod independent_array_joint;
pub mod independent_joint;
pub mod instant;
pub mod samples;
pub mod switched;

pub use conditioned::*;
pub use dependent_joint::*;
pub use discrete_posterior::*;
pub use independent_array_joint::*;
pub use independent_joint::*;
pub use instant::*;
pub use samples::*;
pub use switched::*;

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

/// The trait which all structs of distribution must implement.
/// - `fk`: The kernel part of probability density function `f`. The kernel means that it doesn't need normalization term of probability density function.
pub trait Distribution: Clone + Debug + Send + Sync {
    type Value: RandomVariable;
    type Condition: RandomVariable;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError>;
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError>;
}

pub trait DiscreteDistribution: Distribution {
    fn fm(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        self.fk(x, theta)
    }
}
