pub mod condition_mapped;
pub mod conditionalize_latent;
pub mod continuous_samples;
pub mod degenerate;
pub mod dependent_joint;
pub mod differentiable;
pub mod discrete_posterior;
pub mod discrete_samples;
pub mod event;
pub mod independent_array_joint;
pub mod independent_joint;
pub mod independent_value_array_joint;
pub mod instant;
pub mod instant_condition_differentiable;
pub mod instant_value_differentiable;
pub mod random_variable;
pub mod sampleable;
pub mod switched;
pub mod transformed;
pub mod valued;

pub use condition_mapped::*;
pub use conditionalize_latent::*;
pub use continuous_samples::*;
pub use degenerate::*;
pub use dependent_joint::*;
pub use differentiable::*;
pub use discrete_posterior::*;
pub use discrete_samples::*;
pub use event::*;
pub use independent_array_joint::*;
pub use independent_joint::*;
pub use independent_value_array_joint::*;
pub use instant::*;
pub use random_variable::*;
pub use sampleable::*;
pub use switched::*;
pub use transformed::*;
pub use valued::*;

use opensrdk_kernel_method::KernelError;
use opensrdk_linear_algebra::MatrixError;
use rand::prelude::*;
use std::{error::Error, fmt::Debug};

#[derive(thiserror::Error, Debug)]
pub enum DistributionError {
    #[error("Invalid parameters")]
    InvalidParameters(Box<dyn Error + Send + Sync>),
    #[error("Matrix error")]
    MatrixError(MatrixError),
    #[error("Kernel error")]
    KernelError(KernelError),
    #[error("Invalid restore vector")]
    InvalidRestoreVector,
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
    type Condition: Clone + Debug + Send + Sync;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError>;
    // fn sample(
    //     &self,
    //     theta: &Self::Condition,
    //     rng: &mut dyn RngCore,
    // ) -> Result<Self::Value, DistributionError>;
}

pub trait DiscreteDistribution: Distribution {
    fn fm(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        self.p_kernel(x, theta)
    }
}
