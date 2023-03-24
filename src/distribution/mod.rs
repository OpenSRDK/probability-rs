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
pub mod joint_array_distribution;
pub mod joint_distribution;
pub mod random_variable;
pub mod samplable;
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
pub use joint_array_distribution::*;
pub use joint_distribution::*;
use opensrdk_symbolic_computation::Expression;
pub use random_variable::*;
pub use samplable::*;
pub use switched::*;
pub use transformed::*;
pub use valued::*;

use opensrdk_kernel_method::KernelError;
use opensrdk_linear_algebra::MatrixError;
use rand::prelude::*;
use std::{error::Error, fmt::Debug};

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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
pub trait Distribution: Clone + Debug + Serialize {
    fn value_ids(&self) -> HashSet<&str>;

    fn conditions(&self) -> Vec<&Expression>;

    fn condition_ids(&self) -> HashSet<&str> {
        self.conditions()
            .iter()
            .map(|v| v.variable_ids())
            .flatten()
            .collect::<HashSet<_>>()
            .difference(&self.value_ids())
            .cloned()
            .collect()
    }

    fn pdf(&self) -> Expression;

    fn ln_pdf(&self) -> Expression {
        self.pdf().ln()
    }
}
pub trait ContinuousDistribution: Distribution {
    fn value_ids(&self) -> HashSet<&str>;

    fn conditions(&self) -> Vec<&Expression>;

    fn condition_ids(&self) -> HashSet<&str> {
        self.conditions()
            .iter()
            .map(|v| v.variable_ids())
            .flatten()
            .collect::<HashSet<_>>()
            .difference(&self.value_ids())
            .cloned()
            .collect()
    }

    fn pdf(&self) -> Expression;

    fn ln_pdf(&self) -> Expression {
        self.pdf().ln()
    }
}

pub trait DiscreteDistribution: Distribution {
    fn fm(&self) -> Expression {
        self.pdf()
    }
}
