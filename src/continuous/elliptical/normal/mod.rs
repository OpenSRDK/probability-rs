pub mod multivariate_params;
pub mod multivatiate_normal;
pub mod univariate;
pub mod univariate_params;

pub use multivariate_params::*;
pub use multivatiate_normal::*;
pub use univariate::*;
pub use univariate_params::*;

#[derive(thiserror::Error, Debug)]
pub enum NormalError {
    #[error("'σ' must be positive")]
    SigmaMustBePositive,
}
