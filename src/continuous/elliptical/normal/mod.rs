pub mod multivariate;
pub mod multivariate_params;
pub mod univariate;
pub mod univariate_params;

pub use multivariate::*;
pub use multivariate_params::*;
pub use univariate::*;
pub use univariate_params::*;

#[derive(thiserror::Error, Debug)]
pub enum NormalError {
    #[error("'σ' must be positive")]
    SigmaMustBePositive,
}
