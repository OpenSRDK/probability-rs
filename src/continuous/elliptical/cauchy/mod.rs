pub mod multivariate;
pub mod univariate;
pub mod multivariate_params;
pub mod univariate_params;

pub use multivariate_params::*;
pub use univariate_params::*;
pub use multivariate::*;
pub use univariate::*;

#[derive(thiserror::Error, Debug)]
pub enum CauchyError {
    #[error("'σ' must be positive")]
    SigmaMustBePositive,
}
