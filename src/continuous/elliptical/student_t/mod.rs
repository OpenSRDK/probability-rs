pub mod multivariate;
pub mod univariate;

pub use multivariate::*;
pub use univariate::*;

#[derive(thiserror::Error, Debug)]
pub enum StudentTError {
    #[error("'σ' must be positive")]
    SigmaMustBePositive,
}
