pub mod cauchy;
pub mod normal;
pub mod params;
pub mod student_t;

pub use cauchy::*;
pub use normal::*;
pub use params::*;
pub use student_t::*;

#[derive(thiserror::Error, Debug)]
pub enum EllipticalError {
    #[error("dimension mismatch (Ellptical)")]
    DimensionMismatch,
}
