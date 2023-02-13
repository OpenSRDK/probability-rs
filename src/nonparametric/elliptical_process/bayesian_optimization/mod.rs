pub mod expected_improvement;
pub mod upper_confidence_bound;

pub use expected_improvement::*;
pub use upper_confidence_bound::*;

use crate::NormalParams;

pub trait AcquisitionFunctions {
    fn value(&self, theta: &NormalParams) -> f64;
}
