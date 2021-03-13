pub mod converted;
pub mod dependent_joint;
pub mod independent_array_joint;
pub mod independent_joint;
pub mod instant;

pub use converted::*;
pub use dependent_joint::*;
pub use independent_array_joint::*;
pub use independent_joint::*;
pub use instant::*;
use rand::prelude::*;
use std::{error::Error, fmt::Debug};

pub trait RandomVariable: Clone + Debug + PartialEq {}
impl<T> RandomVariable for T where T: Clone + Debug + PartialEq {}

#[derive(thiserror::Error, Debug)]
pub enum DistributionError {
  #[error("params are not set")]
  ParamsAreNotSet,
}

/// # Distribution
/// ![tex](https://latex.codecogs.com/svg.latex?p%28x%7C\mathbf{\theta}%29)
pub trait Distribution: Clone + Debug {
  type T: RandomVariable;
  type U: RandomVariable;

  fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>>;
  fn ln_p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, Box<dyn Error>> {
    Ok(self.p(x, theta)?.ln())
  }
  fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, Box<dyn Error>>;
}
